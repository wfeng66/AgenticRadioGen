"""PubMed E-utilities client for literature search (no API key required)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from agentic_radiogen.data.http_json import RemoteApiError, get_bytes, get_json

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# NCBI esearch allows retmax up to 10_000 per request.
_MAX_RETMAX = 10_000
# Above this, OR-ing every symbol blows PubMed query length; omit gene clause.
_MAX_GENES_IN_QUERY = 500


@dataclass(frozen=True)
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""


class PubMedClient:
    """Thin NCBI E-utilities wrapper (esearch + efetch)."""

    def __init__(
        self,
        *,
        getter: Callable[..., Any] | None = None,
        bytes_getter: Callable[..., bytes] | None = None,
        timeout: int = 45,
    ) -> None:
        self._get = getter or get_json
        self._get_bytes = bytes_getter or get_bytes
        self.timeout = timeout

    def search(
        self,
        query: str,
        *,
        max_results: int = 500,
    ) -> list[str]:
        """Return PubMed IDs for a free-text query."""
        q = (query or "").strip()
        if not q:
            return []
        retmax = max(1, min(int(max_results), _MAX_RETMAX))
        data = self._get(
            f"{_EUTILS}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": q,
                "retmax": str(retmax),
                "retmode": "json",
                "sort": "relevance",
            },
            timeout=self.timeout,
        )
        try:
            idlist = data.get("esearchresult", {}).get("idlist", [])
        except AttributeError as exc:
            raise RemoteApiError("Unexpected PubMed esearch response") from exc
        return [str(x) for x in idlist if str(x).strip()]

    def fetch(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch title + abstract for PubMed IDs (batched)."""
        ids = [p for p in pmids if p]
        if not ids:
            return []
        # efetch is happier with modest batches
        out: list[PubMedArticle] = []
        batch_size = 200
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            raw = self._get_bytes(
                f"{_EUTILS}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(chunk),
                    "retmode": "xml",
                },
                timeout=self.timeout,
            )
            out.extend(_parse_pubmed_xml(raw))
        return out


def build_radiogenomics_query(
    disease: str,
    question: str | None = None,
    *,
    genes: Iterable[str] | None = None,
) -> str:
    """Build a PubMed query biased toward radiomics ↔ mutation papers.

    Specific gene symbols are included only when ``genes`` is a non-empty
    explicit panel (all of them, or none if the panel is too large for PubMed).
    Never hardcode a shortlist like EGFR/KRAS — that biases retrieval and the LLM.
    """
    disease = (disease or "").strip() or "cancer"
    gene_clause = _gene_or_clause(genes)

    # Keep only non-gene topical cues from the free-text question (no EGFR/KRAS/…).
    extras = ""
    if question:
        stop = {
            "which",
            "what",
            "how",
            "are",
            "the",
            "with",
            "specific",
            "associated",
            "association",
            "imaging",
            "features",
            "feature",
            "genomic",
            "alterations",
            "alteration",
            "mutation",
            "mutations",
            "in",
            "of",
            "and",
            "or",
            "a",
            "an",
            "to",
            "for",
            "by",
            "from",
            "on",
            "this",
            "these",
            "those",
            "is",
            "do",
            "does",
        }
        # Gene-like tokens from the question must not bias the search unless they
        # are part of an explicit ``genes`` panel (handled above).
        words = re.findall(r"[A-Za-z0-9\-+]+", question.lower())
        keep = [
            w
            for w in words
            if w not in stop
            and len(w) > 2
            and w
            in {
                "radiomic",
                "radiomics",
                "texture",
                "biomarker",
                "genotype",
                "genomics",
            }
        ]
        if keep:
            extras = " " + " ".join(sorted(set(keep))[:6])

    mutation_block = (
        gene_clause
        if gene_clause
        else "(mutation OR genomic OR genotype OR radiogenomic OR radiogenomics)"
    )
    return (
        f"({disease}) AND (radiomic OR radiomics OR \"texture analysis\" OR "
        f"\"imaging biomarker\") AND {mutation_block}{extras}"
    )


def _gene_or_clause(genes: Iterable[str] | None) -> str:
    if not genes:
        return ""
    symbols: list[str] = []
    seen: set[str] = set()
    for raw in genes:
        g = str(raw or "").strip().upper().replace("HER2", "ERBB2")
        if g.endswith("_MUT") or g.endswith("_EXPR"):
            g = g.rsplit("_", 1)[0]
        if not g or g in seen:
            continue
        # Skip tokens that look like free text, not gene symbols
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9\-]{0,14}", g):
            continue
        seen.add(g)
        symbols.append(g)
    if not symbols:
        return ""
    if len(symbols) > _MAX_GENES_IN_QUERY:
        # All-or-nothing: too many symbols → omit gene clause entirely.
        return ""
    return "(" + " OR ".join(symbols) + ")"


def _parse_pubmed_xml(raw: bytes) -> list[PubMedArticle]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise RemoteApiError("Invalid PubMed XML") from exc
    articles: list[PubMedArticle] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
        title = " ".join((article.findtext(".//ArticleTitle") or "").split())
        abstract_bits = [
            " ".join((node.text or "").split())
            for node in article.findall(".//Abstract/AbstractText")
            if (node.text or "").strip()
        ]
        abstract = " ".join(abstract_bits)
        journal = (article.findtext(".//Journal/Title") or "").strip()
        year = (
            article.findtext(".//JournalIssue/PubDate/Year")
            or article.findtext(".//PubmedData/History/PubMedPubDate/Year")
            or ""
        ).strip()
        if not pmid and not title:
            continue
        articles.append(
            PubMedArticle(
                pmid=pmid,
                title=title or f"PMID:{pmid}",
                abstract=abstract,
                journal=journal,
                year=year,
            )
        )
    return articles
