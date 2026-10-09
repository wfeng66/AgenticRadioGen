"""PubMed E-utilities client for literature search (no API key required)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Callable

from agentic_radiogen.data.http_json import RemoteApiError, get_bytes, get_json

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


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
        max_results: int = 20,
    ) -> list[str]:
        """Return PubMed IDs for a free-text query."""
        q = (query or "").strip()
        if not q:
            return []
        data = self._get(
            f"{_EUTILS}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": q,
                "retmax": str(max(1, min(int(max_results), 50))),
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
        """Fetch title + abstract for PubMed IDs."""
        ids = [p for p in pmids if p]
        if not ids:
            return []
        raw = self._get_bytes(
            f"{_EUTILS}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
            },
            timeout=self.timeout,
        )
        return _parse_pubmed_xml(raw)


def build_radiogenomics_query(disease: str, question: str | None = None) -> str:
    """Build a PubMed query biased toward radiomics ↔ mutation papers."""
    disease = (disease or "").strip() or "cancer"
    # Keep disease tokens; drop ultra-generic words from the free-text question.
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
        words = re.findall(r"[A-Za-z0-9\-+]+", question.lower())
        keep = [w for w in words if w not in stop and len(w) > 2]
        # Prefer mutation/radiomics cues from the question if present.
        cues = [w for w in keep if w in {"radiomic", "radiomics", "mutation", "mutations", "egfr", "kras", "her2", "erbb2"}]
        if cues:
            extras = " " + " ".join(sorted(set(cues))[:6])
    return (
        f'({disease}) AND (radiomic OR radiomics OR "texture analysis" OR "imaging biomarker") '
        f"AND (mutation OR genomic OR genotype OR EGFR OR KRAS OR TP53 OR HER2 OR BRCA){extras}"
    )


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
