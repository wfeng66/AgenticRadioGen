"""LLM extraction of radiomic-feature ↔ gene-mutation pairs from paper abstracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_radiogen.data.pubmed_client import PubMedArticle
from agentic_radiogen.llm.client import LlmClient


class ExtractedLitPair(BaseModel):
    gene: str = Field(..., description="HGNC-like gene symbol, e.g. EGFR, KRAS, ERBB2.")
    radiomic_terms: list[str] = Field(
        default_factory=list,
        description="Radiomic family/terms mentioned, e.g. glcm, entropy, sphericity, firstorder mean.",
    )
    supports: bool = Field(
        default=True,
        description="True if the paper reports a positive/association; False if negative/null result.",
    )
    note: str = Field(default="", description="Short clause summarizing the claim.")
    paper_title: str = Field(default="")
    pmid: str = Field(default="")


class LiteratureExtraction(BaseModel):
    pairs: list[ExtractedLitPair] = Field(default_factory=list)
    rationale: str = Field(default="")


_SYSTEM = """You extract radiomic imaging feature ↔ gene mutation associations from PubMed abstracts.

Return JSON only:
{
  "pairs": [
    {
      "gene": "EGFR",
      "radiomic_terms": ["glcm", "entropy", "texture"],
      "supports": true,
      "note": "higher GLCM entropy associated with EGFR mutation",
      "paper_title": "...",
      "pmid": "12345678"
    }
  ],
  "rationale": "one sentence"
}

Rules:
- Only include pairs clearly suggested by the abstracts (radiomics/texture/shape/intensity ↔ gene mutation).
- gene: uppercase symbol (ERBB2 not HER2 when both appear; include HER2 as gene ERBB2).
- radiomic_terms: short lowercase tokens useful for matching PyRadiomics names
  (examples: glcm, glrlm, glszm, gldm, ngtdm, entropy, contrast, sphericity, elongation,
   flatness, firstorder, mean, skewness, kurtosis, volume, surface, wavelet, texture, shape).
- supports=false only when the abstract reports failure to associate / negative / conflicting finding.
- Prefer at most 3 pairs per paper. Skip papers with no radiomics–mutation claim.
- Copy paper_title and pmid from the supplied article metadata.
"""


_BATCH_SIZE = 20  # abstracts per LLM call (keeps prompts within context limits)


def extract_pairs_with_llm(
    articles: list[PubMedArticle],
    *,
    disease: str,
    question: str,
    client: LlmClient | None = None,
    batch_size: int = _BATCH_SIZE,
) -> LiteratureExtraction:
    """Ask the LLM to pull radiomic–gene pairs from fetched abstracts (batched)."""
    llm = client or LlmClient()
    if not articles:
        return LiteratureExtraction(pairs=[], rationale="No PubMed articles to extract from")

    all_pairs: list[ExtractedLitPair] = []
    rationales: list[str] = []
    size = max(1, int(batch_size))
    for start in range(0, len(articles), size):
        chunk = articles[start : start + size]
        data = _extract_batch(
            chunk,
            disease=disease,
            question=question,
            client=llm,
            batch_index=start // size + 1,
            batch_total=(len(articles) + size - 1) // size,
        )
        all_pairs.extend(_parse_pairs(data.get("pairs") or []))
        rat = str(data.get("rationale") or "").strip()
        if rat:
            rationales.append(rat)

    # De-duplicate by (gene, pmid, frozenset terms)
    dedup: dict[tuple[str, str, tuple[str, ...]], ExtractedLitPair] = {}
    for p in all_pairs:
        key = (p.gene, p.pmid or "", tuple(sorted(p.radiomic_terms)))
        dedup[key] = p
    merged = list(dedup.values())
    rationale = (
        f"Extracted from {len(articles)} abstracts in "
        f"{(len(articles) + size - 1) // size} batches; {len(merged)} unique pairs. "
        + ("; ".join(rationales[:3]) if rationales else "")
    ).strip()
    return LiteratureExtraction(pairs=merged, rationale=rationale)


def _extract_batch(
    articles: list[PubMedArticle],
    *,
    disease: str,
    question: str,
    client: LlmClient,
    batch_index: int,
    batch_total: int,
) -> dict[str, Any]:
    blocks: list[str] = []
    for i, art in enumerate(articles, start=1):
        abs_txt = (art.abstract or "").strip()
        if len(abs_txt) > 1200:
            abs_txt = abs_txt[:1200] + "..."
        blocks.append(
            f"[{i}] PMID={art.pmid}\n"
            f"Title: {art.title}\n"
            f"Year: {art.year} Journal: {art.journal}\n"
            f"Abstract: {abs_txt or '(no abstract)'}"
        )
    user = (
        f"Disease focus: {disease}\n"
        f"Research question: {question.strip()}\n"
        f"Batch: {batch_index}/{batch_total}\n\n"
        "Articles:\n"
        + "\n\n".join(blocks)
        + "\n\nRespond with JSON keys: pairs, rationale."
    )
    return client.complete_json(system=_SYSTEM, user=user, timeout=90)


def _parse_pairs(raw_pairs: Any) -> list[ExtractedLitPair]:
    if not isinstance(raw_pairs, list):
        return []
    pairs: list[ExtractedLitPair] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        gene = str(item.get("gene") or "").strip().upper().replace("HER2", "ERBB2")
        if gene == "HER2":
            gene = "ERBB2"
        terms = item.get("radiomic_terms") or []
        if isinstance(terms, str):
            terms = [terms]
        terms_norm = sorted(
            {
                str(t).strip().lower().replace("-", "").replace(" ", "")
                for t in terms
                if str(t).strip()
            }
        )
        terms_readable = sorted(
            {
                str(t).strip().lower().replace("-", "_")
                for t in (item.get("radiomic_terms") or [])
                if str(t).strip()
            }
            | set(terms_norm)
        )
        if not gene or not terms_readable:
            continue
        pairs.append(
            ExtractedLitPair(
                gene=gene,
                radiomic_terms=terms_readable,
                supports=bool(item.get("supports", True)),
                note=str(item.get("note") or "").strip(),
                paper_title=str(item.get("paper_title") or "").strip(),
                pmid=str(item.get("pmid") or "").strip(),
            )
        )
    return pairs
