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


def extract_pairs_with_llm(
    articles: list[PubMedArticle],
    *,
    disease: str,
    question: str,
    client: LlmClient | None = None,
) -> LiteratureExtraction:
    """Ask the LLM to pull radiomic–gene pairs from fetched abstracts."""
    llm = client or LlmClient()
    if not articles:
        return LiteratureExtraction(pairs=[], rationale="No PubMed articles to extract from")

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
        f"Research question: {question.strip()}\n\n"
        "Articles:\n"
        + "\n\n".join(blocks)
        + "\n\nRespond with JSON keys: pairs, rationale."
    )
    # Abstracts are longer than orchestrator prompts; allow a bit more time.
    data: dict[str, Any] = llm.complete_json(system=_SYSTEM, user=user, timeout=60)
    raw_pairs = data.get("pairs") or []
    if not isinstance(raw_pairs, list):
        raw_pairs = []
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
        # Keep human-readable variants too for matching
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
    return LiteratureExtraction(
        pairs=pairs,
        rationale=str(data.get("rationale") or "").strip(),
    )
