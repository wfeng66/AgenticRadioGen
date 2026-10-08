from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel, Field

from agentic_radiogen.llm.client import LlmClient


class MatcherLlmPlan(BaseModel):
    """LLM selection of TCIA/GDC sources for DataMatcher."""

    keywords: list[str] = Field(default_factory=list)
    tcia_collections: list[str] = Field(default_factory=list)
    gdc_projects: list[str] = Field(default_factory=list)
    rationale: str = ""


_SYSTEM = """You are the Data Matcher for an agentic radiogenomics pipeline.
Given a disease question and candidate TCIA collections / GDC projects, select the sources
best suited for paired imaging+genomics analysis.

Rules:
- Prefer collections/projects that can share patient IDs (TCGA-* names that exist on both sides).
- For breast cancer prefer TCGA-BRCA; for lung cancer prefer TCGA-LUAD and/or TCGA-LUSC;
  for brain/glioblastoma prefer TCGA-GBM / TCGA-LGG; for sarcoma/bone prefer TCGA-SARC / TARGET-OS when listed.
- You may select multiple sources. Prefer pairable TCGA sources over large non-TCGA imaging-only sets.
- keywords: short search terms useful for diagnosis/site matching (include organ + project codes).
- Only choose names from the provided candidate lists (exact strings).
- Return JSON only with keys: keywords, tcia_collections, gdc_projects, rationale.
"""


def plan_match_sources_with_llm(
    *,
    disease_query: str,
    modality: str,
    question: str | None,
    tcia_candidates: Iterable[str],
    gdc_candidates: Iterable[str],
    client: LlmClient | None = None,
) -> MatcherLlmPlan:
    """Ask the free-tier LLM which TCIA/GDC sources fit the disease request."""
    llm = client or LlmClient()
    tcia_list = [str(x) for x in tcia_candidates if str(x).strip()]
    gdc_list = [str(x) for x in gdc_candidates if str(x).strip()]
    user = (
        f"Disease query: {disease_query}\n"
        f"Modality: {modality}\n"
        f"Research question: {(question or disease_query).strip()}\n\n"
        f"TCIA collection candidates ({len(tcia_list)}):\n"
        + "\n".join(f"- {n}" for n in tcia_list[:60])
        + f"\n\nGDC project candidates ({len(gdc_list)}):\n"
        + "\n".join(f"- {n}" for n in gdc_list[:60])
        + "\n\nRespond with JSON: keywords, tcia_collections, gdc_projects, rationale."
    )
    data: dict[str, Any] = llm.complete_json(system=_SYSTEM, user=user)
    tcia_allowed = {n.lower(): n for n in tcia_list}
    gdc_allowed = {n.lower(): n for n in gdc_list}

    def _pick(raw: Any, allowed: dict[str, str]) -> list[str]:
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for item in raw:
            key = str(item).strip()
            hit = allowed.get(key.lower())
            if hit and hit not in out:
                out.append(hit)
        return out

    keywords_raw = data.get("keywords") or []
    if isinstance(keywords_raw, str):
        keywords_raw = [keywords_raw]
    keywords = [str(k).strip().lower() for k in keywords_raw if str(k).strip()]
    tcia = _pick(data.get("tcia_collections"), tcia_allowed)
    gdc = _pick(data.get("gdc_projects"), gdc_allowed)
    if not tcia and not gdc:
        raise ValueError("LLM matcher returned no valid TCIA/GDC sources from candidates")
    return MatcherLlmPlan(
        keywords=keywords,
        tcia_collections=tcia,
        gdc_projects=gdc,
        rationale=str(data.get("rationale") or "").strip(),
    )
