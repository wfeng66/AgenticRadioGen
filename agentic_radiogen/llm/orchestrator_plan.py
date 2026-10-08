from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_radiogen.llm.client import LlmClient


class OrchestratorLlmPlan(BaseModel):
    """Structured planning output from the Orchestrator LLM."""

    disease_query: str = Field(
        ...,
        description="Short disease phrase for TCIA/GDC keyword match, e.g. 'lung cancer'.",
    )
    modality: str | None = Field(
        default=None,
        description="Preferred imaging modality: CT, MR, PT, etc. Null if unspecified.",
    )
    endpoints: list[str] = Field(
        default_factory=list,
        description="Subset of OS, PFS, subtype mentioned in the question.",
    )
    rationale: str = Field(default="", description="One-sentence reason for the disease choice.")


_SYSTEM = """You are the Orchestrator for an agentic radiogenomics pipeline.
Extract a planning JSON object from the research question.

Rules:
- disease_query: a concise clinical disease phrase used to match TCIA collections and GDC projects
  (examples: "lung cancer", "non-small cell lung cancer", "breast cancer", "glioblastoma", "bone cancer").
  Prefer the disease named in the question; do not invent a TCGA project id unless the question names one.
- modality: CT, MR, PT, or null if the question does not imply one.
- endpoints: only from ["OS", "PFS", "subtype"] when clearly asked (survival → OS).
- Return JSON only. No markdown.
"""


def plan_with_llm(
    question: str,
    *,
    client: LlmClient | None = None,
    disease_hint: str | None = None,
) -> OrchestratorLlmPlan:
    """Ask the free-tier LLM to parse disease / modality / endpoints from the question."""
    llm = client or LlmClient()
    hint = f"\nCaller disease hint: {disease_hint}" if disease_hint else ""
    user = (
        f"Research question:\n{question.strip()}{hint}\n\n"
        "Respond with JSON keys: disease_query, modality, endpoints, rationale."
    )
    data: dict[str, Any] = llm.complete_json(system=_SYSTEM, user=user)
    modality = data.get("modality")
    if isinstance(modality, str):
        modality = modality.strip().upper() or None
        if modality in {"NULL", "NONE", "N/A"}:
            modality = None
    endpoints = data.get("endpoints") or []
    if isinstance(endpoints, str):
        endpoints = [endpoints]
    allowed = {"OS", "PFS", "subtype"}
    endpoints = [str(e).strip() for e in endpoints if str(e).strip() in allowed]
    disease_query = str(data.get("disease_query") or "").strip()
    if not disease_query:
        raise ValueError("LLM plan missing disease_query")
    return OrchestratorLlmPlan(
        disease_query=disease_query,
        modality=modality,
        endpoints=endpoints,
        rationale=str(data.get("rationale") or "").strip(),
    )
