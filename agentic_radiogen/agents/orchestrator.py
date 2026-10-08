from __future__ import annotations

import hashlib
import re
from typing import Any

from agentic_radiogen.data.disease_match import (
    extract_disease_query,
    normalize_question_text,
)
from agentic_radiogen.llm.client import LlmClient, LlmConfig, resolve_llm_config
from agentic_radiogen.llm.orchestrator_plan import OrchestratorLlmPlan, plan_with_llm
from agentic_radiogen.schemas.contracts import DataRequest, ResearchQuestion
from agentic_radiogen.schemas.profiles import get_profile
from agentic_radiogen.util.progress import log


_ENDPOINT_HINTS = {
    "OS": ("survival", "overall survival", "os"),
    "PFS": ("progression", "pfs"),
    "subtype": ("subtype", "molecular subtype"),
}


def infer_disease(text: str) -> str | None:
    """Infer a disease query phrase from free text (no disease register)."""
    lowered = normalize_question_text(text).lower()
    match = re.search(r"\btcga-([a-z0-9]+)\b", lowered)
    if match:
        return f"tcga-{match.group(1)}"
    return extract_disease_query(text)


def _question_id(text: str, disease: str) -> str:
    digest = hashlib.sha1(f"{disease}|{text.strip().lower()}".encode()).hexdigest()[:12]
    return f"q_{digest}"


def _rule_endpoints(text: str, allowed: list[str]) -> list[str]:
    lowered = text.lower()
    return [
        name
        for name, hints in _ENDPOINT_HINTS.items()
        if any(h in lowered for h in hints) and name in allowed
    ]


class OrchestratorAgent:
    """Turns a research question into a minimal DataRequest. Does not download data.

    Uses a free-tier LLM (Gemini 2.5 Flash by default, else Groq gpt-oss-120b) when an
    API key is available; falls back to rule-based parsing otherwise.
    """

    def __init__(
        self,
        *,
        tcga_project: str | None = None,
        tcia_collection: str | None = None,
        modality: str | None = None,
        genes: str | list[str] | None = None,
        use_llm: bool | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_client: LlmClient | None = None,
    ) -> None:
        # genes: None => all genes in paired genomics cohort
        # genes: "auto" => literature/profile candidate genes
        self.tcga_project = tcga_project
        self.tcia_collection = tcia_collection
        self.modality = modality
        self.genes = genes
        self._llm_plan: OrchestratorLlmPlan | None = None
        self._orchestrator_backend = "rules"
        if llm_client is not None:
            self._llm = llm_client
            self._use_llm = True
        else:
            config = resolve_llm_config(
                provider=llm_provider,
                model=llm_model,
                enabled=use_llm,
            )
            self._llm = LlmClient(config)
            # None => auto when a key exists; True requires LLM; False forces rules.
            if use_llm is False:
                self._use_llm = False
            elif use_llm is True:
                self._use_llm = True
            else:
                self._use_llm = self._llm.available

    @property
    def llm_config(self) -> LlmConfig:
        return self._llm.config

    @property
    def last_backend(self) -> str:
        return self._orchestrator_backend

    def parse(self, text: str, disease: str | None = None) -> ResearchQuestion:
        text = normalize_question_text(text)
        self._llm_plan = None
        self._orchestrator_backend = "rules"

        llm_disease: str | None = None
        llm_modality: str | None = None
        llm_endpoints: list[str] | None = None
        if self._use_llm and not disease:
            try:
                log(
                    f"[orchestrator] Calling LLM "
                    f"({self._llm.config.provider}/{self._llm.config.model}) "
                    "(~25s timeout, then rules fallback)..."
                )
                plan = plan_with_llm(text, client=self._llm, disease_hint=disease)
                self._llm_plan = plan
                llm_disease = plan.disease_query.strip() or None
                llm_modality = plan.modality
                llm_endpoints = list(plan.endpoints)
                self._orchestrator_backend = (
                    f"llm:{self._llm.config.provider}:{self._llm.config.model}"
                )
                log(
                    f"[orchestrator] LLM ({self._llm.config.provider}/{self._llm.config.model}) "
                    f"disease={plan.disease_query!r} modality={plan.modality!r}"
                )
            except Exception as exc:  # noqa: BLE001 — fall back to rules
                from agentic_radiogen.llm.client import _redact_secrets

                log(
                    f"[orchestrator] LLM planning failed ({_redact_secrets(str(exc))}); "
                    "using rules"
                )
                self._orchestrator_backend = "rules"

        inferred = llm_disease or infer_disease(text)
        resolved = (disease or inferred or "").strip()
        if not resolved and self.tcga_project:
            resolved = self.tcga_project.strip()
        if not resolved:
            hint = ""
            if self._use_llm and not self._llm.available:
                hint = (
                    " LLM orchestrator is enabled but no API key was found "
                    "(set GEMINI_API_KEY from Google AI Studio, free tier)."
                )
            raise ValueError(
                "Cannot infer disease from the question. Name a disease in the question "
                "(e.g. 'brain cancer', 'NSCLC'), or pass --disease / --tcga-project."
                f"{hint}"
            )
        profile = get_profile(
            resolved,
            tcga_project=self.tcga_project,
            tcia_collection=self.tcia_collection,
            modality=self.modality or llm_modality,
        )
        endpoints = llm_endpoints if llm_endpoints is not None else _rule_endpoints(
            text, profile.endpoints
        )
        endpoints = [e for e in endpoints if e in profile.endpoints]
        return ResearchQuestion(
            text=text,
            disease=profile.name,
            genes=[],
            endpoints=endpoints,
            modality=self.modality or llm_modality or profile.default_modality,
        )

    def plan(self, question: ResearchQuestion) -> DataRequest:
        profile = get_profile(
            question.disease.replace("_", " "),
            tcga_project=self.tcga_project,
            tcia_collection=self.tcia_collection,
            modality=self.modality or question.modality,
        )
        # Prefer LLM disease phrase, then regex extraction from the question text.
        from_text = extract_disease_query(question.text)
        if self._llm_plan and self._llm_plan.disease_query:
            profile = get_profile(
                self._llm_plan.disease_query,
                tcga_project=self.tcga_project,
                tcia_collection=self.tcia_collection,
                modality=self.modality or question.modality,
            )
        elif from_text:
            profile = get_profile(
                from_text,
                tcga_project=self.tcga_project,
                tcia_collection=self.tcia_collection,
                modality=self.modality or question.modality,
            )
        mode = self._gene_mode(self.genes)
        if mode == "literature":
            genes = list(profile.candidate_genes)
            discover = False
            gene_source = "literature"
        else:
            genes = []
            discover = True
            gene_source = "cohort"
        clinical_fields = ["OS_time", "OS_event"] if "OS" in question.endpoints else []
        if "subtype" in question.endpoints:
            clinical_fields.append("subtype")
        disease_query = (
            (self._llm_plan.disease_query if self._llm_plan else None)
            or profile.disease_query
            or from_text
            or question.disease.replace("_", " ")
        )
        # Always match disease text to TCIA/GDC unless both sources are explicitly overridden.
        forced = bool(self.tcga_project and self.tcia_collection)
        keyword_match = not forced
        filters: dict[str, Any] = {
            "question_scoped": True,
            "full_archive": False,
            "discover_genes": discover,
            "gene_source": gene_source,
            "disease_query": disease_query,
            "keyword_match": keyword_match,
            "orchestrator_backend": self._orchestrator_backend,
        }
        if self._llm_plan and self._llm_plan.rationale:
            filters["orchestrator_rationale"] = self._llm_plan.rationale
        return DataRequest(
            question_id=_question_id(question.text, profile.name),
            disease=profile.name,
            tcga_project=("KEYWORD" if keyword_match else profile.tcga_project),
            tcia_collection=("KEYWORD" if keyword_match else profile.tcia_collection),
            modality=question.modality or profile.default_modality,
            genes=genes,
            clinical_fields=clinical_fields,
            immune_signatures=list(profile.immune_signatures),
            filters=filters,
            max_patients=24,
            max_genes=None,
            min_altered=1,
        )

    def parse_and_plan(self, text: str, disease: str | None = None) -> DataRequest:
        return self.plan(self.parse(text, disease=disease))

    @staticmethod
    def _gene_mode(genes: str | list[str] | None) -> str:
        if genes is None:
            return "cohort"
        if isinstance(genes, str):
            token = genes.strip().lower()
        elif len(genes) == 1:
            token = str(genes[0]).strip().lower()
        else:
            raise ValueError(
                "Manual gene lists are not supported. "
                "Omit --genes for all cohort mutations, or pass --genes auto for literature genes."
            )
        if token in {"", "cohort", "all"}:
            return "cohort"
        if token == "auto":
            return "literature"
        raise ValueError(
            f"Unknown --genes value '{genes}'. Use --genes auto (literature) "
            "or omit --genes (all genes in genomics cohort)."
        )
