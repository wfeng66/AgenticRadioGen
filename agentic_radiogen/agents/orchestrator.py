from __future__ import annotations

import hashlib
import re

from agentic_radiogen.data.disease_match import (
    extract_disease_query,
    normalize_question_text,
)
from agentic_radiogen.schemas.contracts import DataRequest, ResearchQuestion
from agentic_radiogen.schemas.profiles import get_profile


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


class OrchestratorAgent:
    """Turns a research question into a minimal DataRequest. Does not download data."""

    def __init__(
        self,
        *,
        tcga_project: str | None = None,
        tcia_collection: str | None = None,
        modality: str | None = None,
        genes: str | list[str] | None = None,
    ) -> None:
        # genes: None => all genes in paired genomics cohort
        # genes: "auto" => literature/profile candidate genes
        self.tcga_project = tcga_project
        self.tcia_collection = tcia_collection
        self.modality = modality
        self.genes = genes

    def parse(self, text: str, disease: str | None = None) -> ResearchQuestion:
        text = normalize_question_text(text)
        inferred = infer_disease(text)
        resolved = (disease or inferred or "").strip()
        if not resolved and self.tcga_project:
            resolved = self.tcga_project.strip()
        if not resolved:
            raise ValueError(
                "Cannot infer disease from the question. Name a disease in the question "
                "(e.g. 'brain cancer', 'NSCLC'), or pass --disease / --tcga-project."
            )
        profile = get_profile(
            resolved,
            tcga_project=self.tcga_project,
            tcia_collection=self.tcia_collection,
            modality=self.modality,
        )
        return ResearchQuestion(
            text=text,
            disease=profile.name,
            genes=[],
            endpoints=[
                name
                for name, hints in _ENDPOINT_HINTS.items()
                if any(h in text.lower() for h in hints) and name in profile.endpoints
            ],
            modality=self.modality or profile.default_modality,
        )

    def plan(self, question: ResearchQuestion) -> DataRequest:
        profile = get_profile(
            question.disease.replace("_", " "),
            tcga_project=self.tcga_project,
            tcia_collection=self.tcia_collection,
            modality=self.modality or question.modality,
        )
        # Preserve disease_query extracted from the full question when richer.
        from_text = extract_disease_query(question.text)
        if from_text:
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
        disease_query = profile.disease_query or from_text or question.disease.replace("_", " ")
        # Always match disease text to TCIA/GDC unless both sources are explicitly overridden.
        forced = bool(self.tcga_project and self.tcia_collection)
        keyword_match = not forced
        return DataRequest(
            question_id=_question_id(question.text, profile.name),
            disease=profile.name,
            tcga_project=("KEYWORD" if keyword_match else profile.tcga_project),
            tcia_collection=("KEYWORD" if keyword_match else profile.tcia_collection),
            modality=question.modality or profile.default_modality,
            genes=genes,
            clinical_fields=clinical_fields,
            immune_signatures=list(profile.immune_signatures),
            filters={
                "question_scoped": True,
                "full_archive": False,
                "discover_genes": discover,
                "gene_source": gene_source,
                "disease_query": disease_query,
                "keyword_match": keyword_match,
            },
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
