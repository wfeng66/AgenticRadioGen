from __future__ import annotations

import hashlib
import re

from agentic_radiogen.schemas.contracts import DataRequest, ResearchQuestion
from agentic_radiogen.schemas.profiles import disease_hint_map, get_profile
from agentic_radiogen.data.disease_match import extract_disease_query


_ENDPOINT_HINTS = {
    "OS": ("survival", "overall survival", "os"),
    "PFS": ("progression", "pfs"),
    "subtype": ("subtype", "molecular subtype"),
}

# Prefer specific phrases over short aliases (e.g. NSCLC must not collapse to "lung"/LUAD).
_SPECIFIC_DISEASE_HINTS: tuple[tuple[str, str], ...] = (
    ("non-small cell lung cancer", "nsclc"),
    ("non small cell lung cancer", "nsclc"),
    ("non-small-cell lung cancer", "nsclc"),
    ("non-small cell lung", "nsclc"),
    ("non small cell lung", "nsclc"),
    ("non-small cell", "nsclc"),
    ("non small cell", "nsclc"),
    ("nsclc", "nsclc"),
)


def infer_disease(text: str) -> str | None:
    """Infer canonical disease from question text using the full TCGA alias map."""
    lowered = text.lower()
    match = re.search(r"\btcga-([a-z0-9]+)\b", lowered)
    if match:
        return f"tcga-{match.group(1)}"
    for hint, name in _SPECIFIC_DISEASE_HINTS:
        if re.search(rf"(?<![a-z0-9_-]){re.escape(hint)}(?![a-z0-9_-])", lowered):
            return name
    ranked: list[tuple[int, str]] = []
    for name, hints in disease_hint_map().items():
        for hint in hints:
            if re.search(rf"(?<![a-z0-9_-]){re.escape(hint)}(?![a-z0-9_-])", lowered):
                ranked.append((len(hint), name))
                break
    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]
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
        inferred = infer_disease(text)
        resolved = (disease or inferred or "").strip().lower()
        if not resolved and self.tcga_project:
            resolved = self.tcga_project.strip().lower()
        if not resolved:
            raise ValueError(
                "Cannot infer disease from the question. Pass --disease (e.g. gbm, pancreas, breast) "
                "or --tcga-project (e.g. TCGA-GBM)."
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
            genes=[],  # gene panel is chosen at plan/fetch time (cohort vs literature)
            endpoints=[
                name
                for name, hints in _ENDPOINT_HINTS.items()
                if any(h in text.lower() for h in hints) and name in profile.endpoints
            ],
            modality=self.modality or profile.default_modality,
        )

    def plan(self, question: ResearchQuestion) -> DataRequest:
        profile = get_profile(
            question.disease,
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
        disease_query = extract_disease_query(question.text) or question.disease
        keyword_match = profile.tcia_collection == "KEYWORD" or profile.name == "nsclc"
        return DataRequest(
            question_id=_question_id(question.text, profile.name),
            disease=profile.name,
            tcga_project=profile.tcga_project,
            tcia_collection=profile.tcia_collection,
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
