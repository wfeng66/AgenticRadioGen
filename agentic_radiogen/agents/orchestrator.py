from __future__ import annotations

import hashlib
import re

from agentic_radiogen.schemas.contracts import DataRequest, ResearchQuestion
from agentic_radiogen.schemas.profiles import DiseaseProfile, get_profile


_DISEASE_HINTS = {
    "lung": ("lung", "luad", "nsclc", "adenocarcinoma of the lung"),
    "breast": ("breast", "brca", "mammary"),
}

_ENDPOINT_HINTS = {
    "OS": ("survival", "overall survival", "os"),
    "PFS": ("progression", "pfs"),
    "subtype": ("subtype", "molecular subtype"),
}


def infer_disease(text: str) -> str | None:
    lowered = text.lower()
    for name, hints in _DISEASE_HINTS.items():
        if any(h in lowered for h in hints):
            return name
    return None


def _question_id(text: str, disease: str) -> str:
    digest = hashlib.sha1(f"{disease}|{text.strip().lower()}".encode()).hexdigest()[:12]
    return f"q_{digest}"


class OrchestratorAgent:
    """Turns a research question into a minimal DataRequest. Does not download data."""

    def parse(self, text: str, disease: str | None = None) -> ResearchQuestion:
        inferred = infer_disease(text)
        resolved = (disease or inferred or "").strip().lower()
        if not resolved:
            raise ValueError(
                "Cannot infer disease from the question. Pass disease= explicitly "
                "(lung and breast are the default profiles)."
            )
        profile = get_profile(resolved)
        genes = self._extract_genes(text, profile)
        endpoints = [
            name
            for name, hints in _ENDPOINT_HINTS.items()
            if any(h in text.lower() for h in hints) and name in profile.endpoints
        ]
        return ResearchQuestion(
            text=text,
            disease=profile.name,
            genes=genes,
            endpoints=endpoints,
            modality=profile.default_modality,
        )

    def plan(self, question: ResearchQuestion) -> DataRequest:
        profile = get_profile(question.disease)
        genes = question.genes or list(profile.candidate_genes)
        clinical_fields = ["OS_time", "OS_event"] if "OS" in question.endpoints else []
        if "subtype" in question.endpoints:
            clinical_fields.append("subtype")
        return DataRequest(
            question_id=_question_id(question.text, profile.name),
            disease=profile.name,
            tcga_project=profile.tcga_project,
            tcia_collection=profile.tcia_collection,
            modality=question.modality or profile.default_modality,
            genes=genes,
            clinical_fields=clinical_fields,
            immune_signatures=list(profile.immune_signatures),
            filters={"question_scoped": True, "full_archive": False},
            max_patients=24,
        )

    def parse_and_plan(self, text: str, disease: str | None = None) -> DataRequest:
        return self.plan(self.parse(text, disease=disease))

    @staticmethod
    def _extract_genes(text: str, profile: DiseaseProfile) -> list[str]:
        lowered = text.lower()
        found: list[str] = []
        for alias, canonical in profile.gene_aliases.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered) and canonical not in found:
                found.append(canonical)
        for gene in profile.candidate_genes:
            if re.search(rf"\b{re.escape(gene.lower())}\b", lowered) and gene not in found:
                found.append(gene)
        return found
