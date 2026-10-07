from __future__ import annotations

import re

from pydantic import BaseModel, Field

# Default literature gene panel when --genes auto (cohort discovery is the default).
PANCAN_GENES = [
    "TP53",
    "KRAS",
    "PIK3CA",
    "PTEN",
    "EGFR",
    "BRAF",
    "IDH1",
    "CDKN2A",
    "MYC",
    "RB1",
    "ERBB2",
    "ALK",
    "STK11",
    "KEAP1",
]

_PANCAN_ALIASES = {
    "tp53": "TP53",
    "p53": "TP53",
    "kras": "KRAS",
    "pik3ca": "PIK3CA",
    "pten": "PTEN",
    "egfr": "EGFR",
    "braf": "BRAF",
    "idh1": "IDH1",
    "cdkn2a": "CDKN2A",
    "myc": "MYC",
    "rb1": "RB1",
    "her2": "ERBB2",
    "erbb2": "ERBB2",
    "alk": "ALK",
}


class DiseaseProfile(BaseModel):
    """Runtime disease plan. Sources are resolved by keyword match, not a fixed register."""

    name: str
    tcga_project: str
    tcia_collection: str
    default_modality: str
    candidate_genes: list[str]
    gene_aliases: dict[str, str] = Field(default_factory=dict)
    immune_signatures: list[str] = Field(default_factory=list)
    endpoints: list[str] = Field(default_factory=list)
    imaging_feature_hints: list[str] = Field(default_factory=list)
    disease_query: str = ""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "disease"


def get_profile(
    name: str,
    *,
    tcga_project: str | None = None,
    tcia_collection: str | None = None,
    modality: str | None = None,
    genes: list[str] | None = None,
) -> DiseaseProfile:
    """Build a disease plan from free text / site code / TCGA id.

    There is no curated disease register. LiveCatalog matches the disease query
    against TCIA collection names and GDC project metadata.
    """
    # Lazy import avoids circular import: data.catalog → profiles → data.*
    from agentic_radiogen.data.disease_match import (
        extract_disease_query,
        guess_modality_for_disease,
        normalize_question_text,
    )

    raw = normalize_question_text(name).strip()
    key = raw.lower()
    if key.startswith("tcga-"):
        query = raw.upper() if raw.upper().startswith("TCGA-") else f"TCGA-{key.split('-', 1)[1].upper()}"
        slug = query.lower().replace("tcga-", "")
        default_mod = modality or guess_modality_for_disease(query)
    else:
        # Prefer a compact disease phrase if the caller passed a whole question.
        query = extract_disease_query(raw) or raw.replace("_", " ")
        slug = _slug(query)
        default_mod = modality or guess_modality_for_disease(query)

    gene_list = list(genes or PANCAN_GENES)
    alias_map = dict(_PANCAN_ALIASES)
    alias_map.update({g.lower(): g for g in gene_list})

    profile = DiseaseProfile(
        name=slug,
        # KEYWORD → LiveCatalog discovers TCIA/GDC sources by matching ``disease_query``.
        tcga_project="KEYWORD",
        tcia_collection="KEYWORD",
        default_modality=default_mod.strip().upper(),
        candidate_genes=gene_list,
        gene_aliases=alias_map,
        immune_signatures=["TMB", "CD8_Tcell"],
        endpoints=["OS", "PFS", "subtype"],
        imaging_feature_hints=["shape", "texture", "intensity"],
        disease_query=query,
    )

    updates: dict = {}
    if tcga_project:
        project = tcga_project.strip().upper()
        if not project.startswith("TCGA-") and not project.startswith("CPTAC"):
            project = f"TCGA-{project}"
        updates["tcga_project"] = project
        if not tcia_collection:
            updates["tcia_collection"] = project
    if tcia_collection:
        updates["tcia_collection"] = tcia_collection.strip()
    if modality:
        updates["default_modality"] = modality.strip().upper()
    if genes:
        updates["candidate_genes"] = list(genes)
        alias_map = dict(profile.gene_aliases)
        alias_map.update({g.lower(): g for g in genes})
        updates["gene_aliases"] = alias_map
    return profile.model_copy(update=updates) if updates else profile


def list_profiles() -> list[str]:
    """No fixed register — return example disease phrases agents can match."""
    return [
        "lung cancer",
        "nsclc",
        "breast cancer",
        "brain cancer",
        "bone cancer",
        "glioblastoma",
        "pancreatic cancer",
        "kidney cancer",
        "prostate cancer",
        "colon cancer",
        "liver cancer",
        "ovarian cancer",
        "melanoma",
    ]


def list_projects() -> list[str]:
    """Example TCGA project ids; live runs discover projects via GDC keyword match."""
    return [
        "TCGA-LUAD",
        "TCGA-LUSC",
        "TCGA-BRCA",
        "TCGA-GBM",
        "TCGA-LGG",
        "TCGA-KIRC",
        "TCGA-PRAD",
        "TCGA-PAAD",
        "TCGA-SARC",
        "CPTAC-3",
    ]


def disease_hint_map() -> dict[str, tuple[str, ...]]:
    """Deprecated no-op kept for imports; inference uses extract_disease_query."""
    return {}


# Back-compat names used by older tests/docs (keyword profiles, not a register).
LUNG_PROFILE = get_profile("lung cancer")
BREAST_PROFILE = get_profile("breast cancer")
