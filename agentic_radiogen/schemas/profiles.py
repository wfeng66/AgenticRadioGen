from __future__ import annotations

from pydantic import BaseModel, Field


class DiseaseProfile(BaseModel):
    """Reusable disease configuration. Agents stay generic; only this changes."""

    name: str
    tcga_project: str
    tcia_collection: str
    default_modality: str
    candidate_genes: list[str]
    gene_aliases: dict[str, str] = Field(default_factory=dict)
    immune_signatures: list[str] = Field(default_factory=list)
    endpoints: list[str] = Field(default_factory=list)
    imaging_feature_hints: list[str] = Field(default_factory=list)


LUNG_PROFILE = DiseaseProfile(
    name="lung",
    tcga_project="TCGA-LUAD",
    tcia_collection="TCGA-LUAD",
    default_modality="CT",
    candidate_genes=["EGFR", "KRAS", "TP53"],
    gene_aliases={"egfr": "EGFR", "kras": "KRAS", "tp53": "TP53", "p53": "TP53"},
    immune_signatures=["TMB", "CD8_Tcell"],
    endpoints=["OS", "PFS", "subtype"],
    imaging_feature_hints=["shape", "texture", "intensity"],
)

BREAST_PROFILE = DiseaseProfile(
    name="breast",
    tcga_project="TCGA-BRCA",
    tcia_collection="TCGA-BRCA",
    default_modality="MR",
    candidate_genes=["BRCA1", "BRCA2", "ESR1", "ERBB2", "PGR"],
    gene_aliases={
        "brca1": "BRCA1",
        "brca2": "BRCA2",
        "esr1": "ESR1",
        "er": "ESR1",
        "her2": "ERBB2",
        "erbb2": "ERBB2",
        "pgr": "PGR",
        "pr": "PGR",
    },
    immune_signatures=["TMB", "immune_score"],
    endpoints=["OS", "PFS", "subtype"],
    imaging_feature_hints=["shape", "texture", "intensity"],
)

_REGISTRY: dict[str, DiseaseProfile] = {
    LUNG_PROFILE.name: LUNG_PROFILE,
    BREAST_PROFILE.name: BREAST_PROFILE,
}


def get_profile(name: str) -> DiseaseProfile:
    key = name.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown disease profile '{name}'. Known: {known}")
    return _REGISTRY[key]


def list_profiles() -> list[str]:
    return sorted(_REGISTRY)
