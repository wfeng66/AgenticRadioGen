from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResearchQuestion(BaseModel):
    text: str
    disease: str
    genes: list[str] = Field(default_factory=list)
    endpoints: list[str] = Field(default_factory=list)
    modality: str | None = None


class DataRequest(BaseModel):
    """Minimal, question-scoped fetch plan. Never a full-archive dump."""

    question_id: str
    disease: str
    tcga_project: str
    tcia_collection: str
    modality: str
    genes: list[str]
    clinical_fields: list[str] = Field(default_factory=list)
    immune_signatures: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    max_patients: int = 24
    max_genes: int | None = None
    min_altered: int = 1
    estimated_patient_count: int | None = None


class ImageSeriesRef(BaseModel):
    patient_id: str
    series_uid: str
    modality: str
    disease: str | None = None
    local_path: str | None = None
    precomputed_features: dict[str, float] = Field(default_factory=dict)
    volume_summary: dict[str, float] = Field(default_factory=dict)


class ImageBundle(BaseModel):
    patient_ids: list[str]
    series: list[ImageSeriesRef]
    metadata: dict[str, Any] = Field(default_factory=dict)


class OmicsBundle(BaseModel):
    patient_ids: list[str]
    expression: dict[str, dict[str, float]] = Field(default_factory=dict)
    mutations: dict[str, dict[str, int]] = Field(default_factory=dict)
    clinical: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RadiomicMatrix(BaseModel):
    patient_ids: list[str]
    feature_names: list[str]
    features: dict[str, dict[str, float]]


class GenomicMatrix(BaseModel):
    patient_ids: list[str]
    feature_names: list[str]
    features: dict[str, dict[str, float]]


class Association(BaseModel):
    imaging_feature: str
    genomic_feature: str
    effect_size: float
    p_value: float
    q_value: float
    n: int


class ModelMetrics(BaseModel):
    auroc: float | None = None
    c_index: float | None = None
    n_train: int = 0
    n_test: int = 0
    target: str | None = None


class ModelDiagnostics(BaseModel):
    patient_split_disjoint: bool = True
    collinear_pairs: list[tuple[str, str, float]] = Field(default_factory=list)
    multiple_testing_method: str = "fdr_bh"
    caveats: list[str] = Field(default_factory=list)


class ModelResult(BaseModel):
    associations: list[Association] = Field(default_factory=list)
    metrics: list[ModelMetrics] = Field(default_factory=list)
    diagnostics: ModelDiagnostics = Field(default_factory=ModelDiagnostics)
    promoted_findings: list[str] = Field(default_factory=list)


class LiteratureSupport(BaseModel):
    finding: str
    papers: list[str] = Field(default_factory=list)
    supported: bool = False
    contradiction: str | None = None


class LiteratureContext(BaseModel):
    supports: list[LiteratureSupport] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)
    proposed_refinements: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    # Prior radiomic–gene claims used for annotation (from PubMed+LLM or static corpus).
    prior_pairs: list[dict] = Field(default_factory=list)
    search_query: str | None = None
    literature_source: str | None = None


class LoopAction(str, Enum):
    FETCH_DIFFERENT_SUBSET = "fetch_different_subset"
    CHANGE_FEATURES = "change_features"
    STOP = "stop"


class RefinementDirective(BaseModel):
    action: LoopAction
    reason: str
    new_data_request: DataRequest | None = None
    human_review_required: bool = True
    auto_promoted: list[str] = Field(default_factory=list)


class MatchPreview(BaseModel):
    n_paired: int
    n_available: int | None = None
    patient_ids: list[str]
    disease: str
    modality: str
    genes: list[str]
    download_required: bool = True
