from __future__ import annotations

import pytest

from agentic_radiogen.agents.orchestrator import OrchestratorAgent, infer_disease
from agentic_radiogen.schemas.contracts import (
    DataRequest,
    GenomicMatrix,
    ImageBundle,
    LiteratureContext,
    OmicsBundle,
    RadiomicMatrix,
    RefinementDirective,
    ResearchQuestion,
)
from agentic_radiogen.schemas.profiles import (
    BREAST_PROFILE,
    LUNG_PROFILE,
    get_profile,
    list_profiles,
    list_projects,
)


def test_profiles_are_keyword_plans_not_a_fixed_register() -> None:
    lung = get_profile("lung cancer")
    breast = get_profile("breast cancer")
    assert lung.tcia_collection == "KEYWORD"
    assert breast.tcia_collection == "KEYWORD"
    assert lung.tcga_project == "KEYWORD"
    assert lung.default_modality == "CT"
    assert breast.default_modality == "MR"
    assert lung.disease_query == "lung cancer"
    assert "lung cancer" in list_profiles()
    assert "TCGA-GBM" in list_projects()


def test_free_text_and_tcga_ids_use_keyword_matching() -> None:
    brain = get_profile("brain cancer")
    assert brain.name == "brain_cancer"
    assert brain.tcia_collection == "KEYWORD"
    assert brain.default_modality == "MR"
    gbm = get_profile("glioblastoma")
    assert gbm.tcia_collection == "KEYWORD"
    assert "glioblastoma" in gbm.disease_query
    xyz = get_profile("TCGA-XYZ")
    assert xyz.tcia_collection == "KEYWORD"
    assert "TCGA-XYZ" in xyz.disease_query.upper()


def test_profile_overrides() -> None:
    profile = get_profile("lung cancer", modality="MR", genes=["EGFR", "ALK"])
    assert profile.default_modality == "MR"
    assert profile.candidate_genes == ["EGFR", "ALK"]
    forced = get_profile(
        "lung cancer",
        tcga_project="TCGA-LUAD",
        tcia_collection="TCGA-LUAD",
    )
    assert forced.tcga_project == "TCGA-LUAD"
    assert forced.tcia_collection == "TCGA-LUAD"


def test_compat_lung_breast_profile_exports() -> None:
    assert LUNG_PROFILE.tcia_collection == "KEYWORD"
    assert BREAST_PROFILE.tcia_collection == "KEYWORD"
    assert LUNG_PROFILE.default_modality == "CT"
    assert BREAST_PROFILE.default_modality == "MR"


def test_orchestrator_infers_disease_phrases() -> None:
    assert infer_disease("IDH1 in glioblastoma MRI") == "glioblastoma"
    assert infer_disease("KRAS in pancreatic cancer CT") == "pancreatic cancer"
    assert infer_disease("features in TCGA-OV") == "tcga-ov"
    req = OrchestratorAgent().parse_and_plan(
        "Which MRI features associate with IDH1 in glioblastoma?"
    )
    assert req.disease == "glioblastoma"
    assert req.tcga_project == "KEYWORD"
    assert req.tcia_collection == "KEYWORD"
    assert req.filters["keyword_match"] is True
    assert req.modality == "MR"


def test_same_contracts_describe_lung_and_breast_requests() -> None:
    lung = DataRequest(
        question_id="q_lung",
        disease="lung_cancer",
        tcga_project="KEYWORD",
        tcia_collection="KEYWORD",
        modality="CT",
        genes=["EGFR"],
        filters={"full_archive": False, "keyword_match": True},
        max_patients=12,
    )
    breast = lung.model_copy(
        update={
            "question_id": "q_breast",
            "disease": "breast_cancer",
            "modality": "MR",
            "genes": ["ERBB2"],
        }
    )
    assert lung.disease != breast.disease
    assert lung.filters["full_archive"] is False
    assert breast.max_patients == 12


def test_matcher_outputs_are_separate_image_and_omics_bundles() -> None:
    images = ImageBundle(patient_ids=["P1"], series=[])
    omics = OmicsBundle(patient_ids=["P1"])
    assert images.model_dump().keys() != omics.model_dump().keys()
    assert "series" in images.model_dump()
    assert "mutations" in omics.model_dump()


def test_join_and_loop_contracts_exist() -> None:
    radio = RadiomicMatrix(patient_ids=["P1"], feature_names=["f"], features={"P1": {"f": 1.0}})
    geno = GenomicMatrix(patient_ids=["P1"], feature_names=["EGFR_mut"], features={"P1": {"EGFR_mut": 1}})
    lit = LiteratureContext(unverified=["f ~ EGFR_mut"])
    question = ResearchQuestion(text="demo", disease="lung_cancer")
    assert radio.feature_names and geno.feature_names
    assert question.disease == "lung_cancer"
    assert "EGFR" in lit.unverified[0]
    directive = RefinementDirective.model_validate(
        {"action": "stop", "reason": "cap", "human_review_required": True}
    )
    assert directive.auto_promoted == []
