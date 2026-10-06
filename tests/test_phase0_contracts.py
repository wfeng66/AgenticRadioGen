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


def test_lung_and_breast_are_registered_profiles() -> None:
    assert set(list_profiles()) >= {"lung", "breast"}
    assert get_profile("lung").tcga_project == "TCGA-LUAD"
    assert get_profile("breast").tcga_project == "TCGA-BRCA"
    assert get_profile("LUNG").default_modality == "CT"
    assert get_profile("breast").default_modality == "MR"


def test_all_major_tcga_diseases_are_available() -> None:
    assert "TCGA-GBM" in list_projects()
    assert "CPTAC-3" in list_projects()
    assert get_profile("pancreas").tcga_project == "CPTAC-3"
    assert get_profile("pancreas").tcia_collection == "CPTAC-PDA"
    assert get_profile("pdac").name == "pancreas"
    assert get_profile("gbm").default_modality == "MR"
    assert get_profile("TCGA-KIRC").name == "kidney"
    assert len(list_profiles()) >= 30


def test_profiles_are_not_hardcoded_to_one_disease() -> None:
    assert LUNG_PROFILE.candidate_genes != BREAST_PROFILE.candidate_genes
    assert "EGFR" in LUNG_PROFILE.candidate_genes
    assert "ERBB2" in BREAST_PROFILE.candidate_genes
    assert LUNG_PROFILE.tcia_collection != BREAST_PROFILE.tcia_collection


def test_unknown_site_code_builds_dynamic_profile() -> None:
    profile = get_profile("TCGA-XYZ")
    assert profile.tcga_project == "TCGA-XYZ"
    assert profile.tcia_collection == "TCGA-XYZ"
    assert "TP53" in profile.candidate_genes


def test_profile_overrides() -> None:
    profile = get_profile("lung", modality="MR", genes=["EGFR", "ALK"])
    assert profile.default_modality == "MR"
    assert profile.candidate_genes == ["EGFR", "ALK"]


def test_orchestrator_infers_many_diseases() -> None:
    assert infer_disease("IDH1 in glioblastoma MRI") == "gbm"
    assert infer_disease("KRAS in pancreatic cancer CT") == "pancreas"
    assert infer_disease("features in TCGA-OV") == "tcga-ov"
    req = OrchestratorAgent().parse_and_plan(
        "Which MRI features associate with IDH1 in glioblastoma?"
    )
    assert req.disease == "gbm"
    assert req.tcga_project == "TCGA-GBM"
    assert req.modality == "MR"


def test_same_contracts_describe_lung_and_breast_requests() -> None:
    lung = DataRequest(
        question_id="q_lung",
        disease="lung",
        tcga_project=LUNG_PROFILE.tcga_project,
        tcia_collection=LUNG_PROFILE.tcia_collection,
        modality="CT",
        genes=["EGFR"],
        filters={"full_archive": False},
        max_patients=12,
    )
    breast = lung.model_copy(
        update={
            "question_id": "q_breast",
            "disease": "breast",
            "tcga_project": BREAST_PROFILE.tcga_project,
            "tcia_collection": BREAST_PROFILE.tcia_collection,
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
    question = ResearchQuestion(text="demo", disease="lung")
    assert radio.feature_names and geno.feature_names
    assert question.disease == "lung"
    assert "EGFR" in lit.unverified[0]
    directive = RefinementDirective.model_validate(
        {"action": "stop", "reason": "cap", "human_review_required": True}
    )
    assert directive.auto_promoted == []
