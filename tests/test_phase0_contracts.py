from __future__ import annotations

import pytest

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
)


def test_lung_and_breast_are_registered_profiles() -> None:
    assert set(list_profiles()) >= {"lung", "breast"}
    assert get_profile("lung").tcga_project == "TCGA-LUAD"
    assert get_profile("breast").tcga_project == "TCGA-BRCA"
    assert get_profile("LUNG").default_modality == "CT"
    assert get_profile("breast").default_modality == "MR"


def test_profiles_are_not_hardcoded_to_one_disease() -> None:
    assert LUNG_PROFILE.candidate_genes != BREAST_PROFILE.candidate_genes
    assert "EGFR" in LUNG_PROFILE.candidate_genes
    assert "ERBB2" in BREAST_PROFILE.candidate_genes
    assert LUNG_PROFILE.tcia_collection != BREAST_PROFILE.tcia_collection


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown disease profile"):
        get_profile("pancreas")


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
