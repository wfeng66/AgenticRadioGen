from __future__ import annotations

from agentic_radiogen.data.disease_match import (
    expand_cohort_label,
    segmentation_disease_label,
)
from agentic_radiogen.imaging.tumor_models.registry import list_models_for_disease


def test_expand_tcga_lusc() -> None:
    assert "squamous" in expand_cohort_label("TCGA-LUSC").lower()
    assert "TCGA-LUSC" in expand_cohort_label("TCGA-LUSC")


def test_question_disease_is_matching_anchor() -> None:
    label = segmentation_disease_label(
        question_disease="lung cancer",
        gdc_project="TCGA-LUSC",
        tcia_collection="TCGA-LUSC",
        primary_diagnosis="Basaloid squamous cell carcinoma",
    )
    # Question site leads — used for organ/tumor model filtering.
    assert label.lower().startswith("lung cancer")
    assert "TCGA-LUSC" in label
    assert "basaloid" in label.lower()


def test_question_disease_keeps_lung_models_not_kidney() -> None:
    label = segmentation_disease_label(
        question_disease="lung cancer",
        gdc_project="TCGA-LUSC",
        primary_diagnosis="Basaloid squamous cell carcinoma",
    )
    ids = {m.model_id for m in list_models_for_disease(label, "CT", direct_only=True)}
    assert "ts_lung" in ids or "nnunet_msd_lung" in ids
    assert "ts_kidney" not in ids


def test_luad_vs_lusc() -> None:
    luad = segmentation_disease_label(
        question_disease="lung cancer", gdc_project="TCGA-LUAD"
    )
    lusc = segmentation_disease_label(
        question_disease="lung cancer", gdc_project="TCGA-LUSC"
    )
    assert luad.lower().startswith("lung cancer")
    assert lusc.lower().startswith("lung cancer")
    assert "adenocarcinoma" in luad.lower()
    assert "squamous" in lusc.lower()
    assert luad != lusc


def test_falls_back_to_cohort_when_no_question() -> None:
    label = segmentation_disease_label(
        question_disease="",
        gdc_project="TCGA-LUSC",
        primary_diagnosis="",
    )
    assert "TCGA-LUSC" in label
    assert "squamous" in label.lower()
