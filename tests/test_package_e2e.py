from __future__ import annotations

from agentic_radiogen import __main__ as cli
from tests.conftest import BREAST_QUESTION, LUNG_QUESTION


def test_all_stages_run_on_the_same_lung_question() -> None:
    common = dict(disease=None, catalog_name="demo", approve_download=False, max_patients=16)
    stage1 = cli.run_stage1(LUNG_QUESTION, **common)
    stage2 = cli.run_stage2(LUNG_QUESTION, **common)
    stage3 = cli.run_stage3(LUNG_QUESTION, **common)
    stage4 = cli.run_stage4(LUNG_QUESTION, **common, max_iterations=3)

    assert stage1["stage"] == 1
    assert stage1["fetch"]["status"] == "ok"
    assert stage1["paired_only"] is True
    assert set(stage1["fetch"]["patient_ids"]) == set(stage1["preview"]["patient_ids"])

    assert stage2["stage"] == 2
    assert stage2["joined"] is False
    assert stage2["specialists"]["status"] == "ok"
    assert "EGFR_mut" not in stage2["specialists"]["radiomics"]["feature_names"]
    assert "original_glcm_Entropy" not in stage2["specialists"]["genomics"]["feature_names"]

    assert stage3["stage"] == 3
    assert stage3["joined"] is True
    assert stage3["looped"] is False
    entropy = next(
        item
        for item in stage3["stats"]["associations"]
        if item["imaging_feature"] == "original_glcm_Entropy"
        and item["genomic_feature"] == "EGFR_mut"
    )
    assert entropy["q_value"] < 0.05
    assert stage3["stats"]["promoted_findings"] == []

    assert stage4["stage"] == 4
    assert stage4["looped"] is True
    assert stage4["stopped"] is True
    assert stage4["human_review_required"] is True
    assert stage4["auto_promoted"] == []
    assert stage4["directive"]["action"] == "stop"
    assert any(item["supported"] for item in stage4["literature"]["supports"])
    assert "EGFR_mut" in stage4["associations_by_gene"]
    assert stage4["mutation_prevalence"]
    assert stage4["patient_table"]["n_patients"] > 0


def test_stage4_also_runs_for_breast() -> None:
    payload = cli.run_stage4(
        BREAST_QUESTION,
        disease=None,
        catalog_name="demo",
        approve_download=False,
        max_patients=16,
        max_iterations=3,
    )
    assert payload["question"]["disease"] == "breast_cancer"
    assert payload["stopped"] is True
    assert payload["auto_promoted"] == []
    shape = next(
        item
        for item in payload["top_associations"]
        if item["imaging_feature"] == "original_shape_Sphericity"
        and item["genomic_feature"] == "ERBB2_mut"
    )
    assert shape["q_value"] < 0.05


def test_stage4_live_requires_approve_download() -> None:
    try:
        cli.run_stage4(
            LUNG_QUESTION,
            disease=None,
            catalog_name="live",
            approve_download=False,
            max_patients=2,
            max_iterations=1,
        )
    except ValueError as exc:
        assert "approve-download" in str(exc)
    else:
        raise AssertionError("live Stage 4 should require --approve-download")
