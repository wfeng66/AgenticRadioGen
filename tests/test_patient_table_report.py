from __future__ import annotations

from agentic_radiogen import __main__ as cli
from agentic_radiogen.pipeline.report_table import build_patient_table, resolve_patient_table
from agentic_radiogen.schemas.contracts import (
    GenomicMatrix,
    ImageBundle,
    ImageSeriesRef,
    OmicsBundle,
    RadiomicMatrix,
)
from tests.conftest import LUNG_QUESTION


def test_patient_table_separates_radiomics_and_genomic_alterations() -> None:
    radio = RadiomicMatrix(
        patient_ids=["P1", "P2"],
        feature_names=["original_glcm_Entropy"],
        features={
            "P1": {"original_glcm_Entropy": 3.1},
            "P2": {"original_glcm_Entropy": 1.2},
        },
    )
    geno = GenomicMatrix(
        patient_ids=["P1", "P2"],
        feature_names=["EGFR_mut", "EGFR_expr", "OS_time"],
        features={
            "P1": {"EGFR_mut": 1.0, "EGFR_expr": 6.0, "OS_time": 20.0},
            "P2": {"EGFR_mut": 0.0, "EGFR_expr": 3.0, "OS_time": 40.0},
        },
    )
    table = build_patient_table(radio, geno)
    assert table["patient_ids"] == ["P1", "P2"]
    assert table["patients"][0]["radiomic_features"]["original_glcm_Entropy"] == 3.1
    assert table["patients"][0]["genomic_alterations"]["altered_genes"] == ["EGFR"]
    assert table["patients"][1]["genomic_alterations"]["mutations"]["EGFR"] == 0
    assert table["patients"][1]["genomic_alterations"]["altered_genes"] == []


def test_resolve_patient_table_falls_back_to_bundles() -> None:
    images = ImageBundle(
        patient_ids=["P1"],
        series=[
            ImageSeriesRef(
                patient_id="P1",
                series_uid="s1",
                modality="CT",
                precomputed_features={"original_glcm_Entropy": 2.0},
            )
        ],
    )
    omics = OmicsBundle(
        patient_ids=["P1"],
        mutations={"P1": {"EGFR": 1}},
        metadata={"genes": ["EGFR"]},
    )
    table = resolve_patient_table(None, None, images, omics)
    assert table["n_patients"] == 1
    assert table["patients"][0]["genomic_alterations"]["altered_genes"] == ["EGFR"]


def test_mutation_prevalence_and_associations_by_gene() -> None:
    from agentic_radiogen.pipeline.report_table import associations_by_gene, mutation_prevalence

    table = {
        "patients": [
            {
                "patient_id": "P1",
                "radiomic_features": {"f1": 1.0},
                "genomic_alterations": {
                    "mutations": {"EGFR": 1, "TP53": 0},
                    "altered_genes": ["EGFR"],
                },
            },
            {
                "patient_id": "P2",
                "radiomic_features": {"f1": 2.0},
                "genomic_alterations": {
                    "mutations": {"EGFR": 0, "TP53": 1},
                    "altered_genes": ["TP53"],
                },
            },
        ]
    }
    prev = mutation_prevalence(table)
    assert prev["EGFR"] == {"altered": 1, "wildtype": 1}
    assert prev["TP53"] == {"altered": 1, "wildtype": 1}
    grouped = associations_by_gene(
        [
            {
                "imaging_feature": "f1",
                "genomic_feature": "EGFR_mut",
                "effect_size": 0.5,
                "q_value": 0.01,
                "n": 2,
            },
            {
                "imaging_feature": "f1",
                "genomic_feature": "TP53_mut",
                "effect_size": 0.1,
                "q_value": 0.9,
                "n": 2,
            },
            {
                "imaging_feature": "f2",
                "genomic_feature": "EGFR_mut",
                "effect_size": 0.4,
                "q_value": 0.02,
                "n": 2,
            },
        ]
    )
    assert list(grouped["EGFR_mut"][0].keys())
    assert grouped["EGFR_mut"][0]["imaging_feature"] == "f1"
    assert "TP53_mut" in grouped


def test_stage4_demo_report_includes_patient_table() -> None:
    payload = cli.run_stage4(
        LUNG_QUESTION,
        disease=None,
        catalog_name="demo",
        approve_download=False,
        max_patients=8,
        max_iterations=1,
    )
    table = payload["patient_table"]
    assert table["n_patients"] == 8
    assert len(table["patients"]) == 8
    assert "original_glcm_Entropy" in table["radiomic_feature_names"]
    first = table["patients"][0]
    assert first["patient_id"].startswith("TCGA-LUNG-")
    assert "EGFR" in first["genomic_alterations"]["mutations"]
    assert "original_glcm_Entropy" in first["radiomic_features"]
