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
    # Ascending |r| among strongest (top_per_gene=3): f2=0.4 then f1=0.5.
    assert grouped["EGFR_mut"][0]["imaging_feature"] == "f2"
    assert grouped["EGFR_mut"][-1]["imaging_feature"] == "f1"
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


def test_write_associations_csv_one_row_per_feature(tmp_path) -> None:
    from agentic_radiogen.pipeline.report_table import write_associations_csv

    path = tmp_path / "assoc.csv"
    n = write_associations_csv(
        path,
        [
            {
                "imaging_feature": "original_glcm_Entropy",
                "genomic_feature": "EGFR_mut",
                "effect_size": 0.254,
                "p_value": 3.81e-2,
                "q_value": 4.29e-1,
                "n": 67,
            },
            {
                "imaging_feature": "original_firstorder_Mean",
                "genomic_feature": "KRAS_mut",
                "effect_size": 0.4,
                "p_value": 0.01,
                "q_value": 0.02,
                "n": 67,
            },
        ],
        disease="lung",
        mutation_prevalence={
            "EGFR": {"altered": 10, "wildtype": 57},
            "KRAS": {"altered": 5, "wildtype": 62},
        },
    )
    assert n == 2
    text = path.read_text(encoding="utf-8")
    assert "category" in text
    assert "EGFR_mut" in text
    assert "supported" in text or "unverified" in text
    assert "contradicted" in text  # KRAS intensity conflict in default corpus
    assert "[mut: altered=10, wildtype=57]" in text
