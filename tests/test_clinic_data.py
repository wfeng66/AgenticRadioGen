"""--clinic_data concatenates clinical covariates into mutation train/eval."""

from __future__ import annotations

from agentic_radiogen.agents.genomics import GenomicsAgent
from agentic_radiogen.agents.statistics import StatisticalCriticalAgent
from agentic_radiogen.schemas.contracts import (
    GenomicMatrix,
    OmicsBundle,
    RadiomicMatrix,
)


def _bundle() -> OmicsBundle:
    patients = [f"P{i}" for i in range(24)]
    return OmicsBundle(
        patient_ids=patients,
        mutations={pid: {"EGFR": i % 2} for i, pid in enumerate(patients)},
        expression={},
        clinical={
            pid: {
                "OS_time": 100.0 - i,
                "OS_event": i % 2,
                "age_at_diagnosis": 50 + i,
                "stage": "Stage I" if i < 8 else ("Stage II" if i < 16 else "Stage III"),
                "subtype": "LUAD",
            }
            for i, pid in enumerate(patients)
        },
        metadata={},
    )


def test_genomics_default_excludes_clinic_predictors() -> None:
    matrix = GenomicsAgent().extract(_bundle())
    assert not any(n.startswith("clinic_") for n in matrix.feature_names)
    assert "OS_time" in matrix.feature_names


def test_genomics_include_clinical_adds_clinic_features() -> None:
    matrix = GenomicsAgent(include_clinical=True).extract(_bundle())
    assert "clinic_age_at_diagnosis" in matrix.feature_names
    assert "clinic_stage" in matrix.feature_names
    assert "clinic_subtype" in matrix.feature_names
    assert matrix.features["P0"]["clinic_age_at_diagnosis"] == 50.0


def test_stats_clinic_data_flag_mentioned_in_caveats() -> None:
    patients = [f"P{i}" for i in range(24)]
    radio = RadiomicMatrix(
        patient_ids=patients,
        feature_names=["f1"],
        features={pid: {"f1": float(i)} for i, pid in enumerate(patients)},
    )
    geno = GenomicsAgent(include_clinical=True).extract(_bundle())
    off = StatisticalCriticalAgent(model="ridge_reg").analyze(radio, geno)
    on = StatisticalCriticalAgent(
        model="ridge_reg", use_clinic_data=True
    ).analyze(radio, geno)
    assert any("Radiomics-only" in c for c in off.diagnostics.caveats)
    assert any("--clinic_data on" in c for c in on.diagnostics.caveats)
    egfr = next(m for m in on.metrics if m.target == "EGFR_mut")
    assert egfr.auprc is not None
