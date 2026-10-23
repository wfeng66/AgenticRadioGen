"""Nonlinear --model backends for radiomics → mutation AUROC."""

from __future__ import annotations

import numpy as np
import pytest

from agentic_radiogen.agents.statistics import StatisticalCriticalAgent
from agentic_radiogen.schemas.contracts import GenomicMatrix, RadiomicMatrix
from agentic_radiogen.stats.predictors import (
    MODEL_CHOICES,
    normalize_model_name,
    predict_proba_mutation,
)


def test_normalize_model_aliases() -> None:
    assert normalize_model_name("XGBoost") == "xgboost"
    assert normalize_model_name("random forest") == "random_forest"
    assert normalize_model_name("SE-CNN") == "se_cnn"
    assert normalize_model_name("ResNeXt") == "resnext"
    assert normalize_model_name("ridge") == "ridge_reg"
    assert normalize_model_name("elastic-net") == "elastic_net"
    assert normalize_model_name("poly_reg") == "poly_reg"
    with pytest.raises(ValueError, match="Unknown"):
        normalize_model_name("logistic")


def test_sklearn_models_predict_proba() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    # Flip a few labels so the task is not perfectly linear.
    y[::7] = 1.0 - y[::7]
    X_train, X_test = X[:28], X[28:]
    y_train = y[:28]
    for name in (
        "svm",
        "random_forest",
        "xgboost",
        "ridge_reg",
        "elastic_net",
        "poly_reg",
    ):
        proba = predict_proba_mutation(name, X_train, y_train, X_test)
        assert proba.shape == (len(X_test),)
        assert np.all(np.isfinite(proba))
        assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_stats_agent_records_model_name() -> None:
    patients = [f"P{i}" for i in range(12)]
    radio = RadiomicMatrix(
        patient_ids=patients,
        feature_names=["f1", "f2"],
        features={
            pid: {"f1": float(i), "f2": float(i % 3)} for i, pid in enumerate(patients)
        },
    )
    geno = GenomicMatrix(
        patient_ids=patients,
        feature_names=["EGFR_mut"],
        features={
            pid: {"EGFR_mut": float(i % 2)} for i, pid in enumerate(patients)
        },
    )
    result = StatisticalCriticalAgent(model="svm").analyze(radio, geno)
    egfr = next(m for m in result.metrics if m.target == "EGFR_mut")
    assert egfr.model == "svm"
    assert egfr.auroc is not None
    assert any("svm" in c for c in result.diagnostics.caveats)


def test_model_choices_cover_cli() -> None:
    assert set(MODEL_CHOICES) == {
        "lstm",
        "svm",
        "random_forest",
        "xgboost",
        "vgg19",
        "resnext",
        "se_cnn",
        "ridge_reg",
        "elastic_net",
        "poly_reg",
    }


@pytest.mark.parametrize("name", ["lstm", "vgg19", "resnext", "se_cnn"])
def test_torch_models_optional(name: str) -> None:
    torch = pytest.importorskip("torch")
    _ = torch
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 8))
    y = (X[:, 0] > 0).astype(float)
    y[::5] = 1.0 - y[::5]
    proba = predict_proba_mutation(name, X[:16], y[:16], X[16:])
    assert proba.shape == (8,)
    assert np.all(np.isfinite(proba))
