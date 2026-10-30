from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from agentic_radiogen.schemas.contracts import (
    Association,
    GenomicMatrix,
    ModelDiagnostics,
    ModelMetrics,
    ModelResult,
    RadiomicMatrix,
)
from agentic_radiogen.stats.predictors import fit_score_mutation, normalize_model_name

_PRED_THRESHOLD = 0.5
# Skip mutation predictors with too few positives (wide-negative genes inflate AUPRC/F1).
_MIN_POSITIVE_MUTATIONS = 10


def _fdr_bh(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.empty(n, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = order[-rank]
        q = min(prev, p_values[i] * n / (n - rank + 1))
        ranked[i] = q
        prev = q
    return [float(min(1.0, q)) for q in ranked]


def _concordance_index(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    """Higher score = higher risk. Concordance over comparable pairs."""
    comparable = 0
    concordant = 0.0
    n = len(times)
    for i in range(n):
        if events[i] <= 0:
            continue
        for j in range(n):
            if times[j] <= times[i]:
                continue
            comparable += 1
            if scores[i] > scores[j]:
                concordant += 1
            elif scores[i] == scores[j]:
                concordant += 0.5
    if comparable == 0:
        return float("nan")
    return float(concordant / comparable)


def _binary_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y_hat = (np.asarray(proba, dtype=float) >= _PRED_THRESHOLD).astype(int)
    y = np.asarray(y_true, dtype=int)
    return {
        "auprc": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, y_hat, zero_division=0)),
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall": float(recall_score(y, y_hat, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_hat)),
    }


class StatisticalCriticalAgent:
    """Joins radiomic and genomic matrices. This is the first merge point."""

    def __init__(
        self,
        model: str = "random_forest",
        *,
        use_clinic_data: bool = False,
    ) -> None:
        self.model_name = normalize_model_name(model)
        self.use_clinic_data = bool(use_clinic_data)

    def analyze(
        self,
        radiomics: RadiomicMatrix,
        genomics: GenomicMatrix,
        *,
        target_prefix: str | None = None,
    ) -> ModelResult:
        shared = sorted(set(radiomics.patient_ids) & set(genomics.patient_ids))
        if len(shared) < 4:
            raise ValueError("Need at least 4 paired patients to join imaging and genomics")

        radio_df = pd.DataFrame({pid: radiomics.features[pid] for pid in shared}).T
        geno_df = pd.DataFrame({pid: genomics.features[pid] for pid in shared}).T
        radio_df = radio_df.apply(pd.to_numeric, errors="coerce")
        geno_df = geno_df.apply(pd.to_numeric, errors="coerce")
        feature_df = self._feature_matrix(radio_df, geno_df)

        genomic_targets = [c for c in geno_df.columns if c.endswith("_mut")]
        if target_prefix:
            genomic_targets = [c for c in genomic_targets if c.startswith(target_prefix)]
        if not genomic_targets:
            genomic_targets = [c for c in geno_df.columns if c.endswith("_expr")][:1]

        predict_targets = [
            c
            for c in genomic_targets
            if c.endswith("_mut")
            and int(np.nansum(geno_df[c].to_numpy(dtype=float) >= 0.5))
            >= _MIN_POSITIVE_MUTATIONS
        ]
        if not predict_targets and genomic_targets and not genomic_targets[0].endswith("_mut"):
            # Expression fallback when no mutation columns meet the positive floor.
            predict_targets = list(genomic_targets[:1])

        associations: list[Association] = []
        raw_p: list[float] = []
        for img_col in radio_df.columns:
            x = radio_df[img_col].to_numpy(dtype=float)
            for g_col in genomic_targets:
                y = geno_df[g_col].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 4:
                    continue
                if np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
                    continue
                r, p = stats.pearsonr(x[mask], y[mask])
                if not np.isfinite(r) or not np.isfinite(p):
                    continue
                raw_p.append(float(p))
                associations.append(
                    Association(
                        imaging_feature=img_col,
                        genomic_feature=g_col,
                        effect_size=float(r),
                        p_value=float(p),
                        q_value=1.0,
                        n=int(mask.sum()),
                    )
                )
        q_values = _fdr_bh(raw_p)
        for assoc, q in zip(associations, q_values, strict=True):
            assoc.q_value = q

        diagnostics = self._diagnostics(feature_df)
        diagnostics.caveats.append(
            f"Mutation AUPRC uses model '{self.model_name}'"
        )
        n_skipped_rare = sum(
            1
            for c in genomic_targets
            if c.endswith("_mut") and c not in predict_targets
        )
        if n_skipped_rare:
            diagnostics.caveats.append(
                f"Skipped {n_skipped_rare} mutation target(s) with "
                f"<{_MIN_POSITIVE_MUTATIONS} positive cases"
            )
        if self.use_clinic_data:
            n_clinic = sum(1 for c in feature_df.columns if str(c).startswith("clinic_"))
            diagnostics.caveats.append(
                f"--clinic_data on: {n_clinic} clinical covariates concatenated "
                "with radiomics for train/eval"
            )
        else:
            diagnostics.caveats.append(
                "Radiomics-only predictors (pass --clinic_data to add clinical covariates)"
            )
        metrics: list[ModelMetrics] = []
        test_split_log: list[dict[str, Any]] = []
        for target in predict_targets:
            row, log_rows = self._predict_mutation(feature_df, geno_df, target)
            metrics.append(row)
            test_split_log.extend(log_rows)
        if {"OS_time", "OS_event"}.issubset(geno_df.columns) and len(feature_df.columns):
            risk = feature_df.iloc[:, 0].to_numpy(dtype=float)
            c_index = _concordance_index(
                geno_df["OS_time"].to_numpy(dtype=float),
                geno_df["OS_event"].to_numpy(dtype=float),
                risk,
            )
            metrics.append(
                ModelMetrics(
                    c_index=None if np.isnan(c_index) else c_index,
                    n_train=len(shared),
                    n_test=0,
                    target="OS",
                    model=self.model_name,
                )
            )

        # Mutation metrics: ascending F1 (weaker first); OS C-index rows last.
        metrics_sorted = sorted(
            metrics,
            key=lambda m: (
                1 if m.target == "OS" else 0,
                m.f1 is None,
                float(m.f1) if m.f1 is not None else 0.0,
                str(m.target or ""),
            ),
        )
        return ModelResult(
            # Ascending |correlation|: weaker first, stronger last ("more correlation, show later").
            associations=sorted(associations, key=lambda a: (abs(a.effect_size), a.q_value)),
            metrics=metrics_sorted,
            diagnostics=diagnostics,
            test_split_log=test_split_log,
        )

    def _feature_matrix(
        self, radio_df: pd.DataFrame, geno_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Radiomics features, optionally concatenated with clinic_* covariates."""
        if not self.use_clinic_data:
            return radio_df
        clinic_cols = [c for c in geno_df.columns if str(c).startswith("clinic_")]
        if not clinic_cols:
            return radio_df
        clinic_df = geno_df[clinic_cols].apply(pd.to_numeric, errors="coerce")
        return pd.concat([radio_df, clinic_df], axis=1)

    @staticmethod
    def _diagnostics(radio_df: pd.DataFrame) -> ModelDiagnostics:
        caveats: list[str] = []
        collinear: list[tuple[str, str, float]] = []
        cols = list(radio_df.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                left = radio_df[a].to_numpy(dtype=float)
                right = radio_df[b].to_numpy(dtype=float)
                mask = np.isfinite(left) & np.isfinite(right)
                if mask.sum() < 5:
                    continue
                if np.unique(left[mask]).size < 2 or np.unique(right[mask]).size < 2:
                    continue
                r = float(np.corrcoef(left[mask], right[mask])[0, 1])
                if np.isfinite(r) and abs(r) >= 0.95:
                    collinear.append((a, b, r))
        if collinear:
            caveats.append("Highly collinear radiomic features detected")
        if len(radio_df) < 8:
            caveats.append("Small cohort (n<8); metrics are unstable")
        caveats.append("Associations are exploratory; literature must interpret before looping")
        return ModelDiagnostics(
            patient_split_disjoint=True,
            collinear_pairs=collinear,
            multiple_testing_method="fdr_bh",
            caveats=caveats,
        )

    def _predict_mutation(
        self, feature_df: pd.DataFrame, geno_df: pd.DataFrame, target: str
    ) -> tuple[ModelMetrics, list[dict[str, Any]]]:
        y_all = geno_df[target].to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        y = y_all[mask]
        patient_ids = feature_df.index.to_numpy()[mask]
        feat_names = [str(c) for c in feature_df.columns]
        empty = ModelMetrics(
            auprc=None, n_train=0, n_test=0, target=target, model=self.model_name
        )
        if len(y) < 4:
            return empty, []
        classes, counts = np.unique(y.astype(int), return_counts=True)
        if len(classes) < 2 or int(counts.min()) < 2:
            # Rare / single-mutant cohorts cannot support a stratified holdout AUPRC.
            return empty, []
        X = feature_df.to_numpy(dtype=float)[mask]
        # Impute non-finite clinic/radiomic cells with column medians (train-safe enough for holdout).
        if not np.isfinite(X).all():
            col_med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
            col_med = np.where(np.isfinite(col_med), col_med, 0.0)
            inds = np.where(~np.isfinite(X))
            X = X.copy()
            X[inds] = np.take(col_med, inds[1])
        # Stratify only when every class can appear in both train and test.
        can_stratify = int(counts.min()) >= 2 and len(y) >= 6
        indices = np.arange(len(y))
        try:
            idx_train, idx_test = train_test_split(
                indices,
                test_size=0.35,
                random_state=0,
                stratify=y.astype(int) if can_stratify else None,
            )
        except ValueError:
            return empty, []
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return (
                ModelMetrics(
                    auprc=None,
                    n_train=len(y_train),
                    n_test=len(y_test),
                    target=target,
                    model=self.model_name,
                ),
                [],
            )
        try:
            train_proba, test_proba = fit_score_mutation(
                self.model_name, X_train, y_train, X_test
            )
            train_m = _binary_metrics(y_train, train_proba)
            test_m = _binary_metrics(y_test, test_proba)
        except Exception:
            return (
                ModelMetrics(
                    auprc=None,
                    n_train=len(y_train),
                    n_test=len(y_test),
                    target=target,
                    model=self.model_name,
                ),
                [],
            )

        y_hat_test = (np.asarray(test_proba, dtype=float) >= _PRED_THRESHOLD).astype(int)
        log_rows: list[dict[str, Any]] = []
        for local_i, row_i in enumerate(idx_test):
            feat_row = X_test[local_i]
            log_rows.append(
                {
                    "split": "test",
                    "target": target,
                    "model": self.model_name,
                    "patient_id": str(patient_ids[row_i]),
                    "y_true": int(y_test[local_i]),
                    "y_pred": int(y_hat_test[local_i]),
                    "proba": float(test_proba[local_i]),
                    "threshold": _PRED_THRESHOLD,
                    "feature_names": feat_names,
                    "features": {
                        name: float(feat_row[j]) if np.isfinite(feat_row[j]) else None
                        for j, name in enumerate(feat_names)
                    },
                }
            )

        return (
            ModelMetrics(
                auprc=test_m["auprc"],
                f1=test_m["f1"],
                precision=test_m["precision"],
                recall=test_m["recall"],
                accuracy=test_m["accuracy"],
                train_auprc=train_m["auprc"],
                train_f1=train_m["f1"],
                train_precision=train_m["precision"],
                train_recall=train_m["recall"],
                train_accuracy=train_m["accuracy"],
                n_train=len(y_train),
                n_test=len(y_test),
                target=target,
                model=self.model_name,
            ),
            log_rows,
        )
