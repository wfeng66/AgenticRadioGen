from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agentic_radiogen.schemas.contracts import (
    Association,
    GenomicMatrix,
    ModelDiagnostics,
    ModelMetrics,
    ModelResult,
    RadiomicMatrix,
)


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


class StatisticalCriticalAgent:
    """Joins radiomic and genomic matrices. This is the first merge point."""

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

        genomic_targets = [c for c in geno_df.columns if c.endswith("_mut")]
        if target_prefix:
            genomic_targets = [c for c in genomic_targets if c.startswith(target_prefix)]
        if not genomic_targets:
            genomic_targets = [c for c in geno_df.columns if c.endswith("_expr")][:1]

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

        diagnostics = self._diagnostics(radio_df)
        metrics: list[ModelMetrics] = []
        for target in genomic_targets:
            metrics.append(self._predict_mutation(radio_df, geno_df, target))
        if {"OS_time", "OS_event"}.issubset(geno_df.columns) and len(radio_df.columns):
            risk = radio_df.iloc[:, 0].to_numpy(dtype=float)
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
                )
            )

        return ModelResult(
            # Ascending |correlation|: weaker first, stronger last ("more correlation, show later").
            associations=sorted(associations, key=lambda a: (abs(a.effect_size), a.q_value)),
            metrics=metrics,
            diagnostics=diagnostics,
        )

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

    @staticmethod
    def _predict_mutation(
        radio_df: pd.DataFrame, geno_df: pd.DataFrame, target: str
    ) -> ModelMetrics:
        y = geno_df[target].to_numpy(dtype=float)
        mask = np.isfinite(y)
        y = y[mask]
        if len(y) < 4:
            return ModelMetrics(auroc=None, n_train=0, n_test=0, target=target)
        classes, counts = np.unique(y.astype(int), return_counts=True)
        if len(classes) < 2 or int(counts.min()) < 2:
            # Rare / single-mutant cohorts cannot support a stratified holdout AUROC.
            return ModelMetrics(auroc=None, n_train=0, n_test=0, target=target)
        X = radio_df.to_numpy(dtype=float)[mask]
        # Stratify only when every class can appear in both train and test.
        can_stratify = int(counts.min()) >= 2 and len(y) >= 6
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.35,
                random_state=0,
                stratify=y.astype(int) if can_stratify else None,
            )
        except ValueError:
            return ModelMetrics(auroc=None, n_train=0, n_test=0, target=target)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return ModelMetrics(auroc=None, n_train=len(y_train), n_test=len(y_test), target=target)
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=200)
        clf.fit(scaler.fit_transform(X_train), y_train)
        proba = clf.predict_proba(scaler.transform(X_test))[:, 1]
        auroc = float(roc_auc_score(y_test, proba))
        return ModelMetrics(
            auroc=auroc,
            n_train=len(y_train),
            n_test=len(y_test),
            target=target,
        )
