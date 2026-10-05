from __future__ import annotations

from pathlib import Path

import numpy as np

from agentic_radiogen.schemas.contracts import ImageBundle, RadiomicMatrix


class ImagingRadiomicsAgent:
    """Consumes an ImageBundle only. Never calls the genomics agent.

    Uses precomputed radiomics from the matcher when present (live fetch).
    Re-reads DICOM only when precomputed features are missing.
    """

    def extract(self, bundle: ImageBundle) -> RadiomicMatrix:
        if not bundle.series:
            raise ValueError("ImageBundle has no series")
        features: dict[str, dict[str, float]] = {}
        names: set[str] = set()
        skipped: list[str] = []
        for series in bundle.series:
            try:
                row = self._series_features(series)
            except Exception as exc:
                skipped.append(f"{series.patient_id}: {exc}")
                continue
            features[series.patient_id] = row
            names.update(row)
        if not features:
            detail = "; ".join(skipped[:5])
            raise ValueError(
                f"No radiomic payload for any series"
                + (f" ({detail})" if detail else "")
            )
        ordered = sorted(names)
        return RadiomicMatrix(
            patient_ids=sorted(features),
            feature_names=ordered,
            features=features,
        )

    def _series_features(self, series) -> dict[str, float]:
        row = {
            k: float(v)
            for k, v in (series.precomputed_features or {}).items()
            if not str(k).startswith("meta_")
        }
        if not row and series.local_path:
            row.update(self._from_local_path(series.local_path))
        if series.volume_summary:
            row.setdefault(
                "original_firstorder_Mean", float(series.volume_summary.get("mean", 0.0))
            )
            row.setdefault(
                "original_firstorder_Std", float(series.volume_summary.get("std", 0.0))
            )
            row.setdefault(
                "original_shape_VoxelVolume", float(series.volume_summary.get("size", 0.0))
            )
        if not row:
            raise ValueError(f"No radiomic payload for series {series.series_uid}")
        return {k: float(v) for k, v in row.items()}

    @staticmethod
    def _from_local_path(local_path: str) -> dict[str, float]:
        path = Path(local_path)
        if path.is_dir():
            from agentic_radiogen.imaging.radiomics_extract import extract_series_features

            extracted = extract_series_features(path)
            return {
                k: float(v)
                for k, v in extracted["features"].items()
                if not str(k).startswith("meta_")
            }
        if path.suffix == ".npy":
            volume = np.load(path)
            return {
                "original_firstorder_Mean": float(np.mean(volume)),
                "original_firstorder_Std": float(np.std(volume)),
                "original_shape_VoxelVolume": float(volume.size),
            }
        raise ValueError(f"Unsupported imaging path: {local_path}")
