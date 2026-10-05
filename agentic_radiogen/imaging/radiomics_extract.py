from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agentic_radiogen.imaging.dicom_io import load_dicom_series
from agentic_radiogen.imaging.segment import make_roi_mask


def extract_series_features(series_dir: str | Path) -> dict[str, Any]:
    volume = load_dicom_series(series_dir)
    features, summary = extract_radiomics_from_volume(volume)
    return {
        "features": features,
        "volume_summary": summary,
        "local_path": str(Path(series_dir).resolve()),
    }


def extract_radiomics_from_volume(volume: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Classical radiomics-style features (not CNN embeddings).

    Uses TotalSegmentator on GPU when available; otherwise a CPU threshold ROI.
    Feature extraction itself is NumPy/CPU (PyRadiomics-compatible names).
    """
    mask, backend = make_roi_mask(volume)
    vals = volume[mask]
    if vals.size == 0:
        vals = volume[np.isfinite(volume)].ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Volume has no finite voxels for radiomics")

    # Downsample mid-slice for cheap GLCM-like entropy
    z = volume.shape[0] // 2
    slice2d = volume[z]
    mask2d = mask[z] if mask.ndim == 3 else mask
    entropy = _slice_entropy(slice2d, mask2d)
    sphericity = _sphericity_proxy(mask)

    features = {
        "original_firstorder_Mean": float(np.mean(vals)),
        "original_firstorder_Std": float(np.std(vals)),
        "original_firstorder_Median": float(np.median(vals)),
        "original_firstorder_Skewness": float(_skewness(vals)),
        "original_glcm_Entropy": float(entropy),
        "original_shape_Sphericity": float(sphericity),
        "original_shape_VoxelVolume": float(mask.sum()),
    }
    summary = {
        "mean": features["original_firstorder_Mean"],
        "std": features["original_firstorder_Std"],
        "size": features["original_shape_VoxelVolume"],
        "segmentation_backend": 1.0 if backend.startswith("totalsegmentator") else 0.0,
    }
    # Keep backend name accessible via imaging metadata path
    features["meta_segmentation_is_gpu_or_ts"] = summary["segmentation_backend"]
    return features, summary


def _skewness(vals: np.ndarray) -> float:
    mu = float(np.mean(vals))
    sigma = float(np.std(vals))
    if sigma < 1e-8:
        return 0.0
    return float(np.mean(((vals - mu) / sigma) ** 3))


def _slice_entropy(slice2d: np.ndarray, mask2d: np.ndarray, bins: int = 32) -> float:
    vals = slice2d[mask2d] if mask2d.any() else slice2d.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=bins)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def _sphericity_proxy(mask: np.ndarray) -> float:
    vol = float(mask.sum())
    if vol <= 0:
        return 0.0
    # Surface approximation via binary gradient magnitude count
    surface = 0.0
    for axis in range(mask.ndim):
        diff = np.diff(mask.astype(np.int8), axis=axis)
        surface += float(np.abs(diff).sum())
    surface = max(surface, 1.0)
    # Ideal sphere: 36*pi*V^2 / A^3  -> use softer proxy in [0, 1]
    ratio = (36.0 * np.pi * (vol**2)) / (surface**3 + 1e-8)
    return float(np.clip(ratio ** (1.0 / 3.0), 0.0, 1.0))
