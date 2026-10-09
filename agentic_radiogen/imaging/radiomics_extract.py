"""Radiomics via PyRadiomics (shape, first-order, GLCM/GLRLM/GLSZM/GLDM/NGTDM)."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from agentic_radiogen.imaging.segment import make_roi_mask

_RADIOMICS_CACHE = "radiomics_features.json"
_CACHE_VERSION = 5  # disease-aware ROI + PyRadiomics
logger = logging.getLogger(__name__)


def _disease_key(disease: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (disease or "unknown").strip().lower()).strip("_") or "unknown"


def extract_series_features(
    series_dir: str | Path,
    *,
    use_cache: bool = True,
    disease: str | None = None,
    modality: str = "CT",
    question: str = "",
    use_llm: bool | None = None,
) -> dict[str, Any]:
    root = Path(series_dir)
    cache_path = root / _RADIOMICS_CACHE
    dkey = _disease_key(disease)
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("features")
                and int(cached.get("version", 0)) >= _CACHE_VERSION
                and str(cached.get("disease_key") or "") == dkey
            ):
                return {
                    "features": {k: float(v) for k, v in cached["features"].items()},
                    "volume_summary": {
                        k: float(v)
                        for k, v in (cached.get("volume_summary") or {}).items()
                        if isinstance(v, (int, float))
                    },
                    "local_path": str(root.resolve()),
                    "from_cache": True,
                    "segmentation_backend": cached.get("segmentation_backend"),
                }
        except Exception:
            pass
    from agentic_radiogen.imaging.dicom_io import load_dicom_series_with_spacing

    volume, spacing_zyx = load_dicom_series_with_spacing(root)
    features, summary = extract_radiomics_from_volume(
        volume,
        disease=disease,
        modality=modality,
        question=question,
        use_llm=use_llm,
        spacing_zyx=spacing_zyx,
    )
    payload = {
        "features": features,
        "volume_summary": summary,
        "local_path": str(root.resolve()),
        "from_cache": False,
        "segmentation_backend": summary.get("segmentation_backend_name"),
    }
    if use_cache:
        try:
            cache_path.write_text(
                json.dumps(
                    {
                        "version": _CACHE_VERSION,
                        "disease_key": dkey,
                        "disease": disease,
                        "modality": modality,
                        "segmentation_backend": summary.get("segmentation_backend_name"),
                        "features": features,
                        "volume_summary": summary,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
    return payload


def extract_radiomics_from_volume(
    volume: np.ndarray,
    *,
    disease: str | None = None,
    modality: str = "CT",
    question: str = "",
    use_llm: bool | None = None,
    spacing_zyx: tuple[float, float, float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract classical radiomics with PyRadiomics on a SegmentationAgent mask."""
    mask, backend = make_roi_mask(
        volume,
        disease=disease,
        modality=modality,
        question=question,
        use_llm=use_llm,
        spacing_zyx=spacing_zyx,
    )
    if not np.asarray(mask).any():
        raise ValueError("Volume has no finite voxels for radiomics")

    features = _pyradiomics_features(volume, mask)
    if "original_firstorder_StandardDeviation" in features:
        features["original_firstorder_Std"] = features[
            "original_firstorder_StandardDeviation"
        ]
    # Legacy alias used by demo catalog / older tests
    if (
        "original_glcm_Entropy" not in features
        and "original_glcm_JointEntropy" in features
    ):
        features["original_glcm_Entropy"] = features["original_glcm_JointEntropy"]

    is_deep = backend.startswith("totalsegmentator") or "tumor" in backend
    summary = {
        "mean": features.get("original_firstorder_Mean", 0.0),
        "std": features.get(
            "original_firstorder_StandardDeviation",
            features.get("original_firstorder_Std", 0.0),
        ),
        "size": features.get("original_shape_VoxelVolume", float(np.asarray(mask).sum())),
        "segmentation_backend": 1.0 if is_deep else 0.0,
        "segmentation_backend_name": backend,
        "n_features": float(len([k for k in features if not k.startswith("meta_")])),
    }
    features["meta_segmentation_is_gpu_or_ts"] = summary["segmentation_backend"]
    return features, summary


@lru_cache(maxsize=1)
def _extractor():
    try:
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyRadiomics is required for radiomics extraction. "
            "Install with: pip install pyradiomics SimpleITK"
        ) from exc

    # Quiet PyRadiomics' verbose logging of every feature.
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    extractor = RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    # Bin width suitable for CT HU ranges; override if needed via settings later.
    extractor.settings["binWidth"] = 25
    extractor.settings["normalize"] = False
    return extractor


def _pyradiomics_features(volume: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    import SimpleITK as sitk

    vol = np.ascontiguousarray(volume, dtype=np.float32)
    msk = np.ascontiguousarray(mask.astype(np.uint8))
    if msk.sum() < 2:
        raise ValueError("ROI mask too small for PyRadiomics")
    # PyRadiomics requires background (0) and label (1); a full-volume ROI is rejected.
    if msk.min() == msk.max() == 1:
        msk = msk.copy()
        msk.flat[0] = 0

    image = sitk.GetImageFromArray(vol)
    label = sitk.GetImageFromArray(msk)
    # Identical geometry (unit spacing); real DICOM spacing can be wired later.
    image.SetSpacing((1.0, 1.0, 1.0))
    label.SetSpacing((1.0, 1.0, 1.0))

    result = _extractor().execute(image, label, label=1)
    features: dict[str, float] = {}
    for key, value in result.items():
        if not str(key).startswith("original_"):
            continue
        try:
            features[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if not features:
        raise ValueError("PyRadiomics returned no original_* features")
    return features
