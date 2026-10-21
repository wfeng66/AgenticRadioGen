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
    store_segm: bool = False,
    segm_dir: str | Path | None = None,
    patient_id: str | None = None,
    force_model_id: str | None = None,
) -> dict[str, Any]:
    root = Path(series_dir)
    cache_path = root / _RADIOMICS_CACHE
    dkey = _disease_key(disease)
    segm_out = Path(segm_dir) if segm_dir else Path.cwd() / "segm_rslt"
    stem = _segm_stem(patient_id=patient_id, series_dir=root)
    existing_mask = _resolve_existing_mask(segm_out, stem)
    # Forced model must recompute (ignore stale monai/etc. cache + mask).
    if force_model_id and existing_mask is not None:
        try:
            existing_mask.unlink()
        except OSError:
            pass
        existing_mask = None
    need_mask = bool(store_segm) and existing_mask is None
    allow_cache = use_cache and not force_model_id

    if allow_cache and cache_path.exists() and not need_mask:
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
                    "segmentation_model_id": cached.get("segmentation_model_id")
                    or cached.get("segmentation_backend"),
                    "models_attempted": cached.get("models_attempted")
                    or cached.get("segmentation_model_id")
                    or cached.get("segmentation_backend")
                    or "",
                    "mask_path": str(existing_mask) if existing_mask else "",
                    "n_mask_voxels": _count_mask_voxels(existing_mask)
                    if existing_mask
                    else "",
                    "segm_status": "success",
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
        store_segm=store_segm,
        segm_dir=segm_out,
        patient_id=patient_id,
        series_dir=root,
        force_model_id=force_model_id,
    )
    mask_saved = str(summary.get("mask_path") or "")
    if not mask_saved and existing_mask:
        mask_saved = str(existing_mask)
    payload = {
        "features": features,
        "volume_summary": summary,
        "local_path": str(root.resolve()),
        "from_cache": False,
        "segmentation_backend": summary.get("segmentation_backend_name"),
        "segmentation_model_id": summary.get("segmentation_model_id")
        or summary.get("segmentation_backend_name"),
        "models_attempted": summary.get("models_attempted")
        or summary.get("segmentation_model_id")
        or summary.get("segmentation_backend_name")
        or "",
        "mask_path": mask_saved,
        "n_mask_voxels": summary.get("n_mask_voxels", ""),
        "segm_status": str(summary.get("segm_status") or "success"),
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
                        "segmentation_model_id": summary.get("segmentation_model_id"),
                        "models_attempted": summary.get("models_attempted"),
                        "features": features,
                        "volume_summary": {
                            k: v
                            for k, v in summary.items()
                            if isinstance(v, (int, float))
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
    return payload


def _resolve_existing_mask(segm_out: Path, stem: str) -> Path | None:
    for cand in (
        segm_out / f"{stem}.nii.gz",
        segm_out / f"{stem}.nii",
        segm_out / f"{stem}.npy",
    ):
        if cand.is_file():
            return cand
    return None


def _count_mask_voxels(path: Path | None) -> int:
    if path is None or not Path(path).is_file():
        return 0
    p = Path(path)
    try:
        if p.suffix.lower() == ".npy":
            return int((np.load(p) > 0).sum())
        try:
            import SimpleITK as sitk

            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
            return int((np.asarray(arr) > 0).sum())
        except Exception:
            import nibabel as nib

            data = np.asanyarray(nib.load(str(p)).dataobj)
            return int((np.asarray(data) > 0).sum())
    except Exception:
        return 0


def _segm_stem(*, patient_id: str | None, series_dir: Path) -> str:
    if patient_id and str(patient_id).strip():
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(patient_id).strip())
    return re.sub(r"[^A-Za-z0-9._-]+", "_", series_dir.name) or "series"


def save_segmentation_mask(
    mask: np.ndarray,
    *,
    out_path: Path,
    spacing_zyx: tuple[float, float, float] | None = None,
) -> Path:
    """Write a boolean/label mask as NIfTI under ``out_path`` (``*.npy`` fallback)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8)
    try:
        from agentic_radiogen.imaging.segment import _nifti_from_volume_zyx
        import nibabel as nib

        img, _ = _nifti_from_volume_zyx(
            m.astype(np.float32),
            spacing_zyx=spacing_zyx or (1.0, 1.0, 1.0),
        )
        img.set_data_dtype(np.uint8)
        nib.save(img, str(out_path))
        return out_path
    except Exception:
        npy_path = out_path.with_suffix("").with_suffix(".npy")
        if npy_path.suffix != ".npy":
            npy_path = Path(str(out_path).removesuffix(".nii.gz") + ".npy")
        np.save(str(npy_path), m)
        return npy_path


def extract_radiomics_from_volume(
    volume: np.ndarray,
    *,
    disease: str | None = None,
    modality: str = "CT",
    question: str = "",
    use_llm: bool | None = None,
    spacing_zyx: tuple[float, float, float] | None = None,
    store_segm: bool = False,
    segm_dir: str | Path | None = None,
    patient_id: str | None = None,
    series_dir: str | Path | None = None,
    force_model_id: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract classical radiomics with PyRadiomics on a SegmentationAgent mask."""
    mask, backend, model_id, models_attempted = make_roi_mask(
        volume,
        disease=disease,
        modality=modality,
        question=question,
        use_llm=use_llm,
        spacing_zyx=spacing_zyx,
        force_model_id=force_model_id,
    )
    n_mask_voxels = int(np.asarray(mask).astype(bool).sum())
    if n_mask_voxels <= 0:
        raise ValueError("Segmentation mask is empty (0 voxels)")

    mask_path = ""
    segm_status = "success"
    if store_segm:
        out_root = Path(segm_dir) if segm_dir else Path.cwd() / "segm_rslt"
        stem = _segm_stem(
            patient_id=patient_id,
            series_dir=Path(series_dir) if series_dir else Path("series"),
        )
        try:
            saved = save_segmentation_mask(
                mask,
                out_path=out_root / f"{stem}.nii.gz",
                spacing_zyx=spacing_zyx,
            )
            mask_path = str(saved.resolve())
            # Verify what was written (catches silent empty saves).
            written = _count_mask_voxels(saved)
            if written <= 0:
                segm_status = "failure: saved mask has 0 voxels"
            elif written != n_mask_voxels:
                # Still usable; keep success but report disk count.
                n_mask_voxels = written
            logger.info(
                "Saved segmentation mask → %s (%s voxels)", saved, n_mask_voxels
            )
        except Exception as exc:
            segm_status = f"mask_save_failed: {exc}"
            logger.warning("Failed to store segmentation mask for %s: %s", stem, exc)

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
        "segmentation_model_id": model_id or backend,
        "models_attempted": models_attempted or (model_id or backend),
        "mask_path": mask_path,
        "n_mask_voxels": n_mask_voxels,
        "segm_status": segm_status,
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
