from __future__ import annotations

import os
from typing import Any

import numpy as np


def segmentation_backend() -> dict[str, Any]:
    """Report which segmentation path is available on this machine."""
    info: dict[str, Any] = {
        "backend": "threshold_cpu",
        "cuda": False,
        "torch": False,
        "totalsegmentator": False,
        "tumor_models_root": str((_default_models_root())),
    }
    try:
        import torch

        info["torch"] = True
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import totalsegmentator  # noqa: F401

        info["totalsegmentator"] = True
    except Exception:
        pass
    if info["totalsegmentator"] and info["cuda"]:
        info["backend"] = "totalsegmentator_gpu"
    elif info["totalsegmentator"]:
        info["backend"] = "totalsegmentator_cpu"
    return info


def _default_models_root():
    from pathlib import Path

    return Path.cwd() / "data_cache" / "seg_models"


def make_roi_mask(
    volume: np.ndarray,
    *,
    disease: str | None = None,
    modality: str = "CT",
    question: str = "",
    use_llm: bool | None = None,
    spacing_zyx: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, str]:
    """Create a segmentation mask via ``SegmentationAgent``.

    Feature extraction is separate (ImagingRadiomicsAgent / PyRadiomics).
    """
    from agentic_radiogen.agents.segmentation import SegmentationAgent

    result = SegmentationAgent(verbose=True, use_llm=use_llm).segment(
        volume,
        disease=disease,
        modality=modality,
        question=question,
        spacing_zyx=spacing_zyx,
    )
    return result.mask, result.backend


def _threshold_mask(volume: np.ndarray) -> np.ndarray:
    """Rough soft-tissue / lesion proxy mask for classical radiomics."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.ones(volume.shape, dtype=bool)
    body = volume > -500
    if body.sum() == 0:
        body = volume > np.percentile(finite, 10)
    if body.sum() == 0:
        return np.isfinite(volume)
    roi_vals = volume[body]
    lo = float(np.percentile(roi_vals, 60))
    hi = float(np.percentile(roi_vals, 99))
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-6:
        mask = body
    else:
        mask = body & (volume >= lo) & (volume <= hi)
    if mask.sum() < max(8, int(0.01 * body.sum())):
        mask = body
    return mask


def _prepare_volume_zyx(
    volume: np.ndarray,
) -> np.ndarray:
    vol = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    # Drop trailing singleton channel dims only — never squeeze away z=1 → 2D.
    while vol.ndim > 3 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    if vol.ndim == 2:
        # Single-slice series: keep as (1, H, W) for NIfTI writers.
        vol = vol[None, ...]
    if vol.ndim != 3:
        raise ValueError(f"Segmentation needs a 3D volume; got shape {vol.shape}")
    if any(int(s) < 1 for s in vol.shape):
        raise ValueError(f"Invalid volume shape {vol.shape}")
    if int(vol.shape[0]) < 2:
        raise ValueError(
            f"Series has only {vol.shape[0]} slice(s); need ≥2 for 3D segmentation "
            f"(got shape {vol.shape})"
        )
    return vol


def _nifti_from_volume_zyx(
    volume: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """Build a RAS NIfTI (x,y,z array order) from our (z,y,x) volume."""
    import nibabel as nib

    vol = _prepare_volume_zyx(volume)
    dz, dy, dx = (float(max(s, 1e-6)) for s in spacing_zyx)
    # nibabel / nnU-Net NibabelIO: array axes are (x, y, z)
    data_xyz = np.transpose(vol, (2, 1, 0))
    affine = np.diag([dx, dy, dz, 1.0]).astype(np.float64)
    img = nib.Nifti1Image(data_xyz, affine)
    img.set_data_dtype(np.float32)
    return img, vol.shape


def _mask_xyz_to_zyx(data: np.ndarray, volume_shape_zyx: tuple[int, ...]) -> np.ndarray:
    """Map a NIfTI-order mask back to (z,y,x)."""
    arr = np.asarray(data)
    arr = np.squeeze(arr)
    if arr.ndim > 3:
        arr = arr[..., 0]
    if arr.shape == volume_shape_zyx:
        return arr
    if arr.ndim == 3 and arr.shape == (
        volume_shape_zyx[2],
        volume_shape_zyx[1],
        volume_shape_zyx[0],
    ):
        return np.transpose(arr, (2, 1, 0))
    # Last resort: nearest resize in zyx if shapes differ slightly after resample.
    if arr.ndim == 3 and arr.shape[::-1] == volume_shape_zyx:
        return np.transpose(arr, (2, 1, 0))
    raise RuntimeError(
        f"Mask shape {arr.shape} incompatible with volume {volume_shape_zyx}"
    )


def _totalsegmentator_mask(
    volume: np.ndarray,
    *,
    organ_keywords: tuple[str, ...] | list[str] | None = None,
    task: str = "total",
    spacing_zyx: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """TotalSegmentator ROI (organ ``task=total`` or specialty tasks e.g. lung_nodules)."""
    from totalsegmentator.python_api import totalsegmentator  # type: ignore
    import tempfile
    import nibabel as nib
    from pathlib import Path

    keywords = [str(k).lower() for k in (organ_keywords or ()) if str(k).strip()]
    task_name = (task or "total").strip() or "total"
    spacing = spacing_zyx or (1.0, 1.0, 1.0)
    img_in, vol_shape = _nifti_from_volume_zyx(volume, spacing_zyx=spacing)

    # Stabilize nnU-Net workers (lung_nodules uses NibabelIOWithReorient + mp).
    os.environ.setdefault("nnUNet_n_proc_DA", "0")
    os.environ.setdefault("nnUNet_def_n_proc", "1")

    device = "gpu" if segmentation_backend()["cuda"] else "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_dir = tmp_path / "seg"
        # Prefer in-memory NIfTI + multilabel return (avoids some disk IO races).
        try:
            seg_img = totalsegmentator(
                img_in,
                str(out_dir),
                task=task_name,
                device=device,
                quiet=True,
                ml=True,
                nr_thr_resamp=1,
                nr_thr_saving=1,
            )
        except TypeError:
            # Older TotalSegmentator without some kwargs.
            seg_img = totalsegmentator(
                img_in, str(out_dir), task=task_name, device=device, quiet=True
            )

        mask = np.zeros(vol_shape, dtype=bool)
        if seg_img is not None:
            try:
                data = np.asanyarray(seg_img.dataobj)
                mask = _mask_xyz_to_zyx(data > 0, vol_shape).astype(bool)
            except Exception:
                mask = np.zeros(vol_shape, dtype=bool)

        if not mask.any():
            candidates = sorted(out_dir.glob("*.nii.gz"))
            if not candidates:
                single = Path(str(out_dir) + ".nii.gz")
                if single.is_file():
                    candidates = [single]
            if not candidates:
                raise RuntimeError(
                    f"TotalSegmentator task={task_name!r} produced no masks"
                )
            preferred: list[Path] = []
            for kw in keywords:
                preferred.extend(p for p in candidates if kw in p.name.lower())
            seen: set[str] = set()
            ordered: list[Path] = []
            for p in preferred + candidates:
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(p)
            for path in ordered:
                data = np.asarray(nib.load(str(path)).dataobj)
                try:
                    arr = _mask_xyz_to_zyx(data > 0, vol_shape)
                except RuntimeError:
                    continue
                mask |= arr.astype(bool)
                if task_name == "total" and not keywords and mask.any():
                    break

        if not mask.any():
            raise RuntimeError(
                f"TotalSegmentator mask empty "
                f"(task={task_name!r}, organs={keywords or ['any']})"
            )
        return mask
