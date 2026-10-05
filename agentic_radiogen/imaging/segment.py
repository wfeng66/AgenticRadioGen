from __future__ import annotations

from typing import Any

import numpy as np


def segmentation_backend() -> dict[str, Any]:
    """Report which segmentation path is available on this machine."""
    info: dict[str, Any] = {
        "backend": "threshold_cpu",
        "cuda": False,
        "torch": False,
        "totalsegmentator": False,
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


def make_roi_mask(volume: np.ndarray) -> tuple[np.ndarray, str]:
    """Create an ROI mask.

    Preferred path (when installed + CUDA): TotalSegmentator.
    Fallback: intensity threshold inside body (CPU) so radiomics can still run.
    """
    backend = segmentation_backend()
    if backend["backend"].startswith("totalsegmentator"):
        try:
            mask = _totalsegmentator_mask(volume)
            return mask, backend["backend"]
        except Exception:
            pass
    return _threshold_mask(volume), "threshold_cpu"


def _threshold_mask(volume: np.ndarray) -> np.ndarray:
    """Rough soft-tissue / lesion proxy mask for classical radiomics."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.ones(volume.shape, dtype=bool)
    # Body vs air
    body = volume > -500
    if body.sum() < 100:
        body = volume > np.percentile(finite, 10)
    roi_vals = volume[body]
    if roi_vals.size == 0:
        return body
    lo = np.percentile(roi_vals, 60)
    hi = np.percentile(roi_vals, 99)
    mask = body & (volume >= lo) & (volume <= hi)
    if mask.sum() < 50:
        mask = body
    return mask


def _totalsegmentator_mask(volume: np.ndarray) -> np.ndarray:
    """Optional deep segmentation. Requires TotalSegmentator + compatible torch."""
    # Lazy import; not required for the default path.
    from totalsegmentator.python_api import totalsegmentator  # type: ignore
    import tempfile
    import nibabel as nib
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        nii_in = tmp_path / "vol.nii.gz"
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), str(nii_in))
        out_dir = tmp_path / "seg"
        device = "gpu" if segmentation_backend()["cuda"] else "cpu"
        totalsegmentator(str(nii_in), str(out_dir), task="total", device=device, quiet=True)
        # Prefer lung or any available label map
        candidates = list(out_dir.glob("*.nii.gz"))
        if not candidates:
            raise RuntimeError("TotalSegmentator produced no masks")
        preferred = [p for p in candidates if "lung" in p.name.lower()]
        mask_img = nib.load(str((preferred or candidates)[0]))
        mask = np.asarray(mask_img.dataobj) > 0
        if mask.shape != volume.shape:
            raise RuntimeError("Segmentation shape mismatch")
        return mask
