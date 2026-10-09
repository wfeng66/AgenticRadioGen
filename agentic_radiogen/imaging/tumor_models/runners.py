"""Run a local / auto-download segmentation backend to produce a boolean ROI mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agentic_radiogen.imaging.tumor_models.registry import TumorModelSpec


def run_model(
    spec: TumorModelSpec,
    volume: np.ndarray,
    *,
    model_dir: Path | None = None,
    spacing_zyx: tuple[float, float, float] | None = None,
    allow_download: bool = True,
    verbose: bool = True,
    modality: str = "CT",
    pet_volume: np.ndarray | None = None,
) -> np.ndarray:
    """Return boolean mask with same shape as ``volume``."""
    if spec.runner == "threshold_proxy":
        from agentic_radiogen.imaging.segment import _threshold_mask

        return _threshold_mask(volume)
    if spec.runner == "totalsegmentator_organ":
        from agentic_radiogen.imaging.segment import _totalsegmentator_mask

        return _totalsegmentator_mask(
            volume,
            organ_keywords=spec.organ_keywords,
            spacing_zyx=spacing_zyx,
        )
    if spec.runner == "totalsegmentator_task":
        from agentic_radiogen.imaging.segment import _totalsegmentator_mask

        task = (spec.task_name or "").strip()
        if not task:
            raise ValueError(f"Model '{spec.model_id}' missing task_name")
        return _totalsegmentator_mask(
            volume,
            organ_keywords=spec.organ_keywords,
            task=task,
            spacing_zyx=spacing_zyx,
        )
    if spec.runner == "nnunet":
        from agentic_radiogen.imaging.tumor_models.nnunet_lung import (
            run_nnunet_msd_lung_mask,
        )

        return run_nnunet_msd_lung_mask(
            volume,
            spacing_zyx=spacing_zyx,
            allow_download=allow_download,
            verbose=verbose,
        )
    if spec.runner == "nnunet_autopet":
        from agentic_radiogen.imaging.tumor_models.nnunet_autopet import (
            run_nnunet_autopet_mask,
        )

        return run_nnunet_autopet_mask(
            volume,
            pet_volume=pet_volume,
            spacing_zyx=spacing_zyx,
            modality=modality,
            allow_download=allow_download,
            verbose=verbose,
        )
    if spec.runner == "monai_bundle":
        from agentic_radiogen.imaging.tumor_models.monai_lung_nodule import (
            run_monai_lung_nodule_mask,
        )

        return run_monai_lung_nodule_mask(
            volume,
            spacing_zyx=spacing_zyx,
            allow_download=allow_download,
            verbose=verbose,
        )
    if spec.runner == "fine_tune":
        raise RuntimeError(
            f"Model '{spec.model_id}' is a fine-tune backbone "
            f"({spec.operational_type}) — not a live segmentor."
        )
    if spec.runner == "promptable":
        raise RuntimeError(
            f"Model '{spec.model_id}' is promptable and needs a click/box prompt "
            f"(interactive GTV) — prompt API not wired yet."
        )
    if spec.runner == "torchscript":
        if model_dir is None:
            raise ValueError("torchscript runner requires model_dir")
        return _run_torchscript(volume, model_dir / (spec.weight_filename or "model.ts"))
    raise ValueError(f"Unknown segmentation runner '{spec.runner}'")


def _run_torchscript(volume: np.ndarray, weight_path: Path) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for TorchScript tumor models."
        ) from exc

    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing model weights: {weight_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(weight_path), map_location=device)
    model.eval()

    vol = np.asarray(volume, dtype=np.float32)
    finite = vol[np.isfinite(vol)]
    if finite.size:
        lo, hi = np.percentile(finite, [0.5, 99.5])
        if hi > lo:
            vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
        else:
            vol = np.zeros_like(vol)
    else:
        vol = np.zeros_like(vol)

    tensor = torch.from_numpy(vol)[None, None].to(device)
    with torch.no_grad():
        out = model(tensor)
    if isinstance(out, (tuple, list)):
        out = out[0]
    arr = out.detach().float().cpu().numpy()
    while arr.ndim > 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 5 and arr.shape[1] in (1, 2):
            arr = arr[0, -1] if arr.shape[1] == 2 else arr[0, 0]
        else:
            arr = arr[0]
    if arr.shape != volume.shape:
        arr = _resize_nearest(arr, volume.shape)
    if arr.dtype != np.bool_:
        mask = arr > 0.0 if (arr.min() < 0 or arr.max() > 1.5) else arr >= 0.5
    else:
        mask = arr
    return np.asarray(mask, dtype=bool)


def _resize_nearest(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    coords = []
    for in_len, out_len in zip(arr.shape, shape):
        idx = (
            (np.linspace(0, in_len - 1, out_len)).astype(int)
            if out_len
            else np.array([], dtype=int)
        )
        coords.append(idx)
    if arr.ndim == 3:
        return arr[np.ix_(coords[0], coords[1], coords[2])]
    raise ValueError(f"Unsupported mask ndim {arr.ndim}")
