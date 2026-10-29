"""nnU-Net v1 runner for the manually downloaded Task998 lung CT model."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

_TASK_NAME = "Task998_LungCTSegmentation"
_TRAINER_NAME = "nnUNetTrainerV2__nnUNetPlansv2.1"
_KAGGLE_URL = "https://www.kaggle.com/models/dejankuhn/ct-lung-tumor-segmentation-model"


def task_root() -> Path:
    """Return the directory searched for the extracted Task998 trainer."""
    override = (os.environ.get("AGENTIC_RADIOGEN_TASK998_MODEL_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.cwd() / "data_cache" / "seg_models"


def _find_trainer(root: Path) -> Path | None:
    from agentic_radiogen.imaging.tumor_models.nnunet_lung import _trainer_ready

    direct = root / _TRAINER_NAME
    if _trainer_ready(direct):
        return direct
    if root.name == _TRAINER_NAME and _trainer_ready(root):
        return root
    for candidate in root.rglob(_TRAINER_NAME):
        if _trainer_ready(candidate):
            return candidate
    return None


def run_nnunet_task998_mask(
    volume: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float] | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Run the manually installed Task998 nnU-Net v1 weights on a CT volume."""
    from agentic_radiogen.imaging.segment import _nifti_from_volume_zyx
    from agentic_radiogen.imaging.tumor_models.nnunet_lung import (
        _prepare_zyx,
        _predict_v1,
        _seg_to_zyx,
    )
    from agentic_radiogen.util.progress import log

    root = task_root()
    model_dir = _find_trainer(root)
    if model_dir is None:
        raise FileNotFoundError(
            "Task998_LungCTSegmentation weights not found. Download the model from "
            f"{_KAGGLE_URL}, then place the extracted "
            f"{_TASK_NAME}/{_TRAINER_NAME} folder under {root}. "
            "Alternatively set AGENTIC_RADIOGEN_TASK998_MODEL_DIR to the "
            f"{_TASK_NAME} directory."
        )

    vol = _prepare_zyx(volume)
    min_slices = int(os.environ.get("AGENTIC_RADIOGEN_NNUNET_MIN_SLICES") or 8)
    if vol.shape[0] < min_slices:
        raise RuntimeError(
            f"too few slices for nnU-Net Task998 ({vol.shape[0]} < {min_slices})"
        )

    import nibabel as nib

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        image, volume_shape = _nifti_from_volume_zyx(
            vol,
            spacing_zyx=spacing_zyx or (1.0, 1.0, 1.0),
        )
        nib.save(image, str(input_dir / "case_0000.nii.gz"))
        log(
            f"[seg-model] Running nnU-Net Task998 from {model_dir}",
            enabled=verbose,
        )
        _predict_v1(model_dir, input_dir, output_dir, verbose=verbose)

        predictions = sorted(output_dir.glob("*.nii.gz"))
        if not predictions:
            raise RuntimeError("nnU-Net Task998 produced no output mask")
        prediction = np.asanyarray(nib.load(str(predictions[0])).dataobj)
        mask = _seg_to_zyx(prediction, volume_shape)
        if not mask.any():
            raise RuntimeError("nnU-Net Task998 returned an empty mask")
        return mask
