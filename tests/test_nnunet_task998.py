import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agentic_radiogen.imaging.tumor_models import nnunet_lung
from agentic_radiogen.imaging.tumor_models.nnunet_task998 import (
    _find_trainer,
    run_nnunet_task998_mask,
    task_root,
)
from agentic_radiogen.imaging.tumor_models.registry import (
    get_model,
    list_models_for_disease,
)
from agentic_radiogen.imaging.tumor_models.store import ModelStore


def test_task998_is_registered_for_lung_ct():
    spec = get_model("nnunet_task998_lungct")

    assert spec is not None
    assert spec.runner == "nnunet_task998"
    assert spec.is_tumor_model
    assert spec in list_models_for_disease("lung adenocarcinoma", "CT")


def test_default_task_root_searches_shared_segmentation_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTIC_RADIOGEN_TASK998_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    trainer = (
        tmp_path
        / "data_cache"
        / "seg_models"
        / "Task998_LungCTSegmentation"
        / "nnUNetTrainerV2__nnUNetPlansv2.1"
    )
    (trainer / "fold_0").mkdir(parents=True)
    (trainer / "plans.pkl").touch()
    (trainer / "fold_0" / "model_final_checkpoint.model").touch()

    assert task_root() == tmp_path / "data_cache" / "seg_models"
    assert _find_trainer(task_root()) == trainer


def test_missing_task998_weights_report_manual_install_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_RADIOGEN_TASK998_MODEL_DIR", str(tmp_path / "weights"))
    volume = np.zeros((8, 16, 16), dtype=np.float32)
    spec = get_model("nnunet_task998_lungct")
    assert spec is not None
    store = ModelStore(root=tmp_path)

    assert not store.is_ready(spec)
    with pytest.raises(FileNotFoundError, match="Task998 weights are not installed"):
        store.ensure_local(spec)
    with pytest.raises(FileNotFoundError, match="Task998_LungCTSegmentation weights not found"):
        run_nnunet_task998_mask(volume, verbose=False)


def test_task998_runner_uses_installed_trainer_and_returns_zyx_mask(
    tmp_path, monkeypatch
):
    task_root = tmp_path / "Task998_LungCTSegmentation"
    trainer = task_root / "nnUNetTrainerV2__nnUNetPlansv2.1"
    fold = trainer / "fold_0"
    fold.mkdir(parents=True)
    (trainer / "plans.pkl").touch()
    (fold / "model_final_checkpoint.model").touch()
    monkeypatch.setenv("AGENTIC_RADIOGEN_TASK998_MODEL_DIR", str(task_root))

    class FakeNifti:
        def set_data_dtype(self, _dtype):
            pass

    fake_nibabel = SimpleNamespace(
        Nifti1Image=lambda *_args: FakeNifti(),
        save=lambda _image, path: Path(path).touch(),
        load=lambda _path: SimpleNamespace(
            dataobj=np.ones((16, 16, 8), dtype=np.uint8)
        ),
    )
    monkeypatch.setitem(sys.modules, "nibabel", fake_nibabel)
    called = []

    def fake_predict(model_dir, _input_dir, output_dir, *, verbose):
        called.append((model_dir, verbose))
        (output_dir / "prediction.nii.gz").touch()

    monkeypatch.setattr(nnunet_lung, "_predict_v1", fake_predict)
    volume = np.zeros((8, 16, 16), dtype=np.float32)

    mask = run_nnunet_task998_mask(volume, verbose=False)

    assert mask.shape == volume.shape
    assert mask.all()
    assert called == [(trainer, False)]
