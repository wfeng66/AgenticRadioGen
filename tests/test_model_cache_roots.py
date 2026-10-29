from pathlib import Path

import pytest

from agentic_radiogen.imaging.tumor_models import (
    monai_lung_nodule,
    monai_vista3d,
    nnunet_autopet,
    nnunet_lung,
)


@pytest.mark.parametrize(
    ("module", "cache_env", "cache_name"),
    [
        (nnunet_lung, "AGENTIC_RADIOGEN_NNUNET_LUNG_CACHE", "nnunet_msd_lung"),
        (nnunet_autopet, "AGENTIC_RADIOGEN_AUTOPET_CACHE", "nnunet_autopet"),
        (monai_vista3d, "AGENTIC_RADIOGEN_VISTA3D_CACHE", "monai_vista3d"),
        (monai_lung_nodule, "AGENTIC_RADIOGEN_MONAI_LUNG_CACHE", "monai_lung_nodule"),
    ],
)
def test_model_cache_roots_default_to_project_cache(
    tmp_path, monkeypatch, module, cache_env, cache_name
):
    expected = tmp_path / "data_cache" / "seg_models" / cache_name
    monkeypatch.delenv(cache_env, raising=False)
    monkeypatch.setattr(module, "_DEFAULT_CACHE", expected)

    assert module.cache_root() == expected


def test_nnunet_msd_model_discovery_uses_project_cache(tmp_path, monkeypatch):
    cache = tmp_path / "data_cache" / "seg_models" / "nnunet_msd_lung"
    trainer = (
        cache
        / "3d_fullres"
        / "Task006_Lung"
        / "nnUNetTrainerV2__nnUNetPlansv2.1"
    )
    fold = trainer / "fold_0"
    fold.mkdir(parents=True)
    (trainer / "plans.pkl").touch()
    (fold / "model_final_checkpoint.model").touch()

    monkeypatch.delenv("AGENTIC_RADIOGEN_NNUNET_LUNG_CACHE", raising=False)
    monkeypatch.setattr(nnunet_lung, "_DEFAULT_CACHE", cache)

    assert nnunet_lung.nnunet_lung_model_dir() == trainer
