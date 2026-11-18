"""Unit tests for segmentation fine-tune discovery + CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agentic_radiogen.imaging.tumor_models.finetune import (
    FineTuneConfig,
    discover_cases,
    get_trainer,
    normalize_model_name,
    resolve_spec,
    run_finetune,
    trainer_backend,
)
from agentic_radiogen.imaging.tumor_models.finetune.datasets import list_datasets
from agentic_radiogen.imaging.tumor_models.finetune.metrics import binary_seg_metrics
from agentic_radiogen.imaging.tumor_models.finetune.torch_loop import (
    normalize_mode,
    split_for_mode,
)
from agentic_radiogen.imaging.tumor_models.finetune.base import CasePair


def _write_nifti(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel required")
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(path))


def test_normalize_model_aliases() -> None:
    assert normalize_model_name("nnunet") == "nnunet_msd_lung"
    assert normalize_model_name("task998") == "nnunet_task998_lungct"
    assert normalize_model_name("vista3d") == "monai_vista3d"


def test_trainer_backend_for_nnunet() -> None:
    spec = resolve_spec("nnunet")
    assert trainer_backend(spec) == "nnunet"
    assert get_trainer("nnunet").name == "nnunet"


def test_trainer_backend_for_swin() -> None:
    spec = resolve_spec("monai_swin_unetr_ssl")
    assert trainer_backend(spec) == "monai"


def test_discover_flat_pairs(tmp_path: Path) -> None:
    img = np.zeros((8, 8, 8), dtype=np.float32)
    lab = np.zeros((8, 8, 8), dtype=np.float32)
    lab[2:5, 2:5, 2:5] = 1
    _write_nifti(tmp_path / "images" / "caseA.nii.gz", img)
    _write_nifti(tmp_path / "labels" / "caseA.nii.gz", lab)
    _write_nifti(tmp_path / "images" / "caseB.nii.gz", img)
    _write_nifti(tmp_path / "labels" / "caseB.nii.gz", lab)

    cases = discover_cases("ntcia", tmp_path)
    assert [c.case_id for c in cases] == ["caseA", "caseB"]


def test_discover_nnunet_raw(tmp_path: Path) -> None:
    img = np.zeros((8, 8, 8), dtype=np.float32)
    lab = np.zeros((8, 8, 8), dtype=np.float32)
    _write_nifti(tmp_path / "imagesTr" / "p1_0000.nii.gz", img)
    _write_nifti(tmp_path / "labelsTr" / "p1.nii.gz", lab)
    cases = discover_cases("nnunet", tmp_path)
    assert len(cases) == 1
    assert cases[0].case_id == "p1"


def test_discover_case_folders(tmp_path: Path) -> None:
    img = np.zeros((8, 8, 8), dtype=np.float32)
    lab = np.zeros((8, 8, 8), dtype=np.float32)
    _write_nifti(tmp_path / "R01-001" / "image.nii.gz", img)
    _write_nifti(tmp_path / "R01-001" / "label.nii.gz", lab)
    cases = discover_cases("generic", tmp_path)
    assert len(cases) == 1
    assert cases[0].case_id == "R01-001"


def test_normalize_mode_aliases() -> None:
    assert normalize_mode("tune-only") == "tune-only"
    assert normalize_mode("evl") == "evl"
    assert normalize_mode("eval") == "evl"


def test_lidc_series_key_overlap() -> None:
    from agentic_radiogen.imaging.tumor_models.finetune.lidc_idri import (
        _normalize_series_token,
        _series_keys,
    )

    dicom = Path("LIDC-IDRI-0001/2000-01-01-30178/3000566-03192")
    manifest = Path(
        "LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192"
    )
    assert set(_series_keys(dicom)) & set(_series_keys(manifest))
    # Descriptive TCIA names that previously failed to match:
    a = Path("x/4-1.25 STANDARD-54861")
    b = Path("x/4.000000-1.25 STANDARD-54861")
    assert set(_series_keys(a)) & set(_series_keys(b))
    assert _normalize_series_token(a.name) == _normalize_series_token(b.name)


def test_looks_like_lidc_root(tmp_path: Path) -> None:
    from agentic_radiogen.imaging.tumor_models.finetune.lidc_idri import (
        looks_like_lidc_root,
    )

    (tmp_path / "LIDC-IDRI-0001").mkdir()
    (tmp_path / "LIDC-IDRI-0002").mkdir()
    assert looks_like_lidc_root(tmp_path) is True
    assert looks_like_lidc_root(tmp_path / "missing") is False


def test_skip_log_roundtrip_and_force_clears_cache(tmp_path: Path) -> None:
    from agentic_radiogen.imaging.tumor_models.finetune.lidc_idri import (
        _clear_patient_cache,
        _load_skip_log,
        _save_skip_log,
    )

    cache = tmp_path / "_finetune_cache"
    cache.mkdir()
    _save_skip_log(cache, {"LIDC-IDRI-0069": "no XML annotation in manifest"})
    loaded = _load_skip_log(cache)
    assert "LIDC-IDRI-0069" in loaded

    out = cache / "LIDC-IDRI-0069"
    out.mkdir()
    (out / "image.nii.gz").write_bytes(b"x")
    (out / "label.nii.gz").write_bytes(b"y")
    _clear_patient_cache(out)
    assert not (out / "image.nii.gz").is_file()
    assert not (out / "label.nii.gz").is_file()


def test_split_tune_only_uses_all() -> None:
    cases = [
        CasePair("a", Path("a.nii.gz"), Path("a_l.nii.gz")),
        CasePair("b", Path("b.nii.gz"), Path("b_l.nii.gz")),
        CasePair("c", Path("c.nii.gz"), Path("c_l.nii.gz")),
        CasePair("d", Path("d.nii.gz"), Path("d_l.nii.gz")),
        CasePair("e", Path("e.nii.gz"), Path("e_l.nii.gz")),
    ]
    train, evl = split_for_mode(cases, mode="tune-only", val_frac=0.2, seed=0)
    assert len(train) == 5 and evl == []


def test_split_evl_holds_out_20pct() -> None:
    cases = [
        CasePair(str(i), Path(f"{i}.nii.gz"), Path(f"{i}_l.nii.gz")) for i in range(10)
    ]
    train, evl = split_for_mode(cases, mode="evl", val_frac=0.2, seed=0)
    assert len(evl) == 2
    assert len(train) == 8
    assert {c.case_id for c in train}.isdisjoint({c.case_id for c in evl})


def test_binary_seg_metrics_perfect() -> None:
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    m = binary_seg_metrics(mask, mask, spacing_zyx=(1.0, 1.0, 1.0))
    assert m["dice"] == pytest.approx(1.0, abs=1e-5)
    assert m["iou"] == pytest.approx(1.0, abs=1e-5)
    assert m["rvd"] == pytest.approx(0.0, abs=1e-5)
    assert m["hd"] == pytest.approx(0.0, abs=1e-5)
    assert m["hd95"] == pytest.approx(0.0, abs=1e-5)
    assert m["assd"] == pytest.approx(0.0, abs=1e-5)


def test_binary_seg_metrics_shift_has_surface_error() -> None:
    gt = np.zeros((16, 16, 16), dtype=bool)
    gt[4:10, 4:10, 4:10] = True
    pred = np.zeros_like(gt)
    pred[5:11, 4:10, 4:10] = True  # shifted by 1 voxel in z
    m = binary_seg_metrics(pred, gt, spacing_zyx=(1.0, 1.0, 1.0))
    assert 0.0 < m["dice"] < 1.0
    assert m["hd"] >= 1.0
    assert m["assd"] > 0.0
    assert m["rvd"] >= 0.0


def test_dry_run_writes_out_ckpt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    img = np.zeros((8, 8, 8), dtype=np.float32)
    lab = np.zeros((8, 8, 8), dtype=np.float32)
    lab[1:4, 1:4, 1:4] = 1
    root = tmp_path / "ntcia"
    _write_nifti(root / "images" / "a.nii.gz", img)
    _write_nifti(root / "labels" / "a.nii.gz", lab)
    _write_nifti(root / "images" / "b.nii.gz", img)
    _write_nifti(root / "labels" / "b.nii.gz", lab)

    fake_ckpt = tmp_path / "pretrained.model"
    fake_ckpt.write_bytes(b"fake")

    from agentic_radiogen.imaging.tumor_models.finetune import nnunet_trainer

    monkeypatch.setattr(
        nnunet_trainer,
        "resolve_in_checkpoint",
        lambda cfg, verbose=True: fake_ckpt,
    )

    out = tmp_path / "out" / "dry.json"
    result = run_finetune(
        FineTuneConfig(
            model="nnunet",
            dataset="ntcia",
            data_path=root,
            out_ckpt=out,
            work_dir=tmp_path / "work",
            mode="evl",
            epochs=3,
            dry_run=True,
            allow_download=False,
        )
    )
    assert result.out_ckpt == out
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["mode"] == "evl"
    assert payload["epochs"] == 3
    assert payload["n_train"] + payload["n_eval"] == 2


def test_cli_epoch_and_mode() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "finetune_segmentation.py"
    spec = importlib.util.spec_from_file_location("finetune_segmentation_cli2", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.build_parser().parse_args(
        [
            "--model",
            "nnunet",
            "--data-path",
            "x",
            "--out-ckpt",
            "y",
            "--mode",
            "evl",
            "--epoch",
            "7",
        ]
    )
    assert args.mode == "evl"
    assert args.epoch == 7


def test_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "finetune_segmentation.py"
    spec = importlib.util.spec_from_file_location("finetune_segmentation_cli", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--list"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "nnunet" in payload["models"]
    assert "ntcia" in payload["datasets"]
    assert set(list_datasets()) == set(payload["datasets"])
