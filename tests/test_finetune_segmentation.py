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


def test_dual_eval_comparison_log(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from agentic_radiogen.imaging.tumor_models.finetune import torch_loop

    lines: list[str] = []
    monkeypatch.setattr(
        torch_loop,
        "log",
        lambda msg, enabled=True: lines.append(str(msg)),
    )
    baseline = {
        "fg_centered": {
            "dice": 0.2, "iou": 0.1, "precision": 0.5, "recall": 0.3,
            "specificity": 0.99, "rvd": 1.0, "hd95": 5.0, "hd": 6.0, "assd": 2.0, "loss": 0.5,
        },
        "full_volume": {
            "dice": 0.7, "iou": 0.6, "precision": 0.9, "recall": 0.6,
            "specificity": 0.99, "rvd": 0.2, "hd95": 2.0, "hd": 3.0, "assd": 1.0, "loss": 0.4,
        },
    }
    final = {
        "fg_centered": {
            "dice": 0.3, "iou": 0.2, "precision": 0.4, "recall": 0.4,
            "specificity": 0.98, "rvd": 0.8, "hd95": 4.0, "hd": 5.0, "assd": 1.5, "loss": 0.45,
        },
        "full_volume": {
            "dice": 0.65, "iou": 0.55, "precision": 0.85, "recall": 0.55,
            "specificity": 0.99, "rvd": 0.3, "hd95": 3.0, "hd": 4.0, "assd": 1.2, "loss": 0.42,
        },
    }
    torch_loop._log_dual_eval_comparison(baseline, final)
    joined = "\n".join(lines)
    assert "Final comparison" in joined
    assert "fg_centered" in joined and "full_volume" in joined
    assert "sliding-window" in joined
    assert "+0.1000" in joined  # fg dice 0.2 → 0.3
    for metric in ("iou", "precision", "recall", "specificity", "rvd", "hd95", "hd", "assd", "loss"):
        assert metric in joined


def test_sliding_window_covers_full_volume() -> None:
    import torch
    from torch import nn

    from agentic_radiogen.imaging.tumor_models.finetune.torch_loop import (
        sliding_window_inference,
    )

    class ConstNet(nn.Module):
        def forward(self, x):  # type: ignore[no-untyped-def]
            # 2-class logits: channel 1 high in the whole patch.
            b, _, z, y, xx = x.shape
            out = torch.zeros(b, 2, z, y, xx, device=x.device, dtype=x.dtype)
            out[:, 1] = 2.0
            out[:, 0] = -2.0
            return out

    model = ConstNet()
    vol = torch.zeros(1, 1, 20, 30, 28)
    logits = sliding_window_inference(
        model, vol, patch_size=(8, 12, 10), overlap=0.5
    )
    assert logits.shape == (1, 2, 20, 30, 28)
    pred = torch.argmax(logits, dim=1)
    assert int(pred.sum()) == 20 * 30 * 28


def test_pristine_catalog_copy_and_out_ckpt_guard(tmp_path: Path) -> None:
    import torch

    from agentic_radiogen.imaging.tumor_models.finetune.resolve import (
        assert_out_ckpt_safe,
        ensure_pristine_catalog_checkpoint,
        _is_agentic_finetune_checkpoint,
    )

    fold = tmp_path / "fold_0"
    fold.mkdir()
    (tmp_path / "plans.pkl").write_bytes(b"plans")
    raw = fold / "model_final_checkpoint.model"
    # Minimal nnU-Net-like blob (not our FT format).
    torch.save({"state_dict": {"w": torch.zeros(1)}, "epoch": 0}, raw)
    pristine = ensure_pristine_catalog_checkpoint(raw, verbose=False)
    assert pristine.is_file()
    assert pristine.name.endswith(".pristine")
    assert raw.is_file()
    # Second call reuses the same pristine file.
    assert ensure_pristine_catalog_checkpoint(raw, verbose=False) == pristine

    with pytest.raises(ValueError, match="catalog"):
        assert_out_ckpt_safe(pristine, pristine)
    with pytest.raises(ValueError, match="catalog model directory"):
        assert_out_ckpt_safe(fold / "oops.model", pristine)
    # Fine-tune outs elsewhere are fine.
    assert_out_ckpt_safe(tmp_path / "outputs" / "ft.model", pristine)

    ft = tmp_path / "outputs" / "ft.model"
    ft.parent.mkdir(parents=True)
    torch.save(
        {
            "state_dict": {"w": torch.zeros(1)},
            "backend": "nnunet",
            "dataset": "ntcia",
            "pretrained": str(pristine),
        },
        ft,
    )
    assert _is_agentic_finetune_checkpoint(ft) is True
    assert _is_agentic_finetune_checkpoint(raw) is False


def test_lidc_requires_pydicom_upfront(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agentic_radiogen.imaging.tumor_models.finetune import lidc_idri

    (tmp_path / "LIDC-IDRI-0001").mkdir()
    (tmp_path / "LIDC-IDRI-0002").mkdir()
    real_import = __import__

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pydicom" or name.startswith("pydicom."):
            raise ImportError("No module named 'pydicom'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    with pytest.raises(ModuleNotFoundError, match="pydicom"):
        lidc_idri.discover_lidc_cases(tmp_path, cache_dir=tmp_path / "_finetune_cache", verbose=False)


def test_find_plans_pkl_next_to_fold_ckpt(tmp_path: Path) -> None:
    from agentic_radiogen.imaging.tumor_models.finetune.nnunet_weights import (
        find_plans_pkl,
    )

    trainer = tmp_path / "nnUNetTrainerV2__nnUNetPlansv2.1"
    fold = trainer / "fold_0"
    fold.mkdir(parents=True)
    plans = trainer / "plans.pkl"
    plans.write_bytes(b"x")
    ckpt = fold / "model_final_checkpoint.model"
    ckpt.write_bytes(b"y")
    assert find_plans_pkl(ckpt) == plans


def test_torch_dice_ce_loss_runs() -> None:
    import torch

    from agentic_radiogen.imaging.tumor_models.finetune.torch_loop import (
        _TorchDiceCELoss,
    )

    loss_fn = _TorchDiceCELoss()
    logits = torch.randn(2, 2, 4, 4, 4)
    target = torch.zeros(2, 1, 4, 4, 4)
    target[:, :, 1:3, 1:3, 1:3] = 1
    val = loss_fn(logits, target)
    assert torch.isfinite(val)
    assert val.ndim == 0


def test_soft_load_reports_low_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from torch import nn

    from agentic_radiogen.imaging.tumor_models.finetune import torch_loop

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(2, 2)

    model = Tiny()
    ckpt = tmp_path / "x.pt"
    # Completely different parameter names → 0% overlap.
    torch.save({"state_dict": {"conv.weight": torch.zeros(1, 1)}}, ckpt)
    logs: list[str] = []
    monkeypatch.setattr(
        torch_loop,
        "log",
        lambda msg, enabled=True: logs.append(str(msg)),
    )
    torch_loop.soft_load(model, ckpt, map_location="cpu")
    assert any("shape_matched=0" in m or "0.0%" in m for m in logs)
    assert any("WARNING" in m and "overlap" in m.lower() for m in logs)


def test_lidc_xml_sop_extract(tmp_path: Path) -> None:
    from agentic_radiogen.imaging.tumor_models.finetune.lidc_idri import _xml_sop_uids

    xml = tmp_path / "ann.xml"
    xml.write_text(
        """<?xml version="1.0"?>
        <LidcReadMessage xmlns="http://www.nih.gov">
          <readingSession><unblindedReadNodule>
            <roi><inclusion>TRUE</inclusion>
              <imageSOP_UID>1.2.3.4</imageSOP_UID>
              <edgeMap><xCoord>1</xCoord><yCoord>2</yCoord></edgeMap>
            </roi>
          </unblindedReadNodule></readingSession>
        </LidcReadMessage>
        """,
        encoding="utf-8",
    )
    assert _xml_sop_uids(xml) == {"1.2.3.4"}


def test_looks_like_lidc_root(tmp_path: Path) -> None:
    from agentic_radiogen.imaging.tumor_models.finetune.lidc_idri import (
        looks_like_lidc_root,
    )

    (tmp_path / "LIDC-IDRI-0001").mkdir()
    (tmp_path / "LIDC-IDRI-0002").mkdir()
    assert looks_like_lidc_root(tmp_path) is True
    assert looks_like_lidc_root(tmp_path / "missing") is False


def test_finetune_cache_is_not_lidc_root(tmp_path: Path) -> None:
    from agentic_radiogen.imaging.tumor_models.finetune.lidc_idri import (
        looks_like_lidc_root,
    )
    from agentic_radiogen.imaging.tumor_models.finetune.datasets import _search_roots

    cache = tmp_path / "_finetune_cache"
    for pid in ("LIDC-IDRI-0001", "LIDC-IDRI-0002"):
        d = cache / pid
        d.mkdir(parents=True)
        (d / "image.nii.gz").write_bytes(b"x")
        (d / "label.nii.gz").write_bytes(b"y")
    assert looks_like_lidc_root(cache) is False
    roots = _search_roots("ntcia", tmp_path)
    assert all(p.name != "_finetune_cache" for p in roots)


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


def test_binary_seg_metrics_empty_pred_is_nan_not_sentinel() -> None:
    from agentic_radiogen.imaging.tumor_models.finetune.metrics import average_metrics

    gt = np.zeros((8, 8, 8), dtype=bool)
    gt[2:5, 2:5, 2:5] = True
    pred = np.zeros_like(gt)
    m = binary_seg_metrics(pred, gt, spacing_zyx=(1.0, 1.0, 1.0))
    assert m["dice"] < 0.1
    assert np.isnan(m["hd"]) and np.isnan(m["hd95"]) and np.isnan(m["assd"])
    assert np.isfinite(m["rvd"])  # GT nonempty → RVD defined (pred empty → ~1)
    avg = average_metrics([m, {"dice": 0.5, "hd": 2.0, "hd95": 1.5, "assd": 1.0, "rvd": 0.1}])
    assert avg["hd"] == pytest.approx(2.0)
    assert avg["dice"] == pytest.approx((m["dice"] + 0.5) / 2.0)


def test_binary_seg_metrics_empty_empty_is_nan() -> None:
    empty = np.zeros((8, 8, 8), dtype=bool)
    m = binary_seg_metrics(empty, empty, spacing_zyx=(1.0, 1.0, 1.0))
    assert np.isnan(m["dice"]) and np.isnan(m["iou"])


def test_fg_crop_keeps_nodule_in_patch() -> None:
    from agentic_radiogen.imaging.tumor_models.finetune.torch_loop import _crop_or_pad

    image = np.zeros((40, 80, 80), dtype=np.float32)
    label = np.zeros_like(image)
    label[5, 10, 10] = 1.0  # nodule far from volume center
    img_c, lab_c = _crop_or_pad(
        image, label, (16, 32, 32), training=False, fg_oversample=1.0
    )
    assert lab_c.sum() >= 1.0
    # Training FG oversample should usually keep the voxel too.
    hits = 0
    for _ in range(20):
        _, lab_t = _crop_or_pad(
            image, label, (16, 32, 32), training=True, fg_oversample=1.0
        )
        hits += int(lab_t.sum() >= 1.0)
    assert hits == 20


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
