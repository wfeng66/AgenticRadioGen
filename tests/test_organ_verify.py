"""Organ FOV gate before tumor segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agentic_radiogen.data.disease_match import expected_organ_keywords
from agentic_radiogen.imaging.organ_verify import OrganVerifyResult, verify_expected_organ


def test_expected_organ_keywords_lung() -> None:
    assert expected_organ_keywords("lung cancer", "TCGA-LUAD") == ("lung",)
    assert expected_organ_keywords("breast cancer") == ("breast",)
    assert expected_organ_keywords("unknown site xyz") == ()


def test_ts_multilabel_filters_by_organ_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Organ gate must not treat any label>0 as lung (ABD FOV false pass)."""
    from agentic_radiogen.imaging import segment as seg

    class _FakeImg:
        def __init__(self, data):
            self.dataobj = data

    # Multilabel volume in NIfTI (x,y,z): class 2 = liver-like, class 13 = lung.
    data_xyz = np.zeros((4, 4, 4), dtype=np.int16)
    data_xyz[:, :, :] = 2
    data_xyz[0, 0, 0] = 13

    monkeypatch.setattr(
        seg,
        "_nifti_from_volume_zyx",
        lambda volume, spacing_zyx=(1, 1, 1): (_FakeImg(volume), volume.shape),
    )
    monkeypatch.setattr(seg, "segmentation_backend", lambda: {"cuda": False})
    monkeypatch.setattr(
        seg,
        "_ts_label_ids_for_keywords",
        lambda _task, keywords: {13} if "lung" in keywords else set(),
    )

    def fake_ts(*_a, **_k):
        return _FakeImg(data_xyz)

    import sys
    import types

    fake_mod = types.ModuleType("totalsegmentator")
    fake_api = types.ModuleType("totalsegmentator.python_api")
    fake_api.totalsegmentator = fake_ts
    sys.modules["totalsegmentator"] = fake_mod
    sys.modules["totalsegmentator.python_api"] = fake_api
    if "nibabel" not in sys.modules:
        sys.modules["nibabel"] = types.ModuleType("nibabel")

    vol = np.zeros((4, 4, 4), dtype=np.float32)
    lung_mask = seg._totalsegmentator_mask(vol, organ_keywords=("lung",))
    assert int(lung_mask.sum()) == 1

    any_mask = seg._totalsegmentator_mask(vol, organ_keywords=())
    assert int(any_mask.sum()) == 64


def test_organ_verify_skips_without_keywords(tmp_path: Path) -> None:
    result = verify_expected_organ(tmp_path, organ_keywords=())
    assert result.ok
    assert "skipped" in result.reason


def test_organ_verify_pass_fail_with_mocked_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vol = np.zeros((8, 8, 8), dtype=np.float32)

    def fake_load(_path):
        return vol, (1.0, 1.0, 1.0)

    monkeypatch.setattr(
        "agentic_radiogen.imaging.dicom_io.load_dicom_series_with_spacing",
        fake_load,
    )

    def fake_mask_ok(volume, **_kwargs):
        m = np.zeros(volume.shape, dtype=bool)
        m[:4, :4, :4] = True  # 64 voxels — below default 1000
        return m

    monkeypatch.setattr(
        "agentic_radiogen.imaging.segment._totalsegmentator_mask",
        fake_mask_ok,
    )
    fail = verify_expected_organ(tmp_path, organ_keywords=("lung",), min_voxels=1000)
    assert isinstance(fail, OrganVerifyResult)
    assert not fail.ok
    assert fail.n_organ_voxels == 64

    def fake_mask_pass(volume, **_kwargs):
        return np.ones(volume.shape, dtype=bool)  # 512 voxels with min=100

    monkeypatch.setattr(
        "agentic_radiogen.imaging.segment._totalsegmentator_mask",
        fake_mask_pass,
    )
    ok = verify_expected_organ(tmp_path, organ_keywords=("lung",), min_voxels=100)
    assert ok.ok
    assert ok.n_organ_voxels == 512


def test_fetch_tries_next_series_after_organ_gate_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentic_radiogen.data.catalog import CatalogRecord
    from agentic_radiogen.data.gdc_client import GdcClient
    from agentic_radiogen.data.live_catalog import LiveCatalog
    from agentic_radiogen.data.tcia_client import TciaClient
    from agentic_radiogen.imaging.organ_verify import OrganVerifyResult

    downloads: list[str] = []

    def download(uid: str):
        downloads.append(uid)
        dest = tmp_path / uid
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ".complete").write_text("ok", encoding="utf-8")
        (dest / "slice.dcm").write_bytes(b"x")
        return dest

    monkeypatch.setattr(
        TciaClient,
        "download_series",
        lambda self, uid: download(uid),
    )

    calls = {"n": 0}

    def fake_verify(series_dir, *, organ_keywords, min_voxels=1000):
        calls["n"] += 1
        uid = Path(series_dir).name
        if uid == "uid-bad":
            return OrganVerifyResult(
                ok=False,
                organ_keywords=tuple(organ_keywords),
                n_organ_voxels=0,
                volume_voxels=100,
                reason="no lung",
            )
        return OrganVerifyResult(
            ok=True,
            organ_keywords=tuple(organ_keywords),
            n_organ_voxels=5000,
            volume_voxels=10000,
            reason="ok",
        )

    monkeypatch.setattr(
        "agentic_radiogen.data.live_catalog.verify_expected_organ",
        fake_verify,
    )
    monkeypatch.setattr(
        "agentic_radiogen.data.live_catalog.expected_organ_keywords",
        lambda *a, **k: ("lung",),
    )
    monkeypatch.setattr(
        "agentic_radiogen.data.live_catalog.extract_series_features",
        lambda *a, **k: {
            "features": {"original_firstorder_Mean": 1.0},
            "volume_summary": {"mean": 1.0},
            "from_cache": False,
            "segmentation_backend": "mock",
            "segmentation_model_id": "mock",
            "segm_status": "success",
            "n_mask_voxels": 10,
            "mask_path": "",
            "models_attempted": "mock",
        },
    )

    catalog = LiveCatalog(
        gdc=GdcClient(poster=lambda *_a, **_k: {"data": {"hits": []}}),
        tcia=TciaClient(cache_dir=tmp_path / "cache"),
        download_dicom=True,
        extract_radiomics=True,
    )
    catalog.last_intermediate = {"disease_query": "lung cancer", "genes": ["EGFR"]}
    catalog._record_cache["TCGA-17-Z021"] = CatalogRecord(
        patient_id="TCGA-17-Z021",
        disease="lung_cancer",
        modality="CT",
        mutations={"EGFR": 0},
        expression={},
        clinical={
            "_gdc_project": "TCGA-LUAD",
            "_tcia_collection": "TCGA-LUAD",
            "_tcia_series_candidates": ["uid-bad", "uid-good"],
        },
        series_uid="uid-bad",
        radiomic_features={},
        volume_summary={},
    )
    out = catalog.fetch_records(["TCGA-17-Z021"])
    assert downloads == ["uid-bad", "uid-good"]
    assert out[0].series_uid == "uid-good"
    assert out[0].radiomic_features.get("original_firstorder_Mean") == 1.0
