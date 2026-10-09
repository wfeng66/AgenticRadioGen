from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from agentic_radiogen.agents.imaging import ImagingRadiomicsAgent
from agentic_radiogen.data.tcia_client import TciaClient
from agentic_radiogen.imaging.radiomics_extract import extract_radiomics_from_volume
from agentic_radiogen.imaging.segment import make_roi_mask, segmentation_backend
from agentic_radiogen.schemas.contracts import ImageBundle, ImageSeriesRef


def test_threshold_radiomics_from_synthetic_volume() -> None:
    rng = np.random.default_rng(0)
    volume = rng.normal(loc=-800, scale=50, size=(16, 32, 32)).astype(np.float32)
    volume[:, 8:24, 8:24] = rng.normal(loc=40, scale=20, size=(16, 16, 16))
    features, summary = extract_radiomics_from_volume(volume)
    assert features["original_shape_VoxelVolume"] > 0
    assert "original_glcm_Entropy" in features
    assert summary["size"] == features["original_shape_VoxelVolume"]
    # PyRadiomics full set: shape + firstorder + GLCM/GLRLM/GLSZM/GLDM/NGTDM
    analysis = {k: v for k, v in features.items() if not str(k).startswith("meta_")}
    assert len(analysis) >= 100
    assert any(k.startswith("original_shape_") for k in analysis)
    assert any(k.startswith("original_firstorder_") for k in analysis)
    assert any(k.startswith("original_glcm_") for k in analysis)
    assert any(k.startswith("original_glrlm_") for k in analysis)
    assert any(k.startswith("original_glszm_") for k in analysis)
    assert any(k.startswith("original_gldm_") for k in analysis)
    assert any(k.startswith("original_ngtdm_") for k in analysis)
    mask, backend = make_roi_mask(volume)
    assert mask.any()
    assert backend in {"threshold_proxy", "threshold_cpu", "totalsegmentator_organ"}
    assert segmentation_backend()["backend"] in {
        "threshold_cpu",
        "totalsegmentator_cpu",
        "totalsegmentator_gpu",
    }


def test_tcia_download_series_is_cached(tmp_path: Path) -> None:
    calls = {"n": 0}

    def fake_download(url: str, params=None, timeout=300):
        calls["n"] += 1
        buf = tmp_path / "payload.zip"
        with ZipFile(buf, "w") as zf:
            zf.writestr("slice1.dcm", b"not-a-real-dicom")
        return buf.read_bytes()

    client = TciaClient(downloader=fake_download, cache_dir=tmp_path / "cache")
    first = client.download_series("1.2.3")
    second = client.download_series("1.2.3")
    assert first == second
    assert calls["n"] == 1
    assert (first / ".complete").exists()


def test_imaging_agent_uses_precomputed_and_npy(tmp_path: Path) -> None:
    npy_path = tmp_path / "vol.npy"
    np.save(npy_path, np.ones((4, 4, 4), dtype=np.float32) * 5)
    agent = ImagingRadiomicsAgent()
    matrix = agent.extract(
        ImageBundle(
            patient_ids=["A", "B"],
            series=[
                ImageSeriesRef(
                    patient_id="A",
                    series_uid="1",
                    modality="CT",
                    precomputed_features={"original_glcm_Entropy": 2.5},
                ),
                ImageSeriesRef(
                    patient_id="B",
                    series_uid="2",
                    modality="CT",
                    local_path=str(npy_path),
                ),
            ],
        )
    )
    assert matrix.features["A"]["original_glcm_Entropy"] == 2.5
    assert matrix.features["B"]["original_firstorder_Mean"] == 5.0


def test_imaging_agent_skips_failed_series() -> None:
    agent = ImagingRadiomicsAgent()
    matrix = agent.extract(
        ImageBundle(
            patient_ids=["A", "B"],
            series=[
                ImageSeriesRef(
                    patient_id="A",
                    series_uid="1",
                    modality="CT",
                    precomputed_features={"original_glcm_Entropy": 1.0},
                ),
                ImageSeriesRef(patient_id="B", series_uid="2", modality="CT"),
            ],
        )
    )
    assert matrix.patient_ids == ["A"]
    assert matrix.features["A"]["original_glcm_Entropy"] == 1.0


def test_imaging_agent_fails_without_payload() -> None:
    agent = ImagingRadiomicsAgent()
    with pytest.raises(ValueError, match="No radiomic payload"):
        agent.extract(
            ImageBundle(
                patient_ids=["A"],
                series=[ImageSeriesRef(patient_id="A", series_uid="1", modality="CT")],
            )
        )
