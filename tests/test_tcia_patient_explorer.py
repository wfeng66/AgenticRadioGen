from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agentic_radiogen.data.gdc_client import GdcClient
from agentic_radiogen.data.tcia_client import TciaSeries
from scripts.explore_tcia_patient import _render_preview, explore_patient


class _FakeTciaClient:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.downloaded: list[str] = []

    def list_series_for_patient(
        self, patient_id: str, modality: str | None = None
    ) -> list[TciaSeries]:
        assert patient_id == "TCGA-17-Z021"
        series = [
            TciaSeries(
                patient_id,
                "uid-ct",
                "CT",
                "TCGA-LUAD",
                image_count=2,
                series_description="Chest CT",
                body_part_examined="CHEST",
                source_metadata={
                    "PatientID": patient_id,
                    "SeriesInstanceUID": "uid-ct",
                    "StudyInstanceUID": "study-1",
                    "Modality": "CT",
                    "Collection": "TCGA-LUAD",
                    "BodyPartExamined": "CHEST",
                    "ImageCount": 2,
                    "FileSize": 1000,
                    "LicenseName": "CC BY 3.0",
                    "Manufacturer": "SIEMENS",
                    "SoftwareVersions": "VB40E",
                    "TimeStamp": "2012-01-01",
                    "DateReleased": "2012-01-01",
                    "CollectionURI": "https://doi.org/example",
                },
            ),
            TciaSeries(
                patient_id,
                "uid-mr",
                "MR",
                "TCGA-LUAD",
                image_count=1,
                series_description="MR series",
                source_metadata={
                    "PatientID": patient_id,
                    "SeriesInstanceUID": "uid-mr",
                    "StudyInstanceUID": "study-1",
                    "Modality": "MR",
                    "Collection": "TCGA-LUAD",
                    "ImageCount": 1,
                    "FileSize": 500,
                },
            ),
        ]
        if modality is None:
            return series
        return [item for item in series if item.modality == modality]

    def list_patient_studies(self, patient_id: str) -> list[dict[str, Any]]:
        return [
            {
                "PatientID": patient_id,
                "PatientSex": "F",
                "PatientAge": "061Y",
                "PatientName": patient_id,
                "StudyInstanceUID": "study-1",
                "StudyDate": "1982-12-19",
                "Collection": "TCGA-LUAD",
                "SeriesCount": 2,
            }
        ]

    def series_metadata(self, series_uid: str) -> dict[str, Any]:
        return {
            "Series UID": series_uid,
            "SOP Class UID": "1.2.840.10008.5.1.4.1.1.2",
            "Annotations Flag": "false",
            "Annotation Size": "0",
        }

    def series_size(self, series_uid: str) -> dict[str, Any]:
        return {"TotalSizeInBytes": 1000 if series_uid == "uid-ct" else 500, "ObjectCount": 2}

    def download_series(self, series_uid: str) -> Path:
        self.downloaded.append(series_uid)
        destination = self.cache_dir / series_uid
        destination.mkdir(parents=True, exist_ok=True)
        return destination


class _FakeGdcClient:
    def __init__(self) -> None:
        self.downloaded: list[str] = []

    def get_case_details(self, submitter_id: str) -> dict[str, Any]:
        return {"submitter_id": submitter_id, "project": {"project_id": "TCGA-LUAD"}}

    def list_clinical_files(self, submitter_id: str) -> list[dict[str, Any]]:
        return [{"file_id": "file-1", "file_name": "clinical.xml"}]

    def download_file(self, file_id: str) -> bytes:
        self.downloaded.append(file_id)
        return b"<clinical/>"


def test_explore_patient_downloads_selected_series_and_clinical_files(
    tmp_path: Path,
) -> None:
    tcia = _FakeTciaClient(tmp_path / "cache")
    gdc = _FakeGdcClient()
    out_dir = tmp_path / "output"

    result = explore_patient(
        patient_id="TCGA-17-Z021",
        out_dir=out_dir,
        modalities={"CT"},
        previews=False,
        disease="lung cancer",
        tcia_client=tcia,  # type: ignore[arg-type]
        gdc_client=gdc,  # type: ignore[arg-type]
    )

    assert tcia.downloaded == ["uid-ct"]
    assert gdc.downloaded == ["file-1"]
    assert len(result["series"]) == 1
    assert result["series"][0]["status"] == "downloaded"
    assert (out_dir / "gdc_clinical_files" / "file-1_clinical.xml").read_bytes() == (
        b"<clinical/>"
    )
    assert (out_dir / "metadata" / "tcia_series.csv").is_file()
    assert (out_dir / "metadata" / "tcia_patient.csv").is_file()
    assert (out_dir / "metadata" / "tcia_studies.json").is_file()
    assert (out_dir / "series.log").is_file()
    series_log = (out_dir / "series.log").read_text(encoding="utf-8")
    assert "patient_id: TCGA-17-Z021" in series_log
    assert "downloaded_series_count: 1" in series_log
    assert "disease_question: lung cancer" in series_log
    assert "metadata_indicates_disease_match: True" in series_log
    assert "SeriesDescription=" in series_log or "BodyPartExamined=" in series_log
    patient_csv = (out_dir / "metadata" / "tcia_patient.csv").read_text(encoding="utf-8")
    assert "getPatientStudy.PatientSex" in patient_csv
    assert "series_records_json" in patient_csv
    assert "F" in patient_csv
    series_csv = (out_dir / "metadata" / "tcia_series.csv").read_text(encoding="utf-8")
    assert "getSeries.FileSize" in series_csv or "file_size" in series_csv
    assert "getSeriesMetaData.SOP_Class_UID" in series_csv
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["series"][0]["modality"] == "CT"
    assert manifest["tcia_patient"]["n_series"] == 1


def test_metadata_only_does_not_download_images_or_clinical_files(
    tmp_path: Path,
) -> None:
    tcia = _FakeTciaClient(tmp_path / "cache")
    gdc = _FakeGdcClient()
    out_dir = tmp_path / "output"

    result = explore_patient(
        patient_id="TCGA-17-Z021",
        out_dir=out_dir,
        modalities={"CT"},
        download=False,
        previews=False,
        disease="lung cancer",
        tcia_client=tcia,  # type: ignore[arg-type]
        gdc_client=gdc,  # type: ignore[arg-type]
    )

    assert tcia.downloaded == []
    assert gdc.downloaded == []
    assert result["series"][0]["status"] == "metadata_only"
    assert result["clinical_file_downloads"] == []
    series_log = (out_dir / "series.log").read_text(encoding="utf-8")
    assert "downloaded_series_count: 0" in series_log
    assert "metadata-only mode" in series_log


def test_series_log_reports_organ_gate_when_metadata_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentic_radiogen.imaging.organ_verify import OrganVerifyResult

    class _BlankMetaTcia(_FakeTciaClient):
        def list_series_for_patient(
            self, patient_id: str, modality: str | None = None
        ) -> list[TciaSeries]:
            series = [
                TciaSeries(
                    patient_id,
                    "uid-ct",
                    "CT",
                    "OTHER",
                    image_count=2,
                    series_description="",
                    body_part_examined="",
                    source_metadata={
                        "PatientID": patient_id,
                        "SeriesInstanceUID": "uid-ct",
                        "StudyInstanceUID": "study-1",
                        "Modality": "CT",
                        "Collection": "OTHER",
                        "ImageCount": 2,
                        "SeriesNumber": 7,
                    },
                )
            ]
            if modality is None:
                return series
            return [item for item in series if item.modality == modality]

    tcia = _BlankMetaTcia(tmp_path / "cache")

    def fake_verify(series_dir, *, organ_keywords, min_voxels=1000):
        return OrganVerifyResult(
            ok=True,
            organ_keywords=tuple(organ_keywords),
            n_organ_voxels=5000,
            volume_voxels=10000,
            reason="organ ['lung'] voxels=5000",
        )

    monkeypatch.setattr(
        "agentic_radiogen.imaging.organ_verify.verify_expected_organ",
        fake_verify,
    )

    out_dir = tmp_path / "output"
    result = explore_patient(
        patient_id="TCGA-17-Z021",
        out_dir=out_dir,
        modalities={"CT"},
        previews=False,
        disease="lung cancer",
        verify_organ=True,
        tcia_client=tcia,  # type: ignore[arg-type]
        gdc_client=_FakeGdcClient(),  # type: ignore[arg-type]
    )
    assert result["series"][0]["disease_evidence"]["metadata_indicates_match"] is False
    series_log = (out_dir / "series.log").read_text(encoding="utf-8")
    assert "metadata_indicates_disease_match: False" in series_log
    assert "organ_gate PASS" in series_log
    assert "no original metadata matched disease keywords" in series_log
    assert "downloaded_series_numbers: 7" in series_log


def test_gdc_patient_case_and_clinical_file_methods() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def poster(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((url, payload))
        if url.endswith("/cases"):
            return {
                "data": {
                    "hits": [
                        {
                            "case_id": "case-uuid",
                            "submitter_id": "TCGA-17-Z021",
                            "diagnoses": [{"primary_diagnosis": "Adenocarcinoma"}],
                        }
                    ]
                }
            }
        return {
            "data": {
                "hits": [{"file_id": "file-uuid", "file_name": "case.xml"}],
                "pagination": {"total": 1},
            }
        }

    downloaded: list[str] = []

    def downloader(url: str, *, timeout: int) -> bytes:
        downloaded.append(url)
        assert timeout == 600
        return b"clinical-file"

    client = GdcClient(poster=poster, downloader=downloader)
    case = client.get_case_details("TCGA-17-Z021")
    files = client.list_clinical_files("TCGA-17-Z021")
    content = client.download_file("file-uuid")

    assert case is not None and case["case_id"] == "case-uuid"
    assert files == [{"file_id": "file-uuid", "file_name": "case.xml"}]
    assert len(calls) == 2
    assert calls[0][1]["fields"] and calls[1][1]["filters"]["op"] == "and"
    assert content == b"clinical-file"
    assert downloaded[0].endswith("/data/file-uuid")


@pytest.mark.parametrize("instance_count", [1, 2])
def test_render_preview_handles_single_image_and_mpr(
    tmp_path: Path, instance_count: int
) -> None:
    pydicom = pytest.importorskip("pydicom")
    pytest.importorskip("matplotlib")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    series_dir = tmp_path / "dicom"
    series_dir.mkdir()
    for number in range(1, instance_count + 1):
        sop_uid = generate_uid()
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(
            str(series_dir / f"{number}.dcm"),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = sop_uid
        dataset.Rows = 8
        dataset.Columns = 8
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.InstanceNumber = number
        dataset.PixelSpacing = [1, 1]
        dataset.ImagePositionPatient = [0, 0, number]
        dataset.PixelData = (
            np.arange(64, dtype=np.uint16).reshape(8, 8) + number
        ).tobytes()
        dataset.save_as(str(series_dir / f"{number}.dcm"))

    series = TciaSeries("TCGA-17-Z021", "uid", "CT", "TCGA-LUAD")
    output = tmp_path / "preview.png"
    thumbnail = _render_preview(series_dir, output, series, instance_count)

    assert output.is_file()
    assert thumbnail.shape == (8, 8)
    assert pydicom.dcmread(str(series_dir / "1.dcm")).Rows == 8
