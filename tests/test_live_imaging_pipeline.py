from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np

from agentic_radiogen.agents.data_matcher import DataMatcherAgent
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.data.gdc_client import GdcClient
from agentic_radiogen.data.gate import FlagGate
from agentic_radiogen.data.live_catalog import LiveCatalog
from agentic_radiogen.data.tcia_client import TciaClient
from agentic_radiogen.pipeline.stage2 import extract_parallel
from tests.conftest import LUNG_QUESTION


def _write_minimal_dicom(path: Path, value: float = 40.0) -> None:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.Rows = 16
    ds.Columns = 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.InstanceNumber = 1
    ds.ImagePositionPatient = [0, 0, 0]
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    arr = (np.ones((16, 16), dtype=np.int16) * int(value))
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), write_like_original=False)


def test_live_fetch_downloads_dicom_and_builds_radiomics(tmp_path: Path) -> None:
    def tcia_get(_url: str, params=None):
        if "getCollectionValues" in _url:
            return [
                {"Collection": "TCGA-LUAD"},
                {"Collection": "TCGA-LUSC"},
                {"Collection": "TCGA-BRCA"},
            ]
        collection = str((params or {}).get("Collection") or "")
        if collection and collection != "TCGA-LUAD":
            return []
        return [
            {
                "PatientID": "TCGA-05-4244",
                "SeriesInstanceUID": "1.2.840.test.4244",
                "Modality": "CT",
                "Collection": "TCGA-LUAD",
            },
            {
                "PatientID": "TCGA-05-4249",
                "SeriesInstanceUID": "1.2.840.test.4249",
                "Modality": "CT",
                "Collection": "TCGA-LUAD",
            },
        ]

    def tcia_download(url: str, params=None, timeout=300):
        uid = (params or {}).get("SeriesInstanceUID", "x")
        zip_path = tmp_path / f"{uid}.zip"
        with ZipFile(zip_path, "w") as zf:
            dcm = tmp_path / f"{uid}.dcm"
            _write_minimal_dicom(dcm, value=55 if "4244" in uid else 10)
            zf.write(dcm, arcname="slice.dcm")
        return zip_path.read_bytes()

    def gdc_post(url: str, payload: dict):
        if "/projects" in url:
            return {
                "data": {
                    "hits": [
                        {
                            "project_id": "TCGA-LUAD",
                            "name": "Lung Adenocarcinoma",
                            "primary_site": ["Lung"],
                            "disease_type": ["Adenomas and Adenocarcinomas"],
                        }
                    ]
                }
            }
        if "facets" in payload:
            return {
                "data": {
                    "aggregations": {
                        "diagnoses.primary_diagnosis": {
                            "buckets": [{"key": "Adenocarcinoma, NOS", "doc_count": 2}]
                        }
                    }
                }
            }
        if "ssm_occurrences" in url or "ssm.consequence" in str(payload):
            return {
                "data": {
                    "hits": [
                        {
                            "case": {"submitter_id": "TCGA-05-4244"},
                            "ssm": {
                                "consequence": [
                                    {"transcript": {"gene": {"symbol": "EGFR"}}}
                                ]
                            },
                        }
                    ]
                }
            }
        return {
            "data": {
                "hits": [
                    {
                        "submitter_id": "TCGA-05-4244",
                        "diagnoses": [{"vital_status": "Dead", "days_to_death": 100}],
                    },
                    {
                        "submitter_id": "TCGA-05-4249",
                        "diagnoses": [{"vital_status": "Alive", "days_to_last_follow_up": 200}],
                    },
                ],
                "pagination": {"total": 2},
            }
        }

    catalog = LiveCatalog(
        gdc=GdcClient(poster=gdc_post),
        tcia=TciaClient(getter=tcia_get, downloader=tcia_download, cache_dir=tmp_path / "cache"),
        download_dicom=True,
        extract_radiomics=True,
    )
    request = OrchestratorAgent().parse_and_plan(LUNG_QUESTION)
    request = request.model_copy(update={"max_patients": 2})
    matcher = DataMatcherAgent(catalog, gate=FlagGate(True))
    images, omics = matcher.fetch(request)
    assert len(images.patient_ids) == 2
    assert all(images.series[i].precomputed_features for i in range(2))
    assert all(images.series[i].local_path for i in range(2))
    assert omics.mutations["TCGA-05-4244"]["EGFR"] == 1

    outputs = extract_parallel(images, omics)
    assert outputs.radiomics is not None
    assert outputs.genomics is not None
    assert "original_firstorder_Mean" in outputs.radiomics.feature_names
    assert "EGFR_mut" in outputs.genomics.feature_names
    # Stats needs >=8 patients; this test only proves the live imaging path.
    assert len(outputs.patient_ids) == 2
