from __future__ import annotations

from agentic_radiogen.agents.data_matcher import DataMatcherAgent, DownloadDeniedError
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.data.gdc_client import GdcClient
from agentic_radiogen.data.gate import FlagGate
from agentic_radiogen.data.live_catalog import LiveCatalog
from agentic_radiogen.data.tcia_client import TciaClient
from agentic_radiogen import __main__ as cli
from tests.conftest import LUNG_QUESTION


_GDC_CASES = {
    "TCGA-05-4244": {
        "submitter_id": "TCGA-05-4244",
        "diagnoses": [
            {
                "vital_status": "Dead",
                "days_to_death": 400,
                "primary_diagnosis": "Adenocarcinoma",
            }
        ],
    },
    "TCGA-05-4249": {
        "submitter_id": "TCGA-05-4249",
        "diagnoses": [
            {
                "vital_status": "Alive",
                "days_to_last_follow_up": 900,
                "primary_diagnosis": "Adenocarcinoma",
            }
        ],
    },
    "TCGA-99-NOIMG": {
        "submitter_id": "TCGA-99-NOIMG",
        "diagnoses": [
            {
                "vital_status": "Alive",
                "days_to_last_follow_up": 100,
            }
        ],
    },
}


def _requested_ids(payload: dict) -> list[str] | None:
    filters = payload.get("filters") or {}
    contents = filters.get("content")
    if isinstance(contents, list):
        for item in contents:
            field = (item.get("content") or {}).get("field")
            if field == "submitter_id":
                return list((item.get("content") or {}).get("value") or [])
    return None


def _gdc_poster(_url: str, payload: dict) -> dict:
    if "ssm_occurrences" in _url or "ssm.consequence" in str(payload):
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
    wanted = _requested_ids(payload)
    # Full-project listing (no submitter_id filter): return all cases, with pagination.
    if wanted is None:
        all_hits = list(_GDC_CASES.values())
        offset = int(payload.get("from") or 0)
        size = int(payload.get("size") or len(all_hits))
        page = all_hits[offset : offset + size]
        return {
            "data": {
                "hits": page,
                "pagination": {"from": offset, "size": size, "total": len(all_hits)},
            }
        }
    hits = [_GDC_CASES[pid] for pid in wanted if pid in _GDC_CASES]
    return {"data": {"hits": hits, "pagination": {"total": len(hits)}}}


def _tcia_getter(_url: str, params: dict) -> list[dict]:
    return [
        {
            "PatientID": "TCGA-05-4244-01A",
            "SeriesInstanceUID": "1.2.840.ct.4244",
            "Modality": "CT",
            "Collection": "TCGA-LUAD",
        },
        {
            "PatientID": "TCGA-05-4249",
            "SeriesInstanceUID": "1.2.840.ct.4249",
            "Modality": "CT",
            "Collection": "TCGA-LUAD",
        },
        {
            "PatientID": "TCGA-88-NOGDC",
            "SeriesInstanceUID": "1.2.840.ct.orphan",
            "Modality": "CT",
            "Collection": "TCGA-LUAD",
        },
    ]


def _live_catalog() -> LiveCatalog:
    return LiveCatalog(
        gdc=GdcClient(poster=_gdc_poster),
        tcia=TciaClient(getter=_tcia_getter),
    )


def test_tcia_normalizes_tcga_barcode_to_submitter_id() -> None:
    client = TciaClient(getter=_tcia_getter)
    series = client.list_series("TCGA-LUAD", "CT")
    assert {item.patient_id for item in series} == {
        "TCGA-05-4244",
        "TCGA-05-4249",
        "TCGA-88-NOGDC",
    }


def test_live_preview_intersects_gdc_and_tcia_without_fetch() -> None:
    catalog = _live_catalog()
    request = OrchestratorAgent().parse_and_plan(LUNG_QUESTION)
    matcher = DataMatcherAgent(catalog, gate=FlagGate(False))
    preview = matcher.preview(request)
    assert catalog.query_count == 1
    assert catalog.fetch_count == 0
    assert set(preview.patient_ids) == {"TCGA-05-4244", "TCGA-05-4249"}
    assert preview.n_available == 2
    assert "TCGA-99-NOIMG" not in preview.patient_ids
    assert "TCGA-88-NOGDC" not in preview.patient_ids
    assert catalog.last_source_counts["tcia_only_dropped"] == 1
    assert catalog.last_source_counts["gdc_only_dropped"] == 1
    assert catalog.last_source_counts["paired_available"] == 2
    assert catalog.last_source_counts["paired_selected"] == 2
    assert catalog.last_source_counts["tcia_series"] == 3
    assert catalog.last_source_counts["tcia_patients"] == 3
    assert catalog.last_source_counts["gdc_patients"] == 3


def test_max_patients_applied_after_full_intersection() -> None:
    catalog = _live_catalog()
    request = OrchestratorAgent().parse_and_plan(LUNG_QUESTION)
    request = request.model_copy(update={"max_patients": 1})
    preview = DataMatcherAgent(catalog, gate=FlagGate(False)).preview(request)
    assert preview.n_available == 2
    assert preview.n_paired == 1
    assert len(preview.patient_ids) == 1
    assert preview.patient_ids[0] in {"TCGA-05-4244", "TCGA-05-4249"}


def test_live_preview_keeps_pairs_without_os_time() -> None:
    """Survival in the question must not drop TCIA∩GDC pairs missing OS_time."""
    from agentic_radiogen.data.catalog import CatalogRecord

    class NoOsCatalog:
        query_count = 0
        fetch_count = 0
        last_source_counts = {"paired_kept": 1}

        def query_metadata(self, **kwargs):
            self.query_count += 1
            assert kwargs.get("require_endpoint") in (None, "OS")
            return [
                CatalogRecord(
                    patient_id="TCGA-NO-OS",
                    disease="lung",
                    modality="CT",
                    mutations={"EGFR": 0},
                    expression={},
                    clinical={"vital_status": ""},
                    series_uid="1.2.noos",
                    radiomic_features={},
                    volume_summary={},
                )
            ]

        def fetch_records(self, patient_ids):
            self.fetch_count += 1
            return self.query_metadata()

    matcher = DataMatcherAgent(NoOsCatalog(), gate=FlagGate(False))
    request = OrchestratorAgent().parse_and_plan(LUNG_QUESTION)
    preview = matcher.preview(request)
    assert preview.patient_ids == ["TCGA-NO-OS"]


def test_live_fetch_is_gated_and_question_scoped() -> None:
    catalog = _live_catalog()
    request = OrchestratorAgent().parse_and_plan(LUNG_QUESTION)
    request = request.model_copy(update={"max_patients": 1})
    blocked = DataMatcherAgent(catalog, gate=FlagGate(False))
    try:
        blocked.fetch(request)
        raise AssertionError("gated fetch should fail")
    except DownloadDeniedError:
        assert catalog.fetch_count == 0

    matcher = DataMatcherAgent(catalog, gate=FlagGate(True))
    images, omics = matcher.fetch(request)
    assert catalog.fetch_count == 1
    assert images.patient_ids == ["TCGA-05-4244"]
    assert omics.mutations["TCGA-05-4244"] == {"EGFR": 1}
    assert omics.clinical["TCGA-05-4244"]["OS_event"] == 1.0
    assert images.series[0].series_uid == "1.2.840.ct.4244"
    assert images.series[0].precomputed_features == {}


def test_matcher_drops_patients_missing_imaging_or_genomics() -> None:
    from agentic_radiogen.data.catalog import CatalogRecord

    class MixedCatalog:
        query_count = 0
        fetch_count = 0

        def query_metadata(self, **_kwargs):
            self.query_count += 1
            return [
                CatalogRecord(
                    patient_id="BOTH",
                    disease="lung",
                    modality="CT",
                    mutations={"EGFR": 1},
                    expression={},
                    clinical={"OS_time": 10.0, "OS_event": 1.0},
                    series_uid="1.2.both",
                    radiomic_features={},
                    volume_summary={},
                ),
                CatalogRecord(
                    patient_id="IMAGE_ONLY",
                    disease="lung",
                    modality="CT",
                    mutations={},
                    expression={},
                    clinical={},
                    series_uid="1.2.image",
                    radiomic_features={},
                    volume_summary={},
                ),
                CatalogRecord(
                    patient_id="GDC_ONLY",
                    disease="lung",
                    modality="CT",
                    mutations={"EGFR": 0},
                    expression={},
                    clinical={"OS_time": 8.0},
                    series_uid="",
                    radiomic_features={},
                    volume_summary={},
                ),
            ]

        def fetch_records(self, patient_ids):
            self.fetch_count += 1
            return [row for row in self.query_metadata() if row.patient_id in patient_ids]

    matcher = DataMatcherAgent(MixedCatalog(), gate=FlagGate(True))
    request = OrchestratorAgent().parse_and_plan(LUNG_QUESTION)
    preview = matcher.preview(request)
    assert preview.patient_ids == ["BOTH"]
    images, omics = matcher.fetch(request)
    assert images.patient_ids == ["BOTH"]
    assert omics.patient_ids == ["BOTH"]


def test_stage1_cli_preview_does_not_require_a_token() -> None:
    payload = cli.run_stage1(
        LUNG_QUESTION,
        disease=None,
        catalog_name="demo",
        approve_download=False,
        max_patients=8,
    )
    assert payload["stage"] == 1
    assert payload["token_required"] is False
    assert payload["preview"]["n_paired"] <= 8
    assert payload["fetch"]["status"] == "ok"
    assert all(pid.startswith("TCGA-LUNG-") for pid in payload["fetch"]["patient_ids"])
