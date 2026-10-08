from __future__ import annotations

from agentic_radiogen.agents.data_matcher import DataMatcherAgent
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.data.gdc_client import GdcClient
from agentic_radiogen.data.gate import FlagGate
from agentic_radiogen.data.live_catalog import LiveCatalog
from agentic_radiogen.data.tcia_client import TciaClient


def test_breast_pairable_project_keeps_full_gdc_cohort() -> None:
    """TCGA-BRCA diagnoses rarely contain the word 'breast'; do not filter them away."""

    def tcia_get(url: str, params=None):
        if "getCollectionValues" in url:
            return [{"Collection": "TCGA-BRCA"}, {"Collection": "QIN-BREAST"}]
        collection = str((params or {}).get("Collection") or "")
        if collection != "TCGA-BRCA":
            return []
        return [
            {
                "PatientID": "TCGA-A2-A0D0",
                "SeriesInstanceUID": "1.2.brca.1",
                "Modality": "MR",
                "Collection": "TCGA-BRCA",
            },
            {
                "PatientID": "TCGA-A2-A0D1",
                "SeriesInstanceUID": "1.2.brca.2",
                "Modality": "MR",
                "Collection": "TCGA-BRCA",
            },
        ]

    def gdc_post(url: str, payload: dict):
        if "/projects" in url:
            return {
                "data": {
                    "hits": [
                        {
                            "project_id": "TCGA-BRCA",
                            "name": "Breast Invasive Carcinoma",
                            "primary_site": ["Breast"],
                            "disease_type": ["Ductal and Lobular Neoplasms"],
                        }
                    ]
                }
            }
        if "facets" in payload:
            return {
                "data": {
                    "aggregations": {
                        "diagnoses.primary_diagnosis": {
                            "buckets": [
                                {
                                    "key": "Invasive mammary carcinoma",
                                    "doc_count": 1,
                                },
                                {
                                    "key": "Infiltrating duct carcinoma, NOS",
                                    "doc_count": 2,
                                },
                            ]
                        }
                    }
                }
            }
        return {
            "data": {
                "hits": [
                    {
                        "submitter_id": "TCGA-A2-A0D0",
                        "diagnoses": [
                            {"primary_diagnosis": "Infiltrating duct carcinoma, NOS"}
                        ],
                    },
                    {
                        "submitter_id": "TCGA-A2-A0D1",
                        "diagnoses": [
                            {"primary_diagnosis": "Infiltrating duct carcinoma, NOS"}
                        ],
                    },
                    {
                        "submitter_id": "TCGA-XX-ONLY",
                        "diagnoses": [{"primary_diagnosis": "Invasive mammary carcinoma"}],
                    },
                ],
                "pagination": {"total": 3},
            }
        }

    catalog = LiveCatalog(
        gdc=GdcClient(poster=gdc_post),
        tcia=TciaClient(getter=tcia_get),
        download_dicom=False,
        extract_radiomics=False,
        show_progress=False,
    )
    request = OrchestratorAgent(use_llm=False).parse_and_plan(
        "Which imaging features are associated with genomic alterations in breast cancer?"
    )
    preview = DataMatcherAgent(catalog, gate=FlagGate(False)).preview(request)
    assert set(preview.patient_ids) == {"TCGA-A2-A0D0", "TCGA-A2-A0D1"}
    assert catalog.last_source_counts["gdc_patients"] == 3
    assert catalog.last_source_counts["i_and_g"] == 2
