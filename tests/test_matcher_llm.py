from __future__ import annotations

from agentic_radiogen.agents.data_matcher import DataMatcherAgent
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.data.gdc_client import GdcClient
from agentic_radiogen.data.gate import FlagGate
from agentic_radiogen.data.live_catalog import LiveCatalog
from agentic_radiogen.data.tcia_client import TciaClient
from agentic_radiogen.llm.client import LlmClient, LlmConfig


def test_matcher_llm_selects_tcga_brca_and_pairs() -> None:
    def tcia_get(url: str, params=None):
        if "getCollectionValues" in url:
            return [{"Collection": "TCGA-BRCA"}, {"Collection": "QIN-BREAST"}]
        if str((params or {}).get("Collection") or "") != "TCGA-BRCA":
            return []
        return [
            {
                "PatientID": "TCGA-A2-A0D0",
                "SeriesInstanceUID": "1.2.brca.1",
                "Modality": "MR",
                "Collection": "TCGA-BRCA",
            }
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
            return {"data": {"aggregations": {}}}
        return {
            "data": {
                "hits": [
                    {
                        "submitter_id": "TCGA-A2-A0D0",
                        "diagnoses": [
                            {"primary_diagnosis": "Infiltrating duct carcinoma, NOS"}
                        ],
                    }
                ],
                "pagination": {"total": 1},
            }
        }

    def llm_poster(url: str, payload: dict, timeout: int = 60, headers=None, **_kwargs):
        _ = url, payload, timeout, headers
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"keywords":["breast","brca","tcga-brca"],'
                                    '"tcia_collections":["TCGA-BRCA"],'
                                    '"gdc_projects":["TCGA-BRCA"],'
                                    '"rationale":"pairable breast TCGA sources"}'
                                )
                            }
                        ]
                    }
                }
            ]
        }

    catalog = LiveCatalog(
        gdc=GdcClient(poster=gdc_post),
        tcia=TciaClient(getter=tcia_get),
        download_dicom=False,
        extract_radiomics=False,
        show_progress=False,
    )
    client = LlmClient(
        LlmConfig(provider="gemini", model="gemini-3.8-flash", api_key="test"),
        poster=llm_poster,
    )
    matcher = DataMatcherAgent(
        catalog, gate=FlagGate(False), use_llm=True, llm_client=client
    )
    request = OrchestratorAgent(use_llm=False).parse_and_plan(
        "Which imaging features are associated with genomic alterations in breast cancer?"
    )
    preview = matcher.preview(request)
    assert preview.patient_ids == ["TCGA-A2-A0D0"]
    assert matcher.last_backend.startswith("llm:gemini:")
    assert catalog.last_intermediate.get("preferred_tcia_collections") == ["TCGA-BRCA"]
