from __future__ import annotations

from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.llm.client import LlmClient, LlmConfig


def test_orchestrator_uses_llm_disease_and_records_backend() -> None:
    def poster(url: str, payload: dict, timeout: int = 60, headers=None, **_kwargs):
        _ = url, payload, timeout, headers
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"disease_query":"bone cancer","modality":"CT",'
                                    '"endpoints":[],"rationale":"named bone cancer"}'
                                )
                            }
                        ]
                    }
                }
            ]
        }

    client = LlmClient(
        LlmConfig(provider="gemini", model="gemini-3.8-flash", api_key="test-key"),
        poster=poster,
    )
    agent = OrchestratorAgent(use_llm=True, llm_client=client)
    question = (
        "Which imaging features are associated with specific genomic alterations in bone cancer?"
    )
    req = agent.parse_and_plan(question)
    assert req.disease == "bone_cancer"
    assert req.filters["disease_query"] == "bone cancer"
    assert req.filters["orchestrator_backend"].startswith("llm:gemini:")
    assert req.modality == "CT"


def test_orchestrator_falls_back_to_rules_when_llm_fails() -> None:
    def poster(url: str, payload: dict, timeout: int = 60, headers=None, **_kwargs):
        raise RuntimeError("network down")

    client = LlmClient(
        LlmConfig(provider="gemini", model="gemini-3.8-flash", api_key="test-key"),
        poster=poster,
    )
    agent = OrchestratorAgent(use_llm=True, llm_client=client)
    req = agent.parse_and_plan(
        "Which imaging features are associated with EGFR mutations in lung cancer?"
    )
    assert req.disease == "lung_cancer"
    assert req.filters["orchestrator_backend"] == "rules"


def test_rules_backend_when_llm_disabled() -> None:
    agent = OrchestratorAgent(use_llm=False)
    req = agent.parse_and_plan(
        "Which imaging features are associated with EGFR mutations in lung cancer?"
    )
    assert req.filters["orchestrator_backend"] == "rules"
