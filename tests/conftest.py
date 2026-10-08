from __future__ import annotations

import os

import pytest

from agentic_radiogen.agents.data_matcher import DataMatcherAgent
from agentic_radiogen.agents.genomics import GenomicsAgent
from agentic_radiogen.agents.imaging import ImagingRadiomicsAgent
from agentic_radiogen.agents.literature import LiteratureAgent
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.agents.statistics import StatisticalCriticalAgent
from agentic_radiogen.data.catalog import DemoCatalog
from agentic_radiogen.data.gate import AlwaysAllowGate

# Keep unit-test output quiet unless a test explicitly enables progress.
os.environ.setdefault("AGENTIC_RADIOGEN_QUIET", "1")
# Tests must not call live LLM APIs even if the developer has keys set.
os.environ["AGENTIC_RADIOGEN_LLM"] = "off"


LUNG_QUESTION = (
    "Which imaging features are associated with EGFR mutations and survival in lung cancer?"
)
BREAST_QUESTION = (
    "Are radiomic features associated with HER2 status and survival in breast cancer?"
)


@pytest.fixture
def catalog() -> DemoCatalog:
    return DemoCatalog()


@pytest.fixture
def orchestrator() -> OrchestratorAgent:
    # Unit tests stay on deterministic rules (no network LLM calls).
    return OrchestratorAgent(use_llm=False)


@pytest.fixture
def matcher(catalog: DemoCatalog) -> DataMatcherAgent:
    return DataMatcherAgent(catalog, gate=AlwaysAllowGate(), use_llm=False)


@pytest.fixture
def imaging() -> ImagingRadiomicsAgent:
    return ImagingRadiomicsAgent()


@pytest.fixture
def genomics() -> GenomicsAgent:
    return GenomicsAgent()


@pytest.fixture
def stats() -> StatisticalCriticalAgent:
    return StatisticalCriticalAgent()


@pytest.fixture
def literature() -> LiteratureAgent:
    return LiteratureAgent()
