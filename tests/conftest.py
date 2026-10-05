from __future__ import annotations

import pytest

from agentic_radiogen.agents.data_matcher import DataMatcherAgent
from agentic_radiogen.agents.genomics import GenomicsAgent
from agentic_radiogen.agents.imaging import ImagingRadiomicsAgent
from agentic_radiogen.agents.literature import LiteratureAgent
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.agents.statistics import StatisticalCriticalAgent
from agentic_radiogen.data.catalog import DemoCatalog
from agentic_radiogen.data.gate import AlwaysAllowGate


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
    return OrchestratorAgent()


@pytest.fixture
def matcher(catalog: DemoCatalog) -> DataMatcherAgent:
    return DataMatcherAgent(catalog, gate=AlwaysAllowGate())


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
