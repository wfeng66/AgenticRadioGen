from __future__ import annotations

import pytest

from agentic_radiogen.agents.data_matcher import DataMatcherAgent, DownloadDeniedError
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.data.catalog import DemoCatalog
from agentic_radiogen.data.gate import AlwaysDenyGate
from tests.conftest import BREAST_QUESTION, LUNG_QUESTION


def test_orchestrator_plans_lung_question_without_downloading(
    orchestrator: OrchestratorAgent, catalog: DemoCatalog
) -> None:
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    assert request.disease == "lung"
    assert request.tcga_project == "TCGA-LUAD"
    assert request.modality == "CT"
    assert request.genes == ["EGFR"]
    assert "OS_time" in request.clinical_fields  # requested if available
    assert request.filters["full_archive"] is False
    assert request.max_patients <= 24
    assert catalog.fetch_count == 0


def test_survival_question_does_not_require_os_time_for_pairing(
    matcher: DataMatcherAgent, orchestrator: OrchestratorAgent
) -> None:
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    assert "OS_time" in request.clinical_fields
    preview = matcher.preview(request)
    assert preview.n_paired > 0
    assert preview.n_paired == len(preview.patient_ids)


def test_orchestrator_plans_breast_with_the_same_agent(
    orchestrator: OrchestratorAgent,
) -> None:
    request = orchestrator.parse_and_plan(BREAST_QUESTION)
    assert request.disease == "breast"
    assert request.tcga_project == "TCGA-BRCA"
    assert request.modality == "MR"
    assert request.genes == ["ERBB2"]
    assert request.tcia_collection != "TCGA-LUAD"


def test_explicit_disease_needed_when_question_is_generic(
    orchestrator: OrchestratorAgent,
) -> None:
    with pytest.raises(ValueError, match="Cannot infer disease"):
        orchestrator.parse("Which imaging features associate with mutations?")
    parsed = orchestrator.parse(
        "Which imaging features associate with BRCA1?", disease="breast"
    )
    assert parsed.disease == "breast"
    assert parsed.genes == ["BRCA1"]


def test_preview_is_metadata_only_and_question_scoped(
    matcher: DataMatcherAgent, orchestrator: OrchestratorAgent, catalog: DemoCatalog
) -> None:
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    preview = matcher.preview(request)
    assert catalog.query_count == 1
    assert catalog.fetch_count == 0
    assert preview.n_paired == len(preview.patient_ids)
    assert preview.n_paired <= request.max_patients
    assert all(pid.startswith("TCGA-LUNG-") for pid in preview.patient_ids)
    assert preview.n_paired < len(catalog._records)


def test_fetch_downloads_only_previewed_patients(
    matcher: DataMatcherAgent, orchestrator: OrchestratorAgent, catalog: DemoCatalog
) -> None:
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    preview = matcher.preview(request)
    images, omics = matcher.fetch(request)
    assert set(images.patient_ids) == set(preview.patient_ids)
    assert set(omics.patient_ids) == set(preview.patient_ids)
    assert set(images.patient_ids).isdisjoint(
        {pid for pid in catalog._records if pid.startswith("TCGA-BRCA-")}
    )
    assert all(set(mut) == {"EGFR"} for mut in omics.mutations.values())
    assert catalog.fetch_count == 1


def test_denied_gate_blocks_download(catalog: DemoCatalog, orchestrator: OrchestratorAgent) -> None:
    matcher = DataMatcherAgent(catalog, gate=AlwaysDenyGate())
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    with pytest.raises(DownloadDeniedError):
        matcher.fetch(request)
    assert catalog.fetch_count == 0


def test_cache_avoids_second_catalog_fetch(
    matcher: DataMatcherAgent, orchestrator: OrchestratorAgent, catalog: DemoCatalog
) -> None:
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    matcher.fetch(request)
    matcher.fetch(request)
    assert catalog.fetch_count == 1


def test_full_archive_flag_is_rejected(
    matcher: DataMatcherAgent, orchestrator: OrchestratorAgent
) -> None:
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    request.filters["full_archive"] = True
    with pytest.raises(ValueError, match="Full-archive"):
        matcher.preview(request)
