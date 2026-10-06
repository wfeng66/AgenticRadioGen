from __future__ import annotations

import pytest

from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.data.gdc_client import GdcClient


def test_default_plans_all_cohort_genes() -> None:
    request = OrchestratorAgent().parse_and_plan(
        "Which imaging features are associated with specific genomic alterations in lung cancer?"
    )
    assert request.genes == []
    assert request.filters.get("discover_genes") is True
    assert request.filters.get("gene_source") == "cohort"
    assert request.max_genes is None
    assert request.min_altered == 1


def test_genes_auto_uses_literature_panel() -> None:
    request = OrchestratorAgent(genes="auto").parse_and_plan(
        "Which imaging features are associated with genomic alterations in lung cancer?"
    )
    assert "EGFR" in request.genes
    assert "KRAS" in request.genes
    assert request.filters.get("discover_genes") is False
    assert request.filters.get("gene_source") == "literature"


def test_manual_gene_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="Manual gene lists"):
        OrchestratorAgent(genes=["EGFR", "KRAS"]).parse_and_plan("lung cancer imaging")


def test_named_gene_in_question_does_not_restrict_cohort_mode() -> None:
    request = OrchestratorAgent().parse_and_plan(
        "Which imaging features are associated with EGFR mutations in lung cancer?"
    )
    assert request.genes == []
    assert request.filters.get("gene_source") == "cohort"


def test_discover_genes_returns_all_cohort_genes_by_default() -> None:
    def poster(url: str, payload: dict):
        assert "ssm_occurrences" in url
        return {
            "data": {
                "hits": [
                    {
                        "case": {"submitter_id": "P1"},
                        "ssm": {
                            "consequence": [{"transcript": {"gene": {"symbol": "TP53"}}}]
                        },
                    },
                    {
                        "case": {"submitter_id": "P2"},
                        "ssm": {
                            "consequence": [{"transcript": {"gene": {"symbol": "TP53"}}}]
                        },
                    },
                    {
                        "case": {"submitter_id": "P1"},
                        "ssm": {
                            "consequence": [{"transcript": {"gene": {"symbol": "ALK"}}}]
                        },
                    },
                    {
                        "case": {"submitter_id": "P3"},
                        "ssm": {
                            "consequence": [{"transcript": {"gene": {"symbol": "RARE"}}}]
                        },
                    },
                ],
                "pagination": {"total": 4},
            }
        }

    genes = GdcClient(poster=poster).discover_genes(["P1", "P2", "P3"])
    assert genes == ["TP53", "ALK", "RARE"]


def test_empty_mutation_placeholder_still_keeps_gdc_paired_records() -> None:
    from agentic_radiogen.data.catalog import CatalogRecord, is_paired_record

    rec = CatalogRecord(
        patient_id="P1",
        disease="lung",
        modality="CT",
        mutations={},
        expression={},
        clinical={"_gdc_paired": True, "_gdc_project": "TCGA-LUAD"},
        series_uid="1.2.3",
        radiomic_features={},
        volume_summary={},
    )
    assert is_paired_record(rec)
