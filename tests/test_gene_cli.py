from __future__ import annotations

from agentic_radiogen.agents.orchestrator import OrchestratorAgent


def test_explicit_single_gene() -> None:
    request = OrchestratorAgent(genes="tp53").parse_and_plan(
        "Which imaging features are associated with genomic alterations in lung cancer?"
    )
    assert request.genes == ["TP53"]
    assert request.filters.get("discover_genes") is False
    assert request.filters.get("gene_source") == "explicit"


def test_explicit_comma_separated_genes() -> None:
    request = OrchestratorAgent(genes="EGFR, kras, HER2").parse_and_plan(
        "radiomics and mutations in lung cancer"
    )
    assert request.genes == ["EGFR", "KRAS", "ERBB2"]
    assert request.filters.get("gene_source") == "explicit"
    assert request.filters.get("discover_genes") is False


def test_gene_all_still_discovers_cohort() -> None:
    request = OrchestratorAgent(genes="all").parse_and_plan(
        "imaging genomic alterations in lung cancer"
    )
    assert request.genes == []
    assert request.filters.get("gene_source") == "cohort"
    assert request.filters.get("discover_genes") is True


def test_genes_auto_uses_literature_panel() -> None:
    request = OrchestratorAgent(genes="auto").parse_and_plan(
        "Which imaging features are associated with genomic alterations in lung cancer?"
    )
    assert "EGFR" in request.genes
    assert request.filters.get("discover_genes") is False
    assert request.filters.get("gene_source") == "literature"
