from __future__ import annotations

from agentic_radiogen.agents.genomics import GenomicsAgent
from agentic_radiogen.agents.imaging import ImagingRadiomicsAgent
from agentic_radiogen.agents.literature import LiteratureAgent, LitPair
from agentic_radiogen.schemas.contracts import Association, ModelDiagnostics, ModelResult


def test_imaging_agent_maps_texture_to_gldm_family() -> None:
    imaging = ImagingRadiomicsAgent()
    ok, reason, score = imaging.relate_to_literature_terms(
        "original_gldm_SmallDependenceLowGrayLevelEmphasis",
        {"texture"},
    )
    assert ok
    assert score > 0
    assert "gldm" in reason or "texture" in reason or "familie" in reason


def test_genomics_agent_maps_erbb_family() -> None:
    genomics = GenomicsAgent()
    ok, reason, score = genomics.relate_to_literature_gene("ERBB2_mut", "EGFR")
    assert ok
    assert score >= 0.7
    assert "family" in reason.lower()
    exact, _, exact_score = genomics.relate_to_literature_gene("EGFR_mut", "EGFR")
    assert exact and exact_score == 1.0
    no, _, _ = genomics.relate_to_literature_gene("ZNF160_mut", "EGFR")
    assert not no


def test_texture_prior_directly_supports_gldm_egfr() -> None:
    """Literature 'texture' should match PyRadiomics GLDM via imaging agent expansion."""
    agent = LiteratureAgent(
        corpus=(),
        verbose=False,
    )
    agent.pairs = [
        LitPair(
            gene="EGFR",
            radiomic_terms=frozenset({"texture"}),
            supports=True,
            paper_title="Demo texture-EGFR paper",
            pmid="1",
        )
    ]
    agent.prepared = True
    agent.last_source = "test"
    result = ModelResult(
        associations=[
            Association(
                imaging_feature="original_gldm_DependenceNonUniformity",
                genomic_feature="EGFR_mut",
                effect_size=0.3,
                p_value=0.01,
                q_value=0.05,
                n=99,
            ),
            Association(
                imaging_feature="original_gldm_DependenceNonUniformity",
                genomic_feature="ZNF160_mut",
                effect_size=0.99,
                p_value=1e-20,
                q_value=1e-18,
                n=99,
            ),
        ],
        diagnostics=ModelDiagnostics(),
    )
    ctx = agent.interpret(result, disease="lung_cancer")
    assert any(s.supported and "EGFR" in s.finding for s in ctx.supports)
    assert any("ZNF160" in u for u in ctx.unverified)
    # Direct support found → soft-map path is not required
    assert ctx.mapped_related == []


def test_soft_map_and_aggregate_when_only_family_gene_matches() -> None:
    """No exact gene overlap with priors → imaging+genomics soft-map + aggregate."""
    agent = LiteratureAgent(corpus=(), verbose=False)
    agent.pairs = [
        LitPair(
            gene="EGFR",
            radiomic_terms=frozenset({"texture"}),
            supports=True,
            paper_title="Habitat radiomics predict EGFR",
            pmid="38982267",
            journal="Scientific reports",
            year="2024",
        )
    ]
    agent.prepared = True
    agent.last_source = "pubmed_llm"
    # Cohort has ERBB2 (family) + texture-like feature, but not EGFR → no direct support
    result = ModelResult(
        associations=[
            Association(
                imaging_feature="original_gldm_DependenceNonUniformity",
                genomic_feature="ERBB2_mut",
                effect_size=0.42,
                p_value=0.01,
                q_value=0.04,
                n=80,
            ),
            Association(
                imaging_feature="original_shape_Sphericity",
                genomic_feature="ZNF160_mut",
                effect_size=0.9,
                p_value=1e-10,
                q_value=1e-8,
                n=80,
            ),
        ],
        diagnostics=ModelDiagnostics(),
    )
    ctx = agent.interpret(result, disease="lung_cancer")
    assert ctx.supports == []
    assert ctx.mapped_related
    assert any("ERBB2" in m.finding for m in ctx.mapped_related)
    assert any(m.literature_gene == "EGFR" for m in ctx.mapped_related)
    assert any("family" in m.gene_reason.lower() for m in ctx.mapped_related)
    assert ctx.pair_aggregates
    agg = ctx.pair_aggregates[-1]
    assert agg.literature_gene == "EGFR"
    assert agg.n_mapped >= 1
    assert agg.mapped_findings
    assert any("mapped" in r.lower() or "soft" in r.lower() for r in ctx.proposed_refinements)
