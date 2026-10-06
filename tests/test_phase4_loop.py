from __future__ import annotations

from agentic_radiogen.agents.data_matcher import DataMatcherAgent
from agentic_radiogen.agents.genomics import GenomicsAgent
from agentic_radiogen.agents.imaging import ImagingRadiomicsAgent
from agentic_radiogen.agents.literature import LiteratureAgent
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.agents.statistics import StatisticalCriticalAgent
from agentic_radiogen.pipeline.loop import DiscoveryLoop
from agentic_radiogen.schemas.contracts import (
    Association,
    DataRequest,
    LiteratureContext,
    LiteratureSupport,
    LoopAction,
    ModelResult,
)
from tests.conftest import LUNG_QUESTION


def _loop(
    orchestrator: OrchestratorAgent,
    matcher: DataMatcherAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
    stats: StatisticalCriticalAgent,
    literature: LiteratureAgent,
    max_iterations: int = 3,
) -> DiscoveryLoop:
    return DiscoveryLoop(
        orchestrator=orchestrator,
        matcher=matcher,
        imaging=imaging,
        genomics=genomics,
        stats=stats,
        literature=literature,
        max_iterations=max_iterations,
    )


def test_unverified_literature_does_not_change_data_result(
    orchestrator: OrchestratorAgent,
    matcher: DataMatcherAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
    stats: StatisticalCriticalAgent,
    literature: LiteratureAgent,
) -> None:
    loop = _loop(orchestrator, matcher, imaging, genomics, stats, literature)
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    strong_stats = ModelResult(
        associations=[
            Association(
                imaging_feature="made_up_wavelet",
                genomic_feature="ZZZFAKE_mut",
                effect_size=0.99,
                p_value=1e-8,
                q_value=1e-8,
                n=16,
            )
        ]
    )
    context = literature.interpret(strong_stats, disease="lung")
    directive = loop.decide(context, request)
    assert context.unverified
    assert directive.action == LoopAction.STOP
    assert directive.new_data_request is None
    assert directive.auto_promoted == []
    assert directive.human_review_required is True
    assert "data-driven" in directive.reason.lower()


def test_supported_literature_stops_without_auto_promote(
    orchestrator: OrchestratorAgent,
    matcher: DataMatcherAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
    stats: StatisticalCriticalAgent,
    literature: LiteratureAgent,
) -> None:
    loop = _loop(orchestrator, matcher, imaging, genomics, stats, literature)
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    context = LiteratureContext(
        supports=[
            LiteratureSupport(
                finding="original_glcm_Entropy ~ EGFR_mut",
                papers=["CT radiomic texture and EGFR mutation status in lung adenocarcinoma"],
                supported=True,
            )
        ]
    )
    directive = loop.decide(context, request)
    assert directive.action == LoopAction.STOP
    assert directive.new_data_request is None
    assert directive.auto_promoted == []
    assert directive.human_review_required is True


def test_literature_contradiction_does_not_shrink_cohort(
    orchestrator: OrchestratorAgent,
    matcher: DataMatcherAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
    stats: StatisticalCriticalAgent,
    literature: LiteratureAgent,
) -> None:
    loop = _loop(orchestrator, matcher, imaging, genomics, stats, literature)
    request = DataRequest(
        question_id="q_test",
        disease="lung",
        tcga_project="TCGA-LUAD",
        tcia_collection="TCGA-LUAD",
        modality="CT",
        genes=["EGFR", "KRAS"],
        max_patients=16,
        filters={"full_archive": False},
    )
    context = LiteratureContext(
        contradictions=["original_firstorder_Mean ~ KRAS_mut: failed to replicate"],
        proposed_refinements=["Conflicts with prior report (kept for discovery)"],
        supports=[LiteratureSupport(finding="original_firstorder_Mean ~ KRAS_mut", supported=False)],
    )
    directive = loop.decide(context, request)
    assert directive.action == LoopAction.STOP
    assert directive.new_data_request is None
    assert directive.auto_promoted == []


def test_stage4_runs_single_data_pass(
    orchestrator: OrchestratorAgent,
    matcher: DataMatcherAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
    stats: StatisticalCriticalAgent,
    literature: LiteratureAgent,
) -> None:
    loop = _loop(orchestrator, matcher, imaging, genomics, stats, literature, max_iterations=3)
    state = loop.run(LUNG_QUESTION)
    assert state.stopped is True
    assert state.iterations == 1
    assert len(state.request_history) == 1
    assert state.literature is not None
    assert state.directive is not None
    assert state.directive.action == LoopAction.STOP
    assert state.directive.new_data_request is None
    assert state.directive.auto_promoted == []
    assert state.directive.human_review_required is True


def test_empty_corpus_still_keeps_full_cohort_once(
    orchestrator: OrchestratorAgent,
    matcher: DataMatcherAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
    stats: StatisticalCriticalAgent,
) -> None:
    loop = DiscoveryLoop(
        orchestrator=orchestrator,
        matcher=matcher,
        imaging=imaging,
        genomics=genomics,
        stats=stats,
        literature=LiteratureAgent(corpus=()),
        max_iterations=2,
        max_patients=16,
    )
    state = loop.run(LUNG_QUESTION)
    assert state.stopped is True
    assert state.iterations == 1
    assert len(state.request_history) == 1
    assert state.request.max_patients == 16
