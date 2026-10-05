from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_radiogen.agents.data_matcher import DataMatcherAgent, DownloadDeniedError
from agentic_radiogen.agents.genomics import GenomicsAgent
from agentic_radiogen.agents.imaging import ImagingRadiomicsAgent
from agentic_radiogen.agents.literature import LiteratureAgent
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.agents.statistics import StatisticalCriticalAgent
from agentic_radiogen.pipeline.stage2 import extract_parallel
from agentic_radiogen.pipeline.stage3 import Stage3Error, join_and_interpret
from agentic_radiogen.schemas.contracts import (
    DataRequest,
    GenomicMatrix,
    ImageBundle,
    LiteratureContext,
    LoopAction,
    ModelResult,
    OmicsBundle,
    RadiomicMatrix,
    RefinementDirective,
    ResearchQuestion,
)


class LoopState(BaseModel):
    question: ResearchQuestion
    request: DataRequest
    literature: LiteratureContext | None = None
    stats: ModelResult | None = None
    radiomics: RadiomicMatrix | None = None
    genomics: GenomicMatrix | None = None
    last_images: ImageBundle | None = None
    last_omics: OmicsBundle | None = None
    imaging_error: str | None = None
    genomics_error: str | None = None
    iterations: int = 0
    stopped: bool = False
    directive: RefinementDirective | None = None
    history: list[LiteratureContext] = Field(default_factory=list)
    request_history: list[DataRequest] = Field(default_factory=list)
    directives: list[RefinementDirective] = Field(default_factory=list)


class DiscoveryLoop:
    """Self-correction consumes Literature, then may send a smaller DataRequest back."""

    def __init__(
        self,
        *,
        orchestrator: OrchestratorAgent,
        matcher: DataMatcherAgent,
        imaging: ImagingRadiomicsAgent,
        genomics: GenomicsAgent,
        stats: StatisticalCriticalAgent,
        literature: LiteratureAgent,
        max_iterations: int = 3,
        max_patients: int | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.matcher = matcher
        self.imaging = imaging
        self.genomics = genomics
        self.stats = stats
        self.literature = literature
        self.max_iterations = max_iterations
        self.max_patients = max_patients

    def run(self, question_text: str, disease: str | None = None) -> LoopState:
        question = self.orchestrator.parse(question_text, disease=disease)
        request = self.orchestrator.plan(question)
        if self.max_patients is not None:
            request = request.model_copy(update={"max_patients": self.max_patients})
        state = LoopState(question=question, request=request)
        while not state.stopped:
            state = self.step(state)
        return state

    def step(self, state: LoopState) -> LoopState:
        if state.iterations >= self.max_iterations:
            return self._stop(state, "Iteration cap reached")

        state.request_history.append(state.request)
        try:
            images, omics = self.matcher.fetch(state.request)
        except DownloadDeniedError:
            return self._stop(state, "Human gate denied fetch")

        state.last_images = images
        state.last_omics = omics
        outputs = extract_parallel(
            images, omics, imaging=self.imaging, genomics=self.genomics
        )
        state.radiomics = outputs.radiomics
        state.genomics = outputs.genomics
        state.imaging_error = outputs.imaging_error
        state.genomics_error = outputs.genomics_error
        try:
            result = join_and_interpret(
                outputs,
                disease=state.question.disease,
                stats=self.stats,
                literature=self.literature,
            )
        except Stage3Error as exc:
            return self._stop(state, str(exc))

        directive = self.decide(result.literature, state.request)
        state.iterations += 1
        state.stats = result.stats
        state.literature = result.literature
        state.radiomics = outputs.radiomics
        state.genomics = outputs.genomics
        state.history.append(result.literature)
        state.directive = directive
        state.directives.append(directive)
        if directive.action == LoopAction.STOP or state.iterations >= self.max_iterations:
            state.stopped = True
        elif directive.new_data_request is not None:
            state.request = directive.new_data_request
        else:
            state.stopped = True
        return state

    def decide(self, context: LiteratureContext, request: DataRequest) -> RefinementDirective:
        """Loop input is literature, not the raw statistical result."""
        supported = [item.finding for item in context.supports if item.supported]
        weak = bool(context.unverified or context.contradictions) or not supported
        if not weak:
            return RefinementDirective(
                action=LoopAction.STOP,
                reason="Literature supports the top findings; stop for human review",
                human_review_required=True,
                auto_promoted=[],
            )
        if context.contradictions:
            narrowed = self._narrow_request(request, drop_gene_substrings=("kras",))
            if narrowed is None:
                return RefinementDirective(
                    action=LoopAction.STOP,
                    reason="Literature contradiction cannot be refined further; human review required",
                    human_review_required=True,
                    auto_promoted=[],
                )
            return RefinementDirective(
                action=LoopAction.FETCH_DIFFERENT_SUBSET,
                reason="; ".join(context.proposed_refinements) or "Literature contradiction",
                new_data_request=narrowed,
                human_review_required=True,
                auto_promoted=[],
            )
        if context.unverified:
            narrowed = self._narrow_request(request)
            return RefinementDirective(
                action=LoopAction.CHANGE_FEATURES,
                reason="Unverified findings must not be auto-promoted",
                new_data_request=narrowed,
                human_review_required=True,
                auto_promoted=[],
            )
        return RefinementDirective(
            action=LoopAction.STOP,
            reason="Weak literature support; human review required",
            human_review_required=True,
            auto_promoted=[],
        )

    @staticmethod
    def _narrow_request(
        request: DataRequest, drop_gene_substrings: tuple[str, ...] = ()
    ) -> DataRequest | None:
        genes = [
            gene
            for gene in request.genes
            if not any(token in gene.lower() for token in drop_gene_substrings)
        ]
        if drop_gene_substrings and not genes:
            return None
        if not genes:
            genes = list(request.genes)
        return request.model_copy(
            update={
                "genes": genes,
                "max_patients": max(8, request.max_patients // 2),
                "filters": {**request.filters, "refined": True, "full_archive": False},
            }
        )

    @staticmethod
    def _stop(state: LoopState, reason: str) -> LoopState:
        state.stopped = True
        state.directive = RefinementDirective(
            action=LoopAction.STOP,
            reason=reason,
            human_review_required=True,
            auto_promoted=[],
        )
        state.directives.append(state.directive)
        return state
