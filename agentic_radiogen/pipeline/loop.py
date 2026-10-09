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
from agentic_radiogen.util.progress import log


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
    """Run specialists + stats once (data-driven); literature annotates but does not refine."""

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
        max_genes: int | None = None,
        min_altered: int | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.matcher = matcher
        self.imaging = imaging
        self.genomics = genomics
        self.stats = stats
        self.literature = literature
        self.max_iterations = max_iterations
        self.max_patients = max_patients
        self.max_genes = max_genes
        self.min_altered = min_altered

    def run(self, question_text: str, disease: str | None = None) -> LoopState:
        question = self.orchestrator.parse(question_text, disease=disease)
        request = self.orchestrator.plan(question)
        updates: dict = {}
        if self.max_patients is not None:
            updates["max_patients"] = self.max_patients
        if self.max_genes is not None:
            updates["max_genes"] = self.max_genes
        if self.min_altered is not None:
            updates["min_altered"] = self.min_altered
        if updates:
            request = request.model_copy(update=updates)
        state = LoopState(question=question, request=request)
        # Search disease literature + extract radiomic–gene priors before fetch/stats.
        self.literature.prepare(
            disease=question.disease,
            question=question_text,
        )
        while not state.stopped:
            state = self.step(state)
        return state

    def step(self, state: LoopState) -> LoopState:
        if state.iterations >= self.max_iterations:
            return self._stop(state, "Iteration cap reached")

        state.request_history.append(state.request)
        log(
            f"[loop] Iteration {state.iterations + 1}/{self.max_iterations}: "
            f"fetch {state.request.modality} from {state.request.tcia_collection} "
            f"+ genomics from {state.request.tcga_project}"
        )
        try:
            images, omics = self.matcher.fetch(state.request)
        except DownloadDeniedError:
            return self._stop(state, "Human gate denied fetch")

        state.last_images = images
        state.last_omics = omics
        if not images.series and not omics.patient_ids:
            counts = getattr(self.matcher.catalog, "last_source_counts", {}) or {}
            detail = (
                f"tcia_only={counts.get('tcia_only_dropped', '?')}, "
                f"gdc_only={counts.get('gdc_only_dropped', '?')}, "
                f"paired={counts.get('paired_kept', 0)}"
            )
            return self._stop(
                state,
                "No paired imaging+genomics patients for "
                f"{state.request.tcia_collection} ∩ {state.request.tcga_project} "
                f"({state.request.modality}). {detail}",
            )
        log("[loop] Running imaging + genomics specialists in parallel")
        outputs = extract_parallel(
            images, omics, imaging=self.imaging, genomics=self.genomics
        )
        state.radiomics = outputs.radiomics
        state.genomics = outputs.genomics
        state.imaging_error = outputs.imaging_error
        state.genomics_error = outputs.genomics_error
        log(
            f"[loop] Specialists done: radiomics="
            f"{0 if outputs.radiomics is None else len(outputs.radiomics.patient_ids)}, "
            f"genomics="
            f"{0 if outputs.genomics is None else len(outputs.genomics.patient_ids)}"
            + (
                f" (imaging_err={outputs.imaging_error})"
                if outputs.imaging_error
                else ""
            )
            + (
                f" (genomics_err={outputs.genomics_error})"
                if outputs.genomics_error
                else ""
            )
        )
        log(
            f"[loop] (5) Testing each gene in gen against radiomics on i&g "
            f"(n_radio={0 if outputs.radiomics is None else len(outputs.radiomics.patient_ids)}, "
            f"n_geno={0 if outputs.genomics is None else len(outputs.genomics.patient_ids)})"
        )
        try:
            log("[loop] Joining matrices + literature annotation (literature does not change results)")
            result = join_and_interpret(
                outputs,
                disease=state.question.disease,
                question=state.question.text,
                stats=self.stats,
                literature=self.literature,
            )
        except Stage3Error as exc:
            return self._stop(state, str(exc))

        n_assoc = len(result.stats.associations)
        log(
            f"[loop] Stats ready: {n_assoc} associations "
            f"(sorted by |r| ascending; strongest last); "
            f"literature supports={len(result.literature.supports)} "
            f"unverified={len(result.literature.unverified)}"
        )
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
        """Literature annotates findings only; it never changes the data-driven result.

        Novel / rare / literature-contradicting associations are kept. The cohort is not
        shrunk to chase paper agreement.
        """
        _ = request
        n_supported = sum(1 for item in context.supports if item.supported)
        n_unverified = len(context.unverified)
        n_contra = len(context.contradictions)
        return RefinementDirective(
            action=LoopAction.STOP,
            reason=(
                "Data-driven stop: keep full-cohort statistical findings. "
                f"Literature is contextual only "
                f"(supported={n_supported}, unverified={n_unverified}, "
                f"contradictions={n_contra}); contradictions are allowed for discovery."
            ),
            new_data_request=None,
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
