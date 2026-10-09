from __future__ import annotations

from dataclasses import dataclass

from agentic_radiogen.agents.literature import LiteratureAgent
from agentic_radiogen.agents.statistics import StatisticalCriticalAgent
from agentic_radiogen.pipeline.stage2 import SpecialistOutputs
from agentic_radiogen.schemas.contracts import LiteratureContext, ModelResult


class Stage3Error(ValueError):
    pass


@dataclass
class Stage3Result:
    stats: ModelResult
    literature: LiteratureContext
    looped: bool = False


def join_and_interpret(
    outputs: SpecialistOutputs,
    *,
    disease: str,
    question: str = "",
    stats: StatisticalCriticalAgent | None = None,
    literature: LiteratureAgent | None = None,
) -> Stage3Result:
    """Stats joins both matrices; literature interprets. This stage does not loop."""
    if outputs.radiomics is None or outputs.genomics is None:
        raise Stage3Error(
            "Stage 3 needs both specialist matrices. "
            f"imaging={outputs.imaging_error or 'ok'}; genomics={outputs.genomics_error or 'ok'}"
        )
    stats_agent = stats or StatisticalCriticalAgent()
    literature_agent = literature or LiteratureAgent()
    model = stats_agent.analyze(outputs.radiomics, outputs.genomics)
    context = literature_agent.interpret(
        model, disease=disease, question=question or disease
    )
    return Stage3Result(stats=model, literature=context, looped=False)
