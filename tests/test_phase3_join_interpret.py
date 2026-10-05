from __future__ import annotations

import pytest

from agentic_radiogen.agents.data_matcher import DataMatcherAgent
from agentic_radiogen.agents.genomics import GenomicsAgent
from agentic_radiogen.agents.imaging import ImagingRadiomicsAgent
from agentic_radiogen.agents.literature import LiteratureAgent
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.agents.statistics import StatisticalCriticalAgent
from agentic_radiogen.schemas.contracts import (
    Association,
    GenomicMatrix,
    ModelDiagnostics,
    ModelResult,
    RadiomicMatrix,
)
from agentic_radiogen import __main__ as cli
from agentic_radiogen.pipeline.stage2 import SpecialistOutputs
from agentic_radiogen.pipeline.stage3 import Stage3Error, join_and_interpret
from tests.conftest import BREAST_QUESTION, LUNG_QUESTION


def _run_to_stats(
    matcher: DataMatcherAgent,
    orchestrator: OrchestratorAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
    stats: StatisticalCriticalAgent,
    question: str,
) -> ModelResult:
    request = orchestrator.parse_and_plan(question)
    images, omics = matcher.fetch(request)
    return stats.analyze(imaging.extract(images), genomics.extract(omics))


def test_stats_requires_both_matrices(stats: StatisticalCriticalAgent) -> None:
    radio = RadiomicMatrix(
        patient_ids=["P1"], feature_names=["f"], features={"P1": {"f": 1.0}}
    )
    with pytest.raises(TypeError):
        stats.analyze(radio)  # type: ignore[misc]


def test_stats_joins_lung_imaging_and_genomics(
    matcher: DataMatcherAgent,
    orchestrator: OrchestratorAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
    stats: StatisticalCriticalAgent,
) -> None:
    result = _run_to_stats(matcher, orchestrator, imaging, genomics, stats, LUNG_QUESTION)
    entropy = next(
        a
        for a in result.associations
        if a.imaging_feature == "original_glcm_Entropy" and a.genomic_feature == "EGFR_mut"
    )
    assert entropy.effect_size > 0.7
    assert entropy.q_value < 0.05
    assert any(m.auroc is not None and m.auroc >= 0.7 for m in result.metrics if m.target == "EGFR_mut")
    assert result.diagnostics.multiple_testing_method == "fdr_bh"
    assert result.promoted_findings == []


def test_stats_joins_breast_imaging_and_genomics(
    matcher: DataMatcherAgent,
    orchestrator: OrchestratorAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
    stats: StatisticalCriticalAgent,
) -> None:
    result = _run_to_stats(matcher, orchestrator, imaging, genomics, stats, BREAST_QUESTION)
    shape = next(
        a
        for a in result.associations
        if a.imaging_feature == "original_shape_Sphericity" and a.genomic_feature == "ERBB2_mut"
    )
    assert shape.effect_size > 0.7
    assert shape.q_value < 0.05


def test_literature_not_stats_marks_unverified_and_supported(
    literature: LiteratureAgent,
) -> None:
    result = ModelResult(
        associations=[
            Association(
                imaging_feature="original_glcm_Entropy",
                genomic_feature="EGFR_mut",
                effect_size=0.8,
                p_value=1e-4,
                q_value=1e-3,
                n=16,
            ),
            Association(
                imaging_feature="made_up_wavelet",
                genomic_feature="TP53_mut",
                effect_size=0.6,
                p_value=1e-3,
                q_value=1e-2,
                n=16,
            ),
        ],
        diagnostics=ModelDiagnostics(),
    )
    context = literature.interpret(result, disease="lung")
    supported = [item.finding for item in context.supports if item.supported]
    assert any("EGFR" in f for f in supported)
    assert any("made_up_wavelet" in u for u in context.unverified)
    assert "made_up_wavelet" not in " ".join(supported)


def test_join_and_interpret_needs_both_matrices() -> None:
    radio = RadiomicMatrix(
        patient_ids=["P1"], feature_names=["f"], features={"P1": {"f": 1.0}}
    )
    with pytest.raises(Stage3Error, match="both specialist matrices"):
        join_and_interpret(SpecialistOutputs(radiomics=radio, genomics=None), disease="lung")


def test_stage3_cli_joins_then_interprets_without_looping() -> None:
    payload = cli.run_stage3(
        LUNG_QUESTION,
        disease=None,
        catalog_name="demo",
        approve_download=False,
        max_patients=16,
    )
    assert payload["stage"] == 3
    assert payload["joined"] is True
    assert payload["looped"] is False
    assert "directive" not in payload
    entropy = next(
        item
        for item in payload["stats"]["associations"]
        if item["imaging_feature"] == "original_glcm_Entropy" and item["genomic_feature"] == "EGFR_mut"
    )
    assert entropy["q_value"] < 0.05
    assert payload["stats"]["promoted_findings"] == []
    supported = [item["finding"] for item in payload["literature"]["supports"] if item["supported"]]
    assert any("EGFR" in finding for finding in supported)


def test_literature_flags_contradictions(literature: LiteratureAgent) -> None:
    result = ModelResult(
        associations=[
            Association(
                imaging_feature="original_firstorder_Mean",
                genomic_feature="KRAS_mut",
                effect_size=0.4,
                p_value=0.01,
                q_value=0.02,
                n=16,
            )
        ]
    )
    context = literature.interpret(result, disease="lung")
    assert context.contradictions
    assert all(not item.supported for item in context.supports)
