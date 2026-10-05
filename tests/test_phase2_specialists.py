from __future__ import annotations

import inspect
from typing import get_type_hints

from agentic_radiogen.agents.data_matcher import DataMatcherAgent
from agentic_radiogen.agents.genomics import GenomicsAgent
from agentic_radiogen.agents.imaging import ImagingRadiomicsAgent
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen import __main__ as cli
from agentic_radiogen.pipeline.stage2 import extract_parallel
from agentic_radiogen.schemas.contracts import (
    GenomicMatrix,
    ImageBundle,
    ImageSeriesRef,
    OmicsBundle,
    RadiomicMatrix,
)
from tests.conftest import BREAST_QUESTION, LUNG_QUESTION


def test_matcher_fans_out_to_both_specialists_independently(
    matcher: DataMatcherAgent,
    orchestrator: OrchestratorAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
) -> None:
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    images, omics = matcher.fetch(request)
    radio = imaging.extract(images)
    geno = genomics.extract(omics)
    assert isinstance(radio, RadiomicMatrix)
    assert isinstance(geno, GenomicMatrix)
    assert set(radio.patient_ids) == set(images.patient_ids)
    assert set(geno.patient_ids) == set(omics.patient_ids)
    assert "original_glcm_Entropy" in radio.feature_names
    assert "EGFR_mut" in geno.feature_names


def test_specialists_do_not_require_each_other(
    matcher: DataMatcherAgent,
    orchestrator: OrchestratorAgent,
    imaging: ImagingRadiomicsAgent,
    genomics: GenomicsAgent,
) -> None:
    request = orchestrator.parse_and_plan(BREAST_QUESTION)
    images, omics = matcher.fetch(request)
    geno_first = genomics.extract(omics)
    radio_second = imaging.extract(images)
    assert "ERBB2_mut" in geno_first.feature_names
    assert radio_second.features[images.patient_ids[0]]
    assert "RadiomicMatrix" not in inspect.signature(genomics.extract).parameters
    assert "GenomicMatrix" not in inspect.signature(imaging.extract).parameters
    assert get_type_hints(imaging.extract)["bundle"] is ImageBundle
    assert get_type_hints(genomics.extract)["bundle"] is OmicsBundle


def test_imaging_does_not_import_genomics() -> None:
    import agentic_radiogen.agents.imaging as imaging_mod
    import agentic_radiogen.agents.genomics as genomics_mod

    assert "agentic_radiogen.agents.genomics" not in imaging_mod.__dict__.get("__name__", "")
    assert "GenomicsAgent" not in imaging_mod.__dict__
    assert "ImagingRadiomicsAgent" not in genomics_mod.__dict__


def test_extract_parallel_does_not_join_matrices(
    matcher: DataMatcherAgent,
    orchestrator: OrchestratorAgent,
) -> None:
    request = orchestrator.parse_and_plan(LUNG_QUESTION)
    outputs = extract_parallel(*matcher.fetch(request))
    assert outputs.imaging_error is None
    assert outputs.genomics_error is None
    assert outputs.radiomics is not None
    assert outputs.genomics is not None
    assert set(outputs.radiomics.patient_ids) == set(outputs.genomics.patient_ids)
    assert "EGFR_mut" not in outputs.radiomics.feature_names
    assert "original_glcm_Entropy" not in outputs.genomics.feature_names


def test_imaging_failure_does_not_block_genomics() -> None:
    images = ImageBundle(
        patient_ids=["P1"],
        series=[ImageSeriesRef(patient_id="P1", series_uid="1.2.x", modality="CT")],
    )
    omics = OmicsBundle(
        patient_ids=["P1"],
        mutations={"P1": {"EGFR": 1}},
        clinical={"P1": {"OS_time": 12.0, "OS_event": 1.0}},
    )
    outputs = extract_parallel(images, omics)
    assert outputs.radiomics is None
    assert outputs.imaging_error
    assert outputs.genomics is not None
    assert outputs.genomics.features["P1"]["EGFR_mut"] == 1.0


def test_stage2_cli_returns_both_matrices_without_stats() -> None:
    payload = cli.run_stage2(
        LUNG_QUESTION,
        disease=None,
        catalog_name="demo",
        approve_download=False,
        max_patients=8,
    )
    assert payload["stage"] == 2
    assert payload["joined"] is False
    assert payload["specialists"]["status"] == "ok"
    assert "original_glcm_Entropy" in payload["specialists"]["radiomics"]["feature_names"]
    assert "EGFR_mut" in payload["specialists"]["genomics"]["feature_names"]
    assert "top_associations" not in payload


def test_empty_bundles_fail_closed(
    imaging: ImagingRadiomicsAgent, genomics: GenomicsAgent
) -> None:
    import pytest

    with pytest.raises(ValueError, match="no series"):
        imaging.extract(ImageBundle(patient_ids=[], series=[]))
    with pytest.raises(ValueError, match="no patients"):
        genomics.extract(OmicsBundle(patient_ids=[]))
