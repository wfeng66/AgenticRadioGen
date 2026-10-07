from __future__ import annotations

from agentic_radiogen.agents.orchestrator import OrchestratorAgent, infer_disease
from agentic_radiogen.data.disease_match import (
    diagnosis_matches,
    keywords_from_query,
    rank_names,
)
from agentic_radiogen.schemas.profiles import get_profile


def test_nsclc_uses_keyword_match_not_luad_register() -> None:
    assert infer_disease(
        "Which imaging features are associated with genomic alterations in Non-Small Cell Lung Cancer - NSCLC?"
    ) == "non-small cell lung cancer"
    profile = get_profile("nsclc")
    assert profile.tcia_collection == "KEYWORD"
    assert profile.tcga_project == "KEYWORD"
    request = OrchestratorAgent().parse_and_plan(
        "Which imaging features are associated with specific genomic alterations in Non-Small Cell Lung Cancer - NSCLC?"
    )
    assert request.filters.get("keyword_match") is True
    assert "non-small" in str(request.filters.get("disease_query") or "").lower() or "nsclc" in str(
        request.filters.get("disease_query") or ""
    ).lower()


def test_keyword_rank_prefers_nsclc_named_collections() -> None:
    keywords = keywords_from_query("non-small cell lung cancer")
    ranked = rank_names(
        ["TCGA-LUAD", "NSCLC-Radiomics", "TCGA-BRCA", "NSCLC Radiogenomics"],
        keywords,
        min_score=10,
    )
    names = [m.name for m in ranked]
    assert names[0].upper().startswith("NSCLC")
    assert diagnosis_matches("adenocarcinoma, nos", keywords)
    assert not diagnosis_matches("small cell carcinoma, nos", keywords)


def test_keyword_tiers_broaden_nsclc_to_lung_cancer() -> None:
    from agentic_radiogen.data.disease_match import broader_disease_query, keyword_match_tiers

    tiers = keyword_match_tiers("non-small cell lung cancer")
    assert len(tiers) >= 2
    assert tiers[0].label == "specific"
    assert tiers[1].label == "broadened"
    assert broader_disease_query("non-small cell lung cancer") == "lung cancer"
    assert "lung" in tiers[1].keywords


def test_brain_cancer_with_nbsp_uses_keyword_match() -> None:
    q = "Which imaging features are associated with specific genomic alterations in\xa0brain cancer?"
    assert infer_disease(q) == "brain cancer"
    request = OrchestratorAgent().parse_and_plan(q)
    assert request.disease == "brain_cancer"
    assert request.filters.get("keyword_match") is True
    assert "brain" in str(request.filters.get("disease_query") or "").lower()
    assert request.modality == "MR"
    profile = get_profile("brain cancer")
    assert profile.tcia_collection == "KEYWORD"


def test_bone_cancer_with_nbsp_infers_disease() -> None:
    from agentic_radiogen.data.disease_match import keywords_from_query

    q = "Which imaging features are associated with specific genomic alterations in\xa0bone cancer?"
    assert infer_disease(q) == "bone cancer"
    request = OrchestratorAgent().parse_and_plan(q)
    assert request.disease == "bone_cancer"
    assert request.filters.get("keyword_match") is True
    assert request.filters.get("disease_query") == "bone cancer"
    assert "sarc" in " ".join(keywords_from_query("bone cancer"))


def test_format_dataset_extraction_note_broadened() -> None:
    from agentic_radiogen.data.disease_match import format_dataset_extraction_note

    text = format_dataset_extraction_note(
        {
            "dataset_keywords_used": ["lung", "cancer", "bronchus"],
            "dataset_match_tier": "broadened",
            "dataset_broadened_from": "non-small cell lung cancer",
            "tcia_collection": "TCGA-LUAD,TCGA-LUSC",
            "gdc_project": "TCGA-LUAD,TCGA-LUSC",
            "keyword_match": True,
        }
    )
    assert text is not None
    assert "broader disease keywords" in text
    assert "non-small cell lung cancer" in text
    assert "lung" in text
