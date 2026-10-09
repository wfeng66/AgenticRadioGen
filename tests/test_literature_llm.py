from __future__ import annotations

from agentic_radiogen.agents.literature import LiteratureAgent, LitPair
from agentic_radiogen.data.pubmed_client import PubMedArticle, build_radiogenomics_query
from agentic_radiogen.llm.literature_plan import LiteratureExtraction, ExtractedLitPair
from agentic_radiogen.schemas.contracts import Association, LiteratureContext, ModelDiagnostics, ModelResult


def test_build_radiogenomics_query_includes_disease_and_radiomics() -> None:
    q = build_radiogenomics_query("lung cancer", "EGFR radiomics mutations")
    assert "lung cancer" in q
    assert "radiomic" in q.lower()
    assert "mutation" in q.lower()


def test_prepare_with_injected_llm_uses_pubmed_and_extraction(monkeypatch) -> None:
    articles = [
        PubMedArticle(
            pmid="111",
            title="GLCM entropy predicts EGFR mutation in NSCLC",
            abstract="Higher GLCM entropy was associated with EGFR mutations in lung adenocarcinoma.",
            journal="Demo",
            year="2020",
        )
    ]

    class FakePubMed:
        def search(self, query, *, max_results=20):
            assert "lung" in query.lower()
            return ["111"]

        def fetch(self, pmids):
            assert pmids == ["111"]
            return [
                PubMedArticle(
                    pmid="111",
                    title=articles[0].title,
                    abstract=articles[0].abstract,
                    journal="Radiology",
                    year="2020",
                )
            ]

    class FakeLlm:
        available = True

        def complete_json(self, *, system, user, timeout=None):
            return {
                "pairs": [
                    {
                        "gene": "EGFR",
                        "radiomic_terms": ["glcm", "entropy"],
                        "supports": True,
                        "note": "entropy linked to EGFR",
                        "paper_title": articles[0].title,
                        "pmid": "111",
                    }
                ],
                "rationale": "one supporting abstract",
            }

    agent = LiteratureAgent(
        use_llm=True,
        llm_client=FakeLlm(),  # type: ignore[arg-type]
        pubmed=FakePubMed(),  # type: ignore[arg-type]
        verbose=False,
        use_cache=False,
    )
    pairs = agent.prepare(disease="lung", question="radiomics and genomic alterations in lung cancer")
    assert agent.last_source == "pubmed_llm"
    assert len(pairs) == 1
    assert pairs[0].gene == "EGFR"
    assert "entropy" in pairs[0].radiomic_terms

    result = ModelResult(
        associations=[
            Association(
                imaging_feature="original_glcm_Entropy",
                genomic_feature="EGFR_mut",
                effect_size=0.5,
                p_value=0.01,
                q_value=0.02,
                n=20,
            ),
            Association(
                imaging_feature="original_shape_Sphericity",
                genomic_feature="ZZZFAKE_mut",
                effect_size=0.4,
                p_value=0.02,
                q_value=0.03,
                n=20,
            ),
        ],
        diagnostics=ModelDiagnostics(),
    )
    ctx = agent.interpret(result, disease="lung", question="lung cancer radiomics")
    assert isinstance(ctx, LiteratureContext)
    assert ctx.literature_source == "pubmed_llm"
    assert any(item.supported and "EGFR" in item.finding for item in ctx.supports)
    assert any("ZZZFAKE" in u for u in ctx.unverified)
    assert ctx.prior_pairs and ctx.prior_pairs[0]["pmid"] == "111"
    assert ctx.prior_pairs[0]["journal"] == "Radiology"
    assert ctx.prior_pairs[0]["year"] == "2020"
    assert any("Radiology" in (p or "") for item in ctx.supports for p in item.papers)


def test_default_agent_without_llm_flag_uses_builtin_corpus() -> None:
    agent = LiteratureAgent(verbose=False)
    agent.prepare(disease="lung", question="lung cancer")
    assert agent.last_source == "static_corpus"
    assert any(p.gene == "EGFR" for p in agent.pairs)


def test_literature_cache_avoids_second_llm_call(tmp_path) -> None:
    calls = {"llm": 0, "search": 0}

    class FakePubMed:
        def search(self, query, *, max_results=20):
            calls["search"] += 1
            return ["111"]

        def fetch(self, pmids):
            return [
                PubMedArticle(
                    pmid="111",
                    title="GLCM entropy and EGFR",
                    abstract="GLCM entropy associated with EGFR mutation.",
                    journal="Eur Radiol",
                    year="2021",
                )
            ]

    class FakeLlm:
        available = True

        def complete_json(self, *, system, user, timeout=None):
            calls["llm"] += 1
            return {
                "pairs": [
                    {
                        "gene": "EGFR",
                        "radiomic_terms": ["glcm", "entropy"],
                        "supports": True,
                        "note": "ok",
                        "paper_title": "GLCM entropy and EGFR",
                        "pmid": "111",
                    }
                ],
                "rationale": "cached later",
            }

    cache_dir = tmp_path / "lit_cache"
    kwargs = dict(
        use_llm=True,
        llm_client=FakeLlm(),
        pubmed=FakePubMed(),
        verbose=False,
        use_cache=True,
        cache_dir=cache_dir,
    )
    first = LiteratureAgent(**kwargs)  # type: ignore[arg-type]
    first.prepare(disease="lung cancer", question="Which features associate with mutations in lung cancer?")
    assert first.last_source == "pubmed_llm"
    assert calls["llm"] == 1
    assert calls["search"] == 1
    assert list(cache_dir.glob("*.json"))

    second = LiteratureAgent(**kwargs)  # type: ignore[arg-type]
    # Different wording / casing, same disease → cache hit, no LLM/PubMed.
    second.prepare(disease="Lung Cancer", question="lung cancer radiogenomics again")
    assert second.last_source == "pubmed_llm_cache"
    assert calls["llm"] == 1
    assert calls["search"] == 1
    assert len(second.pairs) == 1
    assert second.pairs[0].gene == "EGFR"
    assert second.pairs[0].journal == "Eur Radiol"
    assert second.pairs[0].year == "2021"
