"""Literature annotation: PubMed search + LLM pair extraction, then label stats.

Does not alter data-driven associations — only labels them supported /
unverified / contradicted against prior radiomic–gene claims.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from agentic_radiogen.data.pubmed_client import (
    PubMedClient,
    build_radiogenomics_query,
)
from agentic_radiogen.llm.client import LlmClient, resolve_llm_config
from agentic_radiogen.llm.literature_plan import extract_pairs_with_llm
from agentic_radiogen.schemas.contracts import LiteratureContext, LiteratureSupport, ModelResult
from agentic_radiogen.util.progress import log

_LIT_CACHE_VERSION = 2  # includes journal + year on prior pairs
_DEFAULT_LIT_CACHE_DIR = Path.cwd() / "data_cache" / "literature"


def normalize_disease_key(disease: str) -> str:
    """Normalize disease text so 'lung cancer' and 'Lung Cancer' share a cache entry."""
    slug = re.sub(r"[^a-z0-9]+", "_", (disease or "").strip().lower())
    return slug.strip("_") or "cancer"


@dataclass(frozen=True)
class Paper:
    """Legacy static corpus entry (tests / offline fallback)."""

    title: str
    keywords: frozenset[str]
    supports: bool = True
    note: str | None = None


@dataclass(frozen=True)
class LitPair:
    """A prior radiomic-term ↔ gene claim from literature (LLM or static)."""

    gene: str
    radiomic_terms: frozenset[str]
    supports: bool = True
    note: str | None = None
    paper_title: str = ""
    pmid: str | None = None
    journal: str = ""  # journal or conference name from PubMed
    year: str = ""


def format_paper_ref(pair: LitPair) -> str:
    """Human-readable citation: Title (Journal, Year) [PMID:…]."""
    title = pair.paper_title or (f"PMID:{pair.pmid}" if pair.pmid else "prior literature")
    meta = ", ".join(x for x in (pair.journal, pair.year) if x)
    bits = [title]
    if meta:
        bits.append(f"({meta})")
    if pair.pmid:
        bits.append(f"[PMID:{pair.pmid}]")
    return " ".join(bits)


DEFAULT_CORPUS: tuple[Paper, ...] = (
    Paper(
        title="CT radiomic texture and EGFR mutation status in lung adenocarcinoma",
        keywords=frozenset({"egfr", "entropy", "glcm", "lung"}),
    ),
    Paper(
        title="MRI shape features associate with HER2-positive breast cancer",
        keywords=frozenset({"erbb2", "her2", "sphericity", "breast"}),
    ),
    Paper(
        title="Conflicting report: KRAS is not reliably predicted by first-order intensity",
        keywords=frozenset({"kras", "mean", "firstorder"}),
        supports=False,
        note="Prior work failed to replicate intensity-KRAS association",
    ),
)


_DISEASE_TAGS = frozenset(
    {
        "lung",
        "breast",
        "pancreas",
        "gbm",
        "glioblastoma",
        "prostate",
        "kidney",
        "liver",
        "ovarian",
        "melanoma",
        "colon",
        "rectal",
        "bladder",
        "thyroid",
        "sarcoma",
        "stomach",
    }
)


def _fmt_r(r: float) -> str:
    if r < 0:
        return f"r={r:.3f} (|r|={abs(r):.3f})"
    return f"r={r:.3f}"


def _papers_to_pairs(corpus: tuple[Paper, ...]) -> list[LitPair]:
    pairs: list[LitPair] = []
    for paper in corpus:
        genes = {
            k.upper().replace("HER2", "ERBB2")
            for k in paper.keywords
            if k.lower() not in _DISEASE_TAGS
            and k.lower()
            not in {
                "entropy",
                "glcm",
                "sphericity",
                "mean",
                "firstorder",
                "texture",
                "shape",
                "volume",
            }
            and len(k) <= 12
        }
        # Heuristic: known gene-like tokens in the static corpus
        gene_like = {k.upper() for k in paper.keywords if k.lower() in {"egfr", "kras", "erbb2", "her2", "tp53", "brca1", "brca2"}}
        gene_like = {("ERBB2" if g == "HER2" else g) for g in gene_like}
        if not gene_like:
            gene_like = genes
        terms = frozenset(k.lower() for k in paper.keywords) | frozenset(
            k.lower() for k in paper.title.replace("-", " ").split() if len(k) > 3
        )
        for gene in sorted(gene_like) or ["UNKNOWN"]:
            if gene == "UNKNOWN":
                continue
            pairs.append(
                LitPair(
                    gene=gene,
                    radiomic_terms=terms,
                    supports=paper.supports,
                    note=paper.note,
                    paper_title=paper.title,
                    pmid=None,
                    journal="",
                    year="",
                )
            )
    return pairs


class LiteratureAgent:
    """Search disease literature, extract radiomic–gene pairs, annotate stats."""

    def __init__(
        self,
        corpus: tuple[Paper, ...] | None = None,
        *,
        use_llm: bool | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_client: LlmClient | None = None,
        pubmed: PubMedClient | None = None,
        max_papers: int = 15,
        verbose: bool = True,
        use_cache: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        # Explicit corpus (including empty) => offline / test mode, no PubMed.
        self._static_corpus = corpus
        self._use_llm = use_llm
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._llm_client = llm_client
        self.pubmed = pubmed or PubMedClient()
        self.max_papers = max_papers
        self.verbose = verbose
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_LIT_CACHE_DIR

        self.pairs: list[LitPair] = []
        self.prepared = False
        self.last_search_query: str | None = None
        self.last_pmids: list[str] = []
        self.last_source: str = "unprepared"
        self.last_rationale: str = ""
        self.last_cache_path: str | None = None

        # Back-compat attribute used by older tests / callers.
        self.corpus = DEFAULT_CORPUS if corpus is None else corpus

    def prepare(self, *, disease: str, question: str = "") -> list[LitPair]:
        """Search / load prior radiomic–gene pairs for this disease question."""
        disease = (disease or "").strip() or "cancer"
        question = (question or "").strip() or disease

        if self._static_corpus is not None:
            self.pairs = _papers_to_pairs(self._static_corpus)
            self.last_source = "static_corpus"
            self.last_search_query = None
            self.last_pmids = []
            self.last_rationale = f"Static corpus ({len(self.pairs)} pairs)"
            self.prepared = True
            log(
                f"[literature] Using static corpus ({len(self.pairs)} prior pairs)",
                enabled=self.verbose,
            )
            return self.pairs

        # PubMed+LLM only when explicitly enabled (--llm) or a client is injected.
        # Avoid surprise network calls in unit tests when a developer has API keys set.
        want_llm = self._use_llm is True or self._llm_client is not None
        cfg = resolve_llm_config(
            provider=self._llm_provider,
            model=self._llm_model,
            enabled=True if want_llm else False,
        )
        client = self._llm_client or LlmClient(cfg)
        if not want_llm or not client.available:
            self.pairs = _papers_to_pairs(DEFAULT_CORPUS)
            self.last_source = "static_fallback" if want_llm else "static_corpus"
            self.last_search_query = None
            self.last_pmids = []
            self.last_rationale = (
                "LLM unavailable; using built-in corpus"
                if want_llm
                else "Built-in corpus (pass --llm for PubMed + LLM extraction)"
            )
            self.prepared = True
            log(
                f"[literature] {self.last_rationale} ({len(self.pairs)} pairs)",
                enabled=self.verbose,
            )
            return self.pairs

        # Disease-level cache: "lung cancer" / "Lung Cancer" / "lung_cancer" share one entry.
        cache_path = self._cache_path(disease, model=cfg.model or "default")
        self.last_cache_path = str(cache_path)
        if self.use_cache:
            cached = self._load_cache(cache_path)
            if cached is not None:
                self.pairs = cached["pairs"]
                self.last_search_query = cached.get("search_query")
                self.last_pmids = list(cached.get("pmids") or [])
                self.last_source = "pubmed_llm_cache"
                self.last_rationale = str(
                    cached.get("rationale")
                    or f"{len(self.pairs)} pairs from local literature cache"
                )
                self.prepared = True
                log(
                    f"[literature] Cache hit for disease='{normalize_disease_key(disease)}' "
                    f"({len(self.pairs)} pairs) → {cache_path.name}",
                    enabled=self.verbose,
                )
                return self.pairs

        query = build_radiogenomics_query(disease, question)
        self.last_search_query = query
        log(
            f"[literature] Searching PubMed for radiomics–mutation papers "
            f"({cfg.provider}/{cfg.model})...",
            enabled=self.verbose,
        )
        try:
            pmids = self.pubmed.search(query, max_results=self.max_papers)
            self.last_pmids = pmids
            articles = self.pubmed.fetch(pmids) if pmids else []
            log(
                f"[literature] PubMed hits: {len(pmids)} ids, fetched {len(articles)} articles",
                enabled=self.verbose,
            )
            extraction = extract_pairs_with_llm(
                articles,
                disease=disease,
                question=question,
                client=client,
            )
            by_pmid = {a.pmid: a for a in articles if a.pmid}
            self.pairs = []
            for p in extraction.pairs:
                art = by_pmid.get(str(p.pmid or "").strip())
                title = (p.paper_title or (art.title if art else "")).strip()
                self.pairs.append(
                    LitPair(
                        gene=p.gene.upper().replace("HER2", "ERBB2"),
                        radiomic_terms=frozenset(t.lower() for t in p.radiomic_terms),
                        supports=p.supports,
                        note=p.note or None,
                        paper_title=title,
                        pmid=(p.pmid or (art.pmid if art else None)) or None,
                        journal=(art.journal if art else "") or "",
                        year=(art.year if art else "") or "",
                    )
                )
            self.last_source = "pubmed_llm"
            self.last_rationale = extraction.rationale or f"{len(self.pairs)} pairs from PubMed+LLM"
            if not self.pairs:
                # Keep going with empty prior set → everything unverified, but note it.
                self.last_rationale = (
                    extraction.rationale
                    or "PubMed+LLM returned no radiomic–gene pairs"
                )
            if self.use_cache:
                self._save_cache(
                    cache_path,
                    disease=disease,
                    model=cfg.model or "default",
                    search_query=query,
                    pmids=pmids,
                    rationale=self.last_rationale,
                    pairs=self.pairs,
                )
                log(
                    f"[literature] Cached {len(self.pairs)} pairs → {cache_path}",
                    enabled=self.verbose,
                )
            log(
                f"[literature] Extracted {len(self.pairs)} prior radiomic–gene pairs "
                f"from literature",
                enabled=self.verbose,
            )
        except Exception as exc:
            self.pairs = _papers_to_pairs(DEFAULT_CORPUS)
            self.last_source = "static_fallback"
            self.last_rationale = f"PubMed/LLM failed ({exc}); using built-in corpus"
            log(
                f"[literature] PubMed/LLM failed ({exc}); falling back to built-in corpus",
                enabled=self.verbose,
            )
        self.prepared = True
        return self.pairs

    def _cache_path(self, disease: str, *, model: str) -> Path:
        disease_key = normalize_disease_key(disease)
        model_key = normalize_disease_key(model) or "default"
        name = f"{disease_key}__p{self.max_papers}__{model_key}.json"
        return self.cache_dir / name

    def _load_cache(self, path: Path) -> dict | None:
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        if int(raw.get("version", 0)) < _LIT_CACHE_VERSION:
            return None
        if int(raw.get("max_papers", -1)) != int(self.max_papers):
            return None
        pairs_raw = raw.get("pairs")
        if not isinstance(pairs_raw, list):
            return None
        pairs: list[LitPair] = []
        for item in pairs_raw:
            if not isinstance(item, dict) or not item.get("gene"):
                continue
            pairs.append(
                LitPair(
                    gene=str(item["gene"]).upper().replace("HER2", "ERBB2"),
                    radiomic_terms=frozenset(
                        str(t).lower() for t in (item.get("radiomic_terms") or [])
                    ),
                    supports=bool(item.get("supports", True)),
                    note=item.get("note"),
                    paper_title=str(item.get("paper_title") or ""),
                    pmid=item.get("pmid"),
                    journal=str(item.get("journal") or ""),
                    year=str(item.get("year") or ""),
                )
            )
        return {
            "pairs": pairs,
            "search_query": raw.get("search_query"),
            "pmids": list(raw.get("pmids") or []),
            "rationale": raw.get("rationale") or "",
        }

    def _save_cache(
        self,
        path: Path,
        *,
        disease: str,
        model: str,
        search_query: str,
        pmids: list[str],
        rationale: str,
        pairs: list[LitPair],
    ) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": _LIT_CACHE_VERSION,
                "disease": disease,
                "disease_key": normalize_disease_key(disease),
                "model": model,
                "max_papers": self.max_papers,
                "search_query": search_query,
                "pmids": list(pmids),
                "rationale": rationale,
                "pairs": [
                    {
                        "gene": p.gene,
                        "radiomic_terms": sorted(p.radiomic_terms),
                        "supports": p.supports,
                        "note": p.note,
                        "paper_title": p.paper_title,
                        "pmid": p.pmid,
                        "journal": p.journal,
                        "year": p.year,
                    }
                    for p in pairs
                ],
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            log(f"[literature] Cache write failed: {exc}", enabled=self.verbose)

    def interpret(
        self,
        result: ModelResult,
        *,
        disease: str,
        question: str = "",
        top_per_category: int = 8,
    ) -> LiteratureContext:
        if not self.prepared:
            self.prepare(disease=disease, question=question or disease)

        supported_assocs: list = []
        unverified_assocs: list = []
        contradicted_assocs: list = []
        papers_for: dict[tuple[str, str], list[LitPair]] = {}

        for assoc in result.associations:
            hits = self._matching_pairs(assoc.imaging_feature, assoc.genomic_feature, disease)
            key = (assoc.imaging_feature, assoc.genomic_feature)
            papers_for[key] = hits
            if not hits:
                unverified_assocs.append(assoc)
                continue
            supportive = [p for p in hits if p.supports]
            conflicting = [p for p in hits if not p.supports]
            if conflicting and not supportive:
                contradicted_assocs.append(assoc)
            else:
                supported_assocs.append(assoc)

        refinements: list[str] = []
        if not result.associations:
            refinements.append("No associations to annotate")
        if self.last_source.startswith("static"):
            refinements.append(self.last_rationale or "Using static literature corpus")
        elif self.last_search_query:
            refinements.append(
                f"Literature source={self.last_source}; "
                f"pubmed_n={len(self.last_pmids)}; prior_pairs={len(self.pairs)}"
            )

        supports: list[LiteratureSupport] = []
        unverified: list[str] = []
        contradictions: list[str] = []

        for assoc in self._top_by_abs_r(supported_assocs, k=top_per_category):
            hits = papers_for[(assoc.imaging_feature, assoc.genomic_feature)]
            supportive = [p for p in hits if p.supports]
            supports.append(
                LiteratureSupport(
                    finding=self._finding_label(assoc),
                    papers=[format_paper_ref(p) for p in supportive],
                    supported=True,
                )
            )

        for assoc in self._top_by_abs_r(unverified_assocs, k=top_per_category):
            unverified.append(self._finding_label(assoc))

        for assoc in self._top_by_abs_r(contradicted_assocs, k=top_per_category):
            hits = papers_for[(assoc.imaging_feature, assoc.genomic_feature)]
            conflicting = [p for p in hits if not p.supports]
            finding = self._finding_label(assoc)
            note = conflicting[0].note if conflicting else None
            title = conflicting[0].paper_title if conflicting else ""
            contradictions.append(f"{finding}: {note or title}")

        if not supports and unverified:
            refinements.append(
                "Literature support is weak; require human review before another fetch"
            )

        return LiteratureContext(
            supports=supports,
            unverified=unverified,
            proposed_refinements=refinements,
            contradictions=contradictions,
            prior_pairs=[
                {
                    "gene": p.gene,
                    "radiomic_terms": sorted(p.radiomic_terms),
                    "supports": p.supports,
                    "note": p.note,
                    "paper_title": p.paper_title,
                    "pmid": p.pmid,
                    "journal": p.journal,
                    "year": p.year,
                }
                for p in self.pairs
            ],
            search_query=self.last_search_query,
            literature_source=self.last_source,
        )

    def classify(
        self,
        imaging_feature: str,
        genomic_feature: str,
        *,
        disease: str,
    ) -> tuple[str, list[str], str | None]:
        """Return (category, paper_titles, contradiction_note)."""
        if not self.prepared:
            # CSV writer may call classify without prepare; use static/default pairs.
            if not self.pairs:
                if self._static_corpus is not None:
                    self.pairs = _papers_to_pairs(self._static_corpus)
                else:
                    self.pairs = _papers_to_pairs(DEFAULT_CORPUS)
                self.prepared = True
                self.last_source = "static_corpus"
        hits = self._matching_pairs(imaging_feature, genomic_feature, disease)
        if not hits:
            return "unverified", [], None
        supportive = [p for p in hits if p.supports]
        conflicting = [p for p in hits if not p.supports]
        if conflicting and not supportive:
            note = conflicting[0].note or conflicting[0].paper_title
            return "contradicted", [format_paper_ref(p) for p in conflicting], note
        return "supported", [format_paper_ref(p) for p in supportive], None

    def _matching_pairs(
        self, imaging_feature: str, genomic_feature: str, disease: str
    ) -> list[LitPair]:
        gene = genomic_feature
        if gene.endswith("_mut") or gene.endswith("_expr"):
            gene = gene.rsplit("_", 1)[0]
        gene = gene.upper().replace("HER2", "ERBB2")
        img_tokens = self._imaging_tokens(imaging_feature)
        disease_tokens = self._disease_tokens(disease)
        hits: list[LitPair] = []
        for pair in self.pairs:
            if pair.gene.upper().replace("HER2", "ERBB2") != gene:
                continue
            # Optional disease filter when the prior pair carries disease tags
            pair_diseases = {t for t in pair.radiomic_terms if t in _DISEASE_TAGS}
            if pair_diseases and not (pair_diseases & disease_tokens):
                continue
            if self._radiomic_overlap(pair.radiomic_terms, img_tokens):
                hits.append(pair)
        return hits

    @staticmethod
    def _radiomic_overlap(pair_terms: frozenset[str], img_tokens: set[str]) -> bool:
        generic = {
            "original",
            "mut",
            "expr",
            "cancer",
            "tumor",
            "tumour",
            "radiomic",
            "radiomics",
            "feature",
            "features",
            "imaging",
            "ct",
            "mri",
            "mr",
            *_DISEASE_TAGS,
        }
        specific = {t.lower() for t in pair_terms} - generic
        if not specific:
            return False
        # Match if any specific term appears in imaging feature tokens or as substring
        for term in specific:
            if term in img_tokens:
                return True
            if any(term in tok or tok in term for tok in img_tokens if len(term) >= 3):
                return True
        return False

    @staticmethod
    def _imaging_tokens(imaging_feature: str) -> set[str]:
        parts = imaging_feature.lower().replace("-", "_").split("_")
        return {p for p in parts if p}

    @staticmethod
    def _finding_label(assoc) -> str:
        r = float(assoc.effect_size)
        return (
            f"{assoc.imaging_feature} ~ {assoc.genomic_feature} "
            f"({_fmt_r(r)}, p={assoc.p_value:.2e}, q={assoc.q_value:.2e}, n={assoc.n})"
        )

    @staticmethod
    def _top_by_abs_r(items: list, *, k: int = 8) -> list:
        if k <= 0 or not items:
            return []
        ranked = sorted(items, key=lambda a: (abs(a.effect_size), a.q_value))
        return ranked[-k:]

    @staticmethod
    def _disease_tokens(disease: str) -> set[str]:
        raw = disease.lower().replace("-", " ").replace("_", " ").strip()
        parts = {raw, *raw.split()}
        parts.discard("cancer")
        parts.discard("tumor")
        parts.discard("tumour")
        return {p for p in parts if p}
