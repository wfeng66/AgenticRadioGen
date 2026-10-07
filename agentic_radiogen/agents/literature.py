from __future__ import annotations

from dataclasses import dataclass

from agentic_radiogen.schemas.contracts import LiteratureContext, LiteratureSupport, ModelResult


@dataclass(frozen=True)
class Paper:
    title: str
    keywords: frozenset[str]
    supports: bool = True
    note: str | None = None


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
    """Show |r| only when it differs from r (i.e. negative correlation)."""
    if r < 0:
        return f"r={r:.3f} (|r|={abs(r):.3f})"
    return f"r={r:.3f}"


class LiteratureAgent:
    """Annotates statistical findings with prior papers. Does not alter data-driven results."""

    def __init__(self, corpus: tuple[Paper, ...] | None = None) -> None:
        self.corpus = DEFAULT_CORPUS if corpus is None else corpus

    @staticmethod
    def _finding_label(assoc) -> str:
        r = float(assoc.effect_size)
        return (
            f"{assoc.imaging_feature} ~ {assoc.genomic_feature} "
            f"({_fmt_r(r)}, p={assoc.p_value:.2e}, q={assoc.q_value:.2e}, n={assoc.n})"
        )

    @staticmethod
    def _top_by_abs_r(items: list, *, k: int = 8) -> list:
        """Strongest k by |r|, returned in ascending |r| (weaker → stronger)."""
        if k <= 0 or not items:
            return []
        ranked = sorted(items, key=lambda a: (abs(a.effect_size), a.q_value))
        return ranked[-k:]

    def interpret(
        self,
        result: ModelResult,
        *,
        disease: str,
        top_per_category: int = 8,
    ) -> LiteratureContext:
        supported_assocs: list = []
        unverified_assocs: list = []
        contradicted_assocs: list = []
        papers_for: dict[tuple[str, str], list] = {}

        for assoc in result.associations:
            tokens = self._tokens(assoc.imaging_feature, assoc.genomic_feature, disease)
            hits = [p for p in self.corpus if self._paper_matches(p, tokens, disease)]
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
        elif not supported_assocs and not contradicted_assocs and not unverified_assocs:
            refinements.append(
                "No FDR-significant association in this cohort (data-driven; kept for review)"
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
                    papers=[p.title for p in supportive],
                    supported=True,
                )
            )

        for assoc in self._top_by_abs_r(unverified_assocs, k=top_per_category):
            finding = self._finding_label(assoc)
            unverified.append(finding)

        for assoc in self._top_by_abs_r(contradicted_assocs, k=top_per_category):
            hits = papers_for[(assoc.imaging_feature, assoc.genomic_feature)]
            conflicting = [p for p in hits if not p.supports]
            finding = self._finding_label(assoc)
            note = conflicting[0].note if conflicting else None
            title = conflicting[0].title if conflicting else ""
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
        )

    @staticmethod
    def _disease_tokens(disease: str) -> set[str]:
        raw = disease.lower().replace("-", " ").replace("_", " ").strip()
        parts = {raw, *raw.split()}
        parts.discard("cancer")
        parts.discard("tumor")
        parts.discard("tumour")
        return {p for p in parts if p}

    @staticmethod
    def _tokens(imaging_feature: str, genomic_feature: str, disease: str) -> set[str]:
        parts = (
            imaging_feature.lower().replace("-", "_").split("_")
            + genomic_feature.lower().replace("-", "_").split("_")
            + list(LiteratureAgent._disease_tokens(disease))
        )
        aliases = set(parts)
        if "erbb2" in aliases:
            aliases.add("her2")
        return {p for p in aliases if p}

    @staticmethod
    def _paper_matches(paper: Paper, tokens: set[str], disease: str) -> bool:
        disease_tokens = LiteratureAgent._disease_tokens(disease)
        paper_diseases = paper.keywords & _DISEASE_TAGS
        if paper_diseases and not (paper_diseases & disease_tokens):
            return False
        generic = {
            "original",
            "firstorder",
            "shape",
            "mut",
            "expr",
            "cancer",
            "sphericity",
            "volume",
            "mean",
            "median",
            "std",
            "skewness",
            "entropy",
            "glcm",
            *disease_tokens,
        }
        return bool((paper.keywords & tokens) - generic)

    def classify(
        self,
        imaging_feature: str,
        genomic_feature: str,
        *,
        disease: str,
    ) -> tuple[str, list[str], str | None]:
        """Return (category, paper_titles, contradiction_note).

        category is one of: supported, unverified, contradicted.
        """
        tokens = self._tokens(imaging_feature, genomic_feature, disease)
        hits = [p for p in self.corpus if self._paper_matches(p, tokens, disease)]
        if not hits:
            return "unverified", [], None
        supportive = [p for p in hits if p.supports]
        conflicting = [p for p in hits if not p.supports]
        if conflicting and not supportive:
            note = conflicting[0].note or conflicting[0].title
            return "contradicted", [p.title for p in conflicting], note
        return "supported", [p.title for p in supportive], None
