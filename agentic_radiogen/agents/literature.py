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


class LiteratureAgent:
    """Interprets statistical findings. Its output, not stats, feeds the self-correction loop."""

    def __init__(self, corpus: tuple[Paper, ...] | None = None) -> None:
        self.corpus = DEFAULT_CORPUS if corpus is None else corpus

    def interpret(self, result: ModelResult, *, disease: str) -> LiteratureContext:
        supports: list[LiteratureSupport] = []
        unverified: list[str] = []
        contradictions: list[str] = []
        refinements: list[str] = []

        top = [a for a in result.associations if a.q_value <= 0.1][:8]
        if not top:
            top = result.associations[:3]
            refinements.append("No FDR-significant association; request a different feature subset")

        for assoc in top:
            finding = f"{assoc.imaging_feature} ~ {assoc.genomic_feature}"
            tokens = self._tokens(assoc.imaging_feature, assoc.genomic_feature, disease)
            hits = [p for p in self.corpus if self._paper_matches(p, tokens, disease)]
            if not hits:
                unverified.append(finding)
                refinements.append(f"Seek alternative data for unverified finding: {finding}")
                supports.append(LiteratureSupport(finding=finding, supported=False))
                continue
            supportive = [p for p in hits if p.supports]
            conflicting = [p for p in hits if not p.supports]
            if conflicting and not supportive:
                msg = f"{finding}: {conflicting[0].note or conflicting[0].title}"
                contradictions.append(msg)
                refinements.append(f"Drop or re-test contradicted finding: {finding}")
                supports.append(
                    LiteratureSupport(
                        finding=finding,
                        papers=[p.title for p in conflicting],
                        supported=False,
                        contradiction=conflicting[0].note,
                    )
                )
            else:
                supports.append(
                    LiteratureSupport(
                        finding=finding,
                        papers=[p.title for p in supportive],
                        supported=True,
                    )
                )

        if all(not item.supported for item in supports) and "stop after human review" not in refinements:
            refinements.append("Literature support is weak; require human review before another fetch")

        return LiteratureContext(
            supports=supports,
            unverified=unverified,
            proposed_refinements=refinements,
            contradictions=contradictions,
        )

    @staticmethod
    def _tokens(imaging_feature: str, genomic_feature: str, disease: str) -> set[str]:
        parts = (
            imaging_feature.lower().replace("-", "_").split("_")
            + genomic_feature.lower().replace("-", "_").split("_")
            + [disease.lower()]
        )
        aliases = set(parts)
        if "erbb2" in aliases:
            aliases.add("her2")
        return {p for p in aliases if p}

    @staticmethod
    def _paper_matches(paper: Paper, tokens: set[str], disease: str) -> bool:
        generic = {"original", "firstorder", "shape", "mut", "expr", "cancer", disease.lower()}
        return bool((paper.keywords & tokens) - generic)
