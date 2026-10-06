from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Expand clinical shorthand onto terms that actually appear in GDC/TCIA metadata.
_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "nsclc",
            "non-small cell",
            "non small cell",
            "non-small-cell",
            "non-small cell lung",
            "non small cell lung",
            "non-small cell lung cancer",
            "non small cell lung cancer",
            "adenocarcinoma",
            "squamous",
            "squamous cell",
            "squamous cell carcinoma",
            "large cell",
        }
    ),
    frozenset({"luad", "lung adenocarcinoma", "adenocarcinoma of the lung"}),
    frozenset({"lusc", "lung squamous", "squamous lung", "lung squamous cell carcinoma"}),
    frozenset({"gbm", "glioblastoma", "glioblastoma multiforme"}),
    frozenset({"pdac", "pancreatic ductal adenocarcinoma", "pancreas", "pancreatic"}),
)


def extract_disease_query(text: str) -> str | None:
    """Pull a disease phrase from free text (prefer longer, more specific phrases)."""
    lowered = text.lower()
    # Explicit abbreviations / multi-word phrases first.
    patterns = [
        r"non[-\s]?small[-\s]?cell\s+lung\s+cancer",
        r"non[-\s]?small[-\s]?cell\s+lung",
        r"non[-\s]?small[-\s]?cell",
        r"\bnsclc\b",
        r"lung\s+adenocarcinoma",
        r"lung\s+squamous(?:\s+cell)?(?:\s+carcinoma)?",
        r"pancreatic\s+ductal\s+adenocarcinoma",
        r"glioblastoma(?:\s+multiforme)?",
        r"\btcga-[a-z0-9]+\b",
        r"([a-z][a-z\s\-]{2,40}?)\s+cancer",
    ]
    for pat in patterns:
        m = re.search(pat, lowered)
        if m:
            return (m.group(0) if m.lastindex is None else m.group(0)).strip(" -")
    return None


def keywords_from_query(query: str) -> list[str]:
    """Normalized keyword list for matching collection/project/diagnosis strings."""
    q = query.strip().lower()
    if not q:
        return []
    parts = {q}
    parts.add(re.sub(r"[^a-z0-9]+", " ", q).strip())
    parts.add(q.replace("-", " "))
    # Individual tokens longer than 2 chars, skip stopwords.
    stop = {"the", "and", "with", "from", "for", "in", "of", "a", "an", "or", "to"}
    for tok in re.split(r"[^a-z0-9]+", q):
        if len(tok) > 2 and tok not in stop:
            parts.add(tok)
    # Synonym expansion.
    expanded = set(parts)
    for group in _SYNONYM_GROUPS:
        if any(p in group or any(g in p for g in group if len(g) > 3) for p in parts):
            expanded |= set(group)
    # Prefer longer keywords first for scoring.
    return sorted({p for p in expanded if p}, key=len, reverse=True)


def score_keyword_match(text: str, keywords: Iterable[str]) -> int:
    """Higher = better. Literal/phrase hits beat weak single-token hits."""
    hay = text.lower()
    score = 0
    for kw in keywords:
        if not kw:
            continue
        if kw in hay:
            # Length-weighted; whole-word bonus for short tokens.
            score += 10 + len(kw)
            if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", hay):
                score += 5
    return score


@dataclass(frozen=True)
class KeywordTier:
    """One pass of dataset keyword matching (specific → broader parent disease)."""

    label: str
    query: str
    keywords: list[str]
    broadened_from: str | None = None


def broader_disease_query(query: str) -> str | None:
    """Parent disease phrase when the question term is narrower than available paired data."""
    q = query.strip().lower()
    if any(
        token in q
        for token in (
            "non-small",
            "non small",
            "nsclc",
            "non-small-cell",
        )
    ):
        return "lung cancer"
    if "lung adenocarcinoma" in q or re.search(r"\bluad\b", q):
        return "lung cancer"
    if "lung squamous" in q or re.search(r"\blusc\b", q):
        return "lung cancer"
    if "pancreatic ductal" in q or re.search(r"\bpdac\b", q):
        return "pancreatic cancer"
    if "glioblastoma" in q or re.search(r"\bgbm\b", q):
        return "glioblastoma"
    if "hepatocellular" in q or re.search(r"\bhcc\b", q):
        return "liver cancer"
    return None


def keyword_match_tiers(disease_query: str) -> list[KeywordTier]:
    """Ordered tiers: exact question wording first, then broader parent disease."""
    specific = disease_query.strip()
    tiers: list[KeywordTier] = [
        KeywordTier(
            label="specific",
            query=specific,
            keywords=keywords_from_query(specific),
        )
    ]
    broader = broader_disease_query(specific)
    if broader:
        broad_kw = keywords_from_query(broader)
        if broad_kw != tiers[0].keywords or broader.lower() != specific.lower():
            tiers.append(
                KeywordTier(
                    label="broadened",
                    query=broader,
                    keywords=broad_kw,
                    broadened_from=specific,
                )
            )
    return tiers


def format_dataset_extraction_note(intermediate: dict) -> str | None:
    """One-line note for CLI: which keywords selected TCIA/GDC data."""
    if not intermediate:
        return None
    keywords = intermediate.get("dataset_keywords_used") or intermediate.get(
        "disease_keywords"
    )
    if not keywords:
        return None
    kw_text = ", ".join(str(k) for k in keywords[:16])
    if len(keywords) > 16:
        kw_text += f", ... ({len(keywords)} total)"
    tcia = intermediate.get("tcia_collection") or intermediate.get("tcia_collections")
    gdc = intermediate.get("gdc_project") or intermediate.get("gdc_projects")
    sources = ""
    if tcia or gdc:
        sources = f" → TCIA [{tcia}] ∩ GDC [{gdc}]"
    tier = intermediate.get("dataset_match_tier")
    if tier == "broadened" and intermediate.get("dataset_broadened_from"):
        return (
            f"Note: No paired cohort matched {intermediate['dataset_broadened_from']!r}; "
            f"used broader disease keywords [{kw_text}]{sources}."
        )
    if intermediate.get("keyword_match") or intermediate.get("dataset_match_tier"):
        return f"Note: Dataset extraction keywords [{kw_text}]{sources}."
    return f"Note: Dataset extraction keywords [{kw_text}]{sources}."


@dataclass(frozen=True)
class SourceMatch:
    name: str
    score: int
    reason: str


def rank_names(names: Iterable[str], keywords: list[str], *, min_score: int = 12) -> list[SourceMatch]:
    ranked: list[SourceMatch] = []
    for name in names:
        sc = score_keyword_match(name, keywords)
        if sc < min_score:
            continue
        hit = next((kw for kw in keywords if kw in name.lower()), keywords[0] if keywords else "")
        ranked.append(SourceMatch(name=name, score=sc, reason=f"matched '{hit}'"))
    ranked.sort(key=lambda m: (-m.score, m.name.lower()))
    return ranked


def diagnosis_matches(diagnosis: str, keywords: list[str]) -> bool:
    """True if a GDC primary_diagnosis string matches disease keywords."""
    text = (diagnosis or "").lower()
    if not text or text in {"not reported", "unknown", "none"}:
        return False
    # NSCLC must not keep pure small-cell diagnoses.
    nsclc_query = any(
        k in {"nsclc", "non-small cell", "non small cell", "non-small-cell"}
        or "non-small" in k
        or "non small" in k
        for k in keywords
    )
    if nsclc_query and re.search(r"\bsmall\s+cell\b", text) and "non-small" not in text and "non small" not in text:
        return False
    return score_keyword_match(text, keywords) >= 12
