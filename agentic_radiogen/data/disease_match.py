from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Expand clinical shorthand onto terms that actually appear in GDC/TCIA metadata.
# Keep NSCLC / lung / brain groups separate so ranking can prefer the closest label.
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
    frozenset(
        {
            "lung",
            "lung cancer",
            "pulmonary",
            "bronchus",
            "bronchial",
            "luad",
            "lusc",
            "lung adenocarcinoma",
            "adenocarcinoma of the lung",
            "lung squamous",
            "squamous lung",
            "lung squamous cell carcinoma",
            "tcga-luad",
            "tcga-lusc",
        }
    ),
    frozenset({"luad", "lung adenocarcinoma", "adenocarcinoma of the lung", "tcga-luad"}),
    frozenset({"lusc", "lung squamous", "squamous lung", "lung squamous cell carcinoma", "tcga-lusc"}),
    frozenset(
        {
            "brain",
            "brain cancer",
            "cns",
            "central nervous system",
            "glioma",
            "glioblastoma",
            "glioblastoma multiforme",
            "gbm",
            "lgg",
            "lower grade glioma",
            "low grade glioma",
            "tcga-gbm",
            "tcga-lgg",
        }
    ),
    frozenset({"gbm", "glioblastoma", "glioblastoma multiforme", "tcga-gbm"}),
    frozenset({"lgg", "lower grade glioma", "low grade glioma", "tcga-lgg"}),
    frozenset({"breast", "breast cancer", "mammary", "brca", "tcga-brca"}),
    frozenset(
        {
            "bone",
            "bone cancer",
            "osseous",
            "osteosarcoma",
            "chondrosarcoma",
            "ewing",
            "ewing sarcoma",
            "sarcoma",
            "soft tissue sarcoma",
            "tcga-sarc",
        }
    ),
    frozenset(
        {
            "pdac",
            "pancreatic ductal adenocarcinoma",
            "pancreas",
            "pancreatic",
            "pancreatic cancer",
            "tcga-paad",
        }
    ),
)

_QUERY_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "with",
        "from",
        "for",
        "in",
        "of",
        "a",
        "an",
        "or",
        "to",
        "which",
        "what",
        "when",
        "where",
        "that",
        "this",
        "these",
        "those",
        "are",
        "is",
        "associated",
        "features",
        "imaging",
        "specific",
        "genomic",
        "alterations",
        "mutations",
        "patients",
        "between",
        "using",
    }
)


def normalize_question_text(text: str) -> str:
    """Normalize unicode spaces / punctuation that break disease parsing."""
    # NBSP and other unicode spaces → ASCII space (fixes '\\xa0brain').
    out = re.sub(r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]", " ", text)
    out = out.replace("\u2013", "-").replace("\u2014", "-")
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip()


def extract_disease_query(text: str) -> str | None:
    """Pull a disease phrase from free text (prefer longer, more specific phrases)."""
    lowered = normalize_question_text(text).lower()
    patterns = [
        r"non[-\s]?small[-\s]?cell\s+lung\s+cancer",
        r"non[-\s]?small[-\s]?cell\s+lung",
        r"non[-\s]?small[-\s]?cell",
        r"\bnsclc\b",
        r"lung\s+adenocarcinoma",
        r"lung\s+squamous(?:\s+cell)?(?:\s+carcinoma)?",
        r"pancreatic\s+ductal\s+adenocarcinoma",
        r"glioblastoma(?:\s+multiforme)?",
        r"lower\s+grade\s+glioma",
        r"low[-\s]?grade\s+glioma",
        r"\btcga-[a-z0-9]+\b",
        r"\bbrain\s+cancer\b",
        r"\blung\s+cancer\b",
        r"\bbreast\s+cancer\b",
        r"\bpancreatic\s+cancer\b",
        r"\bliver\s+cancer\b",
        r"\bprostate\s+cancer\b",
        r"\bovarian\s+cancer\b",
        r"\bkidney\s+cancer\b",
        r"\bcolon\s+cancer\b",
        r"\bbladder\s+cancer\b",
    ]
    for pat in patterns:
        m = re.search(pat, lowered)
        if m:
            return m.group(0).strip(" -")

    # Generic "<organ> cancer": walk backward from each "cancer" so stopword-heavy
    # spans like "genomic alterations in bone cancer" still yield "bone cancer".
    by_cancer: dict[int, list[tuple[int, str]]] = {}
    for ci, m in enumerate(re.finditer(r"\bcancer\b", lowered)):
        before = lowered[: m.start()].rstrip()
        words = re.findall(r"[a-z][a-z\-]*", before)
        local: list[tuple[int, str]] = []
        for n in range(1, min(5, len(words) + 1)):
            lead = words[-n:]
            if lead[0] in _QUERY_STOPWORDS:
                continue
            if any(w in _QUERY_STOPWORDS for w in lead[1:]):
                continue
            phrase = " ".join(lead) + " cancer"
            if len(phrase) > 48:
                continue
            local.append((n, phrase))
        if local:
            by_cancer[ci] = local
    if by_cancer:
        # Prefer shortest phrase at the last cancer mention (usually the disease named).
        last_ci = max(by_cancer)
        return min(by_cancer[last_ci], key=lambda t: t[0])[1]
    return None


def guess_modality_for_disease(query: str) -> str:
    """Default imaging modality for keyword-matched diseases."""
    q = query.lower()
    if any(
        token in q
        for token in (
            "brain",
            "gbm",
            "glioma",
            "glioblastoma",
            "lgg",
            "breast",
            "prostate",
            "mri",
        )
    ):
        return "MR"
    return "CT"


def keywords_from_query(query: str) -> list[str]:
    """Normalized keyword list for matching collection/project/diagnosis strings."""
    q = normalize_question_text(query).strip().lower()
    if not q:
        return []
    parts = {q}
    parts.add(re.sub(r"[^a-z0-9]+", " ", q).strip())
    parts.add(q.replace("-", " "))
    for tok in re.split(r"[^a-z0-9]+", q):
        if len(tok) > 2 and tok not in _QUERY_STOPWORDS:
            parts.add(tok)
    expanded = set(parts)
    matched: list[tuple[int, frozenset[str]]] = []
    for group in _SYNONYM_GROUPS:
        # Score how specifically this group matches; only expand the best-scoring groups
        # so "non-small cell lung cancer" does not also pull in bare "lung" → TCGA-LUAD.
        best = 0
        for p in parts:
            if p not in group:
                continue
            if len(p) <= 5 and p != q and any(len(x) > len(p) for x in parts):
                continue
            best = max(best, len(p))
        for g in group:
            if len(g) < 6:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", q):
                best = max(best, len(g))
        if best:
            matched.append((best, group))
    if matched:
        max_score = max(score for score, _ in matched)
        for score, group in matched:
            if score >= max_score:
                expanded |= set(group)
    return sorted({p for p in expanded if p}, key=len, reverse=True)


def score_keyword_match(text: str, keywords: Iterable[str]) -> int:
    """Higher = better. Literal/phrase hits beat weak single-token hits."""
    hay = text.lower()
    score = 0
    for kw in keywords:
        if not kw:
            continue
        if kw in hay:
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
    q = normalize_question_text(query).strip().lower()
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
        return "brain cancer"
    if "lower grade glioma" in q or "low grade glioma" in q or re.search(r"\blgg\b", q):
        return "brain cancer"
    if "hepatocellular" in q or re.search(r"\bhcc\b", q):
        return "liver cancer"
    return None


def keyword_match_tiers(disease_query: str) -> list[KeywordTier]:
    """Ordered tiers: exact question wording first, then broader parent disease."""
    specific = normalize_question_text(disease_query).strip()
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


# Generic tokens that match too many unrelated GDC/TCIA labels when ranking sources.
_SOURCE_RANK_STOPWORDS = frozenset(
    {
        "cancer",
        "tumor",
        "tumour",
        "neoplasm",
        "neoplasms",
        "disease",
        "carcinoma",
        "adenocarcinoma",
        "squamous",
    }
)


def source_rank_keywords(keywords: Iterable[str]) -> list[str]:
    """Keywords used to rank GDC/TCIA sources (drop overly generic tokens)."""
    out: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        token = str(kw).strip().lower()
        if not token or token in _SOURCE_RANK_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def site_filter_tokens(disease_query: str) -> list[str]:
    """Primary-site tokens for filtering multi-site GDC projects (e.g. lung → bronchus)."""
    q = normalize_question_text(disease_query).lower()
    tokens: list[str] = []
    if any(t in q for t in ("lung", "nsclc", "luad", "lusc", "pulmonary")):
        tokens.extend(["lung", "bronchus", "bronchial", "pulmonary"])
    if any(t in q for t in ("brain", "gbm", "glioma", "glioblastoma", "lgg")):
        tokens.extend(["brain", "cerebral", "meninges", "central nervous"])
    if any(t in q for t in ("breast", "brca", "mammary")):
        tokens.extend(["breast", "mammary"])
    if any(t in q for t in ("bone", "osteosarcoma", "sarcoma", "osseous")):
        tokens.extend(["bone", "bones", "osseous", "joint"])
    if any(t in q for t in ("pancrea", "pdac")):
        tokens.extend(["pancreas", "pancreatic"])
    return tokens


def primary_site_matches(site: str | None, site_tokens: Iterable[str]) -> bool:
    """True if a GDC primary_site string matches organ tokens for the disease."""
    text = (site or "").lower()
    if not text or text in {"not reported", "unknown", "none"}:
        return False
    return any(tok in text for tok in site_tokens)


def rank_names(names: Iterable[str], keywords: list[str], *, min_score: int = 12) -> list[SourceMatch]:
    ranked: list[SourceMatch] = []
    rank_kw = source_rank_keywords(keywords) or list(keywords)
    for name in names:
        sc = score_keyword_match(name, rank_kw)
        if sc < min_score:
            continue
        hit = next((kw for kw in rank_kw if kw in name.lower()), rank_kw[0] if rank_kw else "")
        ranked.append(SourceMatch(name=name, score=sc, reason=f"matched '{hit}'"))
    ranked.sort(key=lambda m: (-m.score, m.name.lower()))
    return ranked


def diagnosis_matches(diagnosis: str, keywords: list[str]) -> bool:
    """True if a GDC primary_diagnosis string matches disease keywords."""
    text = (diagnosis or "").lower()
    if not text or text in {"not reported", "unknown", "none"}:
        return False
    nsclc_query = any(
        k in {"nsclc", "non-small cell", "non small cell", "non-small-cell"}
        or "non-small" in k
        or "non small" in k
        for k in keywords
    )
    if (
        nsclc_query
        and re.search(r"\bsmall\s+cell\b", text)
        and "non-small" not in text
        and "non small" not in text
    ):
        return False
    return score_keyword_match(text, keywords) >= 12
