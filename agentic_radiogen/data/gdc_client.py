from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agentic_radiogen.data.http_json import post_json

GDC_BASE = "https://api.gdc.cancer.gov"
_CASE_FIELDS = ",".join(
    [
        "submitter_id",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.primary_diagnosis",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.age_at_diagnosis",
    ]
)


@dataclass(frozen=True)
class GdcCase:
    submitter_id: str
    clinical: dict[str, Any]


class GdcClient:
    """Public GDC metadata + small mutation-status queries. No token, no BAM download."""

    def __init__(
        self,
        base_url: str = GDC_BASE,
        poster: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._post = poster or post_json

    def list_cases(
        self,
        project_id: str,
        submitter_ids: list[str] | None = None,
        size: int = 2000,
    ) -> list[GdcCase]:
        if submitter_ids is not None and not submitter_ids:
            return []
        filters: list[dict[str, Any]] = [
            {"op": "=", "content": {"field": "project.project_id", "value": project_id}}
        ]
        if submitter_ids:
            filters.append(
                {"op": "in", "content": {"field": "submitter_id", "value": submitter_ids}}
            )
        payload = {
            "filters": {"op": "and", "content": filters} if len(filters) > 1 else filters[0],
            "fields": _CASE_FIELDS,
            "size": min(size, len(submitter_ids) if submitter_ids else size),
        }
        data = self._post(f"{self.base_url}/cases", payload)
        hits = data.get("data", {}).get("hits", []) if isinstance(data, dict) else []
        cases: list[GdcCase] = []
        for hit in hits:
            submitter = str(hit.get("submitter_id") or "")
            if submitter:
                cases.append(GdcCase(submitter_id=submitter, clinical=_clinical_from_hit(hit)))
        return cases

    def mutation_flags(self, submitter_ids: list[str], genes: list[str]) -> dict[str, dict[str, int]]:
        flags = {pid: {gene: 0 for gene in genes} for pid in submitter_ids}
        if not submitter_ids or not genes:
            return flags
        payload = {
            "filters": {
                "op": "and",
                "content": [
                    {
                        "op": "in",
                        "content": {
                            "field": "ssm.consequence.transcript.gene.symbol",
                            "value": genes,
                        },
                    },
                    {
                        "op": "in",
                        "content": {"field": "case.submitter_id", "value": submitter_ids},
                    },
                ],
            },
            "fields": "case.submitter_id,ssm.consequence.transcript.gene.symbol",
            "size": 10000,
        }
        data = self._post(f"{self.base_url}/ssm_occurrences", payload)
        hits = data.get("data", {}).get("hits", []) if isinstance(data, dict) else []
        for hit in hits:
            case = hit.get("case") or {}
            pid = str(case.get("submitter_id") or "")
            if pid not in flags:
                continue
            for symbol in _gene_symbols(hit):
                if symbol in flags[pid]:
                    flags[pid][symbol] = 1
        return flags

    def expression_means(
        self, submitter_ids: list[str], genes: list[str]
    ) -> dict[str, dict[str, float]]:
        """Best-effort open expression summaries. Returns {} if unavailable."""
        _ = self  # reserved for future gene-expression file queries
        if not submitter_ids or not genes:
            return {}
        # Open RNA-seq bulk pull is heavy; Stage live currently uses mutation flags.
        # Keep the hook so GenomicsAgent can consume expression when added.
        return {pid: {} for pid in submitter_ids}


def _clinical_from_hit(hit: dict[str, Any]) -> dict[str, Any]:
    diagnoses = hit.get("diagnoses") or []
    diagnosis = diagnoses[0] if diagnoses else {}
    demographic = hit.get("demographic") or {}
    vital = str(
        diagnosis.get("vital_status")
        or demographic.get("vital_status")
        or ""
    )
    days_death = diagnosis.get("days_to_death")
    if days_death in (None, ""):
        days_death = demographic.get("days_to_death")
    days_follow = diagnosis.get("days_to_last_follow_up")
    os_time = days_death if days_death not in (None, "") else days_follow
    clinical: dict[str, Any] = {
        "subtype": diagnosis.get("primary_diagnosis"),
        "stage": diagnosis.get("ajcc_pathologic_stage"),
        "vital_status": vital,
        "age_at_diagnosis": diagnosis.get("age_at_diagnosis"),
    }
    if os_time not in (None, ""):
        clinical["OS_time"] = float(os_time)
        clinical["OS_event"] = 1.0 if vital.lower() == "dead" else 0.0
    return {k: v for k, v in clinical.items() if v is not None and v != ""}


def _gene_symbols(hit: dict[str, Any]) -> set[str]:
    symbols: set[str] = set()
    ssm = hit.get("ssm") or {}
    for consequence in ssm.get("consequence") or []:
        transcript = (consequence or {}).get("transcript") or {}
        gene = transcript.get("gene") or {}
        symbol = gene.get("symbol")
        if symbol:
            symbols.add(str(symbol))
    return symbols
