from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agentic_radiogen.data.http_json import post_json

GDC_BASE = "https://api.gdc.cancer.gov"
_CASE_FIELDS = ",".join(
    [
        "submitter_id",
        "primary_site",
        "project.project_id",
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

    def list_projects(self, *, size: int = 200) -> list[dict[str, Any]]:
        """List GDC projects (id, name, primary_site, disease_type)."""
        payload = {
            "fields": "project_id,name,primary_site,disease_type",
            "size": size,
            "from": 0,
        }
        data = self._post(f"{self.base_url}/projects", payload)
        hits = data.get("data", {}).get("hits", []) if isinstance(data, dict) else []
        return [h for h in hits if isinstance(h, dict)]

    def primary_diagnosis_counts(self, project_ids: list[str]) -> dict[str, int]:
        """Facet primary_diagnosis values across one or more GDC projects."""
        if not project_ids:
            return {}
        payload = {
            "filters": {
                "op": "in",
                "content": {"field": "project.project_id", "value": list(project_ids)},
            },
            "facets": "diagnoses.primary_diagnosis",
            "size": 0,
        }
        data = self._post(f"{self.base_url}/cases", payload)
        agg = ((data.get("data") or {}).get("aggregations") or {}) if isinstance(data, dict) else {}
        buckets = (agg.get("diagnoses.primary_diagnosis") or {}).get("buckets") or []
        out: dict[str, int] = {}
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            key = str(bucket.get("key") or "").strip()
            if key:
                out[key] = int(bucket.get("doc_count") or 0)
        return out

    def list_cases(
        self,
        project_id: str,
        submitter_ids: list[str] | None = None,
        size: int = 10000,
        page_size: int = 1000,
    ) -> list[GdcCase]:
        """List cases for a GDC project, paginating until exhausted.

        When ``submitter_ids`` is provided, results are filtered locally after the
        full project listing so intersection with TCIA is complete (not capped by
        an early IN-filter page).
        """
        return self.list_cases_multi(
            [project_id],
            submitter_ids=submitter_ids,
            size=size,
            page_size=page_size,
        )

    def list_cases_multi(
        self,
        project_ids: list[str],
        submitter_ids: list[str] | None = None,
        size: int = 20000,
        page_size: int = 1000,
    ) -> list[GdcCase]:
        """List cases across one or more GDC projects.

        Fetches each project separately so a shared page budget cannot drop
        cases from pairable TCGA projects when many keyword-matched projects
        are requested together.
        """
        projects = [p for p in project_ids if p and p != "KEYWORD"]
        if not projects:
            return []
        if submitter_ids is not None and not submitter_ids:
            return []
        wanted = set(submitter_ids) if submitter_ids is not None else None
        cases: list[GdcCase] = []
        seen: set[str] = set()
        page = max(1, min(page_size, size))
        for project_id in projects:
            offset = 0
            while offset < size:
                payload = {
                    "filters": {
                        "op": "in",
                        "content": {
                            "field": "project.project_id",
                            "value": [project_id],
                        },
                    },
                    "fields": _CASE_FIELDS,
                    "size": min(page, size - offset),
                    "from": offset,
                }
                data = self._post(f"{self.base_url}/cases", payload)
                hits = (
                    data.get("data", {}).get("hits", [])
                    if isinstance(data, dict)
                    else []
                )
                if not hits:
                    break
                for hit in hits:
                    submitter = str(hit.get("submitter_id") or "")
                    if not submitter or submitter in seen:
                        continue
                    if wanted is not None and submitter not in wanted:
                        continue
                    seen.add(submitter)
                    cases.append(
                        GdcCase(
                            submitter_id=submitter,
                            clinical=_clinical_from_hit(hit),
                        )
                    )
                total = (
                    ((data.get("data") or {}).get("pagination") or {}).get("total")
                    if isinstance(data, dict)
                    else None
                )
                offset += len(hits)
                if total is not None and offset >= int(total):
                    break
                if len(hits) < payload["size"]:
                    break
        return cases

    def mutation_flags(self, submitter_ids: list[str], genes: list[str]) -> dict[str, dict[str, int]]:
        flags = {pid: {gene: 0 for gene in genes} for pid in submitter_ids}
        if not submitter_ids or not genes:
            return flags
        # Large gene panels: build flags from an unfiltered cohort scan (avoids huge IN filters).
        if len(genes) > 100:
            per_patient = self._mutation_hits(submitter_ids, genes=None)
            wanted = set(genes)
            for pid, symbols in per_patient.items():
                if pid not in flags:
                    continue
                for symbol in symbols:
                    if symbol in wanted:
                        flags[pid][symbol] = 1
            return flags
        for pid, observed in self._mutation_hits(submitter_ids, genes=genes).items():
            if pid not in flags:
                continue
            for symbol in observed:
                if symbol in flags[pid]:
                    flags[pid][symbol] = 1
        return flags

    def discover_genes(
        self,
        submitter_ids: list[str],
        *,
        min_altered: int = 1,
        max_genes: int | None = None,
    ) -> list[str]:
        """Return every gene mutated in this cohort (optionally frequency-filtered/capped)."""
        genes, _counts = self._gene_counts(submitter_ids, min_altered=min_altered)
        if max_genes is not None:
            genes = genes[: max(0, max_genes)]
        return genes

    def _gene_counts(
        self, submitter_ids: list[str], *, min_altered: int = 1
    ) -> tuple[list[str], dict[str, int]]:
        if not submitter_ids:
            return [], {}
        per_patient = self._mutation_hits(submitter_ids, genes=None)
        counts: dict[str, int] = {}
        for symbols in per_patient.values():
            for symbol in symbols:
                counts[symbol] = counts.get(symbol, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        genes = [gene for gene, n in ranked if n >= min_altered]
        return genes, counts

    def _mutation_hits(
        self,
        submitter_ids: list[str],
        *,
        genes: list[str] | None,
        page_size: int = 2000,
        max_hits: int = 200000,
    ) -> dict[str, set[str]]:
        """Map patient_id → mutated gene symbols observed in GDC SSM data."""
        observed: dict[str, set[str]] = {pid: set() for pid in submitter_ids}
        if not submitter_ids:
            return observed
        filters: list[dict[str, Any]] = [
            {"op": "in", "content": {"field": "case.submitter_id", "value": submitter_ids}}
        ]
        if genes:
            filters.append(
                {
                    "op": "in",
                    "content": {
                        "field": "ssm.consequence.transcript.gene.symbol",
                        "value": genes,
                    },
                }
            )
        offset = 0
        while offset < max_hits:
            payload = {
                "filters": {"op": "and", "content": filters} if len(filters) > 1 else filters[0],
                "fields": "case.submitter_id,ssm.consequence.transcript.gene.symbol",
                "size": min(page_size, max_hits - offset),
                "from": offset,
            }
            data = self._post(f"{self.base_url}/ssm_occurrences", payload)
            hits = data.get("data", {}).get("hits", []) if isinstance(data, dict) else []
            if not hits:
                break
            for hit in hits:
                case = hit.get("case") or {}
                pid = str(case.get("submitter_id") or "")
                if pid not in observed:
                    continue
                observed[pid].update(_gene_symbols(hit))
            total = (
                ((data.get("data") or {}).get("pagination") or {}).get("total")
                if isinstance(data, dict)
                else None
            )
            offset += len(hits)
            if total is not None and offset >= int(total):
                break
            if len(hits) < payload["size"]:
                break
        return observed

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
