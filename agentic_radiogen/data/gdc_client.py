from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agentic_radiogen.data.http_json import get_bytes, post_json

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
_CASE_DETAIL_FIELDS = ",".join(
    [
        "case_id",
        "submitter_id",
        "primary_site",
        "project.project_id",
        "project.disease_type",
        "demographic.gender",
        "demographic.race",
        "demographic.ethnicity",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.tumor_stage",
        "diagnoses.primary_diagnosis",
        "diagnoses.age_at_diagnosis",
    ]
)
_CLINICAL_FILE_FIELDS = ",".join(
    [
        "file_id",
        "file_name",
        "data_category",
        "data_type",
        "experimental_strategy",
        "file_size",
        "md5sum",
        "cases.submitter_id",
    ]
)

# Prespecified MAF Variant_Classification set used for binary gene labels
# (same definition as TP53 cohort labeling from somatic MAF).
QUALIFYING_MAF_VARIANT_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "Splice_Site",
        "In_Frame_Del",
        "In_Frame_Ins",
        "Nonstop_Mutation",
        "Translation_Start_Site",
    }
)

# GDC SSM API uses Sequence Ontology consequence_type; map the MAF classes above.
QUALIFYING_CONSEQUENCE_TYPES: frozenset[str] = frozenset(
    {
        "missense_variant",  # Missense_Mutation
        "stop_gained",  # Nonsense_Mutation
        "frameshift_variant",  # Frame_Shift_Del / Frame_Shift_Ins
        "splice_acceptor_variant",  # Splice_Site
        "splice_donor_variant",  # Splice_Site
        "inframe_deletion",  # In_Frame_Del
        "inframe_insertion",  # In_Frame_Ins
        "stop_lost",  # Nonstop_Mutation
        "start_lost",  # Translation_Start_Site
    }
)

_SSM_FIELDS = ",".join(
    [
        "case.submitter_id",
        "case.observation.mutation_status",
        "case.observation.sample.tumor_sample_barcode",
        "ssm.consequence.transcript.gene.symbol",
        "ssm.consequence.transcript.consequence_type",
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
        downloader: Callable[..., bytes] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._post = poster or post_json
        self._download = downloader or get_bytes

    def get_case_details(self, submitter_id: str) -> dict[str, Any] | None:
        """Return the public GDC case fields useful for a patient-level review."""
        payload = {
            "filters": {
                "op": "in",
                "content": {
                    "field": "submitter_id",
                    "value": [submitter_id],
                },
            },
            "fields": _CASE_DETAIL_FIELDS,
            "size": 2,
            "from": 0,
        }
        data = self._post(f"{self.base_url}/cases", payload)
        hits = data.get("data", {}).get("hits", []) if isinstance(data, dict) else []
        for hit in hits:
            if isinstance(hit, dict) and hit.get("submitter_id") == submitter_id:
                return hit
        return None

    def list_clinical_files(self, submitter_id: str) -> list[dict[str, Any]]:
        """List downloadable public GDC files categorized as clinical."""
        results: list[dict[str, Any]] = []
        offset = 0
        page_size = 1000
        while True:
            payload = {
                "filters": {
                    "op": "and",
                    "content": [
                        {
                            "op": "in",
                            "content": {
                                "field": "cases.submitter_id",
                                "value": [submitter_id],
                            },
                        },
                        {
                            "op": "in",
                            "content": {
                                "field": "data_category",
                                "value": ["Clinical"],
                            },
                        },
                    ],
                },
                "fields": _CLINICAL_FILE_FIELDS,
                "size": page_size,
                "from": offset,
            }
            data = self._post(f"{self.base_url}/files", payload)
            response_data = data.get("data", {}) if isinstance(data, dict) else {}
            hits = response_data.get("hits", [])
            results.extend(hit for hit in hits if isinstance(hit, dict))
            pagination = response_data.get("pagination") or {}
            total = pagination.get("total")
            offset += len(hits)
            if not hits or len(hits) < page_size or (
                total is not None and offset >= int(total)
            ):
                return results

    def download_file(self, file_id: str) -> bytes:
        """Download a public GDC file by UUID."""
        return self._download(f"{self.base_url}/data/{file_id}", timeout=600)

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
        min_altered: int = 4,
        max_genes: int | None = None,
    ) -> list[str]:
        """Return every gene mutated in this cohort (optionally frequency-filtered/capped)."""
        genes, _counts = self._gene_counts(submitter_ids, min_altered=min_altered)
        if max_genes is not None:
            genes = genes[: max(0, max_genes)]
        return genes

    def _gene_counts(
        self, submitter_ids: list[str], *, min_altered: int = 4
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
        """Map patient_id → genes with a qualifying somatic variant (MAF-style).

        A patient is mutant for a gene when any remaining SSM occurrence has a
        ``Variant_Classification``-equivalent consequence in
        ``QUALIFYING_CONSEQUENCE_TYPES`` and ``mutation_status`` somatic.
        Silent / intron / UTR-only calls do not label mutant.
        """
        observed: dict[str, set[str]] = {pid: set() for pid in submitter_ids}
        if not submitter_ids:
            return observed
        # Also index by 12-char TCGA patient IDs so aliquot barcodes resolve.
        by_patient = {pid: pid for pid in submitter_ids}
        for pid in submitter_ids:
            short = _tcga_patient_id(pid)
            if short and short not in by_patient:
                by_patient[short] = pid
        filters: list[dict[str, Any]] = [
            {"op": "in", "content": {"field": "case.submitter_id", "value": submitter_ids}},
            # GDC open SSM quality gate (analogous to somatic-MAF Mutation_Status).
            {
                "op": "in",
                "content": {
                    "field": "case.observation.mutation_status",
                    "value": ["somatic", "Somatic"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "ssm.consequence.transcript.consequence_type",
                    "value": sorted(QUALIFYING_CONSEQUENCE_TYPES),
                },
            },
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
                "filters": {"op": "and", "content": filters},
                "fields": _SSM_FIELDS,
                "size": min(page_size, max_hits - offset),
                "from": offset,
            }
            data = self._post(f"{self.base_url}/ssm_occurrences", payload)
            hits = data.get("data", {}).get("hits", []) if isinstance(data, dict) else []
            if not hits:
                break
            for hit in hits:
                pid = _patient_id_from_hit(hit)
                key = by_patient.get(pid) or by_patient.get(_tcga_patient_id(pid))
                if key is None:
                    continue
                observed[key].update(_qualifying_gene_symbols(hit))
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
    project = hit.get("project") or {}
    clinical: dict[str, Any] = {
        "subtype": diagnosis.get("primary_diagnosis"),
        "primary_diagnosis": diagnosis.get("primary_diagnosis"),
        "stage": diagnosis.get("ajcc_pathologic_stage"),
        "vital_status": vital,
        "age_at_diagnosis": diagnosis.get("age_at_diagnosis"),
        "primary_site": hit.get("primary_site"),
        "_gdc_project": project.get("project_id"),
    }
    if os_time not in (None, ""):
        clinical["OS_time"] = float(os_time)
        clinical["OS_event"] = 1.0 if vital.lower() == "dead" else 0.0
    return {k: v for k, v in clinical.items() if v is not None and v != ""}


def _tcga_patient_id(raw: str) -> str:
    """Map a TCGA aliquot/sample barcode to the 12-char patient ID."""
    s = (raw or "").strip()
    if not s:
        return ""
    if s.startswith("TCGA-") and len(s) >= 12:
        return s[:12]
    return s


def _patient_id_from_hit(hit: dict[str, Any]) -> str:
    case = hit.get("case") or {}
    submitter = str(case.get("submitter_id") or "").strip()
    if submitter:
        return _tcga_patient_id(submitter)
    observation = case.get("observation") or {}
    sample = observation.get("sample") or {}
    barcode = str(sample.get("tumor_sample_barcode") or "").strip()
    return _tcga_patient_id(barcode)


def _is_qualifying_consequence(consequence_type: str) -> bool:
    ct = (consequence_type or "").strip()
    if not ct:
        return False
    if ct in QUALIFYING_CONSEQUENCE_TYPES:
        return True
    # Accept MAF-style names if a caller/fixture supplies them directly.
    return ct in QUALIFYING_MAF_VARIANT_CLASSIFICATIONS


def _qualifying_gene_symbols(hit: dict[str, Any]) -> set[str]:
    """Gene symbols with at least one qualifying coding/splice consequence."""
    symbols: set[str] = set()
    ssm = hit.get("ssm") or {}
    for consequence in ssm.get("consequence") or []:
        transcript = (consequence or {}).get("transcript") or {}
        if not _is_qualifying_consequence(str(transcript.get("consequence_type") or "")):
            continue
        gene = transcript.get("gene") or {}
        symbol = gene.get("symbol")
        if symbol:
            symbols.add(str(symbol))
    return symbols
