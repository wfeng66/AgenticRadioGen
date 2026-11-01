from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from agentic_radiogen.data.catalog import CatalogRecord
from agentic_radiogen.data.disease_match import (
    KeywordTier,
    diagnosis_matches,
    expected_organ_keywords,
    keyword_match_tiers,
    keywords_from_query,
    rank_names,
    segmentation_disease_label,
)
from agentic_radiogen.data.gdc_client import GdcCase, GdcClient
from agentic_radiogen.data.organ_gate_blacklist import (
    default_blacklist_path,
    filter_uids,
    load_failed_uids,
    record_failed_uid,
    remove_cached_series,
)
from agentic_radiogen.data.tcia_client import (
    TciaClient,
    TciaSeries,
    prefer_volumetric_series_ranked,
)
from agentic_radiogen.imaging.organ_verify import verify_expected_organ
from agentic_radiogen.imaging.radiomics_extract import extract_series_features
from agentic_radiogen.util.progress import log, progress_enabled, progress_iter

_MAX_SERIES_ATTEMPTS = 3

_SEGM_LOG_FIELDS = (
    "patient_id",
    "image_path",
    "diagnosis",
    "segmentation_model",
    "models_attempted",
    "mask_path",
    "n_mask_voxels",
    "status",
)

# Lung / chest cohorts: drop abd/pelvis/WB series before thickest-non-scout ranking.
_LUNG_SITE_RE = (
    "lung",
    "nsclc",
    "luad",
    "lusc",
    "pulmonary",
    "chest",
    "thorax",
    "bronchus",
)


def _wants_chest_ct(*texts: str) -> bool:
    blob = " ".join(t for t in texts if t).lower()
    return any(tok in blob for tok in _LUNG_SITE_RE)


@dataclass
class LiveCatalog:
    """Paired-only catalog with question-scoped DICOM + radiomics on fetch."""

    gdc: GdcClient = field(default_factory=GdcClient)
    tcia: TciaClient = field(default_factory=TciaClient)
    download_dicom: bool = True
    extract_radiomics: bool = True
    use_llm: bool | None = None
    show_progress: bool | None = None
    store_segm: bool = False
    segm_dir: Path = field(default_factory=lambda: Path.cwd() / "segm_rslt")
    force_seg_model: str | None = None
    query_count: int = 0
    fetch_count: int = 0
    last_source_counts: dict[str, int] = field(default_factory=dict)
    last_imaging_notes: list[str] = field(default_factory=list)
    last_segmentation_log: list[dict[str, str]] = field(default_factory=list)
    last_segmentation_log_path: str | None = None
    last_intermediate: dict = field(default_factory=dict)
    last_question_text: str = ""
    _record_cache: dict[str, CatalogRecord] = field(default_factory=dict)
    _series_cache: dict[tuple[str, str], list[TciaSeries]] = field(default_factory=dict)

    def apply_request_options(
        self,
        *,
        max_genes: int | None = None,
        min_altered: int = 4,
        disease_query: str | None = None,
        keyword_match: bool = False,
        preferred_tcia_collections: list[str] | None = None,
        preferred_gdc_projects: list[str] | None = None,
        preferred_keywords: list[str] | None = None,
        matcher_backend: str | None = None,
    ) -> None:
        self.last_intermediate["max_genes"] = max_genes
        self.last_intermediate["min_altered"] = int(min_altered)
        if disease_query:
            self.last_intermediate["disease_query"] = disease_query
            self.last_question_text = str(disease_query)
        self.last_intermediate["keyword_match"] = bool(keyword_match)
        if preferred_tcia_collections is not None:
            self.last_intermediate["preferred_tcia_collections"] = list(
                preferred_tcia_collections
            )
        if preferred_gdc_projects is not None:
            self.last_intermediate["preferred_gdc_projects"] = list(preferred_gdc_projects)
        if preferred_keywords is not None:
            self.last_intermediate["preferred_keywords"] = list(preferred_keywords)
        if matcher_backend:
            self.last_intermediate["matcher_backend"] = matcher_backend

    def _progress_on(self) -> bool:
        if self.show_progress is None:
            return progress_enabled()
        return bool(self.show_progress)

    def query_metadata(
        self,
        *,
        disease: str,
        modality: str,
        genes: list[str],
        require_endpoint: str | None = None,
        disease_query: str | None = None,
        keyword_match: bool = False,
        tcia_collection: str | None = None,
        tcga_project: str | None = None,
    ) -> list[CatalogRecord]:
        self.query_count += 1
        verbose = self._progress_on()
        query = (
            disease_query
            or self.last_intermediate.get("disease_query")
            or disease.replace("_", " ")
        )
        forced_tcia = (tcia_collection or "").strip()
        forced_gdc = (tcga_project or "").strip()
        use_keywords = bool(
            keyword_match
            or self.last_intermediate.get("keyword_match")
            or not forced_tcia
            or forced_tcia.upper() == "KEYWORD"
            or not forced_gdc
            or forced_gdc.upper() == "KEYWORD"
        )
        _ = require_endpoint
        if use_keywords:
            return self._query_by_keywords(
                disease=disease,
                modality=modality,
                genes=genes,
                disease_query=str(query),
                verbose=verbose,
            )
        return self._query_fixed_profile(
            profile_name=disease,
            tcia_collection=forced_tcia,
            gdc_project=forced_gdc,
            modality=modality,
            genes=genes,
            verbose=verbose,
        )

    def _query_by_keywords(
        self,
        *,
        disease: str,
        modality: str,
        genes: list[str],
        disease_query: str,
        verbose: bool,
    ) -> list[CatalogRecord]:
        preferred_tcia = list(self.last_intermediate.get("preferred_tcia_collections") or [])
        preferred_gdc = list(self.last_intermediate.get("preferred_gdc_projects") or [])
        preferred_kw = list(self.last_intermediate.get("preferred_keywords") or [])
        if preferred_tcia or preferred_gdc:
            log(
                "[match] Using LLM-selected sources: "
                f"TCIA={preferred_tcia or '(none)'} GDC={preferred_gdc or '(none)'}",
                enabled=verbose,
            )
            keywords = preferred_kw or keywords_from_query(disease_query)
            bundle = self._resolve_selected_sources(
                tcia_collections=preferred_tcia,
                gdc_projects=preferred_gdc,
                keywords=keywords,
                modality=modality,
                verbose=verbose,
                match_tier="llm",
            )
            return self._build_paired_records(
                disease=disease,
                modality=modality,
                genes=genes,
                tcia_patients=bundle["tcia_patients"],
                tcia_ranked=bundle.get("tcia_ranked") or {},
                gdc_cases=bundle["gdc_cases"],
                tcia_collections=bundle["tcia_collections"],
                gdc_projects=bundle["gdc_projects"],
                disease_query=disease_query,
                keywords=keywords,
                matched_diagnoses=bundle["matched_diagnoses"],
                match_tier="llm",
                broadened_from=None,
                tier_query=disease_query,
                verbose=verbose,
            )

        tiers = keyword_match_tiers(disease_query)
        chosen: KeywordTier | None = None
        bundle: dict | None = None
        for tier in tiers:
            log(
                f"[match] Trying {tier.label} keywords for {tier.query!r} "
                f"({', '.join(tier.keywords[:8])}{'...' if len(tier.keywords) > 8 else ''})",
                enabled=verbose,
            )
            candidate = self._resolve_keyword_sources(
                tier=tier,
                modality=modality,
                verbose=verbose,
            )
            n_paired = len(
                set(candidate["tcia_patients"]) & set(candidate["gdc_cases"])
            )
            if n_paired > 0:
                chosen = tier
                bundle = candidate
                break
            if tier.label == "specific" and len(tiers) > 1:
                log(
                    f"[match] No paired patients for {tier.label} keywords (i&g=0); "
                    "broadening to parent disease name",
                    enabled=verbose,
                )
            bundle = candidate
            chosen = tier

        assert bundle is not None and chosen is not None
        return self._build_paired_records(
            disease=disease,
            modality=modality,
            genes=genes,
            tcia_patients=bundle["tcia_patients"],
            tcia_ranked=bundle.get("tcia_ranked") or {},
            gdc_cases=bundle["gdc_cases"],
            tcia_collections=bundle["tcia_collections"],
            gdc_projects=bundle["gdc_projects"],
            disease_query=disease_query,
            keywords=chosen.keywords,
            matched_diagnoses=bundle["matched_diagnoses"],
            match_tier=chosen.label,
            broadened_from=chosen.broadened_from,
            tier_query=chosen.query,
            verbose=verbose,
        )

    def _resolve_keyword_sources(
        self,
        *,
        tier: KeywordTier,
        modality: str,
        verbose: bool,
    ) -> dict:
        keywords = tier.keywords
        collections = self.tcia.list_collections()
        min_score = 12 if tier.label == "broadened" else 15
        ranked_collections = rank_names(collections, keywords, min_score=min_score)
        pairable_collections: list[str] = []
        other_collections: list[str] = []
        for match in ranked_collections[:12]:
            series = self._series(match.name, modality)
            n_tcga = sum(1 for s in series if s.patient_id.startswith("TCGA-"))
            log(
                f"[match] TCIA collection candidate {match.name}: "
                f"score={match.score} ({match.reason}), "
                f"series={len(series)}, tcga_ids={n_tcga}",
                enabled=verbose,
            )
            if n_tcga > 0:
                pairable_collections.append(match.name)
            else:
                other_collections.append(match.name)

        projects_meta = self.gdc.list_projects()
        project_labels: list[tuple[str, str]] = []
        for proj in projects_meta:
            pid = str(proj.get("project_id") or "")
            if not pid:
                continue
            label = " | ".join(
                [
                    pid,
                    str(proj.get("name") or ""),
                    " ".join(proj.get("primary_site") or []),
                    " ".join(proj.get("disease_type") or []),
                ]
            )
            project_labels.append((pid, label))
        ranked_projects = rank_names(
            [label for _, label in project_labels], keywords, min_score=min_score
        )
        label_to_pid = {label: pid for pid, label in project_labels}
        all_project_ids = {pid for pid, _ in project_labels}
        gdc_projects: list[str] = []
        for match in ranked_projects[:12]:
            pid = label_to_pid.get(match.name)
            if pid and pid not in gdc_projects:
                gdc_projects.append(pid)
                log(
                    f"[match] GDC project candidate {pid}: score={match.score} ({match.reason})",
                    enabled=verbose,
                )

        tcia_collections = list(pairable_collections)
        if not tcia_collections:
            for pid in gdc_projects:
                if pid in collections and pid not in tcia_collections:
                    tcia_collections.append(pid)
            if other_collections:
                log(
                    "[match] Keyword-matched TCIA collections lack TCGA barcodes for GDC "
                    f"pairing ({', '.join(other_collections[:4])}{'...' if len(other_collections) > 4 else ''}); "
                    f"using TCGA collections for matched projects: "
                    f"{', '.join(tcia_collections) or '(none)'}",
                    enabled=verbose,
                )

        return self._resolve_selected_sources(
            tcia_collections=tcia_collections,
            gdc_projects=gdc_projects,
            keywords=keywords,
            modality=modality,
            verbose=verbose,
            match_tier=tier.label,
            all_project_ids=all_project_ids,
            collections=collections,
        )

    def _resolve_selected_sources(
        self,
        *,
        tcia_collections: list[str],
        gdc_projects: list[str],
        keywords: list[str],
        modality: str,
        verbose: bool,
        match_tier: str,
        all_project_ids: set[str] | None = None,
        collections: list[str] | None = None,
    ) -> dict:
        """Load imaging + genomics patients for selected sources and prepare intersection inputs."""
        if collections is None:
            collections = self.tcia.list_collections()
        collection_set = set(collections)
        if all_project_ids is None:
            projects_meta = self.gdc.list_projects()
            all_project_ids = {
                str(p.get("project_id") or "")
                for p in projects_meta
                if p.get("project_id")
            }

        tcia_collections = [c for c in tcia_collections if c in collection_set]
        gdc_projects = [p for p in gdc_projects if p in all_project_ids]

        pairable_gdc = [c for c in tcia_collections if c in all_project_ids]
        for pid in reversed(pairable_gdc):
            if pid in gdc_projects:
                gdc_projects.remove(pid)
            gdc_projects.insert(0, pid)
            log(
                f"[match] Prioritizing pairable GDC project {pid} (matches TCIA collection)",
                enabled=verbose,
            )
        for pid in pairable_gdc:
            if pid not in gdc_projects:
                gdc_projects.insert(0, pid)
            if pid in collection_set and pid not in tcia_collections:
                tcia_collections.insert(0, pid)

        if not tcia_collections or not gdc_projects:
            log(
                "[match] WARNING: selected sources incomplete "
                f"(tcia={tcia_collections}, gdc={gdc_projects})",
                enabled=verbose,
            )

        chest_only = _wants_chest_ct(
            *tcia_collections, *gdc_projects, *keywords
        )
        tcia_ranked = self._ranked_series_per_patient(
            tcia_collections,
            modality,
            verbose=verbose,
            chest_only=chest_only,
        )
        tcia_patients = {pid: ranks[0] for pid, ranks in tcia_ranked.items() if ranks}
        log(
            f"[match] (1) Imaging patients with {modality} in selected TCIA "
            f"[{', '.join(tcia_collections) or 'none'}]: {len(tcia_patients)}",
            enabled=verbose,
        )

        primary_projects = [p for p in gdc_projects if p in pairable_gdc] or list(gdc_projects)
        extra_projects = [p for p in gdc_projects if p not in primary_projects]
        gdc_cases = {
            case.submitter_id: case
            for case in self.gdc.list_cases_multi(primary_projects, submitter_ids=None)
        }
        if extra_projects and tcia_patients:
            for case in self.gdc.list_cases_multi(
                extra_projects,
                submitter_ids=list(tcia_patients.keys()),
            ):
                gdc_cases.setdefault(case.submitter_id, case)
        gdc_projects = primary_projects + [p for p in extra_projects if p not in primary_projects]
        diagnosis_facet = self.gdc.primary_diagnosis_counts(gdc_projects)
        matched_dx = sorted(dx for dx in diagnosis_facet if diagnosis_matches(dx, keywords))
        if matched_dx:
            log(
                f"[match] GDC primary_diagnosis values matching keywords "
                f"({len(matched_dx)}): {', '.join(matched_dx[:8])}"
                f"{'...' if len(matched_dx) > 8 else ''}",
                enabled=verbose,
            )
        apply_dx_filter = (
            match_tier == "specific"
            and bool(matched_dx)
            and not pairable_gdc
        )
        if pairable_gdc and matched_dx:
            log(
                "[match] Pairable GDC project(s) "
                f"{', '.join(pairable_gdc)}: keeping full project cohort "
                "(diagnosis keywords are informational only)",
                enabled=verbose,
            )
        elif apply_dx_filter:
            matched_set = {d.lower() for d in matched_dx}
            before = dict(gdc_cases)
            filtered = {
                pid: case
                for pid, case in gdc_cases.items()
                if str(case.clinical.get("subtype") or "").lower() in matched_set
            }
            if filtered:
                if (
                    tcia_patients
                    and not (set(filtered) & set(tcia_patients))
                    and (set(before) & set(tcia_patients))
                ):
                    log(
                        "[match] Diagnosis filter would drop all paired IDs "
                        f"({len(set(before) & set(tcia_patients))} available); "
                        "keeping unfiltered GDC cohort",
                        enabled=verbose,
                    )
                else:
                    gdc_cases = filtered
            else:
                log(
                    "[match] Diagnosis facet matched keywords but no case subtype "
                    "strings aligned; keeping project-level GDC cohort",
                    enabled=verbose,
                )
        elif match_tier == "broadened":
            log(
                "[match] Broadened disease tier: using project-level GDC cohort "
                "(no strict primary_diagnosis filter)",
                enabled=verbose,
            )
        elif match_tier == "llm":
            log(
                "[match] LLM-selected sources: using project-level GDC cohort "
                "(no strict primary_diagnosis filter)",
                enabled=verbose,
            )
        else:
            log(
                "[match] No GDC primary_diagnosis string matched keywords; "
                "keeping selected projects without diagnosis filter",
                enabled=verbose,
            )

        log(
            f"[match] (2) Genomics patients in selected GDC "
            f"[{', '.join(gdc_projects) or 'none'}]: {len(gdc_cases)}",
            enabled=verbose,
        )
        return {
            "tcia_patients": tcia_patients,
            "tcia_ranked": tcia_ranked,
            "gdc_cases": gdc_cases,
            "tcia_collections": tcia_collections,
            "gdc_projects": gdc_projects,
            "matched_diagnoses": matched_dx,
        }

    def _query_fixed_profile(
        self,
        *,
        profile_name: str,
        tcia_collection: str,
        gdc_project: str,
        modality: str,
        genes: list[str],
        verbose: bool,
    ) -> list[CatalogRecord]:
        log(
            f"[match] Querying TCIA collection={tcia_collection} modality={modality}",
            enabled=verbose,
        )
        chest_only = _wants_chest_ct(profile_name, tcia_collection, gdc_project)
        tcia_ranked = self._ranked_series_per_patient(
            [tcia_collection],
            modality,
            verbose=verbose,
            chest_only=chest_only,
        )
        tcia_patients = {pid: ranks[0] for pid, ranks in tcia_ranked.items() if ranks}
        log(
            f"[match] (1) Imaging patients with {modality} in {tcia_collection}: "
            f"{len(tcia_patients)}",
            enabled=verbose,
        )
        gdc_cases = {
            case.submitter_id: case
            for case in self.gdc.list_cases(gdc_project, submitter_ids=None)
        }
        log(
            f"[match] (2) Genomics patients in GDC project {gdc_project}: "
            f"{len(gdc_cases)}",
            enabled=verbose,
        )
        return self._build_paired_records(
            disease=profile_name,
            modality=modality,
            genes=genes,
            tcia_patients=tcia_patients,
            tcia_ranked=tcia_ranked,
            gdc_cases=gdc_cases,
            tcia_collections=[tcia_collection],
            gdc_projects=[gdc_project],
            disease_query=profile_name,
            keywords=keywords_from_query(profile_name.replace("_", " ")),
            matched_diagnoses=[],
            match_tier="profile",
            broadened_from=None,
            tier_query=profile_name,
            verbose=verbose,
        )

    def _build_paired_records(
        self,
        *,
        disease: str,
        modality: str,
        genes: list[str],
        tcia_patients: dict[str, TciaSeries],
        gdc_cases: dict[str, GdcCase],
        tcia_collections: list[str],
        gdc_projects: list[str],
        disease_query: str,
        keywords: list[str],
        matched_diagnoses: list[str],
        match_tier: str,
        broadened_from: str | None,
        tier_query: str,
        verbose: bool,
        tcia_ranked: dict[str, list[TciaSeries]] | None = None,
    ) -> list[CatalogRecord]:
        paired = sorted(set(tcia_patients) & set(gdc_cases))
        image_counts = [int(tcia_patients[pid].image_count or 0) for pid in paired]
        n_ge8 = sum(1 for n in image_counts if n >= 8)
        n_thin = sum(1 for n in image_counts if 0 < n < 8)
        n_unk = sum(1 for n in image_counts if n <= 0)
        self.last_source_counts = {
            "tcia_series": sum(len(self._series(c, modality)) for c in tcia_collections),
            "tcia_patients": len(tcia_patients),
            "gdc_patients": len(gdc_cases),
            "gdc_overlapping": len(paired),
            "tcia_only_dropped": len(set(tcia_patients) - set(gdc_cases)),
            "gdc_only_dropped": len(set(gdc_cases) - set(tcia_patients)),
            "paired_available": len(paired),
            "paired_kept": len(paired),
            "paired_series_imagecount_ge8": n_ge8,
            "paired_series_imagecount_lt8": n_thin,
            "paired_series_imagecount_unknown": n_unk,
            "i_and_g": len(paired),
        }
        self.last_intermediate = {
            **self.last_intermediate,
            "disease": disease,
            "disease_query": disease_query,
            "disease_keywords": list(keywords),
            "dataset_keywords_used": list(keywords),
            "dataset_match_tier": match_tier,
            "dataset_broadened_from": broadened_from,
            "dataset_tier_query": tier_query,
            "keyword_match": match_tier in {"specific", "broadened"},
            "matched_diagnoses": list(matched_diagnoses),
            "tcia_collection": ",".join(tcia_collections),
            "gdc_project": ",".join(gdc_projects),
            "tcia_collections": list(tcia_collections),
            "gdc_projects": list(gdc_projects),
            "modality": modality,
            "tcia_series": self.last_source_counts["tcia_series"],
            "tcia_patients": len(tcia_patients),
            "gdc_patients": len(gdc_cases),
            "paired_available": len(paired),
            "i_and_g": len(paired),
            "tcia_only_dropped": self.last_source_counts["tcia_only_dropped"],
            "gdc_only_dropped": self.last_source_counts["gdc_only_dropped"],
            "genes": list(genes),
        }
        log(
            f"[match] (3) i&g = imaging ∩ genomics = {len(paired)} paired patients "
            f"(modality={modality}; tcia-only={self.last_source_counts['tcia_only_dropped']}, "
            f"gdc-only={self.last_source_counts['gdc_only_dropped']})",
            enabled=verbose,
        )
        series_policy = (
            "thickest non-scout chest series"
            if _wants_chest_ct(
                disease,
                disease_query,
                *tcia_collections,
                *gdc_projects,
                *keywords,
            )
            else "thickest non-scout series"
        )
        log(
            f"[match] Assigned volumetric {modality} series per patient: "
            f"ImageCount>=8: {n_ge8}, 1-7: {n_thin}, unknown: {n_unk} "
            f"(prefer {series_policy}, not first TCIA row)",
            enabled=verbose,
        )
        self._record_cache.clear()
        fallback_project = gdc_projects[0] if gdc_projects else ""
        records = []
        for pid in paired:
            case_clinical = dict(gdc_cases[pid].clinical)
            project = str(
                case_clinical.get("_gdc_project")
                or fallback_project
            )
            series = tcia_patients[pid]
            ranked = list((tcia_ranked or {}).get(pid) or [series])
            if not ranked:
                ranked = [series]
            collection = str(series.collection or "")
            case_clinical["_gdc_paired"] = True
            case_clinical["_gdc_project"] = project
            case_clinical["_tcia_collection"] = collection
            case_clinical["_tcia_series_uid"] = series.series_uid
            case_clinical["_tcia_image_count"] = int(series.image_count or 0)
            case_clinical["_tcia_series_description"] = series.series_description or ""
            case_clinical["_tcia_series_candidates"] = [
                s.series_uid for s in ranked if s.series_uid
            ]
            records.append(
                CatalogRecord(
                    patient_id=pid,
                    disease=disease,
                    modality=modality,
                    mutations={gene: 0 for gene in genes},
                    expression={},
                    clinical=case_clinical,
                    series_uid=series.series_uid,
                    radiomic_features={},
                    volume_summary={},
                )
            )
        for rec in records:
            self._record_cache[rec.patient_id] = rec
        return records

    def fetch_records(self, patient_ids: list[str]) -> list[CatalogRecord]:
        self.fetch_count += 1
        self.last_imaging_notes = []
        self.last_segmentation_log = []
        self.last_segmentation_log_path = None
        verbose = self._progress_on()
        if not patient_ids:
            return []
        if self._record_cache.get(patient_ids[0]) is None:
            raise KeyError("query_metadata must run before fetch_records")
        genes = list(next(iter(self._record_cache.values())).mutations)
        discover = not genes
        if discover:
            raw_max = self.last_intermediate.get("max_genes")
            max_genes = None if raw_max in (None, "", "none") else int(raw_max)
            min_altered = int(self.last_intermediate.get("min_altered") or 4)
            log(
                "[fetch] Discovering ALL mutated genes in the paired genomics cohort"
                + (f" (cap={max_genes})" if max_genes is not None else ""),
                enabled=verbose,
            )
            genes = self.gdc.discover_genes(
                patient_ids, min_altered=min_altered, max_genes=max_genes
            )
            if not genes:
                log(
                    "[fetch] WARNING: no mutated genes returned for this cohort; "
                    "keeping patients paired via GDC case membership",
                    enabled=verbose,
                )
            log(
                f"[fetch] (4) gen = all mutated genes in i&g ({len(genes)} genes): "
                f"{', '.join(genes[:12])}{'...' if len(genes) > 12 else ''}",
                enabled=verbose,
            )
            self.last_intermediate["genes"] = list(genes)
            self.last_intermediate["discover_genes"] = True
            self.last_intermediate["gene_source"] = "cohort"
        else:
            src = str(self.last_intermediate.get("gene_source") or "explicit")
            if src not in {"literature", "explicit"}:
                src = "explicit"
            log(
                f"[fetch] Gene panel ({src}) for {len(patient_ids)} patients "
                f"({len(genes)} genes: {', '.join(genes[:8])}"
                f"{'...' if len(genes) > 8 else ''})",
                enabled=verbose,
            )
            self.last_intermediate["discover_genes"] = False
            self.last_intermediate["gene_source"] = src
            self.last_intermediate["genes"] = list(genes)
        flags = self.gdc.mutation_flags(patient_ids, genes) if genes else {
            pid: {} for pid in patient_ids
        }
        expression = self.gdc.expression_means(patient_ids, genes) if genes else {
            pid: {} for pid in patient_ids
        }
        log("[fetch] Genomics metadata ready", enabled=verbose)

        ok = 0
        failed = 0
        skipped = 0
        fetched: list[CatalogRecord] = []
        iterator = progress_iter(
            patient_ids,
            total=len(patient_ids),
            desc="DICOM+radiomics",
            unit="patient",
            enabled=verbose and self.download_dicom,
        )
        for pid in iterator:
            rec = self._record_cache.get(pid)
            if rec is None:
                raise KeyError(f"Unknown patient IDs: [{pid}]")
            local_path = rec.local_path
            radiomic_features = dict(rec.radiomic_features)
            volume_summary = dict(rec.volume_summary)
            chosen_series_uid = rec.series_uid
            if self.download_dicom and rec.series_uid:
                try:
                    q_disease = str(
                        self.last_intermediate.get("disease_query")
                        or self.last_question_text
                        or rec.disease
                        or ""
                    ).strip()
                    seg_disease = segmentation_disease_label(
                        question_disease=q_disease or rec.disease,
                        gdc_project=str(rec.clinical.get("_gdc_project") or ""),
                        tcia_collection=str(
                            rec.clinical.get("_tcia_collection") or ""
                        ),
                        primary_diagnosis=str(
                            rec.clinical.get("primary_diagnosis")
                            or rec.clinical.get("subtype")
                            or ""
                        ),
                    )
                    diagnosis = str(
                        rec.clinical.get("primary_diagnosis")
                        or rec.clinical.get("subtype")
                        or seg_disease
                        or rec.disease
                        or ""
                    )
                    organs = expected_organ_keywords(
                        q_disease,
                        seg_disease,
                        rec.disease,
                        str(rec.clinical.get("_gdc_project") or ""),
                        str(rec.clinical.get("_tcia_collection") or ""),
                    )
                    candidate_uids = [
                        str(u)
                        for u in (
                            rec.clinical.get("_tcia_series_candidates")
                            or [rec.series_uid]
                        )
                        if str(u).strip()
                    ]
                    if rec.series_uid and rec.series_uid not in candidate_uids:
                        candidate_uids.insert(0, rec.series_uid)
                    blacklist_path = self._organ_gate_blacklist_path()
                    failed_uids = load_failed_uids(blacklist_path)
                    if failed_uids:
                        before = list(candidate_uids)
                        candidate_uids = filter_uids(
                            candidate_uids, failed=failed_uids
                        )
                        skipped = [u for u in before if u not in candidate_uids]
                        if skipped:
                            log(
                                f"[fetch] {pid}: skipping {len(skipped)} blacklisted "
                                f"organ-gate UID(s) "
                                f"({', '.join('…' + u[-12:] for u in skipped)})",
                                enabled=verbose,
                            )
                    candidate_uids = candidate_uids[:_MAX_SERIES_ATTEMPTS]
                    if not candidate_uids:
                        raise RuntimeError(
                            f"No series candidates left for {pid} after organ-gate "
                            f"blacklist filter (see {blacklist_path})"
                        )

                    series_dir: Path | None = None
                    last_organ_reason = ""
                    for attempt, uid in enumerate(candidate_uids, start=1):
                        series_dir = self.tcia.download_series(uid)
                        local_path = str(series_dir)
                        chosen_series_uid = uid
                        if organs and self.extract_radiomics:
                            gate = verify_expected_organ(
                                series_dir, organ_keywords=organs
                            )
                            if not gate.ok:
                                last_organ_reason = gate.reason
                                record_failed_uid(
                                    uid,
                                    reason=gate.reason,
                                    patient_id=pid,
                                    organ_keywords=organs,
                                    path=blacklist_path,
                                )
                                removed = remove_cached_series(
                                    uid,
                                    cache_dir=self.tcia.cache_dir,
                                    series_dir=series_dir,
                                )
                                removed_note = (
                                    f"; removed cache {removed.name}"
                                    if removed is not None
                                    else "; cache already absent"
                                )
                                log(
                                    f"[fetch] {pid}: organ gate FAIL attempt "
                                    f"{attempt}/{len(candidate_uids)} "
                                    f"uid=…{uid[-12:]} ({gate.reason}); "
                                    f"blacklisted → {blacklist_path.name}"
                                    f"{removed_note}",
                                    enabled=verbose,
                                )
                                series_dir = None
                                continue
                            log(
                                f"[fetch] {pid}: organ gate PASS "
                                f"({gate.reason}) uid=…{uid[-12:]}",
                                enabled=verbose,
                            )
                        break
                    else:
                        raise RuntimeError(
                            f"No series passed organ gate for {pid} "
                            f"(expected={list(organs) or 'any'}; "
                            f"last={last_organ_reason or 'n/a'})"
                        )

                    assert series_dir is not None
                    if self.extract_radiomics:
                        force_model = (
                            self.force_seg_model
                            or str(self.last_intermediate.get("force_seg_model") or "")
                            or None
                        )
                        extracted = extract_series_features(
                            series_dir,
                            disease=seg_disease,
                            modality=rec.modality or "CT",
                            question=q_disease
                            or self.last_question_text
                            or seg_disease,
                            use_llm=self.use_llm,
                            store_segm=self.store_segm,
                            segm_dir=self.segm_dir,
                            patient_id=pid,
                            force_model_id=force_model,
                        )
                        radiomic_features = {
                            k: float(v)
                            for k, v in extracted["features"].items()
                            if not str(k).startswith("meta_")
                        }
                        volume_summary = {
                            k: float(v)
                            for k, v in extracted["volume_summary"].items()
                            if isinstance(v, (int, float))
                        }
                        cache_note = "cached" if extracted.get("from_cache") else "computed"
                        seg = extracted.get("segmentation_backend") or "?"
                        model_id = (
                            extracted.get("segmentation_model_id")
                            or extracted.get("segmentation_backend")
                            or "?"
                        )
                        status = str(extracted.get("segm_status") or "success")
                        n_vox = extracted.get("n_mask_voxels", "")
                        self.last_segmentation_log.append(
                            {
                                "patient_id": pid,
                                "image_path": str(Path(series_dir).resolve()),
                                "diagnosis": diagnosis,
                                "segmentation_model": str(model_id),
                                "models_attempted": str(
                                    extracted.get("models_attempted") or model_id or ""
                                ),
                                "mask_path": str(extracted.get("mask_path") or ""),
                                "n_mask_voxels": str(n_vox),
                                "status": status,
                            }
                        )
                        self.last_imaging_notes.append(
                            f"{pid}: radiomics ok ({cache_note}, seg={seg}, "
                            f"model={model_id}, disease={seg_disease}, "
                            f"{Path(series_dir).name})"
                        )
                        ok += 1
                    else:
                        self.last_imaging_notes.append(
                            f"{pid}: dicom downloaded, radiomics skipped"
                        )
                        skipped += 1
                except Exception as exc:
                    diagnosis = str(
                        rec.clinical.get("primary_diagnosis")
                        or rec.clinical.get("subtype")
                        or rec.disease
                        or ""
                    )
                    img_path = local_path or ""
                    if rec.series_uid and not img_path:
                        img_path = f"series_uid={rec.series_uid}"
                    self.last_segmentation_log.append(
                        {
                            "patient_id": pid,
                            "image_path": str(img_path),
                            "diagnosis": diagnosis,
                            "segmentation_model": "",
                            "models_attempted": "",
                            "mask_path": "",
                            "n_mask_voxels": "0",
                            "status": f"failure: {exc}",
                        }
                    )
                    self.last_imaging_notes.append(f"{pid}: imaging failed: {exc}")
                    failed += 1
            else:
                skipped += 1
            updated = CatalogRecord(
                patient_id=rec.patient_id,
                disease=rec.disease,
                modality=rec.modality,
                mutations=flags.get(pid, dict(rec.mutations)),
                expression=expression.get(pid, dict(rec.expression)),
                clinical={
                    **{
                        k: v
                        for k, v in dict(rec.clinical).items()
                        if not str(k).startswith("_")
                    },
                    "_gdc_paired": True,
                    "_gdc_project": rec.clinical.get("_gdc_project"),
                    "_tcia_collection": rec.clinical.get("_tcia_collection"),
                    "_tcia_series_uid": chosen_series_uid,
                    "_tcia_series_candidates": rec.clinical.get(
                        "_tcia_series_candidates"
                    ),
                },
                series_uid=chosen_series_uid,
                radiomic_features=radiomic_features,
                volume_summary=volume_summary,
                local_path=local_path,
            )
            self._record_cache[pid] = updated
            fetched.append(updated)

        segm_log_path = self._write_segmentation_log()
        self.last_intermediate = {
            **self.last_intermediate,
            "fetch_requested": len(patient_ids),
            "radiomics_ok": ok,
            "radiomics_failed": failed,
            "imaging_skipped": skipped,
            "segmentation_log_path": segm_log_path,
            "segmentation_log_n": len(self.last_segmentation_log),
        }
        log(
            f"[fetch] Done: radiomics_ok={ok}, failed={failed}, skipped={skipped} "
            f"(of {len(patient_ids)} patients)",
            enabled=verbose,
        )
        if segm_log_path:
            log(
                f"[fetch] Segmentation log ({len(self.last_segmentation_log)} rows) → "
                f"{segm_log_path}",
                enabled=verbose,
            )
        return fetched

    def _write_segmentation_log(self) -> str | None:
        """Write one-row-per-image segmentation CSV under ``segm_rslt``."""
        rows = list(self.last_segmentation_log)
        if not rows:
            return None
        out_dir = Path(self.segm_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "segmentation_log.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(_SEGM_LOG_FIELDS))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in _SEGM_LOG_FIELDS})
        self.last_segmentation_log_path = str(path.resolve())
        return self.last_segmentation_log_path

    def _ranked_series_per_patient(
        self,
        collections: list[str],
        modality: str,
        *,
        verbose: bool = False,
        chest_only: bool = False,
    ) -> dict[str, list[TciaSeries]]:
        """Ranked CT series per patient (best first); not the first NBIA row.

        When ``chest_only`` is True (lung cohorts), non-chest series (abd/pelvis,
        whole-body, renal, …) are dropped first; empty site_text is demoted
        below labeled CHEST series. Patients with no usable candidates are omitted.
        """
        by_patient: dict[str, list[TciaSeries]] = {}
        for coll in collections:
            for item in self._series(coll, modality):
                by_patient.setdefault(item.patient_id, []).append(item)
        chosen: dict[str, list[TciaSeries]] = {}
        thin = 0
        dropped_non_chest = 0
        failed_uids = load_failed_uids(self._organ_gate_blacklist_path())
        blacklisted_drop = 0
        for pid, cands in by_patient.items():
            if failed_uids:
                kept = [s for s in cands if s.series_uid not in failed_uids]
                blacklisted_drop += len(cands) - len(kept)
                cands = kept
            ranked = prefer_volumetric_series_ranked(cands, chest_only=chest_only)
            if not ranked:
                if chest_only and cands:
                    dropped_non_chest += 1
                continue
            chosen[pid] = ranked
            if 0 < int(ranked[0].image_count) < 8:
                thin += 1
        if blacklisted_drop and verbose:
            log(
                f"[match] Organ-gate blacklist: excluded {blacklisted_drop} series "
                f"UID(s) from ranking ({self._organ_gate_blacklist_path().name})",
                enabled=True,
            )
        if chest_only and verbose:
            log(
                f"[match] Chest-only series filter: kept {len(chosen)} patients, "
                f"dropped {dropped_non_chest} with no usable chest CT "
                f"(non-chest / scout / ImageCount<8); empty-desc demoted vs CHEST text",
                enabled=True,
            )
        if thin and verbose:
            log(
                f"[match] WARNING: {thin} patients still have ImageCount<8 after "
                f"preferring volumetric series (nnU-Net may fall back per case)",
                enabled=True,
            )
        return chosen

    def _series(self, collection: str, modality: str) -> list[TciaSeries]:
        key = (collection, modality)
        if key not in self._series_cache:
            self._series_cache[key] = self.tcia.list_series(collection, modality)
        return self._series_cache[key]

    def _organ_gate_blacklist_path(self) -> Path:
        """JSONL of organ-gate failures, stored next to the TCIA DICOM cache."""
        return default_blacklist_path(self.tcia.cache_dir)
