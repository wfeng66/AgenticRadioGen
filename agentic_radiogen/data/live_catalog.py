from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agentic_radiogen.data.catalog import CatalogRecord
from agentic_radiogen.data.disease_match import (
    KeywordTier,
    diagnosis_matches,
    keyword_match_tiers,
    keywords_from_query,
    rank_names,
)
from agentic_radiogen.data.gdc_client import GdcCase, GdcClient
from agentic_radiogen.data.tcia_client import TciaClient, TciaSeries
from agentic_radiogen.imaging.radiomics_extract import extract_series_features
from agentic_radiogen.util.progress import log, progress_enabled, progress_iter


@dataclass
class LiveCatalog:
    """Paired-only catalog with question-scoped DICOM + radiomics on fetch."""

    gdc: GdcClient = field(default_factory=GdcClient)
    tcia: TciaClient = field(default_factory=TciaClient)
    download_dicom: bool = True
    extract_radiomics: bool = True
    show_progress: bool | None = None
    query_count: int = 0
    fetch_count: int = 0
    last_source_counts: dict[str, int] = field(default_factory=dict)
    last_imaging_notes: list[str] = field(default_factory=list)
    last_intermediate: dict = field(default_factory=dict)
    _record_cache: dict[str, CatalogRecord] = field(default_factory=dict)
    _series_cache: dict[tuple[str, str], list[TciaSeries]] = field(default_factory=dict)

    def apply_request_options(
        self,
        *,
        max_genes: int | None = None,
        min_altered: int = 1,
        disease_query: str | None = None,
        keyword_match: bool = False,
    ) -> None:
        self.last_intermediate["max_genes"] = max_genes
        self.last_intermediate["min_altered"] = int(min_altered)
        if disease_query:
            self.last_intermediate["disease_query"] = disease_query
        self.last_intermediate["keyword_match"] = bool(keyword_match)

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

        # Prefer GDC projects that share IDs with pairable TCIA collections (e.g. TCGA-LUAD).
        # Those are the ones that can grow i&g; list them first and always include them.
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

        if not tcia_collections or not gdc_projects:
            log(
                "[match] WARNING: keyword match found incomplete sources "
                f"(tcia={tcia_collections}, gdc={gdc_projects})",
                enabled=verbose,
            )

        tcia_patients: dict[str, TciaSeries] = {}
        for coll in tcia_collections:
            for item in self._series(coll, modality):
                tcia_patients.setdefault(item.patient_id, item)
        log(
            f"[match] (1) Imaging patients with {modality} in keyword-matched TCIA "
            f"[{', '.join(tcia_collections) or 'none'}]: {len(tcia_patients)}",
            enabled=verbose,
        )

        # Fetch pairable projects in full; other keyword hits only for imaging IDs
        # (avoids huge non-pairable cohorts crowding out LUAD/LUSC cases).
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
        # Keep log / facet over the projects we actually used for pairing.
        gdc_projects = primary_projects + [p for p in extra_projects if p not in primary_projects]
        diagnosis_facet = self.gdc.primary_diagnosis_counts(gdc_projects)
        matched_dx = sorted(dx for dx in diagnosis_facet if diagnosis_matches(dx, keywords))
        apply_dx_filter = tier.label == "specific" and matched_dx
        if matched_dx:
            log(
                f"[match] GDC primary_diagnosis values matching keywords "
                f"({len(matched_dx)}): {', '.join(matched_dx[:8])}"
                f"{'...' if len(matched_dx) > 8 else ''}",
                enabled=verbose,
            )
        if apply_dx_filter:
            matched_set = {d.lower() for d in matched_dx}
            filtered = {
                pid: case
                for pid, case in gdc_cases.items()
                if str(case.clinical.get("subtype") or "").lower() in matched_set
            }
            if filtered:
                gdc_cases = filtered
            else:
                log(
                    "[match] Diagnosis facet matched keywords but no case subtype "
                    "strings aligned; keeping project-level GDC cohort",
                    enabled=verbose,
                )
        elif tier.label == "broadened":
            log(
                "[match] Broadened disease tier: using project-level GDC cohort "
                "(no strict primary_diagnosis filter)",
                enabled=verbose,
            )
        else:
            log(
                "[match] No GDC primary_diagnosis string matched keywords; "
                "keeping keyword-matched projects without diagnosis filter",
                enabled=verbose,
            )

        log(
            f"[match] (2) Genomics patients in keyword-matched GDC "
            f"[{', '.join(gdc_projects) or 'none'}]: {len(gdc_cases)}",
            enabled=verbose,
        )
        return {
            "tcia_patients": tcia_patients,
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
        series = self._series(tcia_collection, modality)
        tcia_patients: dict[str, TciaSeries] = {}
        for item in series:
            tcia_patients.setdefault(item.patient_id, item)
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
    ) -> list[CatalogRecord]:
        paired = sorted(set(tcia_patients) & set(gdc_cases))
        self.last_source_counts = {
            "tcia_series": sum(len(self._series(c, modality)) for c in tcia_collections),
            "tcia_patients": len(tcia_patients),
            "gdc_patients": len(gdc_cases),
            "gdc_overlapping": len(paired),
            "tcia_only_dropped": len(set(tcia_patients) - set(gdc_cases)),
            "gdc_only_dropped": len(set(gdc_cases) - set(tcia_patients)),
            "paired_available": len(paired),
            "paired_kept": len(paired),
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
            f"(tcia-only={self.last_source_counts['tcia_only_dropped']}, "
            f"gdc-only={self.last_source_counts['gdc_only_dropped']})",
            enabled=verbose,
        )
        self._record_cache.clear()
        primary_project = gdc_projects[0] if gdc_projects else ""
        records = [
            CatalogRecord(
                patient_id=pid,
                disease=disease,
                modality=modality,
                mutations={gene: 0 for gene in genes},
                expression={},
                clinical={
                    **dict(gdc_cases[pid].clinical),
                    "_gdc_paired": True,
                    "_gdc_project": primary_project,
                },
                series_uid=tcia_patients[pid].series_uid,
                radiomic_features={},
                volume_summary={},
            )
            for pid in paired
        ]
        for rec in records:
            self._record_cache[rec.patient_id] = rec
        return records

    def fetch_records(self, patient_ids: list[str]) -> list[CatalogRecord]:
        self.fetch_count += 1
        self.last_imaging_notes = []
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
            min_altered = int(self.last_intermediate.get("min_altered") or 1)
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
            log(
                f"[fetch] Literature gene panel for {len(patient_ids)} patients "
                f"({', '.join(genes[:6])}{'...' if len(genes) > 6 else ''})",
                enabled=verbose,
            )
            self.last_intermediate["discover_genes"] = False
            self.last_intermediate["gene_source"] = "literature"
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
            if self.download_dicom and rec.series_uid:
                try:
                    series_dir = self.tcia.download_series(rec.series_uid)
                    local_path = str(series_dir)
                    if self.extract_radiomics:
                        extracted = extract_series_features(series_dir)
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
                        self.last_imaging_notes.append(
                            f"{pid}: radiomics ok ({cache_note}, {Path(series_dir).name})"
                        )
                        ok += 1
                    else:
                        self.last_imaging_notes.append(
                            f"{pid}: dicom downloaded, radiomics skipped"
                        )
                        skipped += 1
                except Exception as exc:
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
            },
                series_uid=rec.series_uid,
                radiomic_features=radiomic_features,
                volume_summary=volume_summary,
                local_path=local_path,
            )
            self._record_cache[pid] = updated
            fetched.append(updated)

        self.last_intermediate = {
            **self.last_intermediate,
            "fetch_requested": len(patient_ids),
            "radiomics_ok": ok,
            "radiomics_failed": failed,
            "imaging_skipped": skipped,
        }
        log(
            f"[fetch] Done: radiomics_ok={ok}, failed={failed}, skipped={skipped} "
            f"(of {len(patient_ids)} patients)",
            enabled=verbose,
        )
        return fetched

    def _series(self, collection: str, modality: str) -> list[TciaSeries]:
        key = (collection, modality)
        if key not in self._series_cache:
            self._series_cache[key] = self.tcia.list_series(collection, modality)
        return self._series_cache[key]
