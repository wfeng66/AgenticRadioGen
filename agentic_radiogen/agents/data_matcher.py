from __future__ import annotations

from agentic_radiogen.data.catalog import CatalogClient, CatalogRecord, is_paired_record
from agentic_radiogen.data.disease_match import (
    keywords_from_query,
    rank_names,
    segmentation_disease_label,
)
from agentic_radiogen.data.gate import AlwaysAllowGate, DownloadGate
from agentic_radiogen.llm.client import LlmClient, resolve_llm_config
from agentic_radiogen.llm.matcher_plan import plan_match_sources_with_llm
from agentic_radiogen.schemas.contracts import (
    DataRequest,
    ImageBundle,
    ImageSeriesRef,
    MatchPreview,
    OmicsBundle,
)
from agentic_radiogen.util.progress import log


class DownloadDeniedError(RuntimeError):
    pass


def _series_seg_disease(rec: CatalogRecord, *, question_disease: str | None = None) -> str:
    """Segmentation label: question disease first, then cohort/histology."""
    clinical = rec.clinical or {}
    return segmentation_disease_label(
        question_disease=(question_disease or rec.disease or "").strip() or rec.disease,
        gdc_project=str(clinical.get("_gdc_project") or ""),
        tcia_collection=str(clinical.get("_tcia_collection") or ""),
        primary_diagnosis=str(
            clinical.get("primary_diagnosis") or clinical.get("subtype") or ""
        ),
    )


class DataMatcherAgent:
    """Metadata preview, then question-scoped fetch of images and omics.

    With an LLM key (or use_llm=True), selects TCIA collections / GDC projects /
    keywords for the disease, then intersects patient IDs across both databases.
    Falls back to rule-based keyword ranking when the LLM is unavailable.
    """

    def __init__(
        self,
        catalog: CatalogClient,
        gate: DownloadGate | None = None,
        *,
        use_llm: bool | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_client: LlmClient | None = None,
    ) -> None:
        self.catalog = catalog
        self.gate = gate or AlwaysAllowGate()
        self._cache: dict[tuple[str, str, tuple[str, ...]], tuple[ImageBundle, OmicsBundle]] = {}
        self._matcher_backend = "rules"
        if llm_client is not None:
            self._llm = llm_client
            self._use_llm = True
        else:
            config = resolve_llm_config(
                provider=llm_provider,
                model=llm_model,
                enabled=use_llm,
            )
            self._llm = LlmClient(config)
            if use_llm is False:
                self._use_llm = False
            elif use_llm is True:
                self._use_llm = True
            else:
                self._use_llm = self._llm.available

    @property
    def last_backend(self) -> str:
        return self._matcher_backend

    def preview(self, request: DataRequest) -> MatchPreview:
        if request.filters.get("full_archive"):
            raise ValueError("Full-archive downloads are not allowed")
        self._prepare_llm_sources(request)
        # Survival fields are optional when present; never require OS_time to keep a pair.
        hits = [
            record
            for record in self.catalog.query_metadata(
                disease=request.disease,
                modality=request.modality,
                genes=request.genes,
                require_endpoint=None,
                disease_query=str(request.filters.get("disease_query") or request.disease),
                keyword_match=bool(request.filters.get("keyword_match")),
                tcia_collection=request.tcia_collection,
                tcga_project=request.tcga_project,
            )
            if is_paired_record(record)
        ]
        # Full intersection first; only then apply --max-patients.
        available = len(hits)
        selected = hits[: request.max_patients]
        patient_ids = [r.patient_id for r in selected]
        coll = request.tcia_collection
        proj = request.tcga_project
        intermediate = getattr(self.catalog, "last_intermediate", None)
        if isinstance(intermediate, dict):
            coll = intermediate.get("tcia_collection") or coll
            proj = intermediate.get("gdc_project") or proj
            intermediate["matcher_backend"] = self._matcher_backend
        log(
            f"[match] Available paired (full TCIA ∩ GDC)={available}; "
            f"selecting {len(patient_ids)} after --max-patients={request.max_patients} "
            f"for {coll} ∩ {proj} (matcher={self._matcher_backend})"
        )
        counts = getattr(self.catalog, "last_source_counts", None)
        if isinstance(counts, dict):
            counts["paired_available"] = available
            counts["paired_selected"] = len(patient_ids)
            counts["max_patients"] = request.max_patients
        if isinstance(intermediate, dict):
            intermediate["paired_available"] = available
            intermediate["paired_selected"] = len(patient_ids)
            intermediate["max_patients"] = request.max_patients
        return MatchPreview(
            n_paired=len(patient_ids),
            n_available=available,
            patient_ids=patient_ids,
            disease=request.disease,
            modality=request.modality,
            genes=list(request.genes),
        )

    def fetch(self, request: DataRequest) -> tuple[ImageBundle, OmicsBundle]:
        preview = self.preview(request)
        cache_key = self._cache_key(request, preview.patient_ids)
        if cache_key in self._cache:
            log("[fetch] Using cached ImageBundle / OmicsBundle")
            return self._cache[cache_key]
        if not self.gate.approve(preview):
            raise DownloadDeniedError("Download gate denied the question-scoped fetch")
        log(f"[fetch] Starting question-scoped fetch for {preview.n_paired} patients")
        apply = getattr(self.catalog, "apply_request_options", None)
        if callable(apply):
            apply(
                max_genes=request.max_genes,
                min_altered=request.min_altered,
                disease_query=str(request.filters.get("disease_query") or request.disease),
                keyword_match=bool(request.filters.get("keyword_match")),
                matcher_backend=self._matcher_backend,
            )
        # Preview already selected TCIA ∩ GDC IDs; do not drop them again after fetch.
        records = list(self.catalog.fetch_records(preview.patient_ids))
        fetched_ids = {record.patient_id for record in records}
        if fetched_ids != set(preview.patient_ids):
            missing = sorted(set(preview.patient_ids) - fetched_ids)
            raise ValueError(
                "Fetch returned missing patients from the paired preview; "
                f"both TCIA and GDC are required. missing={missing[:8]}"
            )
        image_bundle, omics_bundle = self._split_payloads(request, records)
        self._cache[cache_key] = (image_bundle, omics_bundle)
        with_radio = sum(1 for s in image_bundle.series if s.precomputed_features)
        log(
            f"[fetch] Bundles ready: imaging={len(image_bundle.patient_ids)} "
            f"(radiomics={with_radio}), genomics={len(omics_bundle.patient_ids)}"
        )
        return image_bundle, omics_bundle

    def _prepare_llm_sources(self, request: DataRequest) -> None:
        """Ask LLM to pick TCIA/GDC sources; store preferences on the live catalog."""
        self._matcher_backend = "rules"
        apply = getattr(self.catalog, "apply_request_options", None)
        tcia = getattr(self.catalog, "tcia", None)
        gdc = getattr(self.catalog, "gdc", None)
        keyword_match = bool(request.filters.get("keyword_match"))
        forced = (
            request.tcia_collection
            and request.tcga_project
            and request.tcia_collection.upper() != "KEYWORD"
            and request.tcga_project.upper() != "KEYWORD"
        )
        if (
            not self._use_llm
            or not callable(apply)
            or tcia is None
            or gdc is None
            or not keyword_match
            or forced
        ):
            if callable(apply):
                apply(
                    disease_query=str(request.filters.get("disease_query") or request.disease),
                    keyword_match=keyword_match,
                    preferred_tcia_collections=[],
                    preferred_gdc_projects=[],
                    preferred_keywords=[],
                    matcher_backend="rules",
                )
            return

        disease_query = str(request.filters.get("disease_query") or request.disease).replace(
            "_", " "
        )
        try:
            log(
                f"[match] Calling LLM for source selection "
                f"({self._llm.config.provider}/{self._llm.config.model})..."
            )
            seed_kw = keywords_from_query(disease_query)
            collections = list(tcia.list_collections())
            ranked_c = rank_names(collections, seed_kw, min_score=10)[:40]
            tcia_candidates = [m.name for m in ranked_c] or collections[:40]

            projects_meta = list(gdc.list_projects())
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
            ranked_p = rank_names(
                [label for _, label in project_labels], seed_kw, min_score=10
            )[:40]
            label_to_pid = {label: pid for pid, label in project_labels}
            gdc_candidates = [
                label_to_pid[m.name] for m in ranked_p if m.name in label_to_pid
            ] or [pid for pid, _ in project_labels[:40]]

            plan = plan_match_sources_with_llm(
                disease_query=disease_query,
                modality=request.modality,
                question=str(request.filters.get("question_text") or disease_query),
                tcia_candidates=tcia_candidates,
                gdc_candidates=gdc_candidates,
                client=self._llm,
            )
            self._matcher_backend = (
                f"llm:{self._llm.config.provider}:{self._llm.config.model}"
            )
            log(
                f"[match] LLM selected TCIA={plan.tcia_collections} "
                f"GDC={plan.gdc_projects} keywords={plan.keywords[:8]}"
            )
            apply(
                disease_query=disease_query,
                keyword_match=True,
                preferred_tcia_collections=plan.tcia_collections,
                preferred_gdc_projects=plan.gdc_projects,
                preferred_keywords=plan.keywords or seed_kw,
                matcher_backend=self._matcher_backend,
            )
            if plan.rationale:
                intermediate = getattr(self.catalog, "last_intermediate", None)
                if isinstance(intermediate, dict):
                    intermediate["matcher_rationale"] = plan.rationale
        except Exception as exc:  # noqa: BLE001 — fall back to rules
            from agentic_radiogen.llm.client import _redact_secrets

            log(
                f"[match] LLM source selection failed ({_redact_secrets(str(exc))}); "
                "using keyword rules"
            )
            self._matcher_backend = "rules"
            apply(
                disease_query=disease_query,
                keyword_match=True,
                preferred_tcia_collections=[],
                preferred_gdc_projects=[],
                preferred_keywords=[],
                matcher_backend="rules",
            )

    @staticmethod
    def _cache_key(request: DataRequest, patient_ids: list[str]) -> tuple[str, str, tuple[str, ...]]:
        return (request.disease, request.question_id, tuple(patient_ids))

    @staticmethod
    def _split_payloads(
        request: DataRequest, records: list[CatalogRecord]
    ) -> tuple[ImageBundle, OmicsBundle]:
        if request.genes:
            genes = list(request.genes)
        else:
            genes = sorted({gene for rec in records for gene in rec.mutations})
        q_disease = str(
            request.filters.get("disease_query") or request.disease or ""
        ).strip()
        series = [
            ImageSeriesRef(
                patient_id=rec.patient_id,
                series_uid=rec.series_uid,
                modality=rec.modality,
                disease=_series_seg_disease(rec, question_disease=q_disease),
                local_path=rec.local_path,
                precomputed_features=dict(rec.radiomic_features),
                volume_summary=dict(rec.volume_summary),
            )
            for rec in records
        ]
        image_bundle = ImageBundle(
            patient_ids=[rec.patient_id for rec in records],
            series=series,
            metadata={"source": "question_scoped", "collection": request.tcia_collection},
        )
        omics_bundle = OmicsBundle(
            patient_ids=[rec.patient_id for rec in records],
            expression={rec.patient_id: dict(rec.expression) for rec in records},
            mutations={
                rec.patient_id: {g: rec.mutations.get(g, 0) for g in genes} for rec in records
            },
            clinical={
                rec.patient_id: {k: rec.clinical[k] for k in request.clinical_fields if k in rec.clinical}
                for rec in records
            },
            metadata={
                "source": "question_scoped",
                "project": request.tcga_project,
                "genes": genes,
                "discover_genes": bool(request.filters.get("discover_genes")),
            },
        )
        return image_bundle, omics_bundle
