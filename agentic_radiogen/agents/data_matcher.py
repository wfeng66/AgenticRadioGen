from __future__ import annotations

from agentic_radiogen.data.catalog import CatalogClient, CatalogRecord, is_paired_record
from agentic_radiogen.data.gate import AlwaysAllowGate, DownloadGate
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


class DataMatcherAgent:
    """Metadata preview, then question-scoped fetch of images and omics in parallel payloads."""

    def __init__(
        self,
        catalog: CatalogClient,
        gate: DownloadGate | None = None,
    ) -> None:
        self.catalog = catalog
        self.gate = gate or AlwaysAllowGate()
        self._cache: dict[tuple[str, str, tuple[str, ...]], tuple[ImageBundle, OmicsBundle]] = {}

    def preview(self, request: DataRequest) -> MatchPreview:
        if request.filters.get("full_archive"):
            raise ValueError("Full-archive downloads are not allowed")
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
        log(
            f"[match] Available paired (full TCIA ∩ GDC)={available}; "
            f"selecting {len(patient_ids)} after --max-patients={request.max_patients} "
            f"for {coll} ∩ {proj}"
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
        series = [
            ImageSeriesRef(
                patient_id=rec.patient_id,
                series_uid=rec.series_uid,
                modality=rec.modality,
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
