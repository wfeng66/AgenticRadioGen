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
from agentic_radiogen.schemas.profiles import get_profile


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
            )
            if is_paired_record(record)
        ]
        patient_ids = [r.patient_id for r in hits[: request.max_patients]]
        return MatchPreview(
            n_paired=len(patient_ids),
            patient_ids=patient_ids,
            disease=request.disease,
            modality=request.modality,
            genes=list(request.genes),
        )

    def fetch(self, request: DataRequest) -> tuple[ImageBundle, OmicsBundle]:
        preview = self.preview(request)
        cache_key = self._cache_key(request, preview.patient_ids)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not self.gate.approve(preview):
            raise DownloadDeniedError("Download gate denied the question-scoped fetch")
        records = [
            record
            for record in self.catalog.fetch_records(preview.patient_ids)
            if is_paired_record(record)
        ]
        if {record.patient_id for record in records} != set(preview.patient_ids):
            raise ValueError("Fetch returned unpaired or missing patients; both TCIA and GDC are required")
        image_bundle, omics_bundle = self._split_payloads(request, records)
        self._cache[cache_key] = (image_bundle, omics_bundle)
        return image_bundle, omics_bundle

    @staticmethod
    def _cache_key(request: DataRequest, patient_ids: list[str]) -> tuple[str, str, tuple[str, ...]]:
        return (request.disease, request.question_id, tuple(patient_ids))

    @staticmethod
    def _split_payloads(
        request: DataRequest, records: list[CatalogRecord]
    ) -> tuple[ImageBundle, OmicsBundle]:
        profile = get_profile(request.disease)
        genes = request.genes or list(profile.candidate_genes)
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
            metadata={"source": "question_scoped", "project": request.tcga_project, "genes": genes},
        )
        return image_bundle, omics_bundle
