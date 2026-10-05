from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agentic_radiogen.data.catalog import CatalogRecord
from agentic_radiogen.data.gdc_client import GdcClient
from agentic_radiogen.data.tcia_client import TciaClient, TciaSeries
from agentic_radiogen.imaging.radiomics_extract import extract_series_features
from agentic_radiogen.schemas.profiles import get_profile


@dataclass
class LiveCatalog:
    """Paired-only catalog with question-scoped DICOM + radiomics on fetch."""

    gdc: GdcClient = field(default_factory=GdcClient)
    tcia: TciaClient = field(default_factory=TciaClient)
    download_dicom: bool = True
    extract_radiomics: bool = True
    query_count: int = 0
    fetch_count: int = 0
    last_source_counts: dict[str, int] = field(default_factory=dict)
    last_imaging_notes: list[str] = field(default_factory=list)
    _record_cache: dict[str, CatalogRecord] = field(default_factory=dict)
    _series_cache: dict[tuple[str, str], list[TciaSeries]] = field(default_factory=dict)

    def query_metadata(
        self,
        *,
        disease: str,
        modality: str,
        genes: list[str],
        require_endpoint: str | None = None,
    ) -> list[CatalogRecord]:
        self.query_count += 1
        profile = get_profile(disease)
        series = self._series(profile.tcia_collection, modality)
        tcia_patients = {item.patient_id: item for item in series}
        gdc_cases = {
            case.submitter_id: case
            for case in self.gdc.list_cases(
                profile.tcga_project, submitter_ids=sorted(tcia_patients)
            )
        }
        _ = require_endpoint
        paired = sorted(set(tcia_patients) & set(gdc_cases))
        self.last_source_counts = {
            "tcia_only_dropped": len(set(tcia_patients) - set(gdc_cases)),
            "gdc_only_dropped": len(set(gdc_cases) - set(tcia_patients)),
            "paired_kept": len(paired),
        }
        self._record_cache.clear()
        records = [
            CatalogRecord(
                patient_id=pid,
                disease=profile.name,
                modality=modality,
                mutations={gene: 0 for gene in genes},
                expression={},
                clinical=dict(gdc_cases[pid].clinical),
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
        if not patient_ids:
            return []
        if self._record_cache.get(patient_ids[0]) is None:
            raise KeyError("query_metadata must run before fetch_records")
        genes = list(next(iter(self._record_cache.values())).mutations)
        flags = self.gdc.mutation_flags(patient_ids, genes)
        expression = self.gdc.expression_means(patient_ids, genes)
        fetched: list[CatalogRecord] = []
        for pid in patient_ids:
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
                        self.last_imaging_notes.append(f"{pid}: radiomics ok ({Path(series_dir).name})")
                    else:
                        self.last_imaging_notes.append(f"{pid}: dicom downloaded, radiomics skipped")
                except Exception as exc:
                    self.last_imaging_notes.append(f"{pid}: imaging failed: {exc}")
            updated = CatalogRecord(
                patient_id=rec.patient_id,
                disease=rec.disease,
                modality=rec.modality,
                mutations=flags.get(pid, dict(rec.mutations)),
                expression=expression.get(pid, dict(rec.expression)),
                clinical=dict(rec.clinical),
                series_uid=rec.series_uid,
                radiomic_features=radiomic_features,
                volume_summary=volume_summary,
                local_path=local_path,
            )
            self._record_cache[pid] = updated
            fetched.append(updated)
        return fetched

    def _series(self, collection: str, modality: str) -> list[TciaSeries]:
        key = (collection, modality)
        if key not in self._series_cache:
            self._series_cache[key] = self.tcia.list_series(collection, modality)
        return self._series_cache[key]
