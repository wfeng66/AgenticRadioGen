from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CatalogRecord:
    patient_id: str
    disease: str
    modality: str
    mutations: dict[str, int]
    expression: dict[str, float]
    clinical: dict[str, Any]
    series_uid: str
    radiomic_features: dict[str, float]
    volume_summary: dict[str, float]
    local_path: str | None = None


def is_paired_record(record: CatalogRecord) -> bool:
    """Keep a patient only when both imaging and genomics sides exist."""
    has_image = bool(record.series_uid or record.radiomic_features or record.volume_summary)
    has_omics = bool(record.mutations or record.expression or record.clinical)
    return has_image and has_omics


class CatalogClient(Protocol):
    """Metadata-first catalog. Implementations must not dump whole archives."""

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
    ) -> list[CatalogRecord]: ...

    def fetch_records(self, patient_ids: list[str]) -> list[CatalogRecord]: ...


def _lung_records() -> list[CatalogRecord]:
    records: list[CatalogRecord] = []
    for i in range(16):
        egfr = int(i < 8)
        kras = int(8 <= i < 12)
        tp53 = int(i % 3 == 0)
        entropy = 2.2 + 1.4 * egfr + 0.08 * (i % 4)
        sphericity = 0.55 + 0.02 * (i % 5)
        os_time = 40.0 - 18.0 * egfr + (i % 5)
        records.append(
            CatalogRecord(
                patient_id=f"TCGA-LUNG-{i:02d}",
                disease="lung",
                modality="CT",
                mutations={"EGFR": egfr, "KRAS": kras, "TP53": tp53},
                expression={
                    "EGFR": 4.0 + 2.5 * egfr,
                    "KRAS": 3.0 + 1.5 * kras,
                    "CD8A": 1.0 + 0.2 * (i % 4),
                },
                clinical={
                    "OS_time": os_time,
                    "OS_event": int(egfr or i % 2),
                    "subtype": "LUAD",
                },
                series_uid=f"1.2.lung.{i}",
                radiomic_features={
                    "original_glcm_Entropy": entropy,
                    "original_shape_Sphericity": sphericity,
                    "original_firstorder_Mean": 80.0 + i,
                },
                volume_summary={"mean": 80.0 + i, "std": 12.0, "size": 1000.0 + 10 * i},
            )
        )
    return records


def _breast_records() -> list[CatalogRecord]:
    records: list[CatalogRecord] = []
    for i in range(16):
        her2 = int(i < 8)
        brca1 = int(i % 4 == 0)
        esr1 = int(i % 2 == 0)
        sphericity = 0.50 + 0.22 * her2 + 0.01 * (i % 3)
        entropy = 1.8 + 0.05 * i
        os_time = 50.0 - 10.0 * her2 + (i % 4)
        records.append(
            CatalogRecord(
                patient_id=f"TCGA-BRCA-{i:02d}",
                disease="breast",
                modality="MR",
                mutations={"ERBB2": her2, "BRCA1": brca1, "BRCA2": 0, "ESR1": esr1, "PGR": int(not her2)},
                expression={
                    "ERBB2": 3.0 + 2.8 * her2,
                    "ESR1": 2.0 + 1.5 * esr1,
                    "BRCA1": 1.5 + 0.4 * brca1,
                },
                clinical={
                    "OS_time": os_time,
                    "OS_event": int(her2 or i % 3 == 0),
                    "subtype": "HER2+" if her2 else "HR+",
                },
                series_uid=f"1.2.breast.{i}",
                radiomic_features={
                    "original_glcm_Entropy": entropy,
                    "original_shape_Sphericity": sphericity,
                    "original_firstorder_Mean": 60.0 + i,
                },
                volume_summary={"mean": 60.0 + i, "std": 9.0, "size": 800.0 + 8 * i},
            )
        )
    return records


@dataclass
class DemoCatalog:
    """In-memory GDC/TCIA stand-in with planted radiogenomic signal."""

    _records: dict[str, CatalogRecord] = field(default_factory=dict)
    fetch_count: int = 0
    query_count: int = 0

    def __post_init__(self) -> None:
        if not self._records:
            for rec in _lung_records() + _breast_records():
                self._records[rec.patient_id] = rec

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
        _ = keyword_match
        _ = tcia_collection
        _ = tcga_project
        query = (disease_query or disease or "").lower().replace("_", " ")
        # Demo only has lung/breast synthetic rows; map by closest keyword.
        if any(token in query for token in ("breast", "brca", "mammary")):
            wanted = "breast"
        else:
            wanted = "lung"
        hits: list[CatalogRecord] = []
        for rec in self._records.values():
            if rec.disease != wanted:
                continue
            if rec.modality != modality:
                continue
            if genes and not any(g in rec.mutations for g in genes):
                continue
            _ = require_endpoint
            hits.append(rec)
        return hits

    def fetch_records(self, patient_ids: list[str]) -> list[CatalogRecord]:
        self.fetch_count += 1
        missing = [pid for pid in patient_ids if pid not in self._records]
        if missing:
            raise KeyError(f"Unknown patient IDs: {missing}")
        return [self._records[pid] for pid in patient_ids]
