from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from agentic_radiogen.data.http_json import get_bytes, get_json

TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

# Series descriptions that are almost never useful for 3D tumor radiomics.
_SCOUT_RE = re.compile(
    r"scout|localizer|localiser|topogram|surview|pilot|3d\s*recon|dose\s*report",
    re.I,
)

# Body-site series that are not chest/lung CTs (abd/pelvis, whole-body, renal, …).
# Used when ranking series for lung radiogenomics so ImageCount alone cannot
# prefer ABD/PEL or CT WB over CHEST / LUNG series.
_NON_CHEST_RE = re.compile(
    r"(?:"
    r"\babd(?:omen)?\b|"
    r"\bpel(?:vis)?\b|"
    r"\bwhole\s*body\b|"
    r"\bwb\b|"
    r"\brenal\b|"
    r"\bkidney\b"
    r")",
    re.I,
)

# Positive chest/lung markers override abd/pelvis tokens (e.g. "Chest/Abd").
_CHEST_RE = re.compile(
    r"\b(?:chest|lung|thorax|thoracic|pe\s*study)\b",
    re.I,
)

# Minimum ImageCount for a usable 3D CT when chest_only filtering is on.
_MIN_CHEST_SLICES = 8


@dataclass(frozen=True)
class TciaSeries:
    patient_id: str
    series_uid: str
    modality: str
    collection: str
    image_count: int = 0
    series_description: str = ""
    protocol_name: str = ""
    study_description: str = ""
    body_part_examined: str = ""
    source_metadata: dict[str, Any] | None = None

    @property
    def site_text(self) -> str:
        """Text used for site/scout filters (descriptions + protocol + study).

        BodyPartExamined is intentionally omitted here: on TCGA-LUAD it often
        says CHEST/LUNG even for ABD/PEL volumes, so it must not count as
        positive chest evidence. It is still available on the dataclass for
        logging / optional blacklist via ``body_part_examined``.
        """
        parts = [
            self.series_description or "",
            self.protocol_name or "",
            self.study_description or "",
        ]
        return " ".join(p.strip() for p in parts if p and str(p).strip())


class TciaClient:
    """Public TCIA/NBIA metadata + question-scoped DICOM series download."""

    def __init__(
        self,
        base_url: str = TCIA_BASE,
        getter: Callable[..., Any] | None = None,
        downloader: Callable[..., bytes] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._get = getter or get_json
        self._download = downloader or get_bytes
        self.cache_dir = Path(cache_dir or Path.cwd() / "data_cache" / "tcia")

    def list_collections(self) -> list[str]:
        """Return public TCIA collection names."""
        rows = self._get(f"{self.base_url}/getCollectionValues", {})
        if not isinstance(rows, list):
            return []
        names: list[str] = []
        for row in rows:
            if isinstance(row, dict):
                name = str(row.get("Collection") or row.get("collection") or "").strip()
            else:
                name = str(row).strip()
            if name:
                names.append(name)
        return sorted(set(names))

    def list_series(self, collection: str, modality: str) -> list[TciaSeries]:
        rows = self._get(
            f"{self.base_url}/getSeries",
            {"Collection": collection, "Modality": modality},
        )
        return self._parse_series_rows(rows, default_collection=collection, modality=modality)

    def list_series_for_patient(
        self, patient_id: str, modality: str | None = None
    ) -> list[TciaSeries]:
        """Look up imaging series for one patient ID across TCIA collections."""
        params = {"PatientID": patient_id}
        if modality:
            params["Modality"] = modality
        rows = self._get(
            f"{self.base_url}/getSeries",
            params,
        )
        return self._parse_series_rows(rows, default_collection="", modality=modality)

    def _parse_series_rows(
        self,
        rows: Any,
        *,
        default_collection: str,
        modality: str | None,
    ) -> list[TciaSeries]:
        if not isinstance(rows, list):
            return []
        series: list[TciaSeries] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            patient_id = str(row.get("PatientID") or row.get("PatientId") or "")
            uid = str(row.get("SeriesInstanceUID") or "")
            if not patient_id or not uid:
                continue
            series_desc = str(
                row.get("SeriesDescription") or row.get("seriesDescription") or ""
            ).strip()
            protocol = str(row.get("ProtocolName") or row.get("protocolName") or "").strip()
            study_desc = str(
                row.get("StudyDescription") or row.get("studyDescription") or ""
            ).strip()
            body_part = str(
                row.get("BodyPartExamined") or row.get("bodyPartExamined") or ""
            ).strip()
            # Backward-compatible display field: prefer SeriesDescription, else Protocol.
            display_desc = series_desc or protocol
            count = _image_count(row)
            series.append(
                TciaSeries(
                    patient_id=_submitter_id(patient_id),
                    series_uid=uid,
                    modality=str(row.get("Modality") or modality or ""),
                    collection=str(row.get("Collection") or default_collection),
                    image_count=count,
                    series_description=display_desc,
                    protocol_name=protocol,
                    study_description=study_desc,
                    body_part_examined=body_part,
                    source_metadata=dict(row),
                )
            )
        return series

    def list_patient_studies(self, patient_id: str) -> list[dict[str, Any]]:
        """NBIA ``getPatientStudy`` rows for one PatientID (demographics + study UIDs)."""
        rows = self._get(
            f"{self.base_url}/getPatientStudy",
            {"PatientID": patient_id},
        )
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows if isinstance(row, dict)]

    def series_metadata(self, series_uid: str) -> dict[str, Any]:
        """NBIA ``getSeriesMetaData`` for one SeriesInstanceUID (richer series fields)."""
        rows = self._get(
            f"{self.base_url}/getSeriesMetaData",
            {"SeriesInstanceUID": series_uid},
        )
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return dict(rows[0])
        if isinstance(rows, dict):
            return dict(rows)
        return {}

    def series_size(self, series_uid: str) -> dict[str, Any]:
        """NBIA ``getSeriesSize`` (TotalSizeInBytes, ObjectCount)."""
        rows = self._get(
            f"{self.base_url}/getSeriesSize",
            {"SeriesInstanceUID": series_uid},
        )
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return dict(rows[0])
        if isinstance(rows, dict):
            return dict(rows)
        return {}

    def download_series(self, series_uid: str) -> Path:
        """Download one series ZIP and extract it. Cached; never pulls a full collection."""
        dest = self.cache_dir / series_uid
        marker = dest / ".complete"
        if marker.exists() and any(dest.rglob("*")):
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        blob = self._download(
            f"{self.base_url}/getImage",
            {"SeriesInstanceUID": series_uid},
            timeout=600,
        )
        if len(blob) < 100:
            raise RuntimeError(f"Empty DICOM download for series {series_uid}")
        with zipfile.ZipFile(BytesIO(blob)) as zf:
            zf.extractall(dest)
        marker.write_text("ok", encoding="utf-8")
        return dest


def is_scout_series(series: TciaSeries) -> bool:
    """True for scout / localizer / topogram-style series."""
    return bool(_SCOUT_RE.search(series.site_text or series.series_description or ""))


def has_positive_chest_evidence(series: TciaSeries) -> bool:
    """True when description/protocol/study text contains a chest/lung marker."""
    return bool(_CHEST_RE.search(series.site_text or ""))


def is_non_chest_series(series: TciaSeries) -> bool:
    """True when metadata clearly indicates a non-chest body site.

    Combined labels like ``Chest/Abd`` are treated as chest-eligible because a
    chest keyword is present. BodyPartExamined ABD/PELVIS can blacklist when
    description text is empty (but CHEST/LUNG BodyPart alone does not whitelist).
    """
    text = series.site_text or ""
    if _CHEST_RE.search(text):
        return False
    if _NON_CHEST_RE.search(text):
        return True
    body = (series.body_part_examined or "").strip()
    if body and _NON_CHEST_RE.search(body) and not _CHEST_RE.search(body):
        return True
    return False


def prefer_volumetric_series_ranked(
    candidates: list[TciaSeries],
    *,
    chest_only: bool = False,
) -> list[TciaSeries]:
    """Rank CT series for 3D tumor segmentation / radiomics (best first).

    When ``chest_only`` is True:
    1. Drop clearly non-chest series (abd/pelvis, whole-body, renal, …), but
       keep combined chest series such as ``Chest/Abd``.
    2. Drop scouts/topograms and series with ImageCount < 8.
    3. Prefer positive chest/lung text evidence; demote empty site_text so
       ImageCount alone cannot beat a labeled CHEST series.
    4. If nothing remains, return ``[]``.
    """
    if not candidates:
        return []
    pool = candidates
    if chest_only:
        pool = [
            s
            for s in candidates
            if not is_non_chest_series(s)
            and not is_scout_series(s)
            and int(s.image_count or 0) >= _MIN_CHEST_SLICES
        ]
        if not pool:
            return []

    def _score(s: TciaSeries) -> tuple:
        scout = 1 if is_scout_series(s) else 0
        count = int(s.image_count or 0)
        if chest_only:
            # 0 = has CHEST/LUNG text; 1 = empty/ambiguous site text (demoted).
            no_chest_text = 0 if has_positive_chest_evidence(s) else 1
            empty_site = 0 if (s.site_text or "").strip() else 1
            return (scout, no_chest_text, empty_site, -count, s.series_uid)
        return (scout, -count, s.series_uid)

    return sorted(pool, key=_score)


def prefer_volumetric_series(
    candidates: list[TciaSeries],
    *,
    chest_only: bool = False,
) -> TciaSeries | None:
    """Pick the best CT series (first of :func:`prefer_volumetric_series_ranked`)."""
    ranked = prefer_volumetric_series_ranked(candidates, chest_only=chest_only)
    return ranked[0] if ranked else None


def _image_count(row: dict[str, Any]) -> int:
    for key in ("ImageCount", "imageCount", "NumberOfImages", "Instances"):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return 0


_TCGA_SUBMITTER = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})(?:-|$)")


def _submitter_id(patient_id: str) -> str:
    """TCGA imaging IDs are barcodes; GDC cases use the 12-character submitter id."""
    match = _TCGA_SUBMITTER.match(patient_id)
    return match.group(1) if match else patient_id
