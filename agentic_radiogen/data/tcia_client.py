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


@dataclass(frozen=True)
class TciaSeries:
    patient_id: str
    series_uid: str
    modality: str
    collection: str
    image_count: int = 0
    series_description: str = ""


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

    def list_series_for_patient(self, patient_id: str, modality: str) -> list[TciaSeries]:
        """Look up imaging series for one patient ID across TCIA collections."""
        rows = self._get(
            f"{self.base_url}/getSeries",
            {"PatientID": patient_id, "Modality": modality},
        )
        return self._parse_series_rows(rows, default_collection="", modality=modality)

    def _parse_series_rows(
        self,
        rows: Any,
        *,
        default_collection: str,
        modality: str,
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
            desc = str(
                row.get("SeriesDescription")
                or row.get("ProtocolName")
                or row.get("seriesDescription")
                or ""
            )
            count = _image_count(row)
            series.append(
                TciaSeries(
                    patient_id=_submitter_id(patient_id),
                    series_uid=uid,
                    modality=str(row.get("Modality") or modality),
                    collection=str(row.get("Collection") or default_collection),
                    image_count=count,
                    series_description=desc,
                )
            )
        return series

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


def prefer_volumetric_series(candidates: list[TciaSeries]) -> TciaSeries | None:
    """Pick the best CT series for 3D tumor segmentation / radiomics.

    Prefers non-scout series with the largest ImageCount. This is why nnU-Net
    was falling back: ``setdefault`` kept the first (often 2-slice scout) series.
    """
    if not candidates:
        return None

    def _score(s: TciaSeries) -> tuple:
        scout = 1 if _SCOUT_RE.search(s.series_description or "") else 0
        # Unknown count (0) ranks below known thin series only slightly; prefer known thick.
        count = int(s.image_count or 0)
        return (scout, -count, s.series_uid)

    return sorted(candidates, key=_score)[0]


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
