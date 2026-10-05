from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from agentic_radiogen.data.http_json import get_bytes, get_json

TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


@dataclass(frozen=True)
class TciaSeries:
    patient_id: str
    series_uid: str
    modality: str
    collection: str


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

    def list_series(self, collection: str, modality: str) -> list[TciaSeries]:
        rows = self._get(
            f"{self.base_url}/getSeries",
            {"Collection": collection, "Modality": modality},
        )
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
            series.append(
                TciaSeries(
                    patient_id=_submitter_id(patient_id),
                    series_uid=uid,
                    modality=str(row.get("Modality") or modality),
                    collection=str(row.get("Collection") or collection),
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


_TCGA_SUBMITTER = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})(?:-|$)")


def _submitter_id(patient_id: str) -> str:
    """TCGA imaging IDs are barcodes; GDC cases use the 12-character submitter id."""
    match = _TCGA_SUBMITTER.match(patient_id)
    return match.group(1) if match else patient_id
