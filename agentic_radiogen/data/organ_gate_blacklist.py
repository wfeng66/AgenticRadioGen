"""Persistent blacklist of SeriesInstanceUIDs that failed organ-gate verification.

Failed UIDs are skipped on later downloads so we do not re-run TotalSegmentator
(or re-hit NBIA for uncached volumes) on known wrong-site series. The local
DICOM cache folder for a failed UID is also removed.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_NAME = "organ_gate_failed.jsonl"


def default_blacklist_path(cache_dir: str | Path | None = None) -> Path:
    root = Path(cache_dir) if cache_dir else Path.cwd() / "data_cache" / "tcia"
    return Path(root) / _DEFAULT_NAME


def load_failed_uids(path: str | Path | None = None) -> set[str]:
    """Return SeriesInstanceUIDs previously recorded as organ-gate failures."""
    file_path = Path(path) if path else default_blacklist_path()
    if not file_path.is_file():
        return set()
    uids: set[str] = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("{"):
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            uid = str(row.get("series_uid") or "").strip()
            if uid:
                uids.add(uid)
            continue
        uids.add(text)
    return uids


def record_failed_uid(
    series_uid: str,
    *,
    reason: str = "",
    patient_id: str = "",
    organ_keywords: list[str] | tuple[str, ...] | None = None,
    path: str | Path | None = None,
) -> Path:
    """Append one organ-gate failure. Idempotent for the same UID."""
    uid = str(series_uid or "").strip()
    if not uid:
        raise ValueError("series_uid is required")
    file_path = Path(path) if path else default_blacklist_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_failed_uids(file_path)
    if uid in existing:
        return file_path
    row: dict[str, Any] = {
        "series_uid": uid,
        "patient_id": str(patient_id or ""),
        "reason": str(reason or ""),
        "organ_keywords": [str(k) for k in (organ_keywords or ())],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with file_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return file_path


def filter_uids(uids: list[str], *, failed: set[str] | None = None, path: str | Path | None = None) -> list[str]:
    """Drop blacklisted UIDs while preserving order."""
    blocked = failed if failed is not None else load_failed_uids(path)
    if not blocked:
        return list(uids)
    return [u for u in uids if str(u).strip() and str(u).strip() not in blocked]


def remove_cached_series(
    series_uid: str,
    *,
    cache_dir: str | Path | None = None,
    series_dir: str | Path | None = None,
) -> Path | None:
    """Delete the local DICOM folder for a failed series UID.

    Prefer ``series_dir`` when provided (exact download path). Otherwise remove
    ``cache_dir / series_uid``. Returns the removed path, or None if nothing
    existed / UID was empty.
    """
    uid = str(series_uid or "").strip()
    target: Path | None = None
    if series_dir is not None:
        candidate = Path(series_dir)
        # Only delete if the folder name matches the UID (safety).
        if candidate.is_dir() and (not uid or candidate.name == uid):
            target = candidate
    if target is None and uid:
        root = Path(cache_dir) if cache_dir else Path.cwd() / "data_cache" / "tcia"
        candidate = Path(root) / uid
        if candidate.is_dir():
            target = candidate
    if target is None or not target.is_dir():
        return None
    # Never delete the cache root itself.
    if target.resolve() == (
        Path(cache_dir).resolve()
        if cache_dir
        else (Path.cwd() / "data_cache" / "tcia").resolve()
    ):
        return None
    shutil.rmtree(target, ignore_errors=False)
    return target
