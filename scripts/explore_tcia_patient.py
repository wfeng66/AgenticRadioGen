#!/usr/bin/env python3
"""Download and explore TCIA imaging and GDC clinical data for one patient.

Writes under ``--out-dir/metadata``:

- ``tcia_patient.csv`` — **one row per patient** with getPatientStudy
  demographics, aggregates, and ``series_records_json`` (all NBIA series fields)
- ``tcia_series.csv`` — one row per series with every getSeries /
  getSeriesMetaData / getSeriesSize field
- ``tcia_studies.csv`` — one row per study
- ``tcia_all_metadata.csv`` — one row per DICOM instance (headers + TCIA fields)

Examples
--------
python scripts/explore_tcia_patient.py --patient-id TCGA-17-Z021
python scripts/explore_tcia_patient.py --patient-id TCGA-17-Z021 --all-modalities
python scripts/explore_tcia_patient.py --patient-id TCGA-17-Z021 --metadata-only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from agentic_radiogen.data.gdc_client import GdcClient
from agentic_radiogen.data.tcia_client import TciaClient, TciaSeries


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
        stream.write("\n")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unnamed"


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return value


def _flatten_prefixed(prefix: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten a NBIA dict into CSV cells; spaces in keys become underscores."""
    out: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        safe = re.sub(r"\s+", "_", str(key).strip())
        out[f"{prefix}.{safe}"] = _cell(value)
    return out


def _series_metadata(
    series: TciaSeries,
    *,
    series_meta: dict[str, Any] | None = None,
    series_size: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One series record with every NBIA field we can obtain."""
    result = asdict(series)
    source = dict(result.pop("source_metadata") or {})
    # Prefer explicit dataclass fields, then overlay raw getSeries keys.
    flat: dict[str, Any] = {
        "patient_id": series.patient_id,
        "series_uid": series.series_uid,
        "modality": series.modality,
        "collection": series.collection,
        "image_count": series.image_count,
        "series_description": series.series_description,
        "protocol_name": series.protocol_name,
        "study_description": series.study_description,
        "body_part_examined": series.body_part_examined,
    }
    flat.update(_flatten_prefixed("getSeries", source))
    # Convenience aliases (kept for older consumers / studies JSON).
    flat["study_instance_uid"] = source.get("StudyInstanceUID", "")
    flat["study_date"] = source.get("StudyDate", "")
    flat["series_date"] = source.get("SeriesDate", "")
    flat["series_number"] = source.get("SeriesNumber", "")
    flat["manufacturer"] = source.get("Manufacturer", "")
    flat["manufacturer_model_name"] = source.get("ManufacturerModelName", "")
    flat["file_size"] = source.get("FileSize", "")
    flat["license_name"] = source.get("LicenseName", "")
    flat["license_uri"] = source.get("LicenseURI", "")
    flat["collection_uri"] = source.get("CollectionURI", "")
    flat["date_released"] = source.get("DateReleased", "")
    flat["timestamp"] = source.get("TimeStamp", "")
    flat["software_versions"] = source.get("SoftwareVersions", "")
    if series_meta:
        flat.update(_flatten_prefixed("getSeriesMetaData", series_meta))
    if series_size:
        flat.update(_flatten_prefixed("getSeriesSize", series_size))
    flat["source_metadata"] = source
    flat["series_metadata"] = dict(series_meta or {})
    flat["series_size"] = dict(series_size or {})
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]], *, preferred: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    skip = {"source_metadata", "series_metadata", "series_size"}
    preferred = list(preferred or [])
    extra = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in preferred and key not in skip
        }
    )
    columns = preferred + [c for c in extra if c not in preferred]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_tcia_metadata(
    series_rows: list[dict[str, Any]],
    studies_api: list[dict[str, Any]],
    out_dir: Path,
    *,
    patient_id: str,
) -> dict[str, Any]:
    """Write series/study/patient CSVs; return the single patient summary row."""
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    _json_write(meta_dir / "tcia_series.json", series_rows)
    _write_csv(
        meta_dir / "tcia_series.csv",
        series_rows,
        preferred=[
            "patient_id",
            "collection",
            "modality",
            "study_instance_uid",
            "study_date",
            "study_description",
            "series_uid",
            "series_number",
            "series_description",
            "protocol_name",
            "body_part_examined",
            "image_count",
            "file_size",
            "manufacturer",
            "manufacturer_model_name",
            "software_versions",
            "license_name",
            "license_uri",
            "collection_uri",
            "date_released",
            "timestamp",
        ],
    )

    studies: dict[str, dict[str, Any]] = {}
    for row in series_rows:
        study_uid = str(row.get("study_instance_uid") or "unknown")
        study = studies.setdefault(
            study_uid,
            {
                "study_instance_uid": study_uid,
                "study_date": row.get("study_date", ""),
                "study_description": row.get("study_description", ""),
                "collection": row.get("collection", ""),
                "series_uids": [],
            },
        )
        study["series_uids"].append(row["series_uid"])
    # Merge getPatientStudy demographics onto study objects.
    for api_row in studies_api:
        uid = str(api_row.get("StudyInstanceUID") or "")
        if not uid:
            continue
        study = studies.setdefault(
            uid,
            {
                "study_instance_uid": uid,
                "study_date": api_row.get("StudyDate", ""),
                "study_description": "",
                "collection": api_row.get("Collection", ""),
                "series_uids": [],
            },
        )
        study["getPatientStudy"] = dict(api_row)
        for key, value in api_row.items():
            study[f"getPatientStudy.{key}"] = value
    study_list = list(studies.values())
    _json_write(meta_dir / "tcia_studies.json", study_list)
    study_csv_rows = []
    for s in study_list:
        flat = {k: _cell(v) for k, v in s.items() if k != "getPatientStudy"}
        study_csv_rows.append(flat)
    _write_csv(
        meta_dir / "tcia_studies.csv",
        study_csv_rows,
        preferred=[
            "study_instance_uid",
            "study_date",
            "study_description",
            "collection",
            "series_uids",
        ],
    )

    patient_row = _patient_summary_row(patient_id, series_rows, studies_api)
    _json_write(meta_dir / "tcia_patient.json", patient_row)
    _write_csv(meta_dir / "tcia_patient.csv", [patient_row])
    return patient_row


def _unique_join(values: list[Any]) -> str:
    seen: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.append(text)
    return "|".join(seen)


def _patient_summary_row(
    patient_id: str,
    series_rows: list[dict[str, Any]],
    studies_api: list[dict[str, Any]],
) -> dict[str, Any]:
    """One CSV row per patient with demographics + aggregates + full series JSON."""
    row: dict[str, Any] = {"patient_id": patient_id}
    # Flatten unique getPatientStudy demographics (PatientSex, PatientAge, …).
    for api_row in studies_api:
        for key, value in api_row.items():
            col = f"getPatientStudy.{key}"
            if col not in row or not row[col]:
                row[col] = value
            elif str(row[col]) != str(value) and value not in (None, ""):
                # Multiple studies with differing values → pipe-join.
                row[col] = _unique_join([row[col], value])
    row["n_series"] = len(series_rows)
    row["n_studies"] = len(
        {
            str(r.get("study_instance_uid") or r.get("getSeries.StudyInstanceUID") or "")
            for r in series_rows
            if r.get("study_instance_uid") or r.get("getSeries.StudyInstanceUID")
        }
        | {str(s.get("StudyInstanceUID") or "") for s in studies_api if s.get("StudyInstanceUID")}
    )
    row["collections"] = _unique_join(
        [r.get("collection") for r in series_rows]
        + [s.get("Collection") for s in studies_api]
    )
    row["modalities"] = _unique_join([r.get("modality") for r in series_rows])
    row["body_parts"] = _unique_join(
        [
            r.get("body_part_examined") or r.get("getSeries.BodyPartExamined")
            for r in series_rows
        ]
    )
    row["series_uids"] = _unique_join([r.get("series_uid") for r in series_rows])
    row["study_uids"] = _unique_join(
        [r.get("study_instance_uid") for r in series_rows]
        + [s.get("StudyInstanceUID") for s in studies_api]
    )
    total_images = 0
    total_bytes = 0
    for r in series_rows:
        try:
            total_images += int(r.get("image_count") or 0)
        except (TypeError, ValueError):
            pass
        for key in ("file_size", "getSeries.FileSize", "getSeriesSize.TotalSizeInBytes"):
            try:
                total_bytes += int(float(r.get(key) or 0))
                break
            except (TypeError, ValueError):
                continue
    row["total_image_count"] = total_images
    row["total_file_size_bytes"] = total_bytes
    # Full per-series payload (all NBIA fields) as one JSON cell.
    slim_series = []
    for r in series_rows:
        slim = {
            k: v
            for k, v in r.items()
            if k not in {"source_metadata", "series_metadata", "series_size"}
            and not isinstance(v, (dict, list))
        }
        slim["source_metadata"] = r.get("source_metadata") or {}
        slim["series_metadata"] = r.get("series_metadata") or {}
        slim["series_size"] = r.get("series_size") or {}
        slim_series.append(slim)
    row["series_records_json"] = json.dumps(slim_series, ensure_ascii=False, default=str)
    row["studies_records_json"] = json.dumps(studies_api, ensure_ascii=False, default=str)
    return row


_BULK_DATA_KEYWORDS = {
    "AudioSampleData",
    "CurveData",
    "DoubleFloatPixelData",
    "EncapsulatedDocument",
    "FloatPixelData",
    "OverlayData",
    "PixelData",
    "SpectroscopyData",
    "WaveformData",
}


def _dicom_value(element: Any) -> Any:
    if element.VR == "SQ":
        return [
            {
                child.keyword or str(child.tag): _dicom_value(child)
                for child in item
                if child.keyword not in _BULK_DATA_KEYWORDS
            }
            for item in element.value
        ]
    value = element.value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        try:
            return [_dicom_scalar(item) for item in value]
        except TypeError:
            pass
    return _dicom_scalar(value)


def _dicom_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_dicom_header_metadata(path: Path) -> dict[str, Any]:
    import pydicom

    dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
    metadata: dict[str, Any] = {}
    for element in dataset:
        if element.keyword in _BULK_DATA_KEYWORDS or element.tag.group == 0x7FE0:
            continue
        key = element.keyword or f"{element.tag.group:04X}_{element.tag.element:04X}"
        value = _dicom_value(element)
        metadata[f"dicom.{key}"] = (
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
            if isinstance(value, (dict, list))
            else value
        )
    return metadata


def _all_metadata_row(
    series: TciaSeries,
    *,
    dicom_path: Path | None = None,
    download_dir: Path | None = None,
    status: str = "",
    error: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_level": "instance" if dicom_path else "series",
        "tcia.patient_id": series.patient_id,
        "tcia.series_uid": series.series_uid,
        "exploration.status": status,
        "exploration.error": error,
        "download.directory": str(download_dir) if download_dir else "",
        "dicom.file_path": str(dicom_path) if dicom_path else "",
    }
    if series.source_metadata:
        for key, value in series.source_metadata.items():
            row[f"tcia.{key}"] = (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
                if isinstance(value, (dict, list))
                else value
            )
    if dicom_path:
        row.update(_read_dicom_header_metadata(dicom_path))
    return row


def _write_all_metadata_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    output = out_dir / "metadata" / "tcia_all_metadata.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    base_columns = [
        "record_level",
        "tcia.patient_id",
        "tcia.series_uid",
        "exploration.status",
        "exploration.error",
        "download.directory",
        "dicom.file_path",
    ]
    columns = base_columns + sorted(
        {key for row in rows for key in row if key not in base_columns}
    )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return output


def _dicom_instances(series_dir: Path) -> list[Path]:
    import pydicom
    from pydicom.errors import InvalidDicomError

    found: list[tuple[tuple[int, Any], Path]] = []
    for path in series_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except (InvalidDicomError, OSError, EOFError, ValueError):
            continue
        if not getattr(ds, "SOPInstanceUID", None) or not getattr(ds, "Rows", None):
            continue
        instance = getattr(ds, "InstanceNumber", None)
        try:
            sort_key: tuple[int, Any] = (0, int(instance))
        except (TypeError, ValueError):
            sort_key = (1, str(getattr(ds, "SOPInstanceUID", "")))
        found.append((sort_key, path))
    return [path for _key, path in sorted(found)]


def _grayscale_slice(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        array = (
            0.299 * array[..., 0]
            + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]
        )
    elif array.ndim == 3 and array.shape[0] in (3, 4):
        array = (
            0.299 * array[0]
            + 0.587 * array[1]
            + 0.114 * array[2]
        )
    elif array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Unsupported DICOM pixel-array shape: {array.shape}")
    return np.asarray(array, dtype=np.float32)


def _window(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)


def _render_preview(
    series_dir: Path,
    output: Path,
    series: TciaSeries,
    actual_count: int,
) -> np.ndarray:
    import matplotlib.pyplot as plt
    import pydicom

    if actual_count > 1:
        from agentic_radiogen.imaging.dicom_io import load_dicom_series_with_spacing

        volume, spacing = load_dicom_series_with_spacing(series_dir)
        z, y, x = (size // 2 for size in volume.shape)
        slices = [
            (volume[z, :, :], f"Axial (z={z})", spacing[1] / spacing[2]),
            (volume[:, y, :], f"Coronal (y={y})", spacing[0] / spacing[2]),
            (volume[:, :, x], f"Sagittal (x={x})", spacing[0] / spacing[1]),
        ]
    else:
        if actual_count != 1:
            raise ValueError("No image instances available for a preview")
        ds = pydicom.dcmread(str(_dicom_instances(series_dir)[0]))
        image = _grayscale_slice(ds.pixel_array)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        image = image * slope + intercept
        slices = [(image, "Single image (MPR unavailable)", 1.0)]

    fig, axes = plt.subplots(1, len(slices), figsize=(5 * len(slices), 5))
    axes_list = np.atleast_1d(axes).ravel()
    for axis, (image, label, aspect) in zip(axes_list, slices):
        axis.imshow(_window(image), cmap="gray", origin="lower", aspect=aspect)
        axis.set_title(label)
        axis.axis("off")
    description = series.series_description or series.protocol_name or "No description"
    fig.suptitle(
        f"{series.modality} | {description} | "
        f"{series.series_uid} | {actual_count} DICOM instance(s)"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    if actual_count > 1:
        return _window(volume[volume.shape[0] // 2])
    return _window(slices[0][0])


def _write_overview(
    previews: list[tuple[str, np.ndarray]], output: Path, columns: int = 3
) -> None:
    if not previews:
        return
    import matplotlib.pyplot as plt

    rows = (len(previews) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes_list = np.atleast_1d(axes).ravel()
    for axis, (label, image) in zip(axes_list, previews):
        axis.imshow(image, cmap="gray", origin="lower")
        axis.set_title(label)
        axis.axis("off")
    for axis in axes_list[len(previews) :]:
        axis.axis("off")
    fig.suptitle("TCIA series overview")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def explore_patient(
    *,
    patient_id: str,
    out_dir: Path,
    modalities: set[str] | None = None,
    download: bool = True,
    previews: bool = True,
    cache_dir: Path | None = None,
    tcia_client: TciaClient | None = None,
    gdc_client: GdcClient | None = None,
) -> dict[str, Any]:
    """Fetch metadata, optionally download series and clinical files, and preview."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcia = tcia_client or TciaClient(cache_dir=cache_dir)
    gdc = gdc_client or GdcClient()

    if modalities is None:
        all_series = tcia.list_series_for_patient(patient_id)
    else:
        all_series = []
        for modality in sorted(modalities):
            all_series.extend(
                tcia.list_series_for_patient(patient_id, modality=modality)
            )
    selected = [
        item
        for item in all_series
        if modalities is None or item.modality.upper() in modalities
    ]
    by_uid = {item.series_uid: item for item in selected}
    series = sorted(by_uid.values(), key=lambda item: (item.modality, item.series_uid))

    # Enrich each series with getSeriesMetaData + getSeriesSize when the client supports it.
    series_rows: list[dict[str, Any]] = []
    for item in series:
        meta: dict[str, Any] = {}
        size: dict[str, Any] = {}
        if hasattr(tcia, "series_metadata"):
            try:
                meta = tcia.series_metadata(item.series_uid) or {}
            except Exception as exc:
                print(
                    f"WARNING: getSeriesMetaData failed for {item.series_uid}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
        if hasattr(tcia, "series_size"):
            try:
                size = tcia.series_size(item.series_uid) or {}
            except Exception as exc:
                print(
                    f"WARNING: getSeriesSize failed for {item.series_uid}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
        series_rows.append(
            _series_metadata(item, series_meta=meta, series_size=size)
        )

    studies_api: list[dict[str, Any]] = []
    if hasattr(tcia, "list_patient_studies"):
        try:
            studies_api = tcia.list_patient_studies(patient_id) or []
        except Exception as exc:
            print(
                f"WARNING: getPatientStudy failed for {patient_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    patient_row = _write_tcia_metadata(
        series_rows, studies_api, out_dir, patient_id=patient_id
    )

    case = gdc.get_case_details(patient_id)
    _json_write(out_dir / "metadata" / "gdc_case.json", case)
    clinical_files = gdc.list_clinical_files(patient_id)
    _json_write(out_dir / "metadata" / "gdc_clinical_files.json", clinical_files)

    summary: dict[str, Any] = {
        "patient_id": patient_id,
        "selected_modalities": sorted(modalities) if modalities is not None else "all",
        "downloaded": download,
        "gdc_case_found": case is not None,
        "gdc_clinical_file_count": len(clinical_files),
        "tcia_patient": patient_row,
        "series": [],
        "clinical_file_downloads": [],
        "errors": [],
    }
    if case is None:
        print(f"WARNING: no GDC case found for {patient_id}", file=sys.stderr)
    if not series:
        message = (
            f"No TCIA series matched patient {patient_id} and the requested modality."
        )
        summary["errors"].append(message)
        print(f"ERROR: {message}", file=sys.stderr)

    previews_for_overview: list[tuple[str, np.ndarray]] = []
    all_metadata_rows: list[dict[str, Any]] = []
    for index, item in enumerate(series, start=1):
        record: dict[str, Any] = _series_metadata(item)
        record["status"] = "metadata_only" if not download else "pending"
        if download:
            try:
                series_dir = tcia.download_series(item.series_uid)
                instances = _dicom_instances(series_dir)
                actual_count = len(instances)
                record.update(
                    {
                        "status": "downloaded",
                        "download_dir": str(series_dir.resolve()),
                        "actual_image_count": actual_count,
                        "image_count_matches_tcia": (
                            None
                            if not item.image_count
                            else actual_count == item.image_count
                        ),
                    }
                )
                all_metadata_rows.extend(
                    _all_metadata_row(
                        item,
                        dicom_path=instance,
                        download_dir=series_dir,
                        status="downloaded",
                    )
                    for instance in instances
                )
                if record["image_count_matches_tcia"] is False:
                    print(
                        f"WARNING: {item.series_uid} expected {item.image_count} "
                        f"TCIA image(s), found {actual_count}",
                        file=sys.stderr,
                    )
                print(
                    f"[{index}/{len(series)}] {item.modality} {item.series_uid}: "
                    f"{actual_count}/{item.image_count or '?'} DICOM instances"
                )
                if previews:
                    preview_path = (
                        out_dir / "previews" / f"{_safe_name(item.series_uid)}.png"
                    )
                    thumbnail = _render_preview(
                        series_dir, preview_path, item, actual_count
                    )
                    record["preview_path"] = str(preview_path.resolve())
                    previews_for_overview.append(
                        (
                            f"{item.modality} | "
                            f"{item.series_description or item.series_uid}",
                            thumbnail,
                        )
                    )
            except Exception as exc:
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                all_metadata_rows.append(
                    _all_metadata_row(
                        item,
                        download_dir=(
                            series_dir if "series_dir" in locals() else None
                        ),
                        status="error",
                        error=record["error"],
                    )
                )
                summary["errors"].append(
                    f"Series {item.series_uid}: {record['error']}"
                )
                print(
                    f"ERROR: series {item.series_uid}: {record['error']}",
                    file=sys.stderr,
                )
        else:
            all_metadata_rows.append(
                _all_metadata_row(item, status="metadata_only")
            )
        summary["series"].append(record)
        _json_write(out_dir / "manifest.json", summary)

    _write_all_metadata_csv(all_metadata_rows, out_dir)

    if download:
        clinical_dir = out_dir / "gdc_clinical_files"
        clinical_dir.mkdir(parents=True, exist_ok=True)
        for item in clinical_files:
            file_id = str(item.get("file_id") or "")
            file_name = _safe_name(str(item.get("file_name") or file_id))
            if not file_id:
                message = f"Skipped GDC clinical file with no file_id: {file_name}"
                summary["errors"].append(message)
                print(f"ERROR: {message}", file=sys.stderr)
                continue
            destination = clinical_dir / f"{file_id}_{file_name}"
            try:
                if not destination.exists():
                    destination.write_bytes(gdc.download_file(file_id))
                summary["clinical_file_downloads"].append(
                    {"file_id": file_id, "path": str(destination.resolve())}
                )
            except Exception as exc:
                message = f"GDC clinical file {file_id}: {type(exc).__name__}: {exc}"
                summary["errors"].append(message)
                print(f"ERROR: {message}", file=sys.stderr)

    if previews and download:
        _write_overview(
            previews_for_overview, out_dir / "previews" / "series_overview.png"
        )
    _json_write(out_dir / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and explore TCIA series and GDC clinical data for one patient."
    )
    parser.add_argument("--patient-id", default="TCGA-17-Z021")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/tcia_patient/<patient-id>)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="TCIA DICOM cache (default: data_cache/tcia)",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=None,
        help="Modality to include; repeat to select multiple (default: CT)",
    )
    parser.add_argument(
        "--all-modalities",
        action="store_true",
        help="Include every modality returned by TCIA, not just CT.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Fetch metadata but do not download DICOM or GDC clinical files.",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip PNG image previews and the series overview.",
    )
    args = parser.parse_args()
    if args.all_modalities and args.modality:
        parser.error("--all-modalities cannot be combined with --modality")

    modalities = (
        None
        if args.all_modalities
        else {value.upper() for value in (args.modality or ["CT"])}
    )
    out_dir = args.out_dir or (
        Path("outputs") / "tcia_patient" / _safe_name(args.patient_id)
    )
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        if not args.no_previews and not args.metadata_only:
            raise SystemExit(
                "Image previews require matplotlib. Install with "
                '`pip install -e ".[visualization]"` or pass --no-previews.'
            ) from exc

    result = explore_patient(
        patient_id=args.patient_id,
        out_dir=out_dir,
        modalities=modalities,
        download=not args.metadata_only,
        previews=not args.no_previews,
        cache_dir=args.cache_dir,
    )
    failures = len(result["errors"])
    print(
        f"Finished: {len(result['series'])} series, "
        f"{len(result['clinical_file_downloads'])} clinical files downloaded, "
        f"{failures} error(s). Output: {out_dir.resolve()}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
