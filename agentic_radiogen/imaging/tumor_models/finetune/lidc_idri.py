"""LIDC-IDRI TCIA layout → cached NIfTI image/label pairs.

Expected trees
--------------
DICOM (user ``--data-path``)::

    lidc_idri/
      LIDC-IDRI-0001/
        <study>/
          <series>/*.dcm

Annotations often live in the NBIA sibling manifest tree::

    manifest-*/LIDC-IDRI/LIDC-IDRI-0001/<study>/<series>/*.xml

This module pairs each patient's primary CT series with its XML (same folder or
matched manifest series), rasterizes inclusion nodule ROIs (union of readers),
and writes::

    <cache>/<case_id>/image.nii.gz
    <cache>/<case_id>/label.nii.gz
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from agentic_radiogen.imaging.tumor_models.finetune.base import CasePair
from agentic_radiogen.util.progress import log

_NS = "{http://www.nih.gov}"
_PATIENT_RE = re.compile(r"^LIDC-IDRI-\d+$", re.IGNORECASE)
_MIN_SLICES = 8


def looks_like_lidc_root(root: Path) -> bool:
    if not root.is_dir():
        return False
    hits = 0
    for child in root.iterdir():
        if child.is_dir() and _PATIENT_RE.match(child.name):
            hits += 1
            if hits >= 2:
                return True
    return hits == 1 and any(_PATIENT_RE.match(p.name) for p in root.iterdir() if p.is_dir())


def discover_lidc_cases(
    root: Path,
    *,
    cache_dir: Path | None = None,
    max_cases: int | None = None,
    verbose: bool = True,
) -> list[CasePair]:
    """Convert / reuse LIDC DICOM+XML into CasePair NIfTI paths."""
    root = root.expanduser().resolve()
    if not looks_like_lidc_root(root):
        return []

    cache = (cache_dir or (root / "_finetune_cache")).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    xml_index = _build_xml_index(root, verbose=verbose)

    patients = sorted(
        p for p in root.iterdir() if p.is_dir() and _PATIENT_RE.match(p.name)
    )
    if max_cases is not None:
        patients = patients[: max(0, int(max_cases))]

    cases: list[CasePair] = []
    skipped = 0
    skip_reasons: dict[str, int] = {}
    for i, patient in enumerate(patients, start=1):
        try:
            pair = _ensure_patient_pair(
                patient, cache=cache, xml_index=xml_index, verbose=verbose
            )
        except Exception as exc:
            skipped += 1
            reason = str(exc)
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            log(
                f"[finetune:lidc] skip {patient.name}: {exc}",
                enabled=verbose,
            )
            continue
        if pair is None:
            skipped += 1
            skip_reasons["unknown"] = skip_reasons.get("unknown", 0) + 1
            continue
        cases.append(pair)
        if verbose and (i % 25 == 0 or i == len(patients)):
            # prepared=successes, i=patients attempted so far, skipped=failures
            log(
                f"[finetune:lidc] converted {len(cases)} OK / {i} attempted "
                f"({skipped} skipped — no usable nodule mask or series/XML)",
                enabled=True,
            )

    if skip_reasons and verbose:
        summary = "; ".join(f"{n}× {msg}" for msg, n in sorted(skip_reasons.items(), key=lambda x: -x[1]))
        log(f"[finetune:lidc] skip reasons: {summary}", enabled=True)
    log(
        f"[finetune:lidc] Ready {len(cases)} labeled cases under {cache} "
        f"(attempted={len(patients)}, skipped={skipped})",
        enabled=verbose,
    )
    return cases


def _ensure_patient_pair(
    patient: Path,
    *,
    cache: Path,
    xml_index: dict[str, Path],
    verbose: bool,
) -> CasePair | None:
    out_dir = cache / patient.name
    image_path = out_dir / "image.nii.gz"
    label_path = out_dir / "label.nii.gz"
    if image_path.is_file() and label_path.is_file():
        return CasePair(case_id=patient.name, image=image_path, label=label_path)

    series_dir, xml_path = _select_series_and_xml(patient, xml_index=xml_index)
    if series_dir is None or xml_path is None:
        raise FileNotFoundError(
            "no CT series with LIDC XML annotation "
            "(DICOM-only tree needs sibling manifest-*/LIDC-IDRI XML)"
        )

    volume, spacing, sop_to_index, z_positions = _load_series_with_sop(series_dir)
    mask = _rasterize_lidc_xml(
        xml_path,
        volume.shape,
        sop_to_index,
        z_positions=z_positions,
    )
    if not mask.any():
        raise RuntimeError(
            f"empty nodule mask from {xml_path.name} "
            "(XML ROIs did not map onto CT slices — SOP/Z mismatch or no inclusion ROIs)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_nifti(image_path, volume.astype(np.float32), spacing_zyx=spacing)
    _save_nifti(label_path, mask.astype(np.uint8), spacing_zyx=spacing)
    log(
        f"[finetune:lidc] {patient.name}: series={series_dir.name} "
        f"xml={xml_path.name} shape={volume.shape} tumor_voxels={int(mask.sum())}",
        enabled=verbose,
    )
    return CasePair(case_id=patient.name, image=image_path, label=label_path)


def _select_series_and_xml(
    patient: Path, *, xml_index: dict[str, Path]
) -> tuple[Path | None, Path | None]:
    series_list = _iter_series_dirs(patient)
    scored: list[tuple[int, Path, Path | None]] = []
    for series in series_list:
        n_dcm = sum(1 for p in series.iterdir() if p.is_file() and _is_dicom_name(p))
        if n_dcm < _MIN_SLICES:
            continue
        xml_local = _first_xml(series)
        xml = xml_local
        if xml is None:
            xml = _match_xml_from_index(series, xml_index)
        # Prefer annotated series; then larger series.
        score = (1_000_000 if xml is not None else 0) + n_dcm
        scored.append((score, series, xml))
    if not scored:
        return None, None
    scored.sort(key=lambda t: t[0], reverse=True)
    _, series, xml = scored[0]
    return series, xml


def _iter_series_dirs(patient: Path) -> list[Path]:
    """Yield series folders: patient/study/series (depth 2) or nested DICOM dirs."""
    out: list[Path] = []
    for study in sorted(p for p in patient.iterdir() if p.is_dir()):
        children = [p for p in study.iterdir() if p.is_dir()]
        if children:
            out.extend(sorted(children))
        else:
            # Study folder itself may hold DICOM.
            out.append(study)
    return out


def _is_dicom_name(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".dcm") or path.suffix == "" or name.endswith(".dicom")


def _first_xml(folder: Path) -> Path | None:
    xmls = sorted(folder.glob("*.xml"))
    return xmls[0] if xmls else None


def _build_xml_index(root: Path, *, verbose: bool) -> dict[str, Path]:
    """Map series fingerprint → XML path from root and sibling manifest trees."""
    index: dict[str, Path] = {}
    search_roots = _annotation_roots(root)
    for base in search_roots:
        if not base.is_dir():
            continue
        for xml in base.rglob("*.xml"):
            series_dir = xml.parent
            for key in _series_keys(series_dir):
                index.setdefault(key, xml)
    log(
        f"[finetune:lidc] Indexed {len(index)} XML series keys from "
        f"{len(search_roots)} annotation root(s)",
        enabled=verbose,
    )
    return index


def _annotation_roots(dicom_root: Path) -> list[Path]:
    roots = [dicom_root]
    # Sibling / parent NBIA manifest trees.
    for parent in (dicom_root.parent, dicom_root.parent.parent):
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if not child.is_dir():
                continue
            name = child.name.lower()
            if name.startswith("manifest"):
                cand = child / "LIDC-IDRI"
                roots.append(cand if cand.is_dir() else child)
            if name == "lidc-idri":
                roots.append(child)
    # De-dupe
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            key = str(r.resolve())
        except OSError:
            key = str(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _series_keys(series_dir: Path) -> list[str]:
    """Fingerprints shared between flattened lidc_idri and TCIA manifest names."""
    name = series_dir.name
    keys = {name.lower(), re.sub(r"[^0-9]", "", name)}
    # e.g. 3000566-03192 and 3000566.000000-NA-03192 → 300056603192
    m = re.search(r"(\d{5,}).*?(\d{4,})$", name)
    if m:
        keys.add(m.group(1) + m.group(2))
    study = series_dir.parent.name
    sm = re.search(r"(\d{4,})$", study)
    if sm and m:
        keys.add(sm.group(1) + m.group(1) + m.group(2))
    return [k for k in keys if len(k) >= 6]


def _match_xml_from_index(series_dir: Path, xml_index: dict[str, Path]) -> Path | None:
    for key in _series_keys(series_dir):
        hit = xml_index.get(key)
        if hit is not None:
            return hit
    return None


def _load_series_with_sop(
    series_dir: Path,
) -> tuple[
    np.ndarray,
    tuple[float, float, float],
    dict[str, int],
    np.ndarray,
]:
    import pydicom

    files = [p for p in series_dir.iterdir() if p.is_file() and _is_dicom_name(p)]
    datasets = []
    for path in files:
        if path.suffix.lower() == ".xml":
            continue
        try:
            ds = pydicom.dcmread(str(path), force=True)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        mod = str(getattr(ds, "Modality", "CT") or "CT").upper()
        if mod not in {"CT", "DX"}:
            continue
        datasets.append(ds)
    if len(datasets) < _MIN_SLICES:
        raise ValueError(f"too few CT slices in {series_dir} ({len(datasets)})")

    def _sort_key(ds):  # type: ignore[no-untyped-def]
        if getattr(ds, "InstanceNumber", None) is not None:
            try:
                return (0, int(ds.InstanceNumber))
            except Exception:
                pass
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return (1, float(ipp[2]))
        return (2, str(getattr(ds, "SOPInstanceUID", "")))

    datasets.sort(key=_sort_key)
    slices: list[np.ndarray] = []
    sop_to_index: dict[str, int] = {}
    z_positions: list[float] = []
    kept_ds = []
    for i, ds in enumerate(datasets):
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if arr.ndim != 2:
            continue
        idx = len(slices)
        slices.append(arr * slope + intercept)
        sop = str(getattr(ds, "SOPInstanceUID", "") or "")
        if sop:
            sop_to_index[sop] = idx
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            z_positions.append(float(ipp[2]))
        else:
            z_positions.append(float(idx))
        kept_ds.append(ds)
    if len(slices) < _MIN_SLICES:
        raise ValueError(f"could not stack CT volume from {series_dir}")
    volume = np.stack(slices, axis=0)
    from agentic_radiogen.imaging.dicom_io import _estimate_spacing_zyx

    spacing = _estimate_spacing_zyx(kept_ds)
    return volume, spacing, sop_to_index, np.asarray(z_positions, dtype=float)


def _rasterize_lidc_xml(
    xml_path: Path,
    shape_zyx: tuple[int, ...],
    sop_to_index: dict[str, int],
    *,
    z_positions: np.ndarray | None = None,
) -> np.ndarray:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mask = np.zeros(shape_zyx, dtype=np.uint8)
    # Union of all readers' inclusion ROIs (common weak-label consensus).
    for roi in root.iter(f"{_NS}roi"):
        inclusion = (roi.findtext(f"{_NS}inclusion") or "").strip().upper()
        if inclusion not in {"TRUE", "T", "1", "YES"}:
            continue
        sop = (roi.findtext(f"{_NS}imageSOP_UID") or "").strip()
        z = sop_to_index.get(sop)
        if z is None:
            z = _index_from_zposition(roi, z_positions)
        if z is None or not (0 <= z < shape_zyx[0]):
            continue
        xs: list[int] = []
        ys: list[int] = []
        for edge in roi.findall(f"{_NS}edgeMap"):
            try:
                xs.append(int(float(edge.findtext(f"{_NS}xCoord"))))
                ys.append(int(float(edge.findtext(f"{_NS}yCoord"))))
            except Exception:
                continue
        if len(xs) < 3:
            continue
        _fill_polygon(mask[z], xs, ys)
    return mask


def _index_from_zposition(
    roi: ET.Element, z_positions: np.ndarray | None
) -> int | None:
    if z_positions is None or z_positions.size == 0:
        return None
    raw = (roi.findtext(f"{_NS}imageZposition") or "").strip()
    if not raw:
        return None
    try:
        z = float(raw)
    except ValueError:
        return None
    return int(np.argmin(np.abs(z_positions - z)))


def _fill_polygon(mask2d: np.ndarray, xs: list[int], ys: list[int]) -> None:
    """Fill a 2D polygon into ``mask2d`` (H, W) with ones."""
    h, w = mask2d.shape
    try:
        from PIL import Image, ImageDraw

        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(list(zip(xs, ys)), outline=1, fill=1)
        mask2d |= np.asarray(img, dtype=np.uint8)
        return
    except Exception:
        pass
    try:
        from skimage.draw import polygon as sk_polygon

        rr, cc = sk_polygon(ys, xs, shape=mask2d.shape)
        mask2d[rr, cc] = 1
        return
    except Exception:
        pass
    # Bounding-box fallback (still better than skipping).
    x0, x1 = max(0, min(xs)), min(w - 1, max(xs))
    y0, y1 = max(0, min(ys)), min(h - 1, max(ys))
    mask2d[y0 : y1 + 1, x0 : x1 + 1] = 1


def _save_nifti(
    path: Path,
    array_zyx: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import nibabel as nib

        # nibabel expects (x, y, z); our arrays are (z, y, x).
        data = np.transpose(array_zyx, (2, 1, 0))
        affine = np.diag(
            [spacing_zyx[2], spacing_zyx[1], spacing_zyx[0], 1.0]
        ).astype(float)
        nib.save(nib.Nifti1Image(data, affine), str(path))
        return
    except Exception:
        pass
    try:
        import SimpleITK as sitk

        img = sitk.GetImageFromArray(array_zyx)
        img.SetSpacing(
            (float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0]))
        )
        sitk.WriteImage(img, str(path))
        return
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write NIfTI {path}; install nibabel or SimpleITK ({exc})"
        ) from exc


def resolve_data_path(path: Path | str) -> Path:
    """Map WSL ``/mnt/<drive>/...`` paths to Windows when needed."""
    raw = str(path).replace("\\", "/")
    p = Path(path)
    env = (os.environ.get("AGENTIC_RADIOGEN_LIDC_ROOT") or "").strip()
    if env and not p.exists():
        cand = Path(env)
        if cand.exists():
            return cand
    if p.exists():
        return p
    if raw.startswith("/mnt/") and len(raw) > 7 and raw[5].isalpha() and raw[6] == "/":
        mapped = Path(f"{raw[5].upper()}:/") / raw[7:]
        if mapped.exists():
            return mapped
    return p
