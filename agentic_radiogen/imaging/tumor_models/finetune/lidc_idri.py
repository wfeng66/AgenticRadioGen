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

import json
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
_SKIP_LOG_NAME = "_skip_log.json"


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
    xml_index, xml_by_patient = _build_xml_indexes(root, verbose=verbose)
    prev_skips = _load_skip_log(cache)
    if prev_skips and verbose:
        log(
            f"[finetune:lidc] Re-checking {len(prev_skips)} patient(s) skipped "
            f"in a previous run (will not trust stale skip / bad cache).",
            enabled=True,
        )

    patients = sorted(
        p for p in root.iterdir() if p.is_dir() and _PATIENT_RE.match(p.name)
    )
    if max_cases is not None:
        patients = patients[: max(0, int(max_cases))]

    cases: list[CasePair] = []
    n_cache_hit = 0
    n_new = 0
    n_retry_ok = 0
    skipped = 0
    skip_reasons: dict[str, int] = {}
    new_skip_log: dict[str, str] = {}
    for i, patient in enumerate(patients, start=1):
        pid = patient.name
        force_retry = pid in prev_skips or pid.upper() in {
            k.upper() for k in prev_skips
        }
        try:
            pair, status = _ensure_patient_pair(
                patient,
                cache=cache,
                xml_index=xml_index,
                patient_xmls=xml_by_patient.get(patient.name.upper(), []),
                verbose=verbose,
                force_retry=force_retry,
            )
        except Exception as exc:
            skipped += 1
            reason = _short_skip_reason(
                exc, has_xml=bool(xml_by_patient.get(patient.name.upper()))
            )
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            new_skip_log[pid] = reason
            log(
                f"[finetune:lidc] skip {patient.name}: {reason}",
                enabled=verbose,
            )
            continue
        if pair is None:
            skipped += 1
            skip_reasons["unknown"] = skip_reasons.get("unknown", 0) + 1
            new_skip_log[pid] = "unknown"
            continue
        cases.append(pair)
        if status == "cache_hit":
            n_cache_hit += 1
        elif status in {"converted", "retry_converted", "revalidated"}:
            n_new += 1
            if force_retry or status == "retry_converted":
                n_retry_ok += 1
        else:
            n_new += 1
        if verbose and (i % 25 == 0 or i == len(patients)):
            log(
                f"[finetune:lidc] progress {i}/{len(patients)}: "
                f"ok={len(cases)} (cache_hit={n_cache_hit}, newly_converted={n_new}, "
                f"retry_ok={n_retry_ok}), skipped={skipped}",
                enabled=True,
            )

    _save_skip_log(cache, new_skip_log)
    if skip_reasons and verbose:
        summary = "; ".join(
            f"{n}× {msg}"
            for msg, n in sorted(skip_reasons.items(), key=lambda x: -x[1])
        )
        log(f"[finetune:lidc] skip reasons: {summary}", enabled=True)
    log(
        f"[finetune:lidc] Ready {len(cases)} labeled cases under {cache} "
        f"(attempted={len(patients)}, cache_hit={n_cache_hit}, "
        f"newly_converted={n_new}, retry_ok={n_retry_ok}, skipped={skipped}). "
        f"Source DICOM/XML trees are read-only; only {cache.name}/ is written. "
        f"Previous skips are always re-tried; empty/invalid cache is rebuilt.",
        enabled=verbose,
    )
    return cases

def _short_skip_reason(exc: BaseException, *, has_xml: bool) -> str:
    msg = str(exc)
    if "no CT series with LIDC XML" in msg or "no matching" in msg.lower():
        if has_xml:
            return "XML exists in manifest but could not pair to a CT series"
        return "no XML annotation in manifest for this patient (not downloaded / not present)"
    if "empty nodule mask" in msg:
        return "XML present but no mappable inclusion contour/point on CT"
    if "too few" in msg:
        return "incomplete DICOM (too few CT slices)"
    return msg[:160]


def _ensure_patient_pair(
    patient: Path,
    *,
    cache: Path,
    xml_index: dict[str, Path],
    patient_xmls: list[Path],
    verbose: bool,
    force_retry: bool = False,
) -> tuple[CasePair | None, str]:
    """Return (case, status) with status in {cache_hit, converted, retry_converted}.

    Never modifies source DICOM/XML; writes only under ``cache``.
    Previously skipped patients (``force_retry``) and empty/invalid caches are
    always reconverted.
    """
    out_dir = cache / patient.name
    image_path = out_dir / "image.nii.gz"
    label_path = out_dir / "label.nii.gz"

    if force_retry:
        _clear_patient_cache(out_dir)
        log(
            f"[finetune:lidc] retry {patient.name}: was skipped previously; "
            f"re-checking series/XML (ignoring old cache)",
            enabled=verbose,
        )
    elif image_path.is_file() and label_path.is_file():
        if _cached_label_is_valid(label_path):
            return (
                CasePair(case_id=patient.name, image=image_path, label=label_path),
                "cache_hit",
            )
        log(
            f"[finetune:lidc] revalidate {patient.name}: cached label empty/invalid; "
            f"rebuilding",
            enabled=verbose,
        )
        _clear_patient_cache(out_dir)

    series_dir, xml_path = _select_series_and_xml(
        patient, xml_index=xml_index, patient_xmls=patient_xmls
    )
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
        spacing_zyx=spacing,
    )
    if not mask.any():
        raise RuntimeError(
            f"empty nodule mask from {xml_path.name} "
            "(no mappable inclusion contours/points on this CT)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_nifti(image_path, volume.astype(np.float32), spacing_zyx=spacing)
    _save_nifti(label_path, mask.astype(np.uint8), spacing_zyx=spacing)
    status = "retry_converted" if force_retry else "converted"
    log(
        f"[finetune:lidc] convert {patient.name}: series={series_dir.name} "
        f"xml={xml_path.name} shape={volume.shape} tumor_voxels={int(mask.sum())}",
        enabled=verbose,
    )
    return (
        CasePair(case_id=patient.name, image=image_path, label=label_path),
        status,
    )


def _load_skip_log(cache: Path) -> dict[str, str]:
    path = cache / _SKIP_LOG_NAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and k:
            out[k] = str(v)
    return out


def _save_skip_log(cache: Path, skips: dict[str, str]) -> None:
    path = cache / _SKIP_LOG_NAME
    path.write_text(json.dumps(skips, indent=2, sort_keys=True), encoding="utf-8")


def _clear_patient_cache(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for name in ("image.nii.gz", "label.nii.gz", "image.nii", "label.nii"):
        p = out_dir / name
        try:
            p.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            if p.is_file():
                p.unlink()
        except OSError:
            pass
    # Remove empty patient dir; keep if other files exist.
    try:
        if out_dir.is_dir() and not any(out_dir.iterdir()):
            out_dir.rmdir()
    except OSError:
        pass


def _cached_label_is_valid(label_path: Path) -> bool:
    """True when cached label exists and has at least one foreground voxel."""
    try:
        arr, _ = _load_nifti_with_spacing_local(label_path)
        return bool(np.any(arr > 0))
    except Exception:
        return False


def _load_nifti_with_spacing_local(
    path: Path,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Lightweight NIfTI load for cache validation (avoid torch_loop import cycle)."""
    try:
        import nibabel as nib

        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        arr = np.squeeze(np.asarray(data, dtype=np.float32))
        if arr.ndim == 3:
            # Saved as (x,y,z) in this module.
            arr = np.transpose(arr, (2, 1, 0))
        zooms = img.header.get_zooms()[:3]
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
        return arr, spacing
    except Exception:
        pass
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = np.asarray(sitk.GetArrayFromImage(img), dtype=np.float32)
    sp = img.GetSpacing()
    spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
    return np.squeeze(arr), spacing



def _select_series_and_xml(
    patient: Path,
    *,
    xml_index: dict[str, Path],
    patient_xmls: list[Path],
) -> tuple[Path | None, Path | None]:
    series_list = _iter_series_dirs(patient)
    scored: list[tuple[int, Path, Path | None]] = []
    for series in series_list:
        n_dcm = sum(1 for p in series.iterdir() if p.is_file() and _is_dicom_name(p))
        if n_dcm < _MIN_SLICES:
            continue
        xml = _first_xml(series)
        if xml is None:
            xml = _match_xml_from_index(series, xml_index)
        if xml is None:
            xml = _match_xml_for_patient_series(series, patient_xmls)
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


def _build_xml_indexes(
    root: Path, *, verbose: bool
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Map series fingerprints → XML, and patient-id → XML list."""
    index: dict[str, Path] = {}
    by_patient: dict[str, list[Path]] = {}
    search_roots = _annotation_roots(root)
    n_xml = 0
    for base in search_roots:
        if not base.is_dir():
            continue
        for xml in base.rglob("*.xml"):
            n_xml += 1
            series_dir = xml.parent
            for key in _series_keys(series_dir):
                index.setdefault(key, xml)
            # Patient folder is typically .../LIDC-IDRI-XXXX/<study>/<series>/file.xml
            patient_id = _patient_id_from_xml_path(xml)
            if patient_id:
                by_patient.setdefault(patient_id, []).append(xml)
    # De-dupe patient XML lists
    for pid, xmls in list(by_patient.items()):
        uniq: list[Path] = []
        seen: set[str] = set()
        for x in xmls:
            k = str(x.resolve())
            if k in seen:
                continue
            seen.add(k)
            uniq.append(x)
        by_patient[pid] = uniq
    log(
        f"[finetune:lidc] Indexed {len(index)} XML series keys / {n_xml} XML files "
        f"from {len(search_roots)} annotation root(s); "
        f"patients_with_xml={len(by_patient)}",
        enabled=verbose,
    )
    return index, by_patient


def _patient_id_from_xml_path(xml: Path) -> str | None:
    for part in xml.parts:
        if _PATIENT_RE.match(part):
            return part.upper()
    return None


def _annotation_roots(dicom_root: Path) -> list[Path]:
    roots = [dicom_root]
    # Sibling / parent NBIA manifest trees.
    for parent in (dicom_root.parent, dicom_root.parent.parent):
        if not parent.is_dir():
            continue
        try:
            children = list(parent.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            name = child.name.lower()
            if name.startswith("manifest"):
                cand = child / "LIDC-IDRI"
                roots.append(cand if cand.is_dir() else child)
            if name in {"lidc-idri", "lidc_idri"}:
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


def _normalize_series_token(name: str) -> str:
    """Strip TCIA noise so lidc_idri and manifest folder names can match."""
    s = name.lower()
    s = s.replace(".000000", "")
    s = re.sub(r"\bna\b", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _series_keys(series_dir: Path) -> list[str]:
    """Fingerprints shared between flattened lidc_idri and TCIA manifest names."""
    name = series_dir.name
    keys = {
        name.lower(),
        re.sub(r"[^0-9]", "", name),
        _normalize_series_token(name),
    }
    # Leading series id + trailing id, e.g. 5576 … 00410
    nums = re.findall(r"\d+", name)
    if len(nums) >= 2:
        keys.add(nums[0] + nums[-1])
        if len(nums[0]) >= 1 and len(nums[-1]) >= 4:
            keys.add(nums[0] + nums[-1][-5:])
    # e.g. 3000566-03192 and 3000566.000000-NA-03192 → 300056603192
    m = re.search(r"(\d{5,}).*?(\d{4,})$", name)
    if m:
        keys.add(m.group(1) + m.group(2))
    study = series_dir.parent.name
    sm = re.search(r"(\d{4,})$", study)
    if sm and m:
        keys.add(sm.group(1) + m.group(1) + m.group(2))
    if sm and len(nums) >= 2:
        keys.add(sm.group(1) + nums[0] + nums[-1])
    return [k for k in keys if len(k) >= 4]


def _match_xml_from_index(series_dir: Path, xml_index: dict[str, Path]) -> Path | None:
    for key in _series_keys(series_dir):
        hit = xml_index.get(key)
        if hit is not None:
            return hit
    return None


def _match_xml_for_patient_series(
    series_dir: Path, patient_xmls: list[Path]
) -> Path | None:
    """Fuzzy-match a CT series folder to one of this patient's manifest XMLs."""
    if not patient_xmls:
        return None
    series_norm = _normalize_series_token(series_dir.name)
    series_keys = set(_series_keys(series_dir))
    best: tuple[int, Path] | None = None
    for xml in patient_xmls:
        parent = xml.parent
        parent_keys = set(_series_keys(parent))
        parent_norm = _normalize_series_token(parent.name)
        score = 0
        if series_keys & parent_keys:
            score += 100
        if series_norm and parent_norm and (
            series_norm == parent_norm
            or series_norm in parent_norm
            or parent_norm in series_norm
        ):
            score += 50
        # Shared leading+trailing numeric ids
        s_nums = re.findall(r"\d+", series_dir.name)
        p_nums = re.findall(r"\d+", parent.name)
        if s_nums and p_nums and s_nums[0] == p_nums[0] and s_nums[-1] == p_nums[-1]:
            score += 80
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, xml)
    return best[1] if best is not None else None


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
    spacing_zyx: tuple[float, float, float] | None = None,
) -> np.ndarray:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mask = np.zeros(shape_zyx, dtype=np.uint8)
    # ~3 mm ball for LIDC <3mm nodules that are stored as a single point.
    sp = spacing_zyx or (1.0, 1.0, 1.0)
    point_r_yx = max(1, int(round(1.5 / max(float(sp[1]), float(sp[2]), 1e-6))))
    point_r_z = max(0, int(round(1.5 / max(float(sp[0]), 1e-6))))

    # Union of all readers' inclusion ROIs (contours + point marks).
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
        if len(xs) >= 3:
            _fill_polygon(mask[z], xs, ys)
        elif len(xs) >= 1:
            # Small-nodule point mark (<3 mm in LIDC): paint a small disk/ball.
            _paint_point_ball(
                mask,
                z=int(z),
                y=int(ys[0]),
                x=int(xs[0]),
                r_z=point_r_z,
                r_yx=point_r_yx,
            )
    return mask


def _paint_point_ball(
    mask: np.ndarray,
    *,
    z: int,
    y: int,
    x: int,
    r_z: int,
    r_yx: int,
) -> None:
    zz, yy, xx = mask.shape
    z0, z1 = max(0, z - r_z), min(zz, z + r_z + 1)
    y0, y1 = max(0, y - r_yx), min(yy, y + r_yx + 1)
    x0, x1 = max(0, x - r_yx), min(xx, x + r_yx + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return
    zz_i = np.arange(z0, z1)[:, None, None]
    yy_i = np.arange(y0, y1)[None, :, None]
    xx_i = np.arange(x0, x1)[None, None, :]
    # Ellipsoid in index space using radii.
    rz = max(r_z, 0) + 1e-6
    ry = max(r_yx, 1)
    ball = ((zz_i - z) / rz) ** 2 + ((yy_i - y) / ry) ** 2 + ((xx_i - x) / ry) ** 2 <= 1.0
    mask[z0:z1, y0:y1, x0:x1][ball] = 1


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
