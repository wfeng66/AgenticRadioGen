#!/usr/bin/env python3
"""Visualize a CT/MR volume with its segmentation mask.

Examples
--------
# From segmentation_log.csv row paths:
python scripts/visualize_segmentation.py \\
  --image /path/to/dicom_series_or_volume \\
  --mask segm_rslt/TCGA-XX-XXXX.nii.gz

# Pick a patient from the log CSV:
python scripts/visualize_segmentation.py \\
  --log segm_rslt/segmentation_log.csv \\
  --patient-id TCGA-50-5936

# Save figure without opening a window:
python scripts/visualize_segmentation.py --log segm_rslt/segmentation_log.csv \\
  --patient-id TCGA-50-5936 --out overlay.png

# Show 8 mask overlays in two rows of four:
python scripts/visualize_segmentation.py --log segm_rslt/segmentation_log.csv \\
  --patient-id TCGA-50-5936 --mpr --out mpr.png

# Fuse a registered PET series onto the CT/MPR mask views:
python scripts/visualize_segmentation.py --log segm_rslt/segmentation_log.csv \\
  --patient-id TCGA-50-5936 --pet-image data_cache/tcia/<PET-SeriesInstanceUID> \\
  --mpr

# Automatically look up and download a paired PET series from the CT study:
python scripts/visualize_segmentation.py --log segm_rslt/segmentation_log.csv \\
  --patient-id TCGA-50-5936 --mpr
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _resolve_path(path: Path | str) -> Path:
    """Map WSL ``/mnt/g/...`` paths to Windows ``G:/...`` when needed."""
    raw = str(path).replace("\\", "/")
    p = Path(path)
    if p.exists():
        return p
    if raw.startswith("/mnt/") and len(raw) > 7 and raw[5].isalpha() and raw[6] == "/":
        drive = raw[5].upper()
        mapped = Path(f"{drive}:/") / raw[7:]
        if mapped.exists():
            return mapped
    return p


def _load_array_zyx(path: Path) -> np.ndarray:
    """Load volume/mask as (z,y,x). Prefers SimpleITK, then nibabel, then npy."""
    path = _resolve_path(path)
    if path.is_dir():
        from agentic_radiogen.imaging.dicom_io import load_dicom_series_with_spacing

        vol, _ = load_dicom_series_with_spacing(path)
        return np.asarray(vol, dtype=np.float32)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    try:
        import SimpleITK as sitk

        return np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(path))), dtype=np.float32)
    except Exception:
        pass
    try:
        import nibabel as nib

        data = np.asanyarray(nib.load(str(path)).dataobj)
        data = np.squeeze(data)
        # NIfTI array is often (x,y,z); convert to (z,y,x)
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        return np.asarray(data, dtype=np.float32)
    except Exception as exc:
        raise SystemExit(f"Cannot load {path}: {exc}") from exc


def _load_volume(path: Path) -> np.ndarray:
    return _load_array_zyx(path)


def _load_volume_and_spacing(
    path: Path,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a volume as (z,y,x) with physical voxel spacing in the same order."""
    path = _resolve_path(path)
    if path.is_dir():
        from agentic_radiogen.imaging.dicom_io import load_dicom_series_with_spacing

        volume, spacing_zyx = load_dicom_series_with_spacing(path)
        return np.asarray(volume, dtype=np.float32), spacing_zyx
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32), (1.0, 1.0, 1.0)
    try:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(path))
        volume = sitk.GetArrayFromImage(image)
        return np.asarray(volume, dtype=np.float32), tuple(
            float(value) for value in reversed(image.GetSpacing())
        )
    except Exception:
        pass
    try:
        import nibabel as nib

        image = nib.load(str(path))
        data = np.squeeze(np.asanyarray(image.dataobj))
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        spacing_xyz = np.linalg.norm(image.affine[:3, :3], axis=0)
        spacing_zyx = tuple(float(value) for value in reversed(spacing_xyz))
        return np.asarray(data, dtype=np.float32), spacing_zyx
    except Exception as exc:
        raise SystemExit(f"Cannot load {path}: {exc}") from exc


def _read_sitk_image(path: Path):
    """Read a volume and its physical geometry; identify DICOM frame of reference."""
    import SimpleITK as sitk

    frame_uid = ""
    is_dicom_series = path.is_dir()
    if is_dicom_series:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or ()
        if len(series_ids) != 1:
            raise ValueError(
                f"Expected one DICOM series in {path}, found {len(series_ids)}"
            )
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(path), series_ids[0]
        )
        if not files:
            raise ValueError(f"No DICOM images found in {path}")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        reader.MetaDataDictionaryArrayUpdateOn()
        image = reader.Execute()
        try:
            import pydicom

            for file_path in files:
                dataset = pydicom.dcmread(
                    file_path,
                    stop_before_pixels=True,
                    specific_tags=["FrameOfReferenceUID"],
                )
                frame_uid = str(getattr(dataset, "FrameOfReferenceUID", "") or "")
                if frame_uid:
                    break
        except ImportError as exc:
            raise ImportError(
                "pydicom is required to verify PET/CT DICOM frame alignment"
            ) from exc
    else:
        image = sitk.ReadImage(str(path))
    return image, frame_uid, is_dicom_series


def _load_pet_in_ct_space(
    ct_path: Path, pet_path: Path, ct_volume: np.ndarray
) -> np.ndarray:
    """Resample PET onto the CT physical grid, rejecting unverified DICOM frames."""
    import SimpleITK as sitk

    ct_image, ct_frame_uid, ct_is_dicom = _read_sitk_image(ct_path)
    pet_image, pet_frame_uid, pet_is_dicom = _read_sitk_image(pet_path)
    if ct_is_dicom and pet_is_dicom:
        if not ct_frame_uid or not pet_frame_uid:
            raise ValueError(
                "Cannot align PET and CT DICOM series: FrameOfReferenceUID is "
                "missing from one or both series"
            )
        if ct_frame_uid != pet_frame_uid:
            raise ValueError(
                "Cannot align PET and CT DICOM series with different "
                "FrameOfReferenceUID values"
            )

    ct_sitk_volume = np.asarray(sitk.GetArrayFromImage(ct_image), dtype=np.float32)
    if ct_sitk_volume.shape == ct_volume.shape and np.allclose(
        ct_sitk_volume, ct_volume, rtol=0, atol=1e-4, equal_nan=True
    ):
        reverse_z = False
    elif ct_sitk_volume.shape == ct_volume[::-1].shape and np.allclose(
        ct_sitk_volume, ct_volume[::-1], rtol=0, atol=1e-4, equal_nan=True
    ):
        reverse_z = True
    else:
        raise ValueError(
            "Cannot align PET to this CT ordering: the DICOM loader's CT array "
            "does not match the SimpleITK physical reference grid"
        )

    aligned_pet = sitk.Resample(
        pet_image,
        ct_image,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    pet_volume = np.asarray(sitk.GetArrayFromImage(aligned_pet), dtype=np.float32)
    if reverse_z:
        pet_volume = pet_volume[::-1]
    if pet_volume.shape != ct_volume.shape:
        raise ValueError(
            f"Resampled PET shape {pet_volume.shape} does not match CT "
            f"shape {ct_volume.shape}"
        )
    return pet_volume


def _overlay_pet(ax, pet_slice: np.ndarray) -> None:
    """Overlay non-background PET uptake in color on a CT slice."""
    finite = pet_slice[np.isfinite(pet_slice)]
    if finite.size == 0:
        return
    low, high = np.percentile(finite, [5, 99])
    if high <= low:
        return
    uptake = np.ma.masked_where(
        ~np.isfinite(pet_slice) | (pet_slice <= low), pet_slice
    )
    ax.imshow(
        uptake,
        cmap="magma",
        alpha=0.4,
        vmin=low,
        vmax=high,
        origin="lower",
    )


def _load_mask(path: Path, volume_shape: tuple[int, ...]) -> np.ndarray:
    path = _resolve_path(path)
    if not path.is_file():
        raise SystemExit(f"Mask not found: {path}")
    mask = (_load_array_zyx(path) > 0).astype(np.float32)
    if mask.shape != volume_shape:
        raise SystemExit(
            f"Mask shape {mask.shape} does not match volume {volume_shape}"
        )
    return mask


def _best_mask_slice(mask: np.ndarray) -> int:
    """Slice with the largest mask area (avoids empty middle-slice views)."""
    if mask.ndim != 3:
        return 0
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    if float(areas.max()) <= 0:
        return int(mask.shape[0] // 2)
    return int(np.argmax(areas))


def _mpr_indices(
    mask: np.ndarray,
    volume_shape: tuple[int, int, int],
    slice_index: int | None = None,
) -> tuple[int, int, int]:
    """Choose one z/y/x point through the mask for orthogonal MPR views."""
    coordinates = np.argwhere(mask > 0)
    if coordinates.size:
        z, y, x = np.rint(coordinates.mean(axis=0)).astype(int)
    else:
        z, y, x = ((size - 1) // 2 for size in volume_shape)
    if slice_index is not None:
        z = int(slice_index)
    return tuple(
        max(0, min(int(index), size - 1))
        for index, size in zip((z, y, x), volume_shape)
    )


def _mask_slice_indices(
    mask: np.ndarray,
    axis: int,
    count: int,
    preferred_index: int | None = None,
) -> list[int]:
    """Choose slices across a mask's extent, preferring dense planes."""
    foreground = np.asarray(mask) > 0
    axis = int(axis)
    if axis < 0:
        axis += foreground.ndim
    if foreground.ndim != 3 or axis not in (0, 1, 2):
        raise ValueError(f"Expected a 3D mask and axis 0, 1, or 2; got {axis}")
    if count < 1:
        raise ValueError(f"count must be positive; got {count}")

    areas = foreground.sum(axis=tuple(i for i in range(3) if i != axis))
    available = np.flatnonzero(areas > 0)
    if available.size == 0:
        return [int(round(value)) for value in np.linspace(0, foreground.shape[axis] - 1, count)]

    groups = np.array_split(available, min(count, available.size))
    indices = [
        int(group[np.argmax(areas[group])])
        for group in groups
        if group.size
    ]
    if preferred_index is not None:
        preferred = max(0, min(int(preferred_index), foreground.shape[axis] - 1))
        if preferred not in indices:
            replace_at = int(np.argmin([abs(index - preferred) for index in indices]))
            indices[replace_at] = preferred
    indices = sorted(set(indices))
    while len(indices) < count:
        indices.append(indices[int(np.argmax(areas[indices]))])
    return sorted(indices)


def _mpr_aspects(
    spacing_zyx: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return display pixel aspect ratios for axial, coronal, sagittal (z,y,x)."""
    dz, dy, dx = (max(float(value), 1e-6) for value in spacing_zyx)
    return dy / dx, dz / dx, dz / dy


def _window_ct(slice2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice2d[np.isfinite(slice2d)], [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(slice2d)), float(np.nanmax(slice2d))
    if hi <= lo:
        return np.zeros_like(slice2d, dtype=np.float32)
    out = (slice2d - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def _open_csv_text(path: Path):
    """Open a CSV trying UTF-8 first, then common Windows / legacy encodings."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk", "cp1252", "latin-1"):
        try:
            raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        return path.open(encoding=encoding, newline="")
    return path.open(encoding="utf-8", errors="replace", newline="")


def _row_from_log(log_path: Path, patient_id: str | None) -> dict[str, str]:
    with _open_csv_text(log_path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty log: {log_path}")
    if patient_id:
        hits = [r for r in rows if r.get("patient_id") == patient_id]
        if not hits:
            raise SystemExit(f"patient_id={patient_id} not in {log_path}")
        return hits[0]
    # Prefer first successful row with a mask
    for r in rows:
        if (r.get("mask_path") or "").strip() and str(r.get("status", "")).startswith(
            "success"
        ):
            return r
    return rows[0]


def _series_uid_from_row(row: dict[str, str], image_path: Path | None = None) -> str:
    """Best-effort SeriesInstanceUID from log columns or the DICOM cache path."""
    for key in ("series_uid", "SeriesInstanceUID", "tcia_series_uid"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raw = str(row.get("image_path") or "")
    if raw.startswith("series_uid="):
        return raw.split("=", 1)[1].strip()
    path = _resolve_path(image_path or Path(raw)) if (image_path or raw) else None
    if path is None:
        return ""
    # TCIA downloads land in data_cache/tcia/<SeriesInstanceUID>/
    name = path.name if path.is_dir() else path.parent.name
    if name.startswith("1.") or len(name) > 20:
        return name
    return name


def _dicom_series_identity(path: Path) -> tuple[str, str]:
    """Return PatientID and StudyInstanceUID from one DICOM in a series."""
    import pydicom
    from pydicom.errors import InvalidDicomError

    candidates = [path] if path.is_file() else path.rglob("*")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            dataset = pydicom.dcmread(
                str(candidate),
                stop_before_pixels=True,
                specific_tags=["PatientID", "StudyInstanceUID"],
            )
        except (InvalidDicomError, OSError, EOFError, ValueError):
            continue
        patient_id = str(getattr(dataset, "PatientID", "") or "").strip()
        study_uid = str(getattr(dataset, "StudyInstanceUID", "") or "").strip()
        if patient_id or study_uid:
            return patient_id, study_uid
    return "", ""


def _discover_pet_series(
    *,
    image_path: Path,
    patient_id: str = "",
    series_uid: str = "",
) -> tuple[Path | None, str]:
    """Find and download a PT series from the CT's study, if one is available."""
    image_path = _resolve_path(image_path)
    dicom_patient_id, study_uid = _dicom_series_identity(image_path)
    patient_id = (patient_id or dicom_patient_id).strip()
    if not patient_id:
        return None, "PET auto-discovery needs a patient ID or a DICOM CT series"

    from agentic_radiogen.data.tcia_client import TciaClient

    cache_dir = next(
        (
            parent
            for parent in (image_path, *image_path.parents)
            if parent.name.lower() == "tcia"
        ),
        Path.cwd() / "data_cache" / "tcia",
    )
    tcia = TciaClient(cache_dir=cache_dir)
    series = tcia.list_series_for_patient(patient_id)
    if not study_uid and series_uid:
        study_uid = next(
            (
                str((item.source_metadata or {}).get("StudyInstanceUID") or "").strip()
                for item in series
                if item.series_uid == series_uid and item.modality.upper() == "CT"
            ),
            "",
        )
    if not study_uid:
        return None, (
            f"could not determine StudyInstanceUID for CT series {series_uid or image_path}"
        )

    pet_series = [
        item
        for item in series
        if item.modality.upper() in {"PT", "PET"}
        and str((item.source_metadata or {}).get("StudyInstanceUID") or "").strip()
        == study_uid
    ]
    if not pet_series:
        return None, (
            f"TCIA has no PET/PT series in CT study {study_uid} for {patient_id}"
        )
    pet_series.sort(key=lambda item: (-item.image_count, item.series_uid))
    selected = pet_series[0]
    pet_path = tcia.download_series(selected.series_uid)
    return pet_path, (
        f"using PET series {selected.series_uid} from CT study {study_uid}"
    )


def visualize(
    *,
    image_path: Path,
    mask_path: Path,
    slice_index: int | None = None,
    out: Path | None = None,
    title: str = "",
    img_size: str = "small",
    mpr: bool = False,
    series_uid: str = "",
    pet_image_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    image_path = _resolve_path(image_path)
    mask_path = _resolve_path(mask_path)
    volume, spacing_zyx = _load_volume_and_spacing(image_path)
    if volume.ndim != 3:
        raise SystemExit(f"Expected 3D volume, got shape {volume.shape}")
    pet_volume = None
    if pet_image_path is not None:
        pet_image_path = _resolve_path(pet_image_path)
        try:
            pet_volume = _load_pet_in_ct_space(image_path, pet_image_path, volume)
        except (ImportError, ValueError, RuntimeError) as exc:
            raise SystemExit(f"Cannot display PET with CT: {exc}") from exc
    mask = _load_mask(mask_path, volume.shape)
    n_vox = int(mask.sum())
    nonempty = np.where(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)[0]
    uid = (series_uid or "").strip() or _series_uid_from_row({}, image_path)
    if mpr:
        axial_indices = _mask_slice_indices(
            mask, axis=0, count=4, preferred_index=slice_index
        )
        coronal_indices = _mask_slice_indices(mask, axis=1, count=2)
        sagittal_indices = _mask_slice_indices(mask, axis=2, count=2)
        z = axial_indices[0]
    elif slice_index is not None:
        z = int(slice_index)
    else:
        z = _best_mask_slice(mask)
    if not mpr:
        z = max(0, min(z, volume.shape[0] - 1))

    img = _window_ct(volume[z])
    msk = mask[z]
    slice_vox = int(msk.sum())
    if uid:
        print(f"series_uid={uid}")
    if mpr:
        print(
            f"mask voxels={n_vox} on {len(nonempty)}/{volume.shape[0]} slices; "
            f"showing axial z={axial_indices}, coronal y={coronal_indices}, "
            f"sagittal x={sagittal_indices}"
        )
    else:
        print(
            f"mask voxels={n_vox} on {len(nonempty)}/{volume.shape[0]} slices "
            f"(z={list(nonempty[:8])}{'...' if len(nonempty) > 8 else ''}); "
            f"showing z={z} (slice voxels={slice_vox})"
        )
    if n_vox == 0:
        print("WARNING: mask file has zero positive voxels (true empty segmentation).")
    elif not mpr and slice_vox == 0:
        print(
            "WARNING: chosen slice has no mask; pass --slice with a nonempty index."
        )

    if mpr:
        figsize = {"small": (16, 8), "mid": (20, 10), "big": (24, 12)}[img_size]
        fig, axes = plt.subplots(2, 4, figsize=figsize, squeeze=False)
        axial_aspect, coronal_aspect, sagittal_aspect = _mpr_aspects(spacing_zyx)

        mpr_views = [
            (
                axes[0, position],
                volume[index, :, :],
                mask[index, :, :],
                f"Axial (z={index})",
                axial_aspect,
                pet_volume[index, :, :] if pet_volume is not None else None,
            )
            for position, index in enumerate(axial_indices)
        ]
        mpr_views.extend(
            (
                axes[1, position],
                volume[:, index, :],
                mask[:, index, :],
                f"Coronal (y={index})",
                coronal_aspect,
                pet_volume[:, index, :] if pet_volume is not None else None,
            )
            for position, index in enumerate(coronal_indices)
        )
        mpr_views.extend(
            (
                axes[1, position + len(coronal_indices)],
                volume[:, :, index],
                mask[:, :, index],
                f"Sagittal (x={index})",
                sagittal_aspect,
                pet_volume[:, :, index] if pet_volume is not None else None,
            )
            for position, index in enumerate(sagittal_indices)
        )
        for ax, view, view_mask, view_title, aspect, pet_view in mpr_views:
            ax.imshow(_window_ct(view), cmap="gray", origin="lower", aspect=aspect)
            if pet_view is not None:
                _overlay_pet(ax, pet_view)
            ax.imshow(
                np.ma.masked_where(view_mask < 0.5, view_mask),
                cmap="autumn",
                alpha=0.45,
                origin="lower",
                aspect=aspect,
            )
            pet_label = " | PET overlay" if pet_volume is not None else ""
            ax.set_title(
                f"{view_title} | mask voxels={int(view_mask.sum())}{pet_label}"
            )
    else:
        figsize = {"small": (8, 2.7), "mid": (12, 4), "big": (18, 6)}[img_size]
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(img, cmap="gray", origin="lower")
        axes[0].set_title("Image")
        axes[1].imshow(msk, cmap="Reds", origin="lower", vmin=0, vmax=1)
        axes[1].set_title("Mask")
        axes[2].imshow(img, cmap="gray", origin="lower")
        if pet_volume is not None:
            _overlay_pet(axes[2], pet_volume[z])
        axes[2].imshow(
            np.ma.masked_where(msk < 0.5, msk),
            cmap="autumn",
            alpha=0.45,
            origin="lower",
        )
        axes[2].set_title("Overlay")
    uid_bit = f"  |  series_uid={uid}" if uid else ""
    if mpr:
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(
            (title or f"{image_path.name}")
            + uid_bit
            + f"  |  MPR axial={axial_indices}, coronal={coronal_indices}, "
            + f"sagittal={sagittal_indices}  |  mask_voxels={n_vox}"
        )
    else:
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            (title or f"{image_path.name}")
            + uid_bit
            + f"  |  z={z}/{volume.shape[0]-1}  |  mask_voxels={n_vox}"
        )
    fig.tight_layout()
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Wrote {out.resolve()}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize image + segmentation mask")
    parser.add_argument("--image", type=Path, default=None, help="DICOM series dir or .nii/.npy")
    parser.add_argument(
        "--pet-image",
        type=Path,
        default=None,
        help=(
            "Paired PET DICOM series directory or volume; if omitted, the viewer "
            "looks up and downloads PET/PT from the CT study."
        ),
    )
    parser.add_argument(
        "--no-auto-pet",
        action="store_true",
        help="Do not look up or download a paired PET series when --pet-image is omitted.",
    )
    parser.add_argument("--mask", type=Path, default=None, help="Mask .nii.gz or .npy")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="segmentation_log.csv (use with --patient-id or first success row)",
    )
    parser.add_argument("--patient-id", default=None, help="Patient id from the log CSV")
    parser.add_argument("--slice", type=int, default=None, dest="slice_index")
    parser.add_argument(
        "--mpr",
        action="store_true",
        help="Show 8 mask overlays: four axial, two coronal, and two sagittal slices.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Save PNG instead of showing")
    parser.add_argument(
        "--img_size",
        choices=("small", "mid", "big"),
        default="small",
        help="Visualization size",
    )
    args = parser.parse_args()

    image_path = args.image
    mask_path = args.mask
    title = ""
    series_uid = ""
    patient_id = ""
    if args.log:
        row = _row_from_log(args.log, args.patient_id)
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"]) if row.get("mask_path") else None
        series_uid = _series_uid_from_row(row, image_path)
        patient_id = str(row.get("patient_id") or args.patient_id or "").strip()
        title = (
            f"{row.get('patient_id')} | {row.get('diagnosis')} | "
            f"model={row.get('segmentation_model')} | {row.get('status')}"
        )
        if not mask_path or not str(mask_path).strip():
            raise SystemExit(
                f"No mask_path in log for {row.get('patient_id')}. "
                "Re-run with --store-segm."
            )
    if image_path is None or mask_path is None:
        raise SystemExit("Provide --image and --mask, or --log [--patient-id]")
    if not series_uid:
        series_uid = _series_uid_from_row({}, image_path)
    pet_image_path = args.pet_image
    if pet_image_path is None and not args.no_auto_pet:
        try:
            pet_image_path, pet_message = _discover_pet_series(
                image_path=image_path,
                patient_id=patient_id,
                series_uid=series_uid,
            )
        except Exception as exc:
            pet_message = f"PET auto-discovery failed: {type(exc).__name__}: {exc}"
            print(f"WARNING: {pet_message}", file=sys.stderr)
        else:
            if pet_image_path is None:
                print(f"WARNING: {pet_message}", file=sys.stderr)
            else:
                print(f"PET: {pet_message}")

    visualize(
        image_path=image_path,
        mask_path=mask_path,
        slice_index=args.slice_index,
        out=args.out,
        title=title,
        img_size=args.img_size,
        mpr=args.mpr,
        series_uid=series_uid,
        pet_image_path=pet_image_path,
    )


if __name__ == "__main__":
    main()
