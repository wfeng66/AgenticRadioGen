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

# Show axial, coronal, and sagittal MPR overlays alongside the original CT:
python scripts/visualize_segmentation.py --log segm_rslt/segmentation_log.csv \\
  --patient-id TCGA-50-5936 --mpr --out mpr.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


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


def _row_from_log(log_path: Path, patient_id: str | None) -> dict[str, str]:
    rows = list(csv.DictReader(log_path.open(encoding="utf-8")))
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
) -> None:
    import matplotlib.pyplot as plt

    image_path = _resolve_path(image_path)
    mask_path = _resolve_path(mask_path)
    volume, spacing_zyx = _load_volume_and_spacing(image_path)
    if volume.ndim != 3:
        raise SystemExit(f"Expected 3D volume, got shape {volume.shape}")
    mask = _load_mask(mask_path, volume.shape)
    n_vox = int(mask.sum())
    nonempty = np.where(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)[0]
    uid = (series_uid or "").strip() or _series_uid_from_row({}, image_path)
    if mpr:
        z, y, x = _mpr_indices(mask, volume.shape, slice_index)
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
            f"showing MPR through z/y/x={z}/{y}/{x}"
        )
    else:
        print(
            f"mask voxels={n_vox} on {len(nonempty)}/{volume.shape[0]} slices "
            f"(z={list(nonempty[:8])}{'...' if len(nonempty) > 8 else ''}); "
            f"showing z={z} (slice voxels={slice_vox})"
        )
    if n_vox == 0:
        print("WARNING: mask file has zero positive voxels (true empty segmentation).")
    elif slice_vox == 0:
        print(
            "WARNING: chosen slice has no mask; pass --slice with a nonempty index."
        )

    if mpr:
        figsize = {"small": (10, 8), "mid": (14, 10), "big": (18, 13)}[img_size]
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axial_aspect, coronal_aspect, sagittal_aspect = _mpr_aspects(spacing_zyx)
        mpr_views = (
            (volume[z, :, :], mask[z, :, :], f"Axial (z={z})", axial_aspect),
            (volume[:, y, :], mask[:, y, :], f"Coronal (y={y})", coronal_aspect),
            (volume[:, :, x], mask[:, :, x], f"Sagittal (x={x})", sagittal_aspect),
        )
        axes[0, 0].imshow(img, cmap="gray", origin="lower", aspect=axial_aspect)
        axes[0, 0].set_title(f"Original CT — axial (z={z})")
        for ax, (view, view_mask, view_title, aspect) in zip(axes.flat[1:], mpr_views):
            ax.imshow(_window_ct(view), cmap="gray", origin="lower", aspect=aspect)
            ax.imshow(
                np.ma.masked_where(view_mask < 0.5, view_mask),
                cmap="autumn",
                alpha=0.45,
                origin="lower",
                aspect=aspect,
            )
            ax.set_title(view_title)
    else:
        figsize = {"small": (8, 2.7), "mid": (12, 4), "big": (18, 6)}[img_size]
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(img, cmap="gray", origin="lower")
        axes[0].set_title("Image")
        axes[1].imshow(msk, cmap="Reds", origin="lower", vmin=0, vmax=1)
        axes[1].set_title("Mask")
        axes[2].imshow(img, cmap="gray", origin="lower")
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
            + f"  |  MPR z/y/x={z}/{y}/{x}  |  mask_voxels={n_vox}"
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
        help="Show a 4-panel figure: original axial CT plus axial, coronal, and sagittal MPR mask overlays.",
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
    if args.log:
        row = _row_from_log(args.log, args.patient_id)
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"]) if row.get("mask_path") else None
        series_uid = _series_uid_from_row(row, image_path)
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

    visualize(
        image_path=image_path,
        mask_path=mask_path,
        slice_index=args.slice_index,
        out=args.out,
        title=title,
        img_size=args.img_size,
        mpr=args.mpr,
        series_uid=series_uid,
    )


if __name__ == "__main__":
    main()
