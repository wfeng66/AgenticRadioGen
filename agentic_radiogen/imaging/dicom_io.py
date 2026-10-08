from __future__ import annotations

from pathlib import Path

import numpy as np


def load_dicom_series(series_dir: str | Path) -> np.ndarray:
    """Load a DICOM series directory into a 3D float volume (z, y, x).

    RGB / multi-channel frames shaped (h, w, 3) are converted per-slice with
    luminance (uses all channels). Never stack to (n, h, w, 3) or drop axes as
    a fake (n, h, w) volume (that triggers TotalSegmentator 4D warnings and
    wrong radiomics).
    """
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required for live DICOM loading") from exc

    root = Path(series_dir)
    files = sorted(root.rglob("*"))
    datasets = []
    for path in files:
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), force=True)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        datasets.append(ds)
    if not datasets:
        raise ValueError(f"No DICOM pixel data in {series_dir}")

    def _sort_key(ds):
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
    target_shape: tuple[int, int] | None = None
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        slice2d = _as_grayscale_hw(arr) * slope + intercept
        if target_shape is None:
            target_shape = slice2d.shape
        if slice2d.shape != target_shape:
            slice2d = _resize_slice(slice2d, target_shape)
        slices.append(slice2d)
    if not slices:
        raise ValueError(f"No usable DICOM slices in {series_dir}")
    volume = np.stack(slices, axis=0)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume (n,h,w); got shape {volume.shape}")
    return volume


def _as_grayscale_hw(arr: np.ndarray) -> np.ndarray:
    """Reduce a frame to (h, w) using (h, w, 3) RGB when present."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # ITU-R BT.601 luminance from (h, w, 3[+A]); uses full RGB, not channel-0 only.
        rgb = arr[..., :3]
        return (
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        ).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        # Rare planar (c, h, w)
        rgb = arr[:3]
        return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(f"Unsupported DICOM pixel array shape {arr.shape}; expected (h,w) or (h,w,3)")


def _resize_slice(slice2d: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize so heterogeneous slice sizes can still stack."""
    h, w = slice2d.shape
    th, tw = target
    if (h, w) == (th, tw):
        return slice2d
    ys = (np.linspace(0, h - 1, th)).astype(int)
    xs = (np.linspace(0, w - 1, tw)).astype(int)
    return slice2d[np.ix_(ys, xs)]
