from __future__ import annotations

from pathlib import Path

import numpy as np


def load_dicom_series(series_dir: str | Path) -> np.ndarray:
    """Load a DICOM series directory into a 3D float volume (z, y, x)."""
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
        slice2d = arr * slope + intercept
        if target_shape is None:
            target_shape = slice2d.shape
        if slice2d.shape != target_shape:
            slice2d = _resize_slice(slice2d, target_shape)
        slices.append(slice2d)
    if not slices:
        raise ValueError(f"No usable DICOM slices in {series_dir}")
    return np.stack(slices, axis=0)


def _resize_slice(slice2d: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize so heterogeneous slice sizes can still stack."""
    h, w = slice2d.shape
    th, tw = target
    if (h, w) == (th, tw):
        return slice2d
    ys = (np.linspace(0, h - 1, th)).astype(int)
    xs = (np.linspace(0, w - 1, tw)).astype(int)
    return slice2d[np.ix_(ys, xs)]
