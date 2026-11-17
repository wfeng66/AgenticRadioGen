"""Fast high-air slice fraction check for chest CT FOV (--organ-fraction).

Uses body-masked air (HU ≈ −1000…−400). ``--organ-fraction`` is the minimum
share of slices with substantial intrapulmonary air (default 0.3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_HIGH_AIR_SLICE_THR = 0.30
DEFAULT_ORGAN_FRACTION = 0.3


def fraction_high_air_slices(
    volume: np.ndarray,
    *,
    air_body_threshold: float = _HIGH_AIR_SLICE_THR,
) -> float:
    """Share of slices where air/body ≥ ``air_body_threshold`` (downsampled)."""
    v = np.asarray(volume, dtype=np.float32)
    if v.ndim != 3 or min(v.shape) < 2:
        return 0.0
    z, y, x = v.shape
    ys, xs = max(1, y // 96), max(1, x // 96)
    d = v[:, ::ys, ::xs]
    body = d > -900
    air = body & (d >= -1000) & (d <= -400)
    n = 0
    hit = 0
    for zi in range(d.shape[0]):
        b = int(body[zi].sum())
        if b < 20:
            continue
        n += 1
        if float(air[zi].sum()) / float(b) >= float(air_body_threshold):
            hit += 1
    return float(hit) / float(max(1, n))


def series_organ_fraction(series_dir: str | Path) -> float:
    """High-air slice fraction for a DICOM series directory."""
    from agentic_radiogen.imaging.dicom_io import load_dicom_series_with_spacing

    vol, _spacing = load_dicom_series_with_spacing(series_dir)
    return fraction_high_air_slices(vol)


def series_meets_organ_fraction(
    series_dir: str | Path,
    *,
    organ_fraction: float = DEFAULT_ORGAN_FRACTION,
) -> tuple[bool, float]:
    """Return ``(ok, high_air_slice_frac)`` vs ``--organ-fraction``."""
    frac = series_organ_fraction(series_dir)
    return frac >= float(organ_fraction), float(frac)


def volume_looks_like_chest_ct(
    volume: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float] | None = None,
    min_high_air_slice_fraction: float = DEFAULT_ORGAN_FRACTION,
) -> bool:
    """True when high-air slice fraction ≥ threshold (alias for organ-fraction)."""
    del spacing_zyx
    return fraction_high_air_slices(volume) >= float(min_high_air_slice_fraction)


def series_looks_like_chest_ct(
    series_dir: str | Path,
    *,
    min_high_air_slice_fraction: float = DEFAULT_ORGAN_FRACTION,
) -> tuple[bool, float]:
    """Alias of :func:`series_meets_organ_fraction`."""
    return series_meets_organ_fraction(
        series_dir, organ_fraction=min_high_air_slice_fraction
    )


def disease_requires_chest_ct(disease_query: str) -> bool:
    """True for lung/chest disease strings that should enforce chest CT content QC."""
    blob = (disease_query or "").lower()
    return any(
        tok in blob
        for tok in (
            "lung",
            "luad",
            "lusc",
            "nsclc",
            "sclc",
            "pulmonary",
            "chest",
            "thorax",
            "bronchus",
        )
    )
