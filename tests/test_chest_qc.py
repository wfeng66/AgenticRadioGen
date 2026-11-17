"""Content-based chest CT QC via --organ-fraction (high-air slice fraction)."""

from __future__ import annotations

import numpy as np

from agentic_radiogen.imaging.chest_qc import (
    DEFAULT_ORGAN_FRACTION,
    disease_requires_chest_ct,
    fraction_high_air_slices,
    series_meets_organ_fraction,
    volume_looks_like_chest_ct,
)


def _soft_abdomen(z: int = 80, y: int = 64, x: int = 64) -> np.ndarray:
    vol = np.full((z, y, x), 40.0, dtype=np.float32)
    vol[:, :4, :] = -1000
    vol[:, -4:, :] = -1000
    vol[:, :, :4] = -1000
    vol[:, :, -4:] = -1000
    vol[0:4, 16:48, 8:24] = -800
    vol[0:4, 16:48, 40:56] = -800
    rng = np.random.default_rng(0)
    for zi in range(10, z):
        for _ in range(3):
            yy = int(rng.integers(20, 44))
            xx = int(rng.integers(20, 44))
            vol[zi, yy : yy + 2, xx : xx + 2] = -900
    return vol


def _chest_volume(z: int = 80, y: int = 64, x: int = 64) -> np.ndarray:
    vol = np.full((z, y, x), 40.0, dtype=np.float32)
    vol[:, :4, :] = -1000
    vol[:, -4:, :] = -1000
    vol[5:70, 12:52, 6:26] = -750
    vol[5:70, 12:52, 38:58] = -750
    return vol


def _long_torso_sparse_lung(
    z: int = 200, y: int = 64, x: int = 64, *, lung_slices: int = 40
) -> np.ndarray:
    vol = np.full((z, y, x), 40.0, dtype=np.float32)
    vol[:, :4, :] = -1000
    vol[:, -4:, :] = -1000
    vol[0:lung_slices, 12:52, 6:26] = -750
    vol[0:lung_slices, 12:52, 38:58] = -750
    return vol


def test_default_organ_fraction() -> None:
    assert DEFAULT_ORGAN_FRACTION == 0.3


def test_chest_volume_passes() -> None:
    vol = _chest_volume()
    assert fraction_high_air_slices(vol) >= 0.3
    assert volume_looks_like_chest_ct(vol) is True


def test_abdomen_with_lung_bases_fails() -> None:
    vol = _soft_abdomen()
    assert fraction_high_air_slices(vol) < 0.3
    assert volume_looks_like_chest_ct(vol) is False


def test_long_torso_sparse_lung_fails_at_default_and_at_0_5() -> None:
    """TCGA-34-5240-class: lungs present but organ-fraction below 0.3 / 0.5."""
    vol = _long_torso_sparse_lung(z=200, lung_slices=40)
    frac = fraction_high_air_slices(vol)
    assert 0.15 <= frac < 0.3
    assert volume_looks_like_chest_ct(vol) is False
    assert volume_looks_like_chest_ct(vol, min_high_air_slice_fraction=0.5) is False
    assert volume_looks_like_chest_ct(vol, min_high_air_slice_fraction=0.18) is True


def test_disease_requires_chest_for_luad() -> None:
    assert disease_requires_chest_ct(
        "lung cancer; lung adenocarcinoma (TCGA-LUAD)"
    )
    assert not disease_requires_chest_ct("glioblastoma")
