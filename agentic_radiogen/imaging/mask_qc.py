"""Post-segmentation QC: reject empty, tiny, 1D, or anatomically implausible masks."""

from __future__ import annotations

import re

import numpy as np

# Reject ROIs that PyRadiomics / radiomics cannot use meaningfully.
_MIN_VOXELS = 50
_MIN_ROI_DIMS = 2

_LUNG_DISEASE_RE = re.compile(
    r"\b(?:lung|luad|lusc|nsclc|pulmonary|bronchus|bronchial)\b",
    re.I,
)


def is_lung_disease(disease: str | None) -> bool:
    return bool(_LUNG_DISEASE_RE.search(disease or ""))


def mask_qc_failure(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    disease: str | None = None,
    min_voxels: int = _MIN_VOXELS,
    min_dims: int = _MIN_ROI_DIMS,
) -> str | None:
    """Return a short failure reason, or ``None`` if the mask passes QC."""
    m = np.asarray(mask).astype(bool)
    n = int(m.sum())
    if n <= 0:
        return "empty mask"
    if n < int(min_voxels):
        return f"mask too small ({n} voxels; min {int(min_voxels)})"

    coords = np.argwhere(m)
    extents = coords.max(axis=0) - coords.min(axis=0) + 1
    ndims = int(np.sum(extents > 1))
    if ndims < int(min_dims):
        return (
            f"mask has too few dimensions "
            f"(number of dimensions {ndims}, minimum required {int(min_dims)})"
        )

    if is_lung_disease(disease):
        reason = _lung_parenchyma_overlap_failure(volume, m)
        if reason:
            return reason
    return None


def _lung_parenchyma_overlap_failure(
    volume: np.ndarray,
    mask: np.ndarray,
) -> str | None:
    """Reject masks that barely touch dilated lung air (classic wall FP)."""
    vol = np.asarray(volume, dtype=np.float32)
    if vol.shape != mask.shape:
        return None
    # Approximate lung parenchyma on CT HU.
    lung = (vol > -1000.0) & (vol < -200.0)
    n_lung = int(lung.sum())
    if n_lung < 500:
        # Not a usable chest CT / wrong window — skip anatomy QC.
        return None
    try:
        from scipy import ndimage
    except ImportError:
        lung_dil = lung
    else:
        lung_dil = ndimage.binary_dilation(lung, iterations=4)

    n_mask = int(mask.sum())
    overlap = int(np.logical_and(mask, lung_dil).sum())
    # Require at least 5% of mask voxels (and ≥10 absolute) near lung air.
    need = max(10, int(0.05 * n_mask))
    if overlap < need:
        return (
            f"mask outside lung parenchyma "
            f"(overlap={overlap}/{n_mask}; likely wall/soft-tissue false positive)"
        )
    return None
