"""Post-download organ FOV check before tumor segmentation.

Runs TotalSegmentator organ labels for the expected site (e.g. lung for lung
cancer). Series that do not contain enough organ voxels are rejected so the
catalog can try the next TCIA candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Absolute organ voxels required to accept a series (after TS organ mask).
_MIN_ORGAN_VOXELS = 1000


@dataclass(frozen=True)
class OrganVerifyResult:
    ok: bool
    organ_keywords: tuple[str, ...]
    n_organ_voxels: int
    volume_voxels: int
    reason: str = ""


def verify_expected_organ(
    series_dir: str | Path,
    *,
    organ_keywords: tuple[str, ...] | list[str],
    min_voxels: int = _MIN_ORGAN_VOXELS,
) -> OrganVerifyResult:
    """Return whether the DICOM series FOV includes the expected organ(s)."""
    organs = tuple(str(k).strip().lower() for k in organ_keywords if str(k).strip())
    if not organs:
        return OrganVerifyResult(
            ok=True,
            organ_keywords=(),
            n_organ_voxels=0,
            volume_voxels=0,
            reason="no expected organ keywords; organ gate skipped",
        )
    from agentic_radiogen.imaging.dicom_io import load_dicom_series_with_spacing
    from agentic_radiogen.imaging.segment import _totalsegmentator_mask

    volume, spacing_zyx = load_dicom_series_with_spacing(series_dir)
    volume_voxels = int(np.prod(volume.shape))
    try:
        mask = _totalsegmentator_mask(
            volume,
            organ_keywords=organs,
            task="total",
            spacing_zyx=spacing_zyx,
        )
    except Exception as exc:
        return OrganVerifyResult(
            ok=False,
            organ_keywords=organs,
            n_organ_voxels=0,
            volume_voxels=volume_voxels,
            reason=f"organ segmentation failed: {exc}",
        )
    n_vox = int(np.count_nonzero(mask))
    if n_vox < int(min_voxels):
        return OrganVerifyResult(
            ok=False,
            organ_keywords=organs,
            n_organ_voxels=n_vox,
            volume_voxels=volume_voxels,
            reason=(
                f"expected organ {list(organs)} voxels={n_vox} < min={min_voxels} "
                "(series likely wrong body site)"
            ),
        )
    return OrganVerifyResult(
        ok=True,
        organ_keywords=organs,
        n_organ_voxels=n_vox,
        volume_voxels=volume_voxels,
        reason=f"organ {list(organs)} voxels={n_vox}",
    )
