from __future__ import annotations

import numpy as np

from agentic_radiogen.imaging.mask_qc import is_lung_disease, mask_qc_failure


def test_is_lung_disease() -> None:
    assert is_lung_disease("lung adenocarcinoma (TCGA-LUAD)")
    assert is_lung_disease("Squamous cell carcinoma; TCGA-LUSC")
    assert not is_lung_disease("hepatocellular carcinoma")


def test_rejects_tiny_and_1d_masks() -> None:
    vol = np.zeros((16, 32, 32), dtype=np.float32)
    tiny = np.zeros_like(vol, dtype=bool)
    tiny[8, 16, 16] = True
    assert "too small" in (mask_qc_failure(vol, tiny) or "")

    line = np.zeros_like(vol, dtype=bool)
    line[8, 16, 10:30] = True  # 20 voxels, 1D along x
    assert "too few dimensions" in (mask_qc_failure(vol, line, min_voxels=10) or "")


def test_rejects_wall_false_positive_on_lung_ct() -> None:
    """Body-wall blob with no lung-air overlap must fail for lung disease."""
    vol = np.full((24, 64, 64), -100.0, dtype=np.float32)  # soft tissue
    # Central lung air
    vol[:, 16:48, 16:48] = -800.0
    mask = np.zeros_like(vol, dtype=bool)
    # Peripheral wall blob (outside lung air box)
    mask[10:14, 2:8, 30:36] = True
    reason = mask_qc_failure(
        vol, mask, disease="lung cancer; lung adenocarcinoma (TCGA-LUAD)"
    )
    assert reason is not None
    assert "outside lung" in reason


def test_accepts_mask_inside_lung() -> None:
    vol = np.full((24, 64, 64), -100.0, dtype=np.float32)
    vol[:, 16:48, 16:48] = -800.0
    mask = np.zeros_like(vol, dtype=bool)
    mask[10:16, 24:36, 24:36] = True  # inside lung air
    assert mask_qc_failure(vol, mask, disease="lung adenocarcinoma") is None
