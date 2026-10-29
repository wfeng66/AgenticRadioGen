import numpy as np
import pytest

from scripts import visualize_segmentation
from scripts.visualize_segmentation import _mpr_aspects, _mpr_indices


def test_mpr_indices_use_mask_center():
    mask = np.zeros((8, 10, 12), dtype=np.uint8)
    mask[2:5, 3:6, 8:11] = 1

    assert _mpr_indices(mask, mask.shape) == (3, 4, 9)


def test_mpr_indices_allow_axial_slice_override():
    mask = np.zeros((8, 10, 12), dtype=np.uint8)
    mask[2:5, 3:6, 8:11] = 1

    assert _mpr_indices(mask, mask.shape, slice_index=99) == (7, 4, 9)


def test_mpr_indices_fall_back_to_volume_center_without_mask():
    mask = np.zeros((8, 10, 12), dtype=np.uint8)

    assert _mpr_indices(mask, mask.shape) == (3, 4, 5)


def test_mpr_aspects_use_physical_voxel_spacing():
    assert _mpr_aspects((5.0, 0.8, 0.6)) == pytest.approx((4 / 3, 25 / 3, 6.25))


def test_mpr_visualization_writes_four_panel_figure(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    volume = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    mask = np.zeros_like(volume)
    mask[2:5, 3:6, 8:11] = 1
    monkeypatch.setattr(
        visualize_segmentation,
        "_load_volume_and_spacing",
        lambda _: (volume, (5.0, 0.8, 0.6)),
    )
    monkeypatch.setattr(visualize_segmentation, "_load_mask", lambda *_: mask)
    output = tmp_path / "mpr.png"

    visualize_segmentation.visualize(
        image_path=tmp_path / "ct.npy",
        mask_path=tmp_path / "mask.npy",
        out=output,
        mpr=True,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
