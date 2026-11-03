import numpy as np
import pytest

from scripts import visualize_segmentation
from scripts.visualize_segmentation import (
    _mask_slice_indices,
    _mpr_aspects,
    _mpr_indices,
)


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


def test_mpr_slice_indices_select_mask_intersecting_planes_across_each_axis():
    mask = np.zeros((12, 14, 16), dtype=np.uint8)
    mask[2:6, 3:10, 5:12] = 1
    mask[8:10, 11:13, 13:15] = 1

    axial = _mask_slice_indices(mask, axis=0, count=4)
    coronal = _mask_slice_indices(mask, axis=1, count=2)
    sagittal = _mask_slice_indices(mask, axis=2, count=2)

    assert len(axial) == len(set(axial)) == 4
    assert len(coronal) == len(set(coronal)) == 2
    assert len(sagittal) == len(set(sagittal)) == 2
    assert all(mask[index, :, :].sum() > 0 for index in axial)
    assert all(mask[:, index, :].sum() > 0 for index in coronal)
    assert all(mask[:, :, index].sum() > 0 for index in sagittal)


def test_mpr_slice_indices_include_preferred_axial_slice():
    mask = np.zeros((10, 12, 14), dtype=np.uint8)
    mask[2:8, 3:9, 4:10] = 1

    selected = _mask_slice_indices(mask, axis=0, count=4, preferred_index=0)

    assert len(selected) == 4
    assert 0 in selected


def test_mpr_aspects_use_physical_voxel_spacing():
    assert _mpr_aspects((5.0, 0.8, 0.6)) == pytest.approx((4 / 3, 25 / 3, 6.25))


def test_mpr_visualization_writes_eight_view_figure(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volume = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    mask = np.zeros_like(volume)
    mask[2:5, 3:6, 8:11] = 1
    monkeypatch.setattr(
        visualize_segmentation,
        "_load_volume_and_spacing",
        lambda _: (volume, (5.0, 0.8, 0.6)),
    )
    monkeypatch.setattr(visualize_segmentation, "_load_mask", lambda *_: mask)
    original_subplots = plt.subplots
    captured_axes_count = []

    def capture_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        captured_axes_count.append(axes.size)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    output = tmp_path / "mpr.png"

    visualize_segmentation.visualize(
        image_path=tmp_path / "ct.npy",
        mask_path=tmp_path / "mask.npy",
        out=output,
        mpr=True,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert captured_axes_count == [8]
