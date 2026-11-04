import sys

import numpy as np
import pytest

from scripts import visualize_segmentation
from scripts.visualize_segmentation import (
    _discover_pet_series,
    _mask_slice_indices,
    _mpr_aspects,
    _mpr_indices,
)
from agentic_radiogen.data.tcia_client import TciaSeries


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
    captured_axes_layouts = []
    captured_axes = []

    def capture_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        captured_axes_layouts.append(axes.shape)
        captured_axes.append(axes)
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
    assert captured_axes_layouts == [(2, 4)]
    titles = [[axis.get_title() for axis in row] for row in captured_axes[0]]
    assert all(title.startswith("Axial") for title in titles[0])
    assert titles[1][0].startswith("Coronal")
    assert titles[1][1].startswith("Coronal")
    assert titles[1][2].startswith("Sagittal")
    assert titles[1][3].startswith("Sagittal")


def test_pet_volume_is_overlaid_on_ct_mpr_views(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volume = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    mask = np.zeros_like(volume)
    mask[2:5, 3:6, 8:11] = 1
    pet = np.ones_like(volume)
    monkeypatch.setattr(
        visualize_segmentation,
        "_load_volume_and_spacing",
        lambda _: (volume, (5.0, 0.8, 0.6)),
    )
    monkeypatch.setattr(visualize_segmentation, "_load_mask", lambda *_: mask)
    monkeypatch.setattr(
        visualize_segmentation,
        "_load_pet_in_ct_space",
        lambda *_: pet,
    )
    overlays: list[tuple[int, ...]] = []

    def record_overlay(_axis, pet_slice):
        overlays.append(pet_slice.shape)

    monkeypatch.setattr(visualize_segmentation, "_overlay_pet", record_overlay)
    output = tmp_path / "pet_mpr.png"

    visualize_segmentation.visualize(
        image_path=tmp_path / "ct",
        pet_image_path=tmp_path / "pet",
        mask_path=tmp_path / "mask.nii.gz",
        out=output,
        mpr=True,
    )

    assert output.is_file()
    assert len(overlays) == 8
    assert all(len(shape) == 2 for shape in overlays)


def test_pet_is_resampled_to_ct_physical_grid(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from scripts.visualize_segmentation import _load_pet_in_ct_space

    ct_array = np.arange(6 * 8 * 10, dtype=np.float32).reshape(6, 8, 10)
    pet_array = np.full(ct_array.shape, 7.5, dtype=np.float32)
    ct_image = sitk.GetImageFromArray(ct_array)
    pet_image = sitk.GetImageFromArray(pet_array)
    for image in (ct_image, pet_image):
        image.SetSpacing((0.7, 0.8, 2.0))
        image.SetOrigin((12.0, -4.0, 30.0))
    ct_path = tmp_path / "ct.nii.gz"
    pet_path = tmp_path / "pet.nii.gz"
    sitk.WriteImage(ct_image, str(ct_path))
    sitk.WriteImage(pet_image, str(pet_path))

    aligned = _load_pet_in_ct_space(ct_path, pet_path, ct_array)

    np.testing.assert_allclose(aligned, pet_array)


def test_pet_dicom_with_different_frame_uid_is_rejected(tmp_path, monkeypatch):
    sitk = pytest.importorskip("SimpleITK")
    ct_image = sitk.GetImageFromArray(np.zeros((2, 3, 4), dtype=np.float32))
    pet_image = sitk.GetImageFromArray(np.ones((2, 3, 4), dtype=np.float32))
    images = {
        tmp_path / "ct": (ct_image, "frame-ct", True),
        tmp_path / "pet": (pet_image, "frame-pet", True),
    }
    monkeypatch.setattr(
        visualize_segmentation,
        "_read_sitk_image",
        lambda path: images[path],
    )

    with pytest.raises(ValueError, match="different FrameOfReferenceUID"):
        visualize_segmentation._load_pet_in_ct_space(
            tmp_path / "ct", tmp_path / "pet", np.zeros((2, 3, 4))
        )


def test_pet_auto_discovery_downloads_largest_series_from_ct_study(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "data_cache" / "tcia"
    ct_path = cache_dir / "ct-series"
    ct_path.mkdir(parents=True)
    calls = []

    class FakeTciaClient:
        def __init__(self, *, cache_dir):
            assert cache_dir == cache_dir_arg

        def list_series_for_patient(self, patient_id):
            assert patient_id == "TCGA-17-Z017"
            return [
                TciaSeries(
                    patient_id,
                    "pet-other-study",
                    "PT",
                    "TCGA-LUAD",
                    image_count=100,
                    source_metadata={"StudyInstanceUID": "other-study"},
                ),
                TciaSeries(
                    patient_id,
                    "pet-small",
                    "PT",
                    "TCGA-LUAD",
                    image_count=12,
                    source_metadata={"StudyInstanceUID": "ct-study"},
                ),
                TciaSeries(
                    patient_id,
                    "pet-large",
                    "PT",
                    "TCGA-LUAD",
                    image_count=30,
                    source_metadata={"StudyInstanceUID": "ct-study"},
                ),
            ]

        def download_series(self, series_uid):
            calls.append(series_uid)
            path = cache_dir / series_uid
            path.mkdir()
            return path

    cache_dir_arg = cache_dir
    monkeypatch.setattr(
        visualize_segmentation,
        "_dicom_series_identity",
        lambda _: ("TCGA-17-Z017", "ct-study"),
    )
    monkeypatch.setattr(
        "agentic_radiogen.data.tcia_client.TciaClient", FakeTciaClient
    )

    pet_path, message = _discover_pet_series(
        image_path=ct_path,
        patient_id="",
        series_uid="ct-series",
    )

    assert pet_path == cache_dir / "pet-large"
    assert calls == ["pet-large"]
    assert "pet-large" in message


def test_visualizer_main_auto_passes_discovered_pet_to_visualize(
    tmp_path, monkeypatch
):
    pet_path = tmp_path / "pet"
    received = {}
    monkeypatch.setattr(sys, "argv", [
        "visualize_segmentation.py",
        "--image",
        str(tmp_path / "ct.nii.gz"),
        "--mask",
        str(tmp_path / "mask.nii.gz"),
    ])
    monkeypatch.setattr(
        visualize_segmentation,
        "_discover_pet_series",
        lambda **_: (pet_path, "using paired PET"),
    )
    monkeypatch.setattr(
        visualize_segmentation,
        "visualize",
        lambda **kwargs: received.update(kwargs),
    )

    visualize_segmentation.main()

    assert received["pet_image_path"] == pet_path
