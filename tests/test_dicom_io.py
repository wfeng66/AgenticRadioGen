from __future__ import annotations

import numpy as np
import pytest

from agentic_radiogen.imaging.dicom_io import load_dicom_series


pytest.importorskip("pydicom")


def test_load_dicom_series_resizes_mismatched_slices(tmp_path) -> None:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    def write_slice(name: str, shape: tuple[int, int], z: float) -> None:
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = int(z)
        ds.ImagePositionPatient = [0.0, 0.0, z]
        ds.PixelData = np.zeros(shape, dtype=np.uint16).tobytes()
        ds.Rows, ds.Columns = shape
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        pydicom.dcmwrite(str(tmp_path / name), ds)

    write_slice("a.dcm", (4, 4), 0)
    write_slice("b.dcm", (6, 6), 1)
    vol = load_dicom_series(tmp_path)
    assert vol.shape == (2, 4, 4)


def test_rgb_frames_use_hw3_luminance_not_fake_nhw(tmp_path) -> None:
    """RGB slices are (h,w,3); loader must convert via luminance → (n,h,w), not drop to channel-0 as (n,h,w)."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from agentic_radiogen.imaging.dicom_io import _as_grayscale_hw

    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    gray = _as_grayscale_hw(rgb.astype(np.float32))
    assert gray.shape == (8, 10)
    expected = 0.299 * 10 + 0.587 * 20 + 0.114 * 30
    assert abs(float(gray.mean()) - expected) < 1e-3

    def write_rgb(name: str, z: float) -> None:
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = int(z)
        ds.ImagePositionPatient = [0.0, 0.0, z]
        pix = np.zeros((4, 5, 3), dtype=np.uint8)
        pix[..., 0] = 100
        pix[..., 1] = 0
        pix[..., 2] = 0
        ds.Rows, ds.Columns = 4, 5
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = pix.tobytes()
        pydicom.dcmwrite(str(tmp_path / name), ds)

    write_rgb("r0.dcm", 0)
    write_rgb("r1.dcm", 1)
    vol = load_dicom_series(tmp_path)
    assert vol.ndim == 3
    assert vol.shape == (2, 4, 5)
    # Luminance of pure red (100,0,0), not channel-drop artifact leaving raw 100 on wrong axes.
    assert abs(float(vol.mean()) - 0.299 * 100) < 1e-2
