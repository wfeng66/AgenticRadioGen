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
