from __future__ import annotations

from pathlib import Path

from agentic_radiogen.data.live_catalog import LiveCatalog, _SEGM_LOG_FIELDS


def test_write_segmentation_log_csv(tmp_path: Path) -> None:
    catalog = LiveCatalog(store_segm=True, segm_dir=tmp_path / "segm_rslt")
    catalog.last_segmentation_log = [
        {
            "patient_id": "P1",
            "image_path": "/data/P1",
            "diagnosis": "adenocarcinoma",
            "segmentation_model": "nnunet_msd_lung",
            "models_attempted": "nnunet_msd_lung:ok",
            "mask_path": str(tmp_path / "segm_rslt" / "P1.nii.gz"),
            "n_mask_voxels": "1200",
            "status": "success",
        },
        {
            "patient_id": "P2",
            "image_path": "/data/P2",
            "diagnosis": "squamous",
            "segmentation_model": "",
            "models_attempted": "nnunet_msd_lung:fail:empty|monai_lung_nodule:fail:empty",
            "mask_path": "",
            "n_mask_voxels": "0",
            "status": "failure: empty mask",
        },
    ]
    path = catalog._write_segmentation_log()
    assert path is not None
    text = Path(path).read_text(encoding="utf-8")
    for col in _SEGM_LOG_FIELDS:
        assert col in text.splitlines()[0]
    assert "P1" in text and "nnunet_msd_lung" in text
    assert "failure: empty mask" in text
