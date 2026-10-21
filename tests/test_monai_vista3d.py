from __future__ import annotations

from agentic_radiogen.imaging.tumor_models.monai_vista3d import label_prompt_for_disease
from agentic_radiogen.imaging.tumor_models.registry import get_model


def test_vista3d_in_catalog() -> None:
    spec = get_model("monai_vista3d")
    assert spec is not None
    assert spec.model_id == "monai_vista3d"
    assert spec.runner == "monai_vista3d"
    assert spec.is_tumor_model is True
    assert spec.operational_type == "ready_inference"


def test_label_prompt_defaults_to_lung_tumor() -> None:
    assert label_prompt_for_disease("lung adenocarcinoma (TCGA-LUAD)") == [23]
    assert label_prompt_for_disease("TCGA-LUSC squamous") == [23]
    assert label_prompt_for_disease("hepatocellular carcinoma") == [26]
    assert label_prompt_for_disease("pancreatic adenocarcinoma") == [24]
    assert label_prompt_for_disease("colon cancer") == [27]
