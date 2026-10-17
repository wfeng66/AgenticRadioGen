from __future__ import annotations

import pytest

from agentic_radiogen.imaging.tumor_models.lungtumormask import (
    _cli_argv,
    apply_monai_addchanneld_shim,
)
from agentic_radiogen.imaging.tumor_models.registry import (
    get_model,
    list_models_for_disease,
)
from agentic_radiogen.imaging.tumor_models.segmentation_select import is_selectable


def test_lungtumormask_cli_uses_shim_module_entry() -> None:
    argv = _cli_argv("lungtumormask")
    assert argv[1] == "-m"
    assert argv[2] == "agentic_radiogen.imaging.tumor_models.lungtumormask"


def test_monai_addchanneld_shim_allows_import() -> None:
    """LungTumorMask 1.3.1 needs AddChanneld / alias / export; MONAI≥1.0 removed them."""
    pytest.importorskip("monai")
    apply_monai_addchanneld_shim()
    from monai.transforms import AddChanneld, LoadImaged  # type: ignore[attr-defined]
    import monai.utils as mu

    assert AddChanneld is not None
    assert callable(mu.alias)
    assert callable(mu.export)
    # LoadImaged must keep meta_dict for LungTumorMask pixdim/affine reads.
    assert getattr(LoadImaged, "_agentic_meta_dict_compat", False)


def test_lungmask_legacy_get_model_shim() -> None:
    pytest.importorskip("lungmask")
    apply_monai_addchanneld_shim()
    import lungmask.mask as lm
    import lungmask.utils as lu

    assert hasattr(lu, "LungLabelsDS_inf")
    # Legacy 2-arg form used by LungTumorMask must resolve without FileNotFoundError.
    assert getattr(lm.get_model, "_agentic_legacy_compat", False)


def test_lungtumormask_in_lung_ct_catalog() -> None:
    spec = get_model("lungtumormask")
    assert spec is not None
    assert spec.runner == "lungtumormask"
    assert spec.is_tumor_model
    assert spec.direct_inference
    assert is_selectable(spec)

    ids = [s.model_id for s in list_models_for_disease("lung_cancer", "CT", direct_only=True)]
    assert "nnunet_msd_lung" in ids
    assert "monai_lung_nodule" in ids
    assert "lungtumormask" in ids
    # Tumor models should rank ahead of organ ts_lung
    assert ids.index("nnunet_msd_lung") < ids.index("monai_lung_nodule")
    assert ids.index("monai_lung_nodule") < ids.index("lungtumormask")
    assert ids.index("lungtumormask") < ids.index("ts_lung")
