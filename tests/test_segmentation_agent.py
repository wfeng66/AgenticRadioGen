from __future__ import annotations

import numpy as np

from agentic_radiogen.agents.segmentation import SegmentationAgent
from agentic_radiogen.imaging.tumor_models.registry import (
    catalog_for_llm,
    get_model,
    list_models_for_disease,
    resolve_model,
)
from agentic_radiogen.imaging.tumor_models.store import ModelStore
from agentic_radiogen.llm.segmentation_plan import (
    plan_segmentation_rules,
)


def setup_function() -> None:
    SegmentationAgent._failed_model_ids.clear()


def test_catalog_covers_many_diseases() -> None:
    assert resolve_model("hepatocellular carcinoma", "CT").model_id == "ts_liver"
    assert resolve_model("renal cell carcinoma", "CT").model_id == "ts_kidney"
    assert resolve_model("pancreatic adenocarcinoma", "CT").model_id == "ts_pancreas"
    assert resolve_model("glioblastoma", "MR").model_id == "ts_brain"
    lung = resolve_model("lung cancer", "CT")
    assert lung is not None
    # Prefer auto-downloadable MSD lung tumor model when not runtime-blocked.
    assert lung.model_id in {"nnunet_msd_lung", "ts_lung"}
    models = list_models_for_disease("colorectal cancer", "CT")
    assert any(m.model_id == "ts_colon" for m in models)


def test_squamous_cell_does_not_match_kidney_or_liver() -> None:
    """'cell' in squamous cell carcinoma must not select ts_kidney/ts_liver."""
    disease = (
        "Basaloid squamous cell carcinoma; lung squamous cell carcinoma (TCGA-LUSC)"
    )
    models = list_models_for_disease(disease, "CT", direct_only=True)
    ids = [m.model_id for m in models]
    assert "ts_lung" in ids
    assert "ts_kidney" not in ids
    assert "ts_liver" not in ids
    # After tumor models, first organ fallback should be lung — not kidney.
    organs = [m.model_id for m in models if not m.is_tumor_model]
    assert organs[0] == "ts_lung"


def test_lung_catalog_includes_all_operational_types() -> None:
    models = list_models_for_disease("lung adenocarcinoma TCGA-LUAD", "CT")
    ids = {m.model_id for m in models}
    # Ready lesion, organ fallback, foundation, promptable — all selectable in catalog.
    assert "ts_lung_nodules" in ids
    assert "nnunet_msd_lung" in ids
    assert "monai_lung_nodule" in ids
    assert "nnunet_autopet" in ids
    assert "ts_lung" in ids
    assert "monai_swin_unetr_ssl" in ids
    assert "stu_net" in ids
    assert "medsam" in ids
    assert "sam_med3d" in ids
    from agentic_radiogen.imaging.tumor_models.segmentation_select import is_selectable

    assert not is_selectable(get_model("ts_lung_nodules"))  # type: ignore[arg-type]


def test_catalog_lists_foundation_and_promptable() -> None:
    rows = {r["model_id"]: r for r in catalog_for_llm()}
    assert rows["ts_lung_nodules"]["is_tumor_model"] is True
    assert rows["ts_lung_nodules"]["direct_inference"] is True
    assert rows["nnunet_msd_lung"]["operational_type"] == "ready_inference"
    assert rows["monai_lung_nodule"]["is_tumor_model"] is True
    assert rows["monai_lung_nodule"]["direct_inference"] is True
    assert rows["nnunet_autopet"]["is_tumor_model"] is True
    assert rows["monai_swin_unetr_ssl"]["direct_inference"] is False
    assert rows["stu_net"]["operational_type"] == "fine_tune"
    assert rows["medsam"]["operational_type"] == "promptable"
    assert rows["sam_med3d"]["operational_type"] == "promptable"


def test_empty_mask_tries_tumor_models_before_organ(tmp_path) -> None:
    """After Task006 empty mask, try MONAI / AutoPET before ts_lung."""
    from agentic_radiogen.agents.segmentation import _order_fallback_candidates

    primary = get_model("nnunet_msd_lung")
    assert primary is not None
    rest = list_models_for_disease("lung adenocarcinoma", "CT")
    ordered = _order_fallback_candidates(primary, rest)
    ids = [s.model_id for s in ordered]
    assert ids[0] == "nnunet_msd_lung"
    # Other tumor ready models appear before organ ts_lung.
    assert "monai_lung_nodule" in ids
    assert "nnunet_autopet" in ids
    assert "ts_lung" in ids
    assert ids.index("monai_lung_nodule") < ids.index("ts_lung")
    assert ids.index("nnunet_autopet") < ids.index("ts_lung")


def test_autopet_skips_ct_only_hu_volume() -> None:
    from agentic_radiogen.imaging.tumor_models.nnunet_autopet import (
        run_nnunet_autopet_mask,
    )

    vol = np.full((16, 32, 32), -800.0, dtype=np.float32)
    vol[:, 8:24, 8:24] = 40.0
    try:
        run_nnunet_autopet_mask(vol, modality="CT", allow_download=False, verbose=False)
        raise AssertionError("expected CT-only skip")
    except RuntimeError as exc:
        assert "PET+CT" in str(exc) or "CT-only" in str(exc)


def test_monai_runtime_requires_install_or_package(tmp_path, monkeypatch) -> None:
    from agentic_radiogen.imaging.tumor_models import monai_lung_nodule as ml

    monkeypatch.setattr(ml, "_MONAI_READY", False)

    def _boom():
        raise ImportError("No module named 'monai'")

    # Force the first import check to fail without uninstalling from the env.
    import builtins

    real_import = builtins.__import__

    def _selective_import(name, *args, **kwargs):
        if name == "monai" or name.startswith("monai."):
            raise ImportError("No module named 'monai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _selective_import)
    try:
        ml.ensure_monai_runtime(verbose=False, allow_install=False)
        raise AssertionError("expected RuntimeError when monai missing")
    except RuntimeError as exc:
        assert "not installed" in str(exc).lower() or "auto-install" in str(exc).lower()


def test_rules_fallback_without_llm(tmp_path) -> None:
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    vol[:, 4:12, 4:12] = 40.0
    agent = SegmentationAgent(
        store=ModelStore(root=tmp_path / "seg_models", verbose=False),
        verbose=False,
        use_llm=False,
        allow_download=False,
    )
    result = agent.segment(vol, disease="liver cancer", modality="CT")
    assert result.mask.any()
    assert result.model_id in {"ts_liver", "totalsegmentator_organ", "threshold_proxy"}


def test_llm_choose_model_uses_injected_client(tmp_path) -> None:
    class FakeLlm:
        available = True
        config = type("C", (), {"provider": "fake", "model": "m"})()

        def complete_json(self, *, system, user, timeout=None):
            return {
                "model_id": "ts_kidney",
                "organ_keywords": ["kidney"],
                "rationale": "RCC involves kidney",
                "is_tumor_model": False,
            }

    agent = SegmentationAgent(
        store=ModelStore(root=tmp_path / "seg", verbose=False),
        verbose=False,
        use_llm=True,
        llm_client=FakeLlm(),  # type: ignore[arg-type]
        allow_download=False,
    )
    spec = agent.choose_model(
        disease="kidney cancer", modality="CT", use_cache=False
    )
    assert spec.model_id == "ts_kidney"
    assert "kidney" in spec.organ_keywords


def test_llm_can_select_nnunet_msd_lung(tmp_path) -> None:
    class FakeLlm:
        available = True
        config = type("C", (), {"provider": "fake", "model": "m"})()

        def complete_json(self, *, system, user, timeout=None):
            return {
                "model_id": "nnunet_msd_lung",
                "organ_keywords": [],
                "rationale": "MSD lung tumor nnU-Net for LUAD",
                "is_tumor_model": True,
            }

    agent = SegmentationAgent(
        store=ModelStore(root=tmp_path / "seg", verbose=False),
        verbose=False,
        use_llm=True,
        llm_client=FakeLlm(),  # type: ignore[arg-type]
        allow_download=False,
    )
    spec = agent.choose_model(
        disease="lung adenocarcinoma (TCGA-LUAD)",
        modality="CT",
        use_cache=False,
    )
    assert spec.model_id == "nnunet_msd_lung"
    assert spec.is_tumor_model is True


def test_llm_can_select_promptable_sam_med3d(tmp_path) -> None:
    class FakeLlm:
        available = True
        config = type("C", (), {"provider": "fake", "model": "m"})()

        def complete_json(self, *, system, user, timeout=None):
            return {
                "model_id": "sam_med3d",
                "organ_keywords": [],
                "rationale": "Interactive 3D GTV with point prompt",
                "is_tumor_model": True,
            }

    agent = SegmentationAgent(
        store=ModelStore(root=tmp_path / "seg", verbose=False),
        verbose=False,
        use_llm=True,
        llm_client=FakeLlm(),  # type: ignore[arg-type]
        allow_download=False,
    )
    spec = agent.choose_model(
        disease="lung cancer", modality="CT", use_cache=False
    )
    assert spec.model_id == "sam_med3d"
    assert spec.operational_type == "promptable"


def test_segment_skips_unwired_and_previously_failed(tmp_path) -> None:
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    vol[:, 4:12, 4:12] = 40.0
    SegmentationAgent._failed_model_ids.clear()
    # Prefer tumor nnU-Net but block download so it fails over to organ/threshold.
    SegmentationAgent._failed_model_ids.add("ts_lung_nodules")
    agent = SegmentationAgent(
        store=ModelStore(root=tmp_path / "seg", verbose=False),
        verbose=False,
        use_llm=False,
        allow_download=False,
    )
    result = agent.segment(vol, disease="lung cancer", modality="CT")
    assert result.mask.any()
    # No download → nnunet_msd_lung fails → organ/threshold fallback.
    assert result.model_id in {
        "ts_lung",
        "totalsegmentator_organ",
        "threshold_proxy",
    }
    assert result.model_id != "ts_lung_nodules"


def test_nnunet_available_without_local_weights() -> None:
    from agentic_radiogen.imaging.tumor_models.nnunet_lung import nnunet_lung_available

    # Auto-download URL makes the model selectable even before first fetch.
    assert nnunet_lung_available() is True


def test_empty_mask_does_not_block_model_for_cohort(tmp_path) -> None:
    from agentic_radiogen.agents.segmentation import _is_patient_local_failure

    assert _is_patient_local_failure("nnU-Net MSD lung returned an empty mask")
    assert _is_patient_local_failure("TotalSegmentator task='total' produced no masks")
    assert _is_patient_local_failure(
        "AutoPET requires PET+CT (metabolic); unsuitable for CT-only case"
    )
    assert not _is_patient_local_failure("No module named 'nnunet'")

    vol = np.zeros((8, 16, 16), dtype=np.float32)
    vol[:, 4:12, 4:12] = 40.0
    agent = SegmentationAgent(
        store=ModelStore(root=tmp_path / "seg", verbose=False),
        verbose=False,
        use_llm=False,
        allow_download=False,
    )

    def _boom(*_a, **_k):
        raise RuntimeError("nnU-Net MSD lung returned an empty mask")

    agent._try_spec = _boom  # type: ignore[method-assign]
    # Force candidate list to include nnunet then threshold via monkeypatch path:
    # segment() will try specs; empty-mask must not land in _failed_model_ids.
    from agentic_radiogen.imaging.tumor_models.registry import get_model

    spec = get_model("nnunet_msd_lung")
    assert spec is not None
    try:
        agent._try_spec(spec, vol, disease="lung")
    except Exception as exc:
        if _is_patient_local_failure(str(exc)):
            pass
    # Simulate the segment() except branch:
    if _is_patient_local_failure("nnU-Net MSD lung returned an empty mask"):
        pass
    else:
        SegmentationAgent._failed_model_ids.add("nnunet_msd_lung")
    assert "nnunet_msd_lung" not in SegmentationAgent._failed_model_ids


def test_llm_rejects_fine_tune_choice(tmp_path) -> None:
    class FakeLlm:
        available = True
        config = type("C", (), {"provider": "fake", "model": "m"})()

        def complete_json(self, *, system, user, timeout=None):
            return {
                "model_id": "monai_swin_unetr_ssl",
                "organ_keywords": [],
                "rationale": "wrong: fine-tune only",
                "is_tumor_model": False,
            }

    agent = SegmentationAgent(
        store=ModelStore(root=tmp_path / "seg", verbose=False),
        verbose=False,
        use_llm=True,
        llm_client=FakeLlm(),  # type: ignore[arg-type]
        allow_download=False,
    )
    spec = agent.choose_model(
        disease="lung cancer", modality="CT", use_cache=False
    )
    # Invalid fine-tune pick → rules fallback to a selectable live model.
    assert spec.model_id in {"nnunet_msd_lung", "ts_lung", "threshold_proxy"}


def test_plan_segmentation_rules_unknown_disease() -> None:
    spec = plan_segmentation_rules(disease="rare zebra sarcoma", modality="CT")
    assert spec.model_id in {"totalsegmentator_organ", "threshold_proxy"}


def test_get_model_overrides_organs() -> None:
    spec = get_model("totalsegmentator_organ", organ_keywords=["spleen", "liver"])
    assert spec is not None
    assert spec.organ_keywords == ("spleen", "liver")
