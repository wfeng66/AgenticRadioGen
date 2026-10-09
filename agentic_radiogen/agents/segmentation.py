"""LLM-guided, disease-aware segmentation agent (masks only).

Flow:
1. Send disease (+ modality) to an LLM planner with the full model catalog.
2. Run the chosen segmentor (ready tumor/lesion, organ, or promptable when wired).
3. Return a boolean mask. Feature extraction is ImagingRadiomicsAgent's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentic_radiogen.imaging.tumor_models.registry import (
    TumorModelSpec,
    list_models_for_disease,
)
from agentic_radiogen.imaging.tumor_models.runners import run_model
from agentic_radiogen.imaging.tumor_models.store import ModelStore
from agentic_radiogen.llm.client import LlmClient, resolve_llm_config
from agentic_radiogen.llm.segmentation_plan import (
    plan_segmentation_rules,
    plan_segmentation_with_llm,
)
from agentic_radiogen.util.progress import log


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    backend: str
    model_id: str
    is_tumor_model: bool
    note: str = ""
    organ_keywords: tuple[str, ...] = ()


# Runners that can produce a mask today without prompts / extra wiring.
_LIVE_RUNNERS = frozenset(
    {
        "totalsegmentator_organ",
        "totalsegmentator_task",
        "threshold_proxy",
        "torchscript",
        "nnunet",
        "nnunet_autopet",
        "monai_bundle",
    }
)


def _is_patient_local_failure(message: str) -> bool:
    """True when failure is about this volume, not a broken installation."""
    m = (message or "").lower()
    needles = (
        "empty mask",
        "returned an empty",
        "produced no masks",
        "mask empty",
        "only 1 slice",
        "only 2 slice",
        "need ≥2",
        "need >=2",
        "too few slices",
        "series has only",
        "got shape (512, 512)",
        "needs a 3d volume",
        "unsuitable for 3d",
        "unsuitable for this case",
        "unsuitable for ct-only",
        "requires pet+ct",
        "requires pet",
    )
    return any(n in m for n in needles)


def _order_fallback_candidates(
    primary: TumorModelSpec, rest: list[TumorModelSpec]
) -> list[TumorModelSpec]:
    """Keep primary first; then other tumor models; organs / threshold last.

    Empty-mask from Task006 must try MONAI / AutoPET before ``ts_lung``.
    Preserve disease-ranked order from ``list_models_for_disease`` within each
    group (do not re-sort organs by catalog priority — that put ts_kidney
    ahead of ts_lung after a weak false match).
    """
    seen = {primary.model_id}
    tumors: list[TumorModelSpec] = []
    organs: list[TumorModelSpec] = []
    for spec in rest:
        if spec.model_id in seen:
            continue
        seen.add(spec.model_id)
        if spec.is_tumor_model and spec.operational_type == "ready_inference":
            tumors.append(spec)
        elif spec.model_id == "threshold_proxy":
            organs.append(spec)
        elif not spec.is_tumor_model:
            organs.append(spec)
        else:
            # promptable / fine_tune etc. — after tumors, before organs
            tumors.append(spec)
    return [primary, *tumors, *organs]


class SegmentationAgent:
    """LLM chooses model for disease → download/reuse → segment."""

    # Process-level: skip models that already crashed this run (e.g. lung_nodules).
    _failed_model_ids: set[str] = set()

    def __init__(
        self,
        *,
        store: ModelStore | None = None,
        models: tuple[TumorModelSpec, ...] | None = None,
        verbose: bool = True,
        allow_download: bool = True,
        use_llm: bool | None = None,
        llm_client: LlmClient | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.store = store or ModelStore(verbose=verbose)
        self.models = models
        self.verbose = verbose
        self.allow_download = allow_download
        self._use_llm = use_llm
        self._llm_client = llm_client
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self.last_plan_rationale: str = ""

    def available_models(self, disease: str, modality: str = "CT") -> list[TumorModelSpec]:
        return list_models_for_disease(disease, modality, models=self.models)

    def choose_model(
        self,
        *,
        disease: str,
        modality: str = "CT",
        question: str = "",
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ) -> TumorModelSpec:
        """LLM (or rules) selects a catalog model for this disease."""
        disease = (disease or "").strip() or "unknown"
        modality = (modality or "CT").strip().upper()
        want_llm = self._want_llm()
        if want_llm:
            client = self._llm_client or LlmClient(
                resolve_llm_config(
                    provider=self._llm_provider,
                    model=self._llm_model,
                    enabled=True,
                )
            )
            if client.available:
                try:
                    spec, plan = plan_segmentation_with_llm(
                        disease=disease,
                        modality=modality,
                        question=question,
                        client=client,
                        models=self.models,
                        verbose=self.verbose,
                        use_cache=use_cache,
                        cache_dir=cache_dir,
                        blocked_model_ids=self._blocked_ids(),
                    )
                    self.last_plan_rationale = plan.rationale
                    return spec
                except Exception as exc:
                    log(
                        f"[segmentation] LLM model choice failed ({exc}); using rules",
                        enabled=self.verbose,
                    )
        blocked = self._blocked_ids()
        from agentic_radiogen.imaging.tumor_models.registry import list_models_for_disease
        from agentic_radiogen.imaging.tumor_models.registry import get_model

        picks = list_models_for_disease(
            disease, modality, models=self.models, direct_only=True
        )
        spec = None
        for cand in picks:
            if cand.model_id in blocked:
                continue
            if cand.operational_type == "fine_tune":
                continue
            spec = cand
            break
        if spec is None:
            spec = plan_segmentation_rules(
                disease=disease, modality=modality, models=self.models
            )
            if spec.model_id in blocked:
                spec = get_model("threshold_proxy", models=self.models)  # type: ignore[assignment]
        assert spec is not None
        self.last_plan_rationale = "rules heuristic (LLM off or preferred model blocked)"
        log(
            f"[segmentation] Rules chose '{spec.model_id}' for disease='{disease}'",
            enabled=self.verbose,
        )
        return spec

    def segment(
        self,
        volume: np.ndarray,
        *,
        disease: str | None = None,
        modality: str = "CT",
        question: str = "",
        spacing_zyx: tuple[float, float, float] | None = None,
    ) -> SegmentationResult:
        disease = (disease or "").strip() or "unknown"
        modality = (modality or "CT").strip().upper()

        primary = self.choose_model(
            disease=disease, modality=modality, question=question
        )
        # Try LLM/rules choice first, then other tumor models, then organ/threshold.
        rest: list[TumorModelSpec] = []
        for spec in self.available_models(disease, modality):
            if spec.model_id != primary.model_id:
                rest.append(spec)
        for mid in ("totalsegmentator_organ", "threshold_proxy"):
            from agentic_radiogen.imaging.tumor_models.registry import get_model

            extra = get_model(mid, models=self.models)
            if extra and extra.model_id != primary.model_id:
                rest.append(extra)
        candidates = _order_fallback_candidates(primary, rest)

        # Only attempt backends that can produce a mask in this process.
        live: list[TumorModelSpec] = []
        skipped: list[str] = []
        from agentic_radiogen.imaging.tumor_models.segmentation_select import (
            is_selectable,
        )

        mod_u = modality.upper()
        if mod_u == "MRI":
            mod_u = "MR"
        for spec in candidates:
            if spec.model_id in self._failed_model_ids:
                skipped.append(f"{spec.model_id}: skipped (failed earlier this run)")
                continue
            if not is_selectable(spec):
                skipped.append(f"{spec.model_id}: not selectable (blocked/disabled)")
                continue
            if spec.runner not in _LIVE_RUNNERS:
                skipped.append(f"{spec.model_id}: runner={spec.runner} not live yet")
                continue
            # AutoPET needs PET; skip quietly on CT-only so we don't waste a slot.
            if spec.runner == "nnunet_autopet" and mod_u not in {"PT", "PET"}:
                skipped.append(
                    f"{spec.model_id}: skipped (needs PET+CT; modality={modality})"
                )
                continue
            live.append(spec)
        candidates = live

        errors: list[str] = list(skipped)
        for spec in candidates:
            try:
                result = self._try_spec(
                    spec,
                    volume,
                    disease=disease,
                    modality=modality,
                    spacing_zyx=spacing_zyx,
                )
                # Only rewrite plan cache for durable fallbacks (not one empty-mask case).
                if (
                    result.model_id != primary.model_id
                    and result.model_id != "threshold_proxy"
                    and result.is_tumor_model
                ):
                    self._remember_working_plan(
                        disease=disease,
                        modality=modality,
                        spec=spec,
                        rationale=(
                            f"fallback after {primary.model_id} failed; "
                            f"using {spec.model_id}"
                        ),
                    )
                return result
            except Exception as exc:
                msg = str(exc)
                errors.append(f"{spec.model_id}: {exc}")
                # Empty ROI / thin series are patient-specific — do NOT block the
                # model for the rest of the cohort (that forced everyone to threshold).
                if _is_patient_local_failure(msg):
                    log(
                        f"[segmentation] Model '{spec.model_id}' empty/unsuitable "
                        f"for this case ({exc}); trying next (model stays enabled)",
                        enabled=self.verbose,
                    )
                else:
                    self._failed_model_ids.add(spec.model_id)
                    self._block_model(spec.model_id, msg)
                    log(
                        f"[segmentation] Model '{spec.model_id}' unavailable ({exc}); "
                        f"trying next",
                        enabled=self.verbose,
                    )

        from agentic_radiogen.imaging.segment import _threshold_mask

        mask = _threshold_mask(volume)
        note = "; ".join(errors[:3])
        log(
            f"[segmentation] All models failed for disease='{disease}'; "
            f"using threshold_proxy",
            enabled=self.verbose,
        )
        return SegmentationResult(
            mask=mask,
            backend="threshold_proxy",
            model_id="threshold_proxy",
            is_tumor_model=False,
            note=note or "fallback",
        )

    @classmethod
    def _blocked_ids(cls) -> set[str]:
        from agentic_radiogen.imaging.tumor_models.segmentation_select import (
            load_runtime_blocked,
        )

        return set(cls._failed_model_ids) | load_runtime_blocked()

    @staticmethod
    def _block_model(model_id: str, reason: str) -> None:
        from agentic_radiogen.imaging.tumor_models.segmentation_select import (
            persist_blocked,
        )

        persist_blocked(model_id, reason=reason)

    def _remember_working_plan(
        self,
        *,
        disease: str,
        modality: str,
        spec: TumorModelSpec,
        rationale: str,
    ) -> None:
        """Rewrite seg-plan cache so later patients skip a known-broken primary."""
        try:
            from agentic_radiogen.llm.segmentation_plan import write_segmentation_plan_cache

            write_segmentation_plan_cache(
                disease=disease,
                modality=modality,
                model_id=spec.model_id,
                organ_keywords=list(spec.organ_keywords),
                is_tumor_model=spec.is_tumor_model,
                rationale=rationale,
            )
            log(
                f"[segmentation] Updated plan cache → {spec.model_id} ({rationale})",
                enabled=self.verbose,
            )
        except Exception:
            pass

    def _want_llm(self) -> bool:
        if self._use_llm is True or self._llm_client is not None:
            return True
        if self._use_llm is False:
            return False
        cfg = resolve_llm_config(
            provider=self._llm_provider, model=self._llm_model, enabled=None
        )
        return cfg.provider != "off" and bool(cfg.api_key)

    def _try_spec(
        self,
        spec: TumorModelSpec,
        volume: np.ndarray,
        *,
        disease: str,
        modality: str = "CT",
        spacing_zyx: tuple[float, float, float] | None = None,
        pet_volume: np.ndarray | None = None,
    ) -> SegmentationResult:
        model_dir: Path | None = None
        if spec.runner == "torchscript":
            if self.store.is_ready(spec):
                model_dir = self.store.model_dir(spec)
            elif self.allow_download:
                model_dir = self.store.ensure_local(spec)
            else:
                raise FileNotFoundError(
                    f"Local weights missing for '{spec.model_id}' and download disabled"
                )
        elif spec.runner == "nnunet":
            log(
                f"[segmentation] Using nnU-Net '{spec.model_id}' "
                f"(auto-download Task006 from Zenodo on first use if needed)",
                enabled=self.verbose,
            )
        elif spec.runner == "nnunet_autopet":
            log(
                f"[segmentation] Using AutoPET nnU-Net '{spec.model_id}' "
                f"(PET+CT metabolic lesions; skips CT-only)",
                enabled=self.verbose,
            )
        elif spec.runner == "monai_bundle":
            log(
                f"[segmentation] Using MONAI bundle '{spec.model_id}' "
                f"(lung_nodule_ct_detection; auto-download on first use if needed)",
                enabled=self.verbose,
            )
        elif spec.runner in {"totalsegmentator_organ", "totalsegmentator_task"}:
            try:
                import totalsegmentator  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    "TotalSegmentator not installed (pip install totalsegmentator)"
                ) from exc
            task = spec.task_name or "total"
            log(
                f"[segmentation] Using TotalSegmentator for '{spec.model_id}' "
                f"(task={task}, auto-download on first use if needed)",
                enabled=self.verbose,
            )

        log(
            f"[segmentation] disease='{disease}' → model='{spec.model_id}' "
            f"organs={list(spec.organ_keywords)} "
            f"(tumor={spec.is_tumor_model}, type={spec.operational_type}, "
            f"runner={spec.runner})",
            enabled=self.verbose,
        )
        mask = run_model(
            spec,
            volume,
            model_dir=model_dir,
            spacing_zyx=spacing_zyx,
            allow_download=self.allow_download,
            verbose=self.verbose,
            modality=modality,
            pet_volume=pet_volume,
        )
        if not np.asarray(mask).any():
            raise RuntimeError(f"Model '{spec.model_id}' returned an empty mask")
        return SegmentationResult(
            mask=np.asarray(mask, dtype=bool),
            backend=spec.model_id,
            model_id=spec.model_id,
            is_tumor_model=spec.is_tumor_model,
            note=self.last_plan_rationale or spec.description,
            organ_keywords=tuple(spec.organ_keywords),
        )
