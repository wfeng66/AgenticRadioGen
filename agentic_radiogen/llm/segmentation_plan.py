"""LLM chooses a segmentation backend for the SegmentationAgent (masks only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic_radiogen.imaging.tumor_models.registry import (
    TumorModelSpec,
    catalog_for_llm,
    get_model,
    normalize_disease_token,
    resolve_model,
)
from agentic_radiogen.llm.client import LlmClient
from agentic_radiogen.util.progress import log

_PLAN_CACHE_VERSION = 9  # AutoPET + MONAI lung nodule in catalog; tumor-before-organ
_DEFAULT_PLAN_CACHE = Path.cwd() / "data_cache" / "seg_plans"


class SegmentationLlmPlan(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str = Field(..., description="Id from the provided catalog.")
    organ_keywords: list[str] = Field(
        default_factory=list,
        description="Organ/site filename hints for TotalSegmentator label maps.",
    )
    rationale: str = Field(default="")
    is_tumor_model: bool = Field(
        default=False,
        description="True if the chosen catalog entry is a lesion/tumor segmentor.",
    )


_SYSTEM = """You are the Segmentation planner. Your job is ONLY to pick a segmentation
model that produces a mask for the patient disease + modality. Another agent handles
radiomics / feature extraction — do NOT optimize for features, only for segmentation.

Disease string is most-specific → least-specific (histology, cohort, then coarse phrase).

Catalog operational_type meanings:
- ready_inference: can run now and output a lesion/tumor (or task) mask.
- organ: whole-organ anatomy mask (fallback when no lesion model fits).
- fine_tune: pre-trained backbone for training — do NOT pick for live segmentation.
- promptable: needs a click/box prompt — only pick if the question clearly asks for
  interactive / promptable segmentation; otherwise skip.

Rules:
- Prefer the MOST SPECIFIC anatomy/histology cues over a coarse phrase like "lung cancer".
- For lung cancer / LUAD / LUSC on CT, prefer ready tumor models in this order of fit:
  1) nnunet_msd_lung — large primary NSCLC masses (Zenodo Task006; auto-download).
  2) monai_lung_nodule — pulmonary nodules/masses (MONAI Model Zoo RetinaNet; CT).
  3) nnunet_autopet — only when PET (or PET+CT) is available; metabolic / infiltrating
     lesions (lab-midas AutoPET). Do NOT pick for CT-only cases.
- Do NOT pick ts_lung_nodules unless explicitly enabled.
- Swin UNETR / STU-Net are fine_tune only — never for live segmentation.
- Organ models (ts_*) are last-resort anatomy fallbacks after tumor models fail
  at runtime — do not prefer ts_lung over nnunet_msd_lung / monai_lung_nodule.
- You may pick totalsegmentator_organ with custom organ_keywords when needed.
- Never pick fine_tune for live segmentation.
- Set is_tumor_model to match the catalog entry.
- Return JSON only: model_id, organ_keywords, rationale, is_tumor_model.
- Do NOT invent model_ids outside the catalog (except totalsegmentator_organ with organs).
"""


def plan_segmentation_with_llm(
    *,
    disease: str,
    modality: str = "CT",
    question: str = "",
    client: LlmClient | None = None,
    models: tuple[TumorModelSpec, ...] | None = None,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
    blocked_model_ids: set[str] | None = None,
) -> tuple[TumorModelSpec, SegmentationLlmPlan]:
    """Ask LLM which catalog model to use; cache by normalized disease+modality."""
    disease = (disease or "").strip() or "unknown"
    modality = (modality or "CT").strip().upper()
    cache_root = Path(cache_dir) if cache_dir else _DEFAULT_PLAN_CACHE
    cache_path = cache_root / f"{normalize_disease_token(disease)}__{modality.lower()}.json"

    blocked = set(blocked_model_ids or ())

    if use_cache and cache_path.is_file():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if int(raw.get("version", 0)) >= _PLAN_CACHE_VERSION and raw.get("model_id"):
                plan = SegmentationLlmPlan(
                    model_id=str(raw["model_id"]),
                    organ_keywords=list(raw.get("organ_keywords") or []),
                    rationale=str(raw.get("rationale") or "from cache"),
                    is_tumor_model=bool(raw.get("is_tumor_model", False)),
                )
                if plan.model_id == "threshold_proxy" or plan.model_id in blocked:
                    log(
                        f"[segmentation-llm] Ignoring cached '{plan.model_id}' "
                        f"(heuristic/blocked); re-selecting",
                        enabled=verbose,
                    )
                    return _rules_plan(
                        disease=disease,
                        modality=modality,
                        models=models,
                        blocked=blocked | {"threshold_proxy"},
                        use_cache=use_cache,
                        cache_path=cache_path,
                        verbose=verbose,
                    )
                spec = _spec_from_plan(plan, models=models)
                if spec is not None:
                    log(
                        f"[segmentation-llm] Cache hit disease='{disease}' → {spec.model_id}",
                        enabled=verbose,
                    )
                    return spec, plan
        except Exception:
            pass

    llm = client or LlmClient()
    catalog = [
        row
        for row in catalog_for_llm(models)
        if str(row.get("model_id")) not in blocked
    ]
    user = (
        f"Disease: {disease}\n"
        f"Modality: {modality}\n"
        f"Research question (optional): {(question or '').strip() or '(none)'}\n"
        f"Do NOT pick these blocked model_ids: {sorted(blocked) or '(none)'}\n\n"
        f"Catalog JSON:\n{json.dumps(catalog, indent=2)}\n\n"
        "Pick the best segmentation model for this case. "
        "Respond with JSON keys: model_id, organ_keywords, rationale, is_tumor_model."
    )
    log(
        f"[segmentation-llm] Asking LLM which segmentation model to use for '{disease}' "
        f"({llm.config.provider}/{llm.config.model})...",
        enabled=verbose,
    )
    data: dict[str, Any] = llm.complete_json(system=_SYSTEM, user=user, timeout=40)
    organs = data.get("organ_keywords") or []
    if isinstance(organs, str):
        organs = [organs]
    plan = SegmentationLlmPlan(
        model_id=str(data.get("model_id") or "").strip(),
        organ_keywords=[str(x).strip().lower() for x in organs if str(x).strip()],
        rationale=str(data.get("rationale") or "").strip(),
        is_tumor_model=bool(data.get("is_tumor_model", False)),
    )
    if plan.model_id in blocked:
        log(
            f"[segmentation-llm] LLM picked blocked '{plan.model_id}'; using rules",
            enabled=verbose,
        )
        return _rules_plan(
            disease=disease,
            modality=modality,
            models=models,
            blocked=blocked,
            use_cache=use_cache,
            cache_path=cache_path,
            verbose=verbose,
        )
    spec = _spec_from_plan(plan, models=models)
    if spec is None:
        raise ValueError(f"LLM chose unknown/unusable model_id '{plan.model_id}'")

    if use_cache:
        _write_plan_cache(cache_path, disease, modality, plan)

    log(
        f"[segmentation-llm] Chose {spec.model_id} organs={list(spec.organ_keywords)} "
        f"type={spec.operational_type} — {plan.rationale}",
        enabled=verbose,
    )
    return spec, plan


def _rules_plan(
    *,
    disease: str,
    modality: str,
    models: tuple[TumorModelSpec, ...] | None,
    blocked: set[str],
    use_cache: bool,
    cache_path: Path,
    verbose: bool,
) -> tuple[TumorModelSpec, SegmentationLlmPlan]:
    from agentic_radiogen.imaging.tumor_models.registry import list_models_for_disease

    picks = list_models_for_disease(disease, modality, models=models, direct_only=True)
    spec = None
    for cand in picks:
        if cand.model_id in blocked:
            continue
        if cand.operational_type == "fine_tune":
            continue
        spec = cand
        break
    if spec is None:
        spec = get_model("threshold_proxy", models=models)  # type: ignore[assignment]
    assert spec is not None
    plan = SegmentationLlmPlan(
        model_id=spec.model_id,
        organ_keywords=list(spec.organ_keywords),
        rationale="rules fallback (preferred model blocked this run)",
        is_tumor_model=spec.is_tumor_model,
    )
    if use_cache and spec.model_id != "threshold_proxy":
        _write_plan_cache(cache_path, disease, modality, plan)
    log(
        f"[segmentation-llm] Rules chose {spec.model_id} (blocked={sorted(blocked)})",
        enabled=verbose,
    )
    return spec, plan


def _write_plan_cache(
    cache_path: Path, disease: str, modality: str, plan: SegmentationLlmPlan
) -> None:
    if plan.model_id == "threshold_proxy":
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "version": _PLAN_CACHE_VERSION,
                    "disease": disease,
                    "modality": modality,
                    "model_id": plan.model_id,
                    "organ_keywords": plan.organ_keywords,
                    "rationale": plan.rationale,
                    "is_tumor_model": plan.is_tumor_model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def plan_segmentation_rules(
    *,
    disease: str,
    modality: str = "CT",
    models: tuple[TumorModelSpec, ...] | None = None,
) -> TumorModelSpec:
    """Offline fallback when LLM is unavailable (direct-inference only)."""
    spec = resolve_model(disease, modality, models=models)
    if spec is not None:
        return spec
    return get_model("threshold_proxy", models=models)  # type: ignore[return-value]


def write_segmentation_plan_cache(
    *,
    disease: str,
    modality: str,
    model_id: str,
    organ_keywords: list[str] | None = None,
    is_tumor_model: bool = False,
    rationale: str = "",
    cache_dir: str | Path | None = None,
) -> Path:
    """Persist a working segmentation plan (used after runtime fallback)."""
    cache_root = Path(cache_dir) if cache_dir else _DEFAULT_PLAN_CACHE
    cache_root.mkdir(parents=True, exist_ok=True)
    modality = (modality or "CT").strip().upper()
    path = cache_root / f"{normalize_disease_token(disease)}__{modality.lower()}.json"
    if model_id == "threshold_proxy":
        # Never lock a disease onto the intensity heuristic.
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
        return path
    path.write_text(
        json.dumps(
            {
                "version": _PLAN_CACHE_VERSION,
                "disease": disease,
                "modality": modality,
                "model_id": model_id,
                "organ_keywords": list(organ_keywords or []),
                "rationale": rationale or "runtime fallback",
                "is_tumor_model": bool(is_tumor_model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _spec_from_plan(
    plan: SegmentationLlmPlan,
    *,
    models: tuple[TumorModelSpec, ...] | None,
) -> TumorModelSpec | None:
    from agentic_radiogen.imaging.tumor_models.segmentation_select import is_selectable

    if not plan.model_id:
        return None
    spec = get_model(
        plan.model_id, models=models, organ_keywords=plan.organ_keywords or None
    )
    if spec is None and plan.organ_keywords:
        spec = get_model(
            "totalsegmentator_organ",
            models=models,
            organ_keywords=plan.organ_keywords,
        )
    if spec is not None and not is_selectable(spec):
        return None
    return spec
