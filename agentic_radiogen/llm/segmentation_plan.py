"""LLM chooses a segmentation backend for the SegmentationAgent (masks only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agentic_radiogen.imaging.tumor_models.registry import (
    TumorModelSpec,
    catalog_for_llm,
    get_model,
    normalize_disease_token,
    resolve_model,
)
from agentic_radiogen.llm.client import LlmClient
from agentic_radiogen.util.progress import log

_PLAN_CACHE_VERSION = 13  # include monai_vista3d in lung ready-tumor expansion
_DEFAULT_PLAN_CACHE = Path.cwd() / "data_cache" / "seg_plans"

# Ready CT lung tumor models that must remain in the try-list for LUAD/LUSC.
_LUNG_READY_TUMOR_IDS = (
    "nnunet_msd_lung",
    "monai_vista3d",
    "monai_lung_nodule",
    "lungtumormask",
)


class SegmentationLlmPlan(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str = Field(
        default="",
        description="Primary / first model id from the ranked list.",
    )
    model_ids: list[str] = Field(
        default_factory=list,
        description="Ranked catalog model ids to try for this diagnosis (best first).",
    )
    organ_keywords: list[str] = Field(
        default_factory=list,
        description="Organ/site filename hints for TotalSegmentator label maps.",
    )
    rationale: str = Field(default="")
    is_tumor_model: bool = Field(
        default=False,
        description="True if the primary catalog entry is a lesion/tumor segmentor.",
    )

    @field_validator("model_ids", mode="before")
    @classmethod
    def _coerce_model_ids(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            out: list[str] = []
            for x in value:
                s = str(x).strip()
                if s and s not in out:
                    out.append(s)
            return out
        return []


_SYSTEM = """You are the Segmentation planner. Your job is ONLY to pick segmentation
models that produce a lesion/tumor mask for this patient's disease + modality.
Another agent handles radiomics / feature extraction — do NOT optimize for features.

Disease string is most-specific → least-specific (histology, cohort, then coarse phrase).

Catalog operational_type meanings:
- ready_inference: can run now and output a lesion/tumor (or task) mask.
- organ: whole-organ anatomy mask (fallback when no lesion model fits).
- fine_tune: pre-trained backbone for training — do NOT pick for live segmentation.
- promptable: needs a click/box prompt — only pick if the question clearly asks for
  interactive / promptable segmentation; otherwise skip.

Rules:
- Prefer the MOST SPECIFIC anatomy/histology cues over a coarse phrase like "lung cancer".
- Return a RANKED list ``model_ids`` (2–5 entries when possible): try order for THIS
  diagnosis. Different lung subtypes / presentations may need different first models
  (large mass vs small nodule vs metabolic PET lesion). Do NOT always put the same
  model first for every lung case.
- Also set ``model_id`` to the first entry of ``model_ids`` (primary choice).
- Available lung-oriented ready tumor models (pick/order by fit to the diagnosis):
  - nnunet_msd_lung — large primary NSCLC masses (MSD Lung / Zenodo Task006).
    Prefer FIRST for TCGA-LUAD / TCGA-LUSC / primary lung adenocarcinoma or squamous
    carcinoma on CT (typical solid primary masses — not a nodule-detection task).
  - monai_vista3d — NVIDIA/MONAI VISTA-3D CT foundation model (lung tumor class 23).
    Strong alternate when nnU-Net is empty; good for primary lung tumors on CT.
  - monai_lung_nodule — pulmonary *nodules* (MONAI RetinaNet / LUNA16). Use early only
    when the disease/question clearly implies nodule detection; do NOT default this as
    the sole/first model for LUAD/LUSC primary masses.
  - lungtumormask — end-to-end CT tumor segmentor (good alternate CT tumor model).
  - nnunet_autopet — only when PET (or PET+CT) is available; never for CT-only.
- Always return at least 2–3 ready tumor model_ids for lung CT (never a singleton).
- Do NOT pick ts_lung_nodules unless explicitly enabled.
- Swin UNETR / STU-Net are fine_tune only — never for live segmentation.
- Organ models (ts_*) are last-resort anatomy fallbacks — put them after tumor models
  if you include them at all.
- You may pick totalsegmentator_organ with custom organ_keywords when needed.
- Never pick fine_tune for live segmentation.
- Set is_tumor_model to match the primary catalog entry.
- Return JSON only: model_ids, model_id, organ_keywords, rationale, is_tumor_model.
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
    """Ask LLM which catalog model to use; cache by normalized disease+modality.

    Returns the primary (first) selectable spec plus the full plan (including
    ranked ``model_ids``).
    """
    specs, plan = plan_segmentation_order_with_llm(
        disease=disease,
        modality=modality,
        question=question,
        client=client,
        models=models,
        use_cache=use_cache,
        cache_dir=cache_dir,
        verbose=verbose,
        blocked_model_ids=blocked_model_ids,
    )
    return specs[0], plan


def plan_segmentation_order_with_llm(
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
) -> tuple[list[TumorModelSpec], SegmentationLlmPlan]:
    """Ask LLM for a ranked model list for this disease + modality."""
    disease = (disease or "").strip() or "unknown"
    modality = (modality or "CT").strip().upper()
    cache_root = Path(cache_dir) if cache_dir else _DEFAULT_PLAN_CACHE
    cache_path = cache_root / f"{normalize_disease_token(disease)}__{modality.lower()}.json"

    blocked = set(blocked_model_ids or ())

    if use_cache and cache_path.is_file():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if int(raw.get("version", 0)) >= _PLAN_CACHE_VERSION and (
                raw.get("model_id") or raw.get("model_ids")
            ):
                plan = _plan_from_raw(raw)
                if plan.model_id == "threshold_proxy" or (
                    plan.model_id in blocked and not plan.model_ids
                ):
                    log(
                        f"[segmentation-llm] Ignoring cached '{plan.model_id}' "
                        f"(heuristic/blocked); re-selecting",
                        enabled=verbose,
                    )
                    return _rules_plan_order(
                        disease=disease,
                        modality=modality,
                        models=models,
                        blocked=blocked | {"threshold_proxy"},
                        use_cache=use_cache,
                        cache_path=cache_path,
                        verbose=verbose,
                    )
                if _is_incomplete_lung_plan(disease, modality, plan):
                    log(
                        f"[segmentation-llm] Ignoring incomplete lung cache "
                        f"{plan.model_ids or [plan.model_id]} (need ranked tumor list); "
                        f"re-selecting",
                        enabled=verbose,
                    )
                else:
                    specs = specs_from_plan(plan, models=models, blocked=blocked)
                    specs = expand_ranked_tumor_specs(
                        specs, disease=disease, modality=modality, models=models, blocked=blocked
                    )
                    if specs:
                        log(
                            f"[segmentation-llm] Cache hit disease='{disease}' → "
                            f"{[s.model_id for s in specs]}",
                            enabled=verbose,
                        )
                        return specs, plan.model_copy(
                            update={
                                "model_ids": [s.model_id for s in specs],
                                "model_id": specs[0].model_id,
                            }
                        )
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
        "Pick a RANKED list of segmentation models for this diagnosis "
        "(best first). Respond with JSON keys: model_ids, model_id, "
        "organ_keywords, rationale, is_tumor_model."
    )
    log(
        f"[segmentation-llm] Asking LLM which segmentation models to use for '{disease}' "
        f"({llm.config.provider}/{llm.config.model})...",
        enabled=verbose,
    )
    data: dict[str, Any] = llm.complete_json(system=_SYSTEM, user=user, timeout=40)
    plan = _plan_from_llm_data(data)
    if not plan.model_ids and not plan.model_id:
        raise ValueError("LLM returned no model_id / model_ids")
    if plan.model_id in blocked and not any(m not in blocked for m in plan.model_ids):
        log(
            f"[segmentation-llm] LLM picked blocked '{plan.model_id}'; using rules",
            enabled=verbose,
        )
        return _rules_plan_order(
            disease=disease,
            modality=modality,
            models=models,
            blocked=blocked,
            use_cache=use_cache,
            cache_path=cache_path,
            verbose=verbose,
        )
    specs = specs_from_plan(plan, models=models, blocked=blocked)
    specs = expand_ranked_tumor_specs(
        specs, disease=disease, modality=modality, models=models, blocked=blocked
    )
    if not specs:
        raise ValueError(
            f"LLM chose unknown/unusable model_ids "
            f"{plan.model_ids or [plan.model_id]}"
        )

    # Normalize plan fields from resolved specs.
    plan = plan.model_copy(
        update={
            "model_ids": [s.model_id for s in specs],
            "model_id": specs[0].model_id,
            "is_tumor_model": specs[0].is_tumor_model,
            "organ_keywords": list(specs[0].organ_keywords)
            if not plan.organ_keywords
            else plan.organ_keywords,
        }
    )

    if use_cache:
        _write_plan_cache(cache_path, disease, modality, plan)

    log(
        f"[segmentation-llm] Ranked {[s.model_id for s in specs]} "
        f"organs={list(plan.organ_keywords)} — {plan.rationale}",
        enabled=verbose,
    )
    return specs, plan


def specs_from_plan(
    plan: SegmentationLlmPlan,
    *,
    models: tuple[TumorModelSpec, ...] | None,
    blocked: set[str] | None = None,
) -> list[TumorModelSpec]:
    """Resolve ranked plan ids to selectable ``TumorModelSpec`` entries."""
    blocked = set(blocked or ())
    ids = list(plan.model_ids)
    if plan.model_id and plan.model_id not in ids:
        ids.insert(0, plan.model_id)
    out: list[TumorModelSpec] = []
    seen: set[str] = set()
    for i, mid in enumerate(ids):
        if mid in blocked or mid in seen:
            continue
        organs = plan.organ_keywords if i == 0 else None
        pseudo = SegmentationLlmPlan(
            model_id=mid,
            model_ids=[mid],
            organ_keywords=list(organs or []),
            rationale=plan.rationale,
            is_tumor_model=plan.is_tumor_model,
        )
        spec = _spec_from_plan(pseudo, models=models)
        if spec is None:
            continue
        seen.add(spec.model_id)
        out.append(spec)
    return out


def _is_lung_ct_disease(disease: str, modality: str) -> bool:
    mod = (modality or "CT").strip().upper()
    if mod not in {"CT", "CTPT", "PT"}:
        return False
    d = (disease or "").lower()
    return any(
        tok in d
        for tok in (
            "lung",
            "luad",
            "lusc",
            "nsclc",
            "pulmonary",
            "adenocarcinoma",
            "squamous",
        )
    )


def _is_incomplete_lung_plan(
    disease: str, modality: str, plan: SegmentationLlmPlan
) -> bool:
    """Singleton / monai-only lung caches are treated as incomplete."""
    if not _is_lung_ct_disease(disease, modality):
        return False
    ids = list(plan.model_ids)
    if plan.model_id and plan.model_id not in ids:
        ids.insert(0, plan.model_id)
    ready = [m for m in ids if m in _LUNG_READY_TUMOR_IDS]
    if len(ready) < 2:
        return True
    # Legacy poison: monai alone as first+only ready tumor for LUAD/LUSC masses.
    d = (disease or "").lower()
    primary_mass = any(t in d for t in ("luad", "lusc", "adenocarcinoma", "squamous"))
    if primary_mass and ready[0] == "monai_lung_nodule" and len(ready) < 3:
        return True
    return False


def expand_ranked_tumor_specs(
    specs: list[TumorModelSpec],
    *,
    disease: str,
    modality: str,
    models: tuple[TumorModelSpec, ...] | None,
    blocked: set[str] | None = None,
) -> list[TumorModelSpec]:
    """Append missing ready tumor models from the disease catalog (keep order)."""
    from agentic_radiogen.imaging.tumor_models.registry import list_models_for_disease
    from agentic_radiogen.imaging.tumor_models.segmentation_select import is_selectable

    blocked = set(blocked or ())
    out = list(specs)
    seen = {s.model_id for s in out}
    for cand in list_models_for_disease(
        disease, modality, models=models, direct_only=True
    ):
        if cand.model_id in seen or cand.model_id in blocked:
            continue
        if not cand.is_tumor_model or cand.operational_type != "ready_inference":
            continue
        if not is_selectable(cand):
            continue
        out.append(cand)
        seen.add(cand.model_id)
    return out


def _plan_from_llm_data(data: dict[str, Any]) -> SegmentationLlmPlan:
    organs = data.get("organ_keywords") or []
    if isinstance(organs, str):
        organs = [organs]
    model_ids = data.get("model_ids") or []
    model_id = str(data.get("model_id") or "").strip()
    plan = SegmentationLlmPlan(
        model_id=model_id,
        model_ids=list(model_ids) if model_ids else ([model_id] if model_id else []),
        organ_keywords=[str(x).strip().lower() for x in organs if str(x).strip()],
        rationale=str(data.get("rationale") or "").strip(),
        is_tumor_model=bool(data.get("is_tumor_model", False)),
    )
    if not plan.model_id and plan.model_ids:
        plan = plan.model_copy(update={"model_id": plan.model_ids[0]})
    if plan.model_id and plan.model_id not in plan.model_ids:
        plan = plan.model_copy(update={"model_ids": [plan.model_id, *plan.model_ids]})
    return plan


def _plan_from_raw(raw: dict[str, Any]) -> SegmentationLlmPlan:
    return _plan_from_llm_data(raw)


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
    specs, plan = _rules_plan_order(
        disease=disease,
        modality=modality,
        models=models,
        blocked=blocked,
        use_cache=use_cache,
        cache_path=cache_path,
        verbose=verbose,
    )
    return specs[0], plan


def _rules_plan_order(
    *,
    disease: str,
    modality: str,
    models: tuple[TumorModelSpec, ...] | None,
    blocked: set[str],
    use_cache: bool,
    cache_path: Path,
    verbose: bool,
) -> tuple[list[TumorModelSpec], SegmentationLlmPlan]:
    from agentic_radiogen.imaging.tumor_models.registry import list_models_for_disease

    picks = list_models_for_disease(disease, modality, models=models, direct_only=True)
    specs: list[TumorModelSpec] = []
    for cand in picks:
        if cand.model_id in blocked:
            continue
        if cand.operational_type == "fine_tune":
            continue
        specs.append(cand)
    if not specs:
        proxy = get_model("threshold_proxy", models=models)
        assert proxy is not None
        specs = [proxy]
    specs = expand_ranked_tumor_specs(
        specs, disease=disease, modality=modality, models=models, blocked=blocked
    )
    plan = SegmentationLlmPlan(
        model_id=specs[0].model_id,
        model_ids=[s.model_id for s in specs[:5]],
        organ_keywords=list(specs[0].organ_keywords),
        rationale="rules fallback (preferred model blocked this run)",
        is_tumor_model=specs[0].is_tumor_model,
    )
    if use_cache and specs[0].model_id != "threshold_proxy":
        _write_plan_cache(cache_path, disease, modality, plan)
    log(
        f"[segmentation-llm] Rules ranked {[s.model_id for s in specs[:5]]} "
        f"(blocked={sorted(blocked)})",
        enabled=verbose,
    )
    return specs, plan


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
                    "model_ids": list(plan.model_ids or [plan.model_id]),
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
    model_ids: list[str] | None = None,
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
    ids = list(model_ids or [])
    if model_id and model_id not in ids:
        ids.insert(0, model_id)
    path.write_text(
        json.dumps(
            {
                "version": _PLAN_CACHE_VERSION,
                "disease": disease,
                "modality": modality,
                "model_id": model_id,
                "model_ids": ids,
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
