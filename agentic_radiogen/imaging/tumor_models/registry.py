"""Catalog of segmentation backends for the SegmentationAgent.

The planner picks a model for **segmentation only**. Feature extraction is a
separate agent (ImagingRadiomicsAgent) and must not drive model choice.

Operational types
-----------------
- ``ready_inference``: can produce a mask with direct inference (no training/prompts).
- ``fine_tune``: foundation / SSL weights — initialize training; not a live segmentor.
- ``promptable``: needs a click / box prompt for interactive GTV.
- ``organ``: whole-organ labels (site mask, not lesion).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TumorModelSpec:
    """One segmentation backend in the catalog."""

    model_id: str
    diseases: tuple[str, ...]  # keywords; ("*",) = any
    modalities: tuple[str, ...]
    runner: str
    # totalsegmentator_organ | totalsegmentator_task | threshold_proxy
    # | torchscript | huggingface | nnunet | nnunet_autopet | monai_bundle
    # | promptable | fine_tune
    description: str
    organ_keywords: tuple[str, ...] = ()
    weight_filename: str = ""
    source: str = "builtin"  # builtin | totalsegmentator | huggingface | url | zenodo | gdrive
    hf_repo: str | None = None
    download_url_env: str | None = None
    download_url: str | None = None
    is_tumor_model: bool = False
    priority: int = 100
    # ready_inference | fine_tune | promptable | organ
    operational_type: str = "organ"
    # True → ``segment()`` can run this unattended without prompts/training.
    direct_inference: bool = True
    # TotalSegmentator ``task=`` name when runner is totalsegmentator_task.
    task_name: str = ""
    # Optional extras (e.g. Google Drive file id, MONAI bundle name).
    extra: dict[str, str] | None = None


def _env_url(env_name: str | None) -> str | None:
    if not env_name:
        return None
    return (os.environ.get(env_name) or "").strip() or None


def model_download_url(spec: TumorModelSpec) -> str | None:
    return _env_url(spec.download_url_env) or (spec.download_url or None)


def _ts(
    model_id: str,
    *,
    organs: tuple[str, ...],
    diseases: tuple[str, ...],
    description: str,
    modalities: tuple[str, ...] = ("CT",),
    priority: int = 40,
) -> TumorModelSpec:
    return TumorModelSpec(
        model_id=model_id,
        diseases=diseases,
        modalities=modalities,
        runner="totalsegmentator_organ",
        description=description,
        organ_keywords=organs,
        source="totalsegmentator",
        is_tumor_model=False,
        priority=priority,
        operational_type="organ",
        direct_inference=True,
    )


DEFAULT_MODELS: tuple[TumorModelSpec, ...] = (
    # --- 1. Ready-to-use lesion / tumor (direct inference) ---
    TumorModelSpec(
        model_id="nnunet_msd_lung",
        diseases=("lung", "nsclc", "sclc", "luad", "lusc", "pulmonary"),
        modalities=("CT",),
        runner="nnunet",
        description=(
            "nnU-Net Task006_Lung (Zenodo #4003545). Top pick for large primary NSCLC "
            "masses on CT. Auto-downloads ~5GB weights on first use."
        ),
        source="zenodo",
        download_url=(
            "https://zenodo.org/records/4003545/files/Task006_Lung.zip?download=1"
        ),
        is_tumor_model=True,
        priority=5,
        operational_type="ready_inference",
        direct_inference=True,
    ),
    TumorModelSpec(
        model_id="monai_lung_nodule",
        diseases=("lung", "nsclc", "sclc", "luad", "lusc", "pulmonary", "nodule"),
        modalities=("CT",),
        runner="monai_bundle",
        description=(
            "MONAI Model Zoo lung_nodule_ct_detection (LUNA16 RetinaNet). Detects "
            "pulmonary nodules/masses on CT and rasterizes boxes to a lesion mask. "
            "Auto-downloads from HuggingFace MONAI/lung_nodule_ct_detection."
        ),
        source="huggingface",
        hf_repo="MONAI/lung_nodule_ct_detection",
        is_tumor_model=True,
        priority=12,
        operational_type="ready_inference",
        direct_inference=True,
        extra={"bundle_name": "lung_nodule_ct_detection"},
    ),
    TumorModelSpec(
        model_id="nnunet_autopet",
        diseases=(
            "lung",
            "nsclc",
            "sclc",
            "luad",
            "lusc",
            "pulmonary",
            "lymphoma",
            "melanoma",
            "fdg",
        ),
        modalities=("PT", "CT"),
        runner="nnunet_autopet",
        description=(
            "AutoPET nnU-Net (lab-midas/autoPET / AutoPET II). Metabolic FDG lesion "
            "segmentation for infiltrating/heterogeneous thoracic tumors. Needs "
            "PET+CT (skips CT-only). Weights: Google Drive baseline or Zenodo "
            "MIC-DKFZ AutoPET II."
        ),
        source="gdrive",
        download_url_env="AGENTIC_RADIOGEN_AUTOPET_URL",
        download_url=(
            "https://zenodo.org/records/8362371/files/"
            "nnUNetTrainer__nnUNetPlans__3d_fullres_resenc_bs80_exported.zip?download=1"
        ),
        is_tumor_model=True,
        priority=18,
        operational_type="ready_inference",
        direct_inference=True,
        extra={"gdrive_id": "1G0HGHzQMXzslGDxFSNs5fq3RCeAu7M6l"},
    ),
    TumorModelSpec(
        model_id="ts_lung_nodules",
        diseases=("lung", "nsclc", "sclc", "luad", "lusc", "pulmonary", "nodule"),
        modalities=("CT",),
        runner="totalsegmentator_task",
        description=(
            "TotalSegmentator task=lung_nodules (experimental). Off by default due to "
            "nnU-Net crop IO errors in batch runs. Enable with "
            "AGENTIC_RADIOGEN_ENABLE_TS_LUNG_NODULES=1."
        ),
        source="totalsegmentator",
        is_tumor_model=True,
        priority=45,
        operational_type="ready_inference",
        direct_inference=True,
        task_name="lung_nodules",
    ),
    # --- 2. Foundation / SSL (fine-tune only; not live segmentation) ---
    TumorModelSpec(
        model_id="monai_swin_unetr_ssl",
        diseases=("lung", "nsclc", "luad", "lusc", "*"),
        modalities=("CT", "MR"),
        runner="fine_tune",
        description=(
            "MONAI Swin UNETR SSL checkpoint (~50k CT/MRI). Encoder init for "
            "fine-tuning — not a live segmentor."
        ),
        source="huggingface",
        hf_repo="Project-MONAI/model-zoo",
        weight_filename="model_swinvit.pt",
        is_tumor_model=False,
        priority=200,
        operational_type="fine_tune",
        direct_inference=False,
    ),
    TumorModelSpec(
        model_id="stu_net",
        diseases=("lung", "nsclc", "luad", "lusc", "*"),
        modalities=("CT",),
        runner="fine_tune",
        description=(
            "STU-Net (uni-medical/STU-Net): pre-trained on large-scale CT. "
            "Fine-tune only — not live segmentation."
        ),
        source="huggingface",
        hf_repo="uni-medical/STU-Net",
        is_tumor_model=False,
        priority=210,
        operational_type="fine_tune",
        direct_inference=False,
    ),
    # --- 3. Promptable medical foundation (interactive GTV) ---
    TumorModelSpec(
        model_id="medsam",
        diseases=("lung", "nsclc", "luad", "lusc", "*"),
        modalities=("CT", "MR"),
        runner="promptable",
        description=(
            "MedSAM / MedSAM-2: 2D box-prompt lesion segmentation. Requires a "
            "bounding-box prompt (not unattended)."
        ),
        source="huggingface",
        weight_filename="medsam_vit_b.pth",
        is_tumor_model=True,
        priority=220,
        operational_type="promptable",
        direct_inference=False,
    ),
    TumorModelSpec(
        model_id="sam_med3d",
        diseases=("lung", "nsclc", "luad", "lusc", "*"),
        modalities=("CT", "MR"),
        runner="promptable",
        description=(
            "SAM-Med3D: 3D point/box prompt → GTV. Requires an interactive prompt."
        ),
        source="url",
        weight_filename="sam_med3d_turbo.pth",
        is_tumor_model=True,
        priority=230,
        operational_type="promptable",
        direct_inference=False,
    ),
    # --- Organ site masks (fallback anatomy, not lesion) ---
    _ts(
        "ts_lung",
        organs=("lung",),
        diseases=("lung", "nsclc", "sclc", "luad", "lusc", "pulmonary"),
        description="TotalSegmentator lung organ labels (whole lung, not tumor).",
        priority=50,
    ),
    _ts(
        "ts_liver",
        organs=("liver",),
        diseases=("liver", "hcc", "hepatocellular", "hepatic"),
        description="TotalSegmentator liver (organ ROI).",
    ),
    _ts(
        "ts_kidney",
        organs=("kidney",),
        diseases=("kidney", "renal", "rcc", "renal_cell"),
        description="TotalSegmentator kidney (organ ROI).",
    ),
    _ts(
        "ts_pancreas",
        organs=("pancreas",),
        diseases=("pancreas", "pancreatic", "pdac"),
        description="TotalSegmentator pancreas (organ ROI).",
    ),
    _ts(
        "ts_colon",
        organs=("colon",),
        diseases=("colon", "colorectal", "rectal", "crc"),
        description="TotalSegmentator colon (organ ROI).",
    ),
    _ts(
        "ts_stomach",
        organs=("stomach",),
        diseases=("stomach", "gastric"),
        description="TotalSegmentator stomach (organ ROI).",
    ),
    _ts(
        "ts_spleen",
        organs=("spleen",),
        diseases=("spleen", "splenic"),
        description="TotalSegmentator spleen (organ ROI).",
    ),
    _ts(
        "ts_bladder",
        organs=("urinary_bladder", "bladder"),
        diseases=("bladder", "urothelial"),
        description="TotalSegmentator bladder (organ ROI).",
    ),
    _ts(
        "ts_prostate",
        organs=("prostate",),
        diseases=("prostate", "prad"),
        description="TotalSegmentator prostate (organ ROI).",
        modalities=("CT", "MR"),
    ),
    _ts(
        "ts_heart",
        organs=("heart",),
        diseases=("heart", "cardiac"),
        description="TotalSegmentator heart (organ ROI).",
    ),
    _ts(
        "ts_brain",
        organs=("brain",),
        diseases=("brain", "gbm", "glioblastoma", "glioma", "meningioma"),
        description="TotalSegmentator brain (organ ROI).",
        modalities=("CT", "MR"),
        priority=45,
    ),
    _ts(
        "ts_breast",
        organs=("breast",),
        diseases=("breast", "brca"),
        description="TotalSegmentator breast (organ ROI).",
        modalities=("CT", "MR"),
        priority=45,
    ),
    _ts(
        "ts_esophagus",
        organs=("esophagus",),
        diseases=("esophagus", "esophageal"),
        description="TotalSegmentator esophagus (organ ROI).",
    ),
    _ts(
        "ts_thyroid",
        organs=("thyroid",),
        diseases=("thyroid",),
        description="TotalSegmentator thyroid (organ ROI).",
    ),
    _ts(
        "ts_adrenal",
        organs=("adrenal",),
        diseases=("adrenal",),
        description="TotalSegmentator adrenal gland (organ ROI).",
    ),
    TumorModelSpec(
        model_id="totalsegmentator_organ",
        diseases=("*",),
        modalities=("CT", "MR"),
        runner="totalsegmentator_organ",
        description="TotalSegmentator with LLM-chosen organ keywords.",
        organ_keywords=(),
        source="totalsegmentator",
        is_tumor_model=False,
        priority=70,
        operational_type="organ",
        direct_inference=True,
    ),
    TumorModelSpec(
        model_id="threshold_proxy",
        diseases=("*",),
        modalities=("CT", "MR", "MRI", "PT"),
        runner="threshold_proxy",
        description="Intensity soft-tissue proxy when no DL model is available.",
        source="builtin",
        is_tumor_model=False,
        priority=90,
        operational_type="organ",
        direct_inference=True,
    ),
)


# Generic histology / filler words — must not match organ keywords like renal_cell.
_DISEASE_STOPWORDS = frozenset(
    {
        "cancer",
        "tumor",
        "tumour",
        "cell",
        "carcinoma",
        "adenocarcinoma",
        "squamous",
        "basaloid",
        "neoplasm",
        "malignant",
        "metastatic",
        "primary",
        "lesion",
        "mass",
        "disease",
        "type",
        "nos",
        "the",
        "and",
        "with",
        "from",
    }
)


def normalize_disease_token(disease: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (disease or "").strip().lower()).strip("_")


def _disease_tokens(disease: str) -> set[str]:
    key = normalize_disease_token(disease)
    parts = {key, *key.split("_")}
    return {p for p in parts if p and p not in _DISEASE_STOPWORDS and len(p) >= 3}


def catalog_for_llm(
    models: tuple[TumorModelSpec, ...] | None = None,
) -> list[dict[str, object]]:
    """Full catalog for the Segmentation planner (all operational types)."""
    rows = []
    for spec in models or DEFAULT_MODELS:
        rows.append(
            {
                "model_id": spec.model_id,
                "modalities": list(spec.modalities),
                "organ_keywords": list(spec.organ_keywords),
                "runner": spec.runner,
                "is_tumor_model": spec.is_tumor_model,
                "operational_type": spec.operational_type,
                "direct_inference": spec.direct_inference,
                "task_name": spec.task_name or None,
                "hf_repo": spec.hf_repo,
                "download_url": spec.download_url,
                "auto_download": spec.source
                in {"totalsegmentator", "huggingface", "zenodo", "gdrive"}
                or bool(spec.download_url)
                or bool(spec.hf_repo),
                "description": spec.description,
                "example_diseases": [d for d in spec.diseases if d != "*"][:8],
            }
        )
    return rows


def get_model(
    model_id: str,
    *,
    models: tuple[TumorModelSpec, ...] | None = None,
    organ_keywords: list[str] | tuple[str, ...] | None = None,
) -> TumorModelSpec | None:
    catalog = models or DEFAULT_MODELS
    for spec in catalog:
        if spec.model_id == model_id:
            if organ_keywords:
                return replace(
                    spec,
                    organ_keywords=tuple(
                        str(x).lower() for x in organ_keywords if str(x).strip()
                    ),
                )
            return spec
    return None


def _overlap_score(tokens: set[str], diseases: set[str]) -> int:
    """Higher = better disease match. Ignores catch-all '*'."""
    specific = {d for d in diseases if d != "*"}
    if not specific:
        return 0
    # Exact keyword hits (e.g. lung, lusc, kidney).
    score = 10 * len(tokens & specific)
    # Soft match only on longer, non-stopword tokens against whole disease keywords
    # or their underscore parts — never "cell" ⊂ "renal_cell".
    for t in tokens:
        if len(t) < 5 or t in _DISEASE_STOPWORDS:
            continue
        for d in specific:
            d_parts = {d, *d.split("_")} - _DISEASE_STOPWORDS
            if t == d or t in d_parts:
                score += 3
                continue
            # Bounded substring: token must be a substantial prefix/suffix of keyword.
            if len(d) >= 5 and (d.startswith(t) or d.endswith(t) or t.startswith(d)):
                score += 3
    return score


def list_models_for_disease(
    disease: str,
    modality: str = "CT",
    *,
    models: tuple[TumorModelSpec, ...] | None = None,
    direct_only: bool = False,
) -> list[TumorModelSpec]:
    """Heuristic candidates for the SegmentationAgent.

    Returns the full matching catalog by default (ready, organ, fine_tune,
    promptable). Set ``direct_only=True`` when building an unattended run queue.
    """
    catalog = models or DEFAULT_MODELS
    tokens = _disease_tokens(disease)
    mod = (modality or "CT").strip().upper()
    if mod == "MRI":
        mod = "MR"
    scored: list[tuple[int, TumorModelSpec]] = []
    for spec in catalog:
        if direct_only and not spec.direct_inference:
            continue
        mods = tuple("MR" if m.upper() == "MRI" else m.upper() for m in spec.modalities)
        if mod not in mods and "*" not in mods:
            continue
        diseases = {normalize_disease_token(d) for d in spec.diseases}
        if "*" in diseases:
            scored.append((1, spec))
            continue
        score = _overlap_score(tokens, diseases)
        if score <= 0:
            continue
        # Organ fallbacks need a real anatomy/cohort hit (score≥10), not a weak
        # substring like "cell" → renal_cell from "squamous cell carcinoma".
        if not spec.is_tumor_model and spec.operational_type == "organ" and score < 10:
            continue
        scored.append((score, spec))
    scored.sort(
        key=lambda item: (
            -item[0],
            0 if item[1].is_tumor_model else 1,
            0 if item[1].operational_type == "ready_inference" else 1,
            0 if item[1].direct_inference else 1,
            item[1].priority,
            item[1].model_id,
        )
    )
    return [spec for _, spec in scored]


def resolve_model(
    disease: str,
    modality: str = "CT",
    *,
    models: tuple[TumorModelSpec, ...] | None = None,
) -> TumorModelSpec | None:
    """Default unattended pick: best direct-inference match for disease."""
    from agentic_radiogen.imaging.tumor_models.segmentation_select import is_selectable

    hits = list_models_for_disease(
        disease, modality, models=models, direct_only=True
    )
    for spec in hits:
        if not is_selectable(spec):
            continue
        return spec
    return None
