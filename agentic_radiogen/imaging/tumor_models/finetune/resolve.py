"""Map ``--model`` aliases to catalog specs and pretrained checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path

from agentic_radiogen.imaging.tumor_models.finetune.base import FineTuneConfig
from agentic_radiogen.imaging.tumor_models.registry import (
    DEFAULT_MODELS,
    TumorModelSpec,
    get_model,
)
from agentic_radiogen.util.progress import log

# Sidecar copy of the Zenodo/catalog download; baseline always loads this.
_PRISTINE_SUFFIX = ".pristine"

# Short CLI names → catalog model_id (or trainer backend key).
MODEL_ALIASES: dict[str, str] = {
    "nnunet": "nnunet_msd_lung",
    "nnunet_msd": "nnunet_msd_lung",
    "nnunet_msd_lung": "nnunet_msd_lung",
    "task006": "nnunet_msd_lung",
    "nnunet_task998": "nnunet_task998_lungct",
    "task998": "nnunet_task998_lungct",
    "nnunet_task998_lungct": "nnunet_task998_lungct",
    "nnunet_autopet": "nnunet_autopet",
    "autopet": "nnunet_autopet",
    "vista3d": "monai_vista3d",
    "monai_vista3d": "monai_vista3d",
    "swin_unetr": "monai_swin_unetr_ssl",
    "monai_swin_unetr_ssl": "monai_swin_unetr_ssl",
    "stu_net": "stu_net",
    "stunet": "stu_net",
}


def normalize_model_name(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_")
    return MODEL_ALIASES.get(key, key)


def resolve_spec(model: str) -> TumorModelSpec:
    model_id = normalize_model_name(model)
    spec = get_model(model_id)
    if spec is not None:
        return spec
    known = ", ".join(sorted({*MODEL_ALIASES, *(m.model_id for m in DEFAULT_MODELS)}))
    raise KeyError(f"Unknown model '{model}'. Known: {known}")


def trainer_backend(spec: TumorModelSpec) -> str:
    """Which fine-tune implementation to use for this catalog entry."""
    if spec.runner in {"nnunet", "nnunet_task998", "nnunet_autopet"}:
        return "nnunet"
    if spec.runner in {"monai_vista3d", "monai_bundle", "fine_tune"}:
        return "monai"
    if spec.model_id in {"monai_swin_unetr_ssl", "stu_net", "monai_vista3d"}:
        return "monai"
    raise ValueError(
        f"Model '{spec.model_id}' (runner={spec.runner}) is not supported for "
        "fine-tuning yet. Supported backends: nnunet, monai."
    )


def resolve_catalog_pretrained(
    cfg: FineTuneConfig,
    *,
    verbose: bool = True,
) -> Path:
    """Always the downloaded catalog weights (never ``--out-ckpt`` / prior FT).

    Writes a one-time ``*.pristine`` sibling copy and returns that path so the
    original download stays untouched and baseline is reproducible.
    """
    spec = resolve_spec(cfg.model)
    raw = _ensure_pretrained(spec, allow_download=cfg.allow_download, verbose=verbose)
    path = ensure_pristine_catalog_checkpoint(raw, verbose=verbose)
    log(
        f"[finetune] Catalog pretrained (baseline source, immutable) → {path}",
        enabled=verbose,
    )
    return path


def resolve_train_init_checkpoint(
    cfg: FineTuneConfig,
    *,
    catalog: Path,
    verbose: bool = True,
) -> Path:
    """Weights used to *start fine-tuning* (catalog by default).

    ``--in-ckpt`` continues from a prior fine-tune; baseline still uses ``catalog``.
    """
    if cfg.in_ckpt is None:
        return catalog
    path = Path(cfg.in_ckpt).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"--in-ckpt not found: {path}")
    if path == Path(catalog).resolve():
        return catalog
    if _looks_like_catalog_tree(path) and path.name.endswith(_PRISTINE_SUFFIX):
        return path
    log(
        f"[finetune] Train init from --in-ckpt (baseline still uses catalog) → {path}",
        enabled=verbose,
    )
    return path


def resolve_in_checkpoint(
    cfg: FineTuneConfig,
    *,
    verbose: bool = True,
) -> Path:
    """Backward-compatible: catalog pristine weights (ignores prior FT outs).

    Prefer :func:`resolve_catalog_pretrained` + :func:`resolve_train_init_checkpoint`.
    """
    return resolve_catalog_pretrained(cfg, verbose=verbose)


def ensure_pristine_catalog_checkpoint(
    ckpt_path: Path,
    *,
    verbose: bool = True,
) -> Path:
    """Copy catalog weights once to ``<name>.pristine`` and prefer that file.

    Refuses to treat an AgenticRadioGen fine-tune dump as the catalog source.
    """
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Catalog checkpoint missing: {ckpt_path}")
    if _is_agentic_finetune_checkpoint(ckpt_path):
        raise RuntimeError(
            f"Catalog path looks like a fine-tuned AgenticRadioGen checkpoint "
            f"(not the original nnU-Net download): {ckpt_path}. "
            "Delete/re-download Task006 under data_cache/seg_models/nnunet_msd_lung/ "
            "and re-run. Fine-tuned weights belong under outputs/finetune/ only."
        )
    pristine = ckpt_path.with_name(ckpt_path.name + _PRISTINE_SUFFIX)
    if not pristine.is_file():
        shutil.copy2(ckpt_path, pristine)
        log(
            f"[finetune] Saved immutable catalog copy → {pristine} "
            f"(original left unchanged at {ckpt_path})",
            enabled=verbose,
        )
    elif _is_agentic_finetune_checkpoint(pristine):
        raise RuntimeError(
            f"Corrupt pristine catalog copy (looks fine-tuned): {pristine}. "
            "Remove it and re-run so it is recreated from the Zenodo download."
        )
    return pristine


def assert_out_ckpt_safe(out_ckpt: Path, catalog: Path) -> None:
    """Refuse to overwrite catalog / pristine downloads with fine-tuned weights."""
    out = Path(out_ckpt).expanduser().resolve()
    cat = Path(catalog).expanduser().resolve()
    protected = {cat}
    if cat.name.endswith(_PRISTINE_SUFFIX):
        protected.add(Path(str(cat)[: -len(_PRISTINE_SUFFIX)]))
    else:
        protected.add(cat.with_name(cat.name + _PRISTINE_SUFFIX))
    if out in protected:
        raise ValueError(
            f"--out-ckpt must not overwrite catalog weights ({out}). "
            "Write fine-tuned checkpoints under outputs/finetune/ (or another path "
            "outside data_cache/seg_models/)."
        )
    # Block writes into the same fold/ or trainer/ directory as the download
    # (not the entire ancestor tree — project roots often sit above data_cache/).
    fold_dir = cat.parent.resolve()
    trainer_dir = (
        fold_dir.parent
        if fold_dir.name.startswith("fold_")
        else fold_dir
    )
    out_parent = out.parent.resolve()
    if out_parent == fold_dir or out_parent == trainer_dir.resolve():
        raise ValueError(
            f"--out-ckpt {out} is inside the catalog model directory "
            f"({out_parent}). Use a path under outputs/finetune/ instead."
        )


def _is_agentic_finetune_checkpoint(path: Path) -> bool:
    """True when file is our torch.save dump (has backend/dataset), not nnU-Net v1."""
    try:
        import torch

        try:
            blob = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(str(path), map_location="cpu")
    except Exception:
        return False
    if not isinstance(blob, dict):
        return False
    # Our _save_ckpt payload.
    return (
        "state_dict" in blob
        and "backend" in blob
        and "dataset" in blob
        and "pretrained" in blob
    )


def _looks_like_catalog_tree(path: Path) -> bool:
    p = path if path.is_dir() else path.parent
    return (p / "plans.pkl").is_file() or (p.parent / "plans.pkl").is_file()


def _ensure_pretrained(
    spec: TumorModelSpec,
    *,
    allow_download: bool,
    verbose: bool,
) -> Path:
    if spec.runner == "nnunet":
        from agentic_radiogen.imaging.tumor_models.nnunet_lung import (
            ensure_nnunet_msd_lung,
        )

        trainer = ensure_nnunet_msd_lung(
            allow_download=allow_download, verbose=verbose
        )
        ckpt = _find_nnunet_checkpoint(trainer)
        if ckpt is None:
            raise FileNotFoundError(f"No nnU-Net checkpoint under {trainer}")
        return ckpt

    if spec.runner == "nnunet_task998":
        from agentic_radiogen.imaging.tumor_models.nnunet_task998 import (
            _find_trainer,
            task_root,
        )

        trainer = _find_trainer(task_root())
        if trainer is None:
            raise FileNotFoundError(
                "Task998 weights missing. Install manually and set "
                "AGENTIC_RADIOGEN_TASK998_MODEL_DIR."
            )
        ckpt = _find_nnunet_checkpoint(trainer)
        if ckpt is None:
            raise FileNotFoundError(f"No Task998 checkpoint under {trainer}")
        return ckpt

    if spec.runner == "nnunet_autopet":
        from agentic_radiogen.imaging.tumor_models.nnunet_autopet import (
            ensure_nnunet_autopet,
        )

        trainer = ensure_nnunet_autopet(
            allow_download=allow_download, verbose=verbose
        )
        ckpt = _find_nnunet_checkpoint(trainer)
        if ckpt is None:
            raise FileNotFoundError(f"No AutoPET checkpoint under {trainer}")
        return ckpt

    if spec.runner == "monai_vista3d":
        from agentic_radiogen.imaging.tumor_models.monai_vista3d import (
            ensure_monai_vista3d,
        )

        bundle = ensure_monai_vista3d(
            allow_download=allow_download, verbose=verbose
        )
        for rel in (
            "models/model.pt",
            "models/model.ts",
            "models/model.pt.ckpt",
        ):
            hit = bundle / rel
            if hit.is_file():
                return hit
        raise FileNotFoundError(f"No VISTA-3D weights under {bundle}")

    if spec.model_id == "monai_swin_unetr_ssl":
        return _ensure_hf_weight(
            spec,
            allow_download=allow_download,
            verbose=verbose,
            default_name="model_swinvit.pt",
        )

    if spec.model_id == "stu_net":
        return _ensure_hf_weight(
            spec,
            allow_download=allow_download,
            verbose=verbose,
            default_name="base_ep4k.model",
        )

    raise ValueError(f"No pretrained resolver for model '{spec.model_id}'")


def _find_nnunet_checkpoint(trainer_dir: Path) -> Path | None:
    """Prefer fold_0 final checkpoint (v1 .model or v2 .pth)."""
    fold0 = trainer_dir / "fold_0"
    candidates = [
        fold0 / "model_final_checkpoint.model",
        fold0 / "checkpoint_final.pth",
        fold0 / "model_latest.model",
        fold0 / "checkpoint_latest.pth",
        trainer_dir / "checkpoint_final.pth",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # Any fold
    for fold in sorted(trainer_dir.glob("fold_*")):
        for name in (
            "model_final_checkpoint.model",
            "checkpoint_final.pth",
            "model_latest.model",
            "checkpoint_latest.pth",
        ):
            hit = fold / name
            if hit.is_file():
                return hit
    return None


def _ensure_hf_weight(
    spec: TumorModelSpec,
    *,
    allow_download: bool,
    verbose: bool,
    default_name: str,
) -> Path:
    from agentic_radiogen.imaging.tumor_models.store import ModelStore

    store = ModelStore(verbose=verbose)
    dest = store.model_dir(spec)
    weight_name = spec.weight_filename or default_name
    local = dest / weight_name
    if local.is_file():
        return local
    if not allow_download:
        raise FileNotFoundError(
            f"Missing {local}; re-run with download enabled or pass --in-ckpt"
        )
    if not spec.hf_repo:
        raise FileNotFoundError(
            f"No local weight at {local} and no hf_repo configured for {spec.model_id}"
        )
    log(
        f"[finetune] Downloading HuggingFace repo {spec.hf_repo} → {dest}",
        enabled=verbose,
    )
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download fine-tune backbones "
            "(pip install huggingface_hub)"
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    try:
        path = Path(
            hf_hub_download(
                repo_id=spec.hf_repo,
                filename=weight_name,
                local_dir=str(dest),
            )
        )
        if path.is_file():
            return path
    except Exception:
        # Fall back to a full snapshot and search for the weight file.
        snapshot_download(repo_id=spec.hf_repo, local_dir=str(dest))
    hits = list(dest.rglob(weight_name))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"Downloaded {spec.hf_repo} but could not find {weight_name} under {dest}"
    )


def list_finetune_models() -> list[str]:
    """Models the fine-tune CLI accepts (aliases + catalog ids)."""
    ids = {
        "nnunet",
        "nnunet_msd_lung",
        "nnunet_task998",
        "nnunet_task998_lungct",
        "nnunet_autopet",
        "monai_vista3d",
        "vista3d",
        "monai_swin_unetr_ssl",
        "swin_unetr",
        "stu_net",
    }
    return sorted(ids)
