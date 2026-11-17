"""Fine-tune tumor segmentation models on labeled datasets (e.g. NTCIA)."""

from __future__ import annotations

from pathlib import Path

from agentic_radiogen.imaging.tumor_models.finetune.base import (
    CasePair,
    FineTuneConfig,
    FineTuneResult,
)
from agentic_radiogen.imaging.tumor_models.finetune.datasets import (
    discover_cases,
    list_datasets,
)
from agentic_radiogen.imaging.tumor_models.finetune.resolve import (
    list_finetune_models,
    normalize_model_name,
    resolve_spec,
    trainer_backend,
)
from agentic_radiogen.imaging.tumor_models.finetune.torch_loop import normalize_mode
from agentic_radiogen.util.progress import log


def get_trainer(model: str):
    """Return the fine-tune backend for ``--model``."""
    spec = resolve_spec(model)
    backend = trainer_backend(spec)
    if backend == "nnunet":
        from agentic_radiogen.imaging.tumor_models.finetune.nnunet_trainer import (
            NnUNetFineTuneTrainer,
        )

        return NnUNetFineTuneTrainer()
    if backend == "monai":
        from agentic_radiogen.imaging.tumor_models.finetune.monai_trainer import (
            MonaiFineTuneTrainer,
        )

        return MonaiFineTuneTrainer()
    raise ValueError(f"No trainer for backend '{backend}'")


def run_finetune(cfg: FineTuneConfig) -> FineTuneResult:
    """Discover cases, initialize weights, fine-tune, write ``--out-ckpt``."""
    model = normalize_model_name(cfg.model)
    cfg.model = model
    cfg.mode = normalize_mode(cfg.mode)
    spec = resolve_spec(model)
    log(
        f"[finetune] model={model} mode={cfg.mode} runner={spec.runner} "
        f"dataset={cfg.dataset} data_path={cfg.data_path} epochs={cfg.epochs}",
        enabled=True,
    )
    cases = discover_cases(cfg.dataset, Path(cfg.data_path))
    trainer = get_trainer(model)
    result = trainer.run(cfg, cases)
    log(
        f"[finetune] Done. out_ckpt={result.out_ckpt} "
        f"n_train={result.n_train} n_val={result.n_val} {result.notes}",
        enabled=True,
    )
    return result


__all__ = [
    "CasePair",
    "FineTuneConfig",
    "FineTuneResult",
    "discover_cases",
    "get_trainer",
    "list_datasets",
    "list_finetune_models",
    "normalize_model_name",
    "resolve_spec",
    "run_finetune",
    "trainer_backend",
]
