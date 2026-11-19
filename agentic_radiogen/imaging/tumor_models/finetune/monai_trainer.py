"""Fine-tune MONAI / foundation segmentation weights on NIfTI pairs."""

from __future__ import annotations

from pathlib import Path

from agentic_radiogen.imaging.tumor_models.finetune.base import (
    CasePair,
    FineTuneConfig,
    FineTuneResult,
)
from agentic_radiogen.imaging.tumor_models.finetune.resolve import (
    assert_out_ckpt_safe,
    resolve_catalog_pretrained,
    resolve_train_init_checkpoint,
)
from agentic_radiogen.imaging.tumor_models.finetune.torch_loop import run_torch_finetune


class MonaiFineTuneTrainer:
    name = "monai"

    def resolve_pretrained(self, cfg: FineTuneConfig) -> Path:
        return resolve_catalog_pretrained(cfg)

    def run(self, cfg: FineTuneConfig, cases: list[CasePair]) -> FineTuneResult:
        from agentic_radiogen.imaging.tumor_models.finetune.torch_loop import (
            normalize_mode,
        )

        mode = normalize_mode(cfg.mode)
        cfg.mode = mode
        if len(cases) < 1:
            raise ValueError(f"Need ≥1 labeled cases; got {len(cases)}")
        if mode == "evl" and len(cases) < 2:
            raise ValueError("mode=evl needs ≥2 labeled cases for a holdout split")
        catalog = resolve_catalog_pretrained(cfg)
        train_init = resolve_train_init_checkpoint(cfg, catalog=catalog)
        assert_out_ckpt_safe(cfg.out_ckpt, catalog)
        return run_torch_finetune(
            cfg,
            cases,
            pretrained=catalog,
            train_init=train_init,
            backend_name=self.name,
        )
