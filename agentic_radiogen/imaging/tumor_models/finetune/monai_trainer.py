"""Fine-tune MONAI / foundation segmentation weights on NIfTI pairs."""

from __future__ import annotations

from pathlib import Path

from agentic_radiogen.imaging.tumor_models.finetune.base import (
    CasePair,
    FineTuneConfig,
    FineTuneResult,
)
from agentic_radiogen.imaging.tumor_models.finetune.resolve import resolve_in_checkpoint
from agentic_radiogen.imaging.tumor_models.finetune.torch_loop import run_torch_finetune


class MonaiFineTuneTrainer:
    name = "monai"

    def resolve_pretrained(self, cfg: FineTuneConfig) -> Path:
        return resolve_in_checkpoint(cfg)

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
        pretrained = self.resolve_pretrained(cfg)
        return run_torch_finetune(
            cfg, cases, pretrained=pretrained, backend_name=self.name
        )
