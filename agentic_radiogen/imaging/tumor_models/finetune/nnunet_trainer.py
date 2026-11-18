"""Fine-tune nnU-Net catalog checkpoints via the shared torch epoch loop.

Rebuilds the native nnU-Net v1 ``Generic_UNet`` from ``plans.pkl`` next to the
checkpoint (soft-loading into MONAI UNet is invalid and yields ~0 Dice).
``--mode`` and ``--epoch`` behave the same as the MONAI backend.
"""

from __future__ import annotations

from pathlib import Path

from agentic_radiogen.imaging.tumor_models.finetune.base import (
    CasePair,
    FineTuneConfig,
    FineTuneResult,
)
from agentic_radiogen.imaging.tumor_models.finetune.resolve import resolve_in_checkpoint
from agentic_radiogen.imaging.tumor_models.finetune.torch_loop import run_torch_finetune


class NnUNetFineTuneTrainer:
    name = "nnunet"

    def resolve_pretrained(self, cfg: FineTuneConfig) -> Path:
        return resolve_in_checkpoint(cfg)

    def run(self, cfg: FineTuneConfig, cases: list[CasePair]) -> FineTuneResult:
        from agentic_radiogen.imaging.tumor_models.finetune.torch_loop import (
            normalize_mode,
        )

        mode = normalize_mode(cfg.mode)
        cfg.mode = mode
        if len(cases) < 1:
            raise ValueError(
                f"nnU-Net fine-tune needs ≥1 labeled cases; got {len(cases)}"
            )
        if mode == "evl" and len(cases) < 2:
            raise ValueError("mode=evl needs ≥2 labeled cases for a holdout split")
        pretrained = self.resolve_pretrained(cfg)
        return run_torch_finetune(
            cfg, cases, pretrained=pretrained, backend_name=self.name
        )
