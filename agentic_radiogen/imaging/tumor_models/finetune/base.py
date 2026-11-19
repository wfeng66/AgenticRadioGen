"""Shared types for tumor-segmentation fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class CasePair:
    """One labeled training case (image path + label path)."""

    case_id: str
    image: Path
    label: Path


@dataclass
class FineTuneConfig:
    """CLI-resolved fine-tune job."""

    model: str
    dataset: str
    data_path: Path
    out_ckpt: Path
    in_ckpt: Path | None = None
    work_dir: Path | None = None
    # tune-only: train on all cases. evl: hold out val_frac for evaluation.
    mode: str = "tune-only"
    epochs: int = 50
    fold: int = 0
    batch_size: int = 2
    lr: float = 1e-4
    val_frac: float = 0.2
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda"
    allow_download: bool = True
    dry_run: bool = False
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class FineTuneResult:
    """Artifact locations after a successful fine-tune."""

    out_ckpt: Path
    work_dir: Path
    n_train: int
    n_val: int
    model: str
    dataset: str
    mode: str = "tune-only"
    notes: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    # Flat metrics dict, or dual-protocol map {fg_centered|volume_center: metrics}.
    baseline_metrics: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None


class FineTuneTrainer(Protocol):
    """Backend that can initialize a model and run fine-tuning."""

    name: str

    def resolve_pretrained(self, cfg: FineTuneConfig) -> Path:
        """Return the checkpoint used to initialize training."""

    def run(self, cfg: FineTuneConfig, cases: list[CasePair]) -> FineTuneResult:
        """Fine-tune and write ``cfg.out_ckpt``."""
