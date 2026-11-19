#!/usr/bin/env python3
"""Fine-tune a tumor segmentation model on a labeled dataset.

Examples
--------
# LIDC-IDRI DICOM tree (XML pulled from sibling manifest-*/LIDC-IDRI):
python scripts/finetune_segmentation.py \\
  --model nnunet --dataset ntcia \\
  --data-path /mnt/i/Data/LungCancer/LIDC-IDRI/lidc_idri \\
  --out-ckpt outputs/finetune/nnunet_ntcia/best.pt \\
  --mode tune-only --epoch 50

# Hold out 20% for evaluation; baseline + per-epoch metrics; best ckpt:
python scripts/finetune_segmentation.py \\
  --model nnunet --dataset ntcia \\
  --data-path /mnt/i/Data/LungCancer/LIDC-IDRI/lidc_idri \\
  --out-ckpt outputs/finetune/nnunet_ntcia/best.pt \\
  --mode evl --epoch 50

Dataset layouts under ``--data-path``
-------------------------------------
- LIDC-IDRI: ``LIDC-IDRI-*/<study>/<series>/*.dcm`` (+ XML in-tree or
  sibling ``manifest-*/LIDC-IDRI``). NIfTI cache: ``<root>/_finetune_cache/``.
- ``imagesTr/`` + ``labelsTr/`` (nnU-Net raw)
- ``images/`` + ``labels/`` (matched stems)
- ``<case_id>/{image,label}.nii.gz`` (per-case folders)

Optional: ``AGENTIC_RADIOGEN_LIDC_MAX_CASES=N`` limits how many patients are
converted on first run (useful for smoke tests).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_radiogen.imaging.tumor_models.finetune import (
    FineTuneConfig,
    list_datasets,
    list_finetune_models,
    run_finetune,
)


def build_parser() -> argparse.ArgumentParser:
    models = ", ".join(list_finetune_models())
    datasets = ", ".join(sorted(list_datasets()))
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune a tumor segmentation model. "
            f"Models: {models}. Datasets: {datasets}."
        )
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model to fine-tune (e.g. nnunet, nnunet_msd_lung, monai_vista3d, "
        "monai_swin_unetr_ssl). Loads catalog pretrained weights unless --in-ckpt.",
    )
    p.add_argument(
        "--dataset",
        default="ntcia",
        help="Dataset name / layout alias (default: ntcia). "
        f"Known: {datasets}.",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Root folder of the labeled dataset (NIfTI pairs).",
    )
    p.add_argument(
        "--out-ckpt",
        type=Path,
        default=None,
        help="Where to write the fine-tuned checkpoint "
        "(best eval-Dice in --mode evl; last epoch in tune-only).",
    )
    p.add_argument(
        "--in-ckpt",
        type=Path,
        default=None,
        help=(
            "Optional prior fine-tune checkpoint to CONTINUE training from. "
            "Baseline metrics always use the immutable catalog download "
            "(data_cache/.../*.pristine), never this file. "
            "When omitted, fine-tune also starts from the catalog weights."
        ),
    )
    p.add_argument(
        "--mode",
        default="tune-only",
        choices=["tune-only", "evl", "eval"],
        help=(
            "tune-only: train on all data. "
            "evl/eval: hold out 20%% for evaluation; print baseline metrics "
            "before fine-tune and train/eval metrics each epoch; save best ckpt."
        ),
    )
    p.add_argument(
        "--epoch",
        "--epochs",
        dest="epoch",
        type=int,
        default=50,
        help="Number of fine-tune epochs (default: 50).",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Scratch / history directory "
        "(default: outputs/finetune/<model>_<dataset>_<mode>).",
    )
    p.add_argument("--fold", type=int, default=0, help="Reserved (default 0).")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (nnU-Net backend auto-lowers 1e-4 → 1e-5 unless overridden).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Eval holdout fraction when --mode evl (default: 0.2).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Do not auto-download pretrained weights if missing.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover cases + resolve checkpoint; skip actual training.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print known models/datasets and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print(
            json.dumps(
                {"models": list_finetune_models(), "datasets": list_datasets()},
                indent=2,
            )
        )
        return 0

    missing = [
        name
        for name, val in (
            ("--model", args.model),
            ("--data-path", args.data_path),
            ("--out-ckpt", args.out_ckpt),
        )
        if val is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))

    mode = "evl" if args.mode == "eval" else args.mode

    cfg = FineTuneConfig(
        model=args.model,
        dataset=args.dataset,
        data_path=args.data_path,
        out_ckpt=args.out_ckpt,
        in_ckpt=args.in_ckpt,
        work_dir=args.work_dir,
        mode=mode,
        epochs=args.epoch,
        fold=args.fold,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        allow_download=not args.no_download,
        dry_run=args.dry_run,
    )
    result = run_finetune(cfg)
    payload = {
        "out_ckpt": str(result.out_ckpt),
        "work_dir": str(result.work_dir),
        "mode": result.mode,
        "n_train": result.n_train,
        "n_eval": result.n_val,
        "model": result.model,
        "dataset": result.dataset,
        "notes": result.notes,
        "baseline_metrics": result.baseline_metrics,
        "best_metrics": result.best_metrics,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
