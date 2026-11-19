"""Shared PyTorch fine-tune loop with tune-only / evl modes."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from agentic_radiogen.imaging.tumor_models.finetune.base import (
    CasePair,
    FineTuneConfig,
    FineTuneResult,
)
from agentic_radiogen.imaging.tumor_models.finetune.metrics import (
    average_metrics,
    binary_seg_metrics,
    format_metrics,
)
from agentic_radiogen.util.progress import log

_PATCH = (96, 96, 96)


def normalize_mode(mode: str | None) -> str:
    raw = (mode or "tune-only").strip().lower().replace("_", "-")
    if raw in {"tune-only", "tuneonly", "train", "train-only"}:
        return "tune-only"
    if raw in {"evl", "eval", "evaluate", "evaluation"}:
        return "evl"
    raise ValueError(
        f"Unknown --mode '{mode}'. Use 'tune-only' (all data) or 'evl' (20% holdout)."
    )


def split_for_mode(
    cases: list[CasePair],
    *,
    mode: str,
    val_frac: float,
    seed: int,
) -> tuple[list[CasePair], list[CasePair]]:
    mode = normalize_mode(mode)
    if mode == "tune-only":
        return list(cases), []
    rng = random.Random(seed)
    order = list(cases)
    rng.shuffle(order)
    frac = float(val_frac) if val_frac > 0 else 0.2
    n_val = max(1, int(round(len(order) * frac)))
    if n_val >= len(order):
        n_val = 1
    val = order[:n_val]
    train = order[n_val:]
    if not train:
        train, val = order[:-1], order[-1:]
    return train, val


def run_torch_finetune(
    cfg: FineTuneConfig,
    cases: list[CasePair],
    *,
    pretrained: Path,
    backend_name: str,
    train_init: Path | None = None,
) -> FineTuneResult:
    """Fine-tune from catalog ``pretrained`` (baseline) then optional ``train_init``.

    ``pretrained`` must be the immutable catalog download (``.pristine``). Baseline
    metrics always use those weights. Fine-tuning starts from ``train_init`` when
    set (e.g. ``--in-ckpt``), otherwise from the same catalog weights.
    """
    mode = normalize_mode(cfg.mode)
    train_cases, eval_cases = split_for_mode(
        cases, mode=mode, val_frac=cfg.val_frac, seed=cfg.seed
    )
    work = _work_dir(cfg)
    work.mkdir(parents=True, exist_ok=True)
    catalog = Path(pretrained).expanduser().resolve()
    init_path = Path(train_init or catalog).expanduser().resolve()

    if cfg.dry_run:
        out = Path(cfg.out_ckpt)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dry_run": True,
            "mode": mode,
            "model": cfg.model,
            "dataset": cfg.dataset,
            "backend": backend_name,
            "n_train": len(train_cases),
            "n_eval": len(eval_cases),
            "epochs": int(cfg.epochs),
            "pretrained_catalog": str(catalog),
            "train_init": str(init_path),
            "pretrained": str(catalog),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return FineTuneResult(
            out_ckpt=out,
            work_dir=work,
            n_train=len(train_cases),
            n_val=len(eval_cases),
            model=cfg.model,
            dataset=cfg.dataset,
            mode=mode,
            notes="dry_run",
            history=[payload],
        )

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required (pip install torch)") from exc

    device = torch.device(
        cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    monai_ver = "n/a"
    try:
        import monai

        monai_ver = monai.__version__
    except ImportError:
        monai = None  # type: ignore[assignment]

    log(
        f"[finetune] backend={backend_name} mode={mode} device={device} "
        f"monai={monai_ver} epochs={cfg.epochs} "
        f"n_train={len(train_cases)} n_eval={len(eval_cases)}",
        enabled=True,
    )

    patch_size = _PATCH
    intensity = None
    backend_l = str(backend_name).lower()
    log(
        f"[finetune] baseline_weights={catalog} | train_init={init_path}",
        enabled=True,
    )
    if backend_l in {"nnunet", "nnunetv1", "nnunet-v1"}:
        from agentic_radiogen.imaging.tumor_models.finetune.nnunet_weights import (
            load_nnunet_v1_network,
        )

        # Always build + load catalog first so baseline is the downloaded model.
        model, net_meta = load_nnunet_v1_network(
            catalog, device=device, verbose=True
        )
        patch_size = tuple(net_meta.get("patch_size") or _PATCH)
        intensity = net_meta.get("intensity")
        log(
            f"[finetune] nnU-Net patch_size={patch_size} "
            f"(native Generic_UNet; catalog baseline)",
            enabled=True,
        )
    else:
        if monai is None:
            raise RuntimeError(
                "MONAI is required for the MONAI fine-tune backend. Install with:\n"
                "  pip install -r requirements.txt\n"
                "or: pip install 'monai[nibabel]'"
            )
        from monai.networks.nets import UNet

        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
        soft_load(model, catalog, map_location=device)

    loss_fn = _build_dice_ce_loss(monai)

    train_loader = DataLoader(
        NiftiSegDataset(
            train_cases,
            patch_size=patch_size,
            training=True,
            intensity=intensity,
            fg_oversample=0.67,
        ),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        num_workers=max(0, int(cfg.num_workers)),
    )
    # Dual holdout protocols: FG-centered patch + full-volume sliding window.
    eval_loader_fg = None
    if eval_cases:
        eval_loader_fg = DataLoader(
            NiftiSegDataset(
                eval_cases,
                patch_size=patch_size,
                training=False,
                intensity=intensity,
                fg_oversample=1.0,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    # Epoch model-selection uses FG-centered (tumor always in patch).
    eval_loader = eval_loader_fg

    history: list[dict[str, Any]] = []
    baseline: dict[str, float] | None = None
    baseline_both: dict[str, dict[str, float]] | None = None
    out = Path(cfg.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    if mode == "evl" and eval_loader_fg is not None and eval_cases:
        log(
            "[finetune] Running dual baseline: fg_centered + full_volume "
            "(sliding-window whole scan)…",
            enabled=True,
        )
        baseline_both = {
            "fg_centered": evaluate_loader(model, eval_loader_fg, loss_fn, device),
            "full_volume": evaluate_full_volume(
                model,
                eval_cases,
                loss_fn=loss_fn,
                device=device,
                patch_size=patch_size,
                intensity=intensity,
            ),
        }
        baseline = baseline_both["fg_centered"]
        log(
            format_metrics(
                "[finetune] baseline fg_centered (catalog, before fine-tune)",
                baseline_both["fg_centered"],
            ),
            enabled=True,
        )
        log(
            format_metrics(
                "[finetune] baseline full_volume (catalog, before fine-tune)",
                baseline_both["full_volume"],
            ),
            enabled=True,
        )
        history.append(
            {
                "epoch": 0,
                "phase": "baseline",
                "eval": baseline_both,
                "weights": str(catalog),
            }
        )
        # Snapshot catalog weights into out_ckpt (not the download tree).
        _save_ckpt(
            out,
            model=model,
            cfg=cfg,
            epoch=0,
            pretrained=catalog,
            train_metrics=None,
            eval_metrics=baseline,
            baseline=baseline_both,
            mode=mode,
            backend_name=backend_name,
        )
        log(
            f"[finetune] saved catalog-baseline snapshot "
            f"(fg_dice={float(baseline_both['fg_centered'].get('dice', float('nan'))):.4f}, "
            f"full_dice={float(baseline_both['full_volume'].get('dice', float('nan'))):.4f}) "
            f"→ {out} (download unchanged: {catalog})",
            enabled=True,
        )

    # Optional: continue FT from a previous out_ckpt AFTER catalog baseline.
    if init_path != catalog:
        log(
            f"[finetune] Loading train_init for fine-tune (after catalog baseline) → {init_path}",
            enabled=True,
        )
        soft_load(model, init_path, map_location=device)

    # Fine-tune defaults: low LR + weight decay so pretrained weights are not wiped.
    lr = float(cfg.lr)
    if backend_l in {"nnunet", "nnunetv1", "nnunet-v1"} and abs(lr - 1e-4) < 1e-15:
        lr = 1e-5
        log(
            f"[finetune] nnU-Net fine-tune LR lowered {cfg.lr} → {lr} "
            "(pass --lr to override)",
            enabled=True,
        )
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, int(cfg.epochs)), eta_min=lr * 0.1
    )

    best_score = (
        float(baseline["dice"])
        if baseline is not None and np.isfinite(baseline.get("dice", float("nan")))
        else -1.0
    )
    best_epoch = 0
    last_train: dict[str, float] = {}
    last_eval: dict[str, float] | None = baseline

    for epoch in range(1, int(cfg.epochs) + 1):
        train_m = train_one_epoch(model, train_loader, loss_fn, optim, device)
        scheduler.step()
        last_train = train_m
        row: dict[str, Any] = {"epoch": epoch, "train": train_m, "lr": lr}

        if mode == "evl" and eval_loader is not None:
            eval_m = evaluate_loader(model, eval_loader, loss_fn, device)
            last_eval = eval_m
            row["eval"] = eval_m
            log(
                format_metrics(f"[finetune] epoch {epoch}/{cfg.epochs} train", train_m)
                + " | "
                + format_metrics("eval", eval_m),
                enabled=True,
            )
            score = float(eval_m.get("dice", float("nan")))
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_epoch = epoch
                _save_ckpt(
                    out,
                    model=model,
                    cfg=cfg,
                    epoch=epoch,
                    pretrained=catalog,
                    train_metrics=train_m,
                    eval_metrics=eval_m,
                    baseline=baseline,
                    mode=mode,
                    backend_name=backend_name,
                )
                log(
                    f"[finetune] new best checkpoint (eval_dice={best_score:.4f}) → {out}",
                    enabled=True,
                )
            elif np.isfinite(score) and baseline is not None and score < float(
                baseline.get("dice", best_score)
            ):
                log(
                    f"[finetune] epoch {epoch}: eval_dice={score:.4f} < baseline "
                    f"{float(baseline.get('dice', float('nan'))):.4f} "
                    "(keeping best ckpt)",
                    enabled=True,
                )
        else:
            log(
                format_metrics(f"[finetune] epoch {epoch}/{cfg.epochs} train", train_m),
                enabled=True,
            )
            # tune-only: keep the latest weights as the output checkpoint
            _save_ckpt(
                out,
                model=model,
                cfg=cfg,
                epoch=epoch,
                pretrained=catalog,
                train_metrics=train_m,
                eval_metrics=None,
                baseline=None,
                mode=mode,
                backend_name=backend_name,
            )
            best_epoch = epoch
            best_score = float(train_m.get("dice", 0.0))

        history.append(row)

    final_both: dict[str, dict[str, float]] | None = None
    if mode == "evl" and eval_loader_fg is not None and eval_cases and out.is_file():
        # Reload best checkpoint (may be catalog baseline if FT never improved).
        soft_load(model, out, map_location=device)
        log(
            "[finetune] Running dual final_test on best ckpt: "
            "fg_centered + full_volume…",
            enabled=True,
        )
        final_both = {
            "fg_centered": evaluate_loader(model, eval_loader_fg, loss_fn, device),
            "full_volume": evaluate_full_volume(
                model,
                eval_cases,
                loss_fn=loss_fn,
                device=device,
                patch_size=patch_size,
                intensity=intensity,
            ),
        }
        log(
            format_metrics(
                "[finetune] final_test fg_centered (best ckpt)",
                final_both["fg_centered"],
            ),
            enabled=True,
        )
        log(
            format_metrics(
                "[finetune] final_test full_volume (best ckpt)",
                final_both["full_volume"],
            ),
            enabled=True,
        )
        history.append(
            {
                "epoch": best_epoch,
                "phase": "final_test",
                "eval": final_both,
                "weights": str(out),
            }
        )
        if baseline_both is not None:
            _log_dual_eval_comparison(baseline_both, final_both)

    (work / "history.json").write_text(
        json.dumps(history, indent=2, default=str), encoding="utf-8"
    )
    if not out.is_file():
        _save_ckpt(
            out,
            model=model,
            cfg=cfg,
            epoch=int(cfg.epochs),
            pretrained=catalog,
            train_metrics=last_train,
            eval_metrics=last_eval,
            baseline=baseline_both if baseline_both is not None else baseline,
            mode=mode,
            backend_name=backend_name,
        )

    notes = (
        f"mode={mode}; best_epoch={best_epoch}; "
        f"catalog={catalog}; train_init={init_path}"
    )
    if mode == "evl":
        notes += f"; best_eval_dice_fg={best_score:.6f}"
        if baseline_both is not None:
            notes += (
                f"; baseline_fg={baseline_both['fg_centered'].get('dice', 0.0):.6f}"
                f"; baseline_full={baseline_both['full_volume'].get('dice', 0.0):.6f}"
            )
        if final_both is not None:
            notes += (
                f"; final_fg={final_both['fg_centered'].get('dice', 0.0):.6f}"
                f"; final_full={final_both['full_volume'].get('dice', 0.0):.6f}"
            )
    else:
        notes += f"; final_train_dice={best_score:.6f}"

    return FineTuneResult(
        out_ckpt=out,
        work_dir=work,
        n_train=len(train_cases),
        n_val=len(eval_cases),
        model=cfg.model,
        dataset=cfg.dataset,
        mode=mode,
        notes=notes,
        history=history,
        baseline_metrics=baseline_both if baseline_both is not None else baseline,
        best_metrics=(
            final_both
            if final_both is not None
            else (last_eval if mode == "evl" else last_train)
        ),
    )


def train_one_epoch(model, loader, loss_fn, optim, device) -> dict[str, float]:
    import torch

    model.train()
    losses: list[float] = []
    metric_rows: list[dict[str, float]] = []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
        optim.step()
        losses.append(float(loss.item()))
        # Train: overlap metrics only (surface distances are expensive / noisy on crops).
        metric_rows.append(_batch_metrics(logits, y, compute_surface=False))
    out = average_metrics(metric_rows)
    out["loss"] = float(np.mean(losses)) if losses else 0.0
    return out


def evaluate_loader(model, loader, loss_fn, device) -> dict[str, float]:
    import torch

    model.eval()
    losses: list[float] = []
    metric_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            loss = loss_fn(logits, y)
            losses.append(float(loss.item()))
            spacing = batch.get("spacing_zyx")
            sp = None
            if spacing is not None:
                sp = tuple(float(v) for v in spacing[0].tolist())
            metric_rows.append(
                _batch_metrics(logits, y, spacing_zyx=sp, compute_surface=True)
            )
    out = average_metrics(metric_rows)
    out["loss"] = float(np.mean(losses)) if losses else 0.0
    return out


def evaluate_full_volume(
    model,
    cases: list[CasePair],
    *,
    loss_fn,
    device,
    patch_size: tuple[int, int, int],
    intensity: dict[str, float] | None = None,
    overlap: float = 0.5,
) -> dict[str, float]:
    """Whole-scan metrics via sliding-window inference (nnU-Net-style)."""
    import torch

    from agentic_radiogen.imaging.tumor_models.finetune.nnunet_weights import (
        normalize_ct_volume,
    )

    model.eval()
    losses: list[float] = []
    metric_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for case in cases:
            image, spacing = _load_nifti_with_spacing(case.image)
            label, _ = _load_nifti_with_spacing(case.label)
            if label.shape != image.shape:
                raise ValueError(
                    f"Shape mismatch for {case.case_id}: "
                    f"image {image.shape} vs label {label.shape}"
                )
            label = (label > 0).astype(np.float32)
            if intensity is not None:
                image = normalize_ct_volume(image, intensity)
            else:
                image = _zscore(image)

            x = torch.from_numpy(image[None, None]).float().to(device)
            y = torch.from_numpy(label[None, None]).float().to(device)
            logits = sliding_window_inference(
                model, x, patch_size=patch_size, overlap=overlap
            )
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            try:
                loss = loss_fn(logits, y)
                losses.append(float(loss.item()))
            except Exception:
                # Very large volumes may OOM on full DiceCE; metrics still computed.
                pass
            metric_rows.append(
                _batch_metrics(
                    logits, y, spacing_zyx=spacing, compute_surface=True
                )
            )
            del x, y, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out = average_metrics(metric_rows)
    out["loss"] = float(np.mean(losses)) if losses else float("nan")
    return out


def sliding_window_inference(
    model,
    x,
    *,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
):
    """Slide ``patch_size`` over ``x`` (1,1,Z,Y,X), stitch logits with overlap weights.

    Mirrors nnU-Net / MONAI sliding-window: overlapping patches, weighted average
    of logits, then caller applies argmax / loss on the full volume.
    """
    import torch
    import torch.nn.functional as F

    if x.ndim != 5:
        raise ValueError(f"Expected x shape (B,C,Z,Y,X); got {tuple(x.shape)}")
    if int(x.shape[0]) != 1:
        raise ValueError("sliding_window_inference currently supports batch size 1")

    pz, py, px = (int(v) for v in patch_size)
    _, _, zz, yy, xx = x.shape
    # Pad so each axis is at least the patch size.
    pad_z = max(0, pz - zz)
    pad_y = max(0, py - yy)
    pad_x = max(0, px - xx)
    if pad_z or pad_y or pad_x:
        x = F.pad(x, (0, pad_x, 0, pad_y, 0, pad_z))
    _, _, zz_p, yy_p, xx_p = x.shape

    step_z = max(1, int(round(pz * (1.0 - overlap))))
    step_y = max(1, int(round(py * (1.0 - overlap))))
    step_x = max(1, int(round(px * (1.0 - overlap))))

    # Probe channel count.
    with torch.no_grad():
        probe = model(x[:, :, :pz, :py, :px])
        if isinstance(probe, (list, tuple)):
            probe = probe[0]
        n_class = int(probe.shape[1])
        del probe

    acc = torch.zeros((1, n_class, zz_p, yy_p, xx_p), device=x.device, dtype=torch.float32)
    weight = torch.zeros((1, 1, zz_p, yy_p, xx_p), device=x.device, dtype=torch.float32)
    win = _gaussian_patch_weight((pz, py, px), device=x.device, dtype=torch.float32)

    z_starts = list(range(0, max(zz_p - pz, 0) + 1, step_z))
    y_starts = list(range(0, max(yy_p - py, 0) + 1, step_y))
    x_starts = list(range(0, max(xx_p - px, 0) + 1, step_x))
    if not z_starts or z_starts[-1] != zz_p - pz:
        z_starts.append(max(0, zz_p - pz))
    if not y_starts or y_starts[-1] != yy_p - py:
        y_starts.append(max(0, yy_p - py))
    if not x_starts or x_starts[-1] != xx_p - px:
        x_starts.append(max(0, xx_p - px))
    # De-dupe while preserving order.
    z_starts = list(dict.fromkeys(z_starts))
    y_starts = list(dict.fromkeys(y_starts))
    x_starts = list(dict.fromkeys(x_starts))

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = x[:, :, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
                logits = model(patch)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                acc[:, :, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += logits * win
                weight[:, :, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += win

    weight = torch.clamp(weight, min=1e-8)
    logits_full = acc / weight
    # Crop back to original size (before pad).
    return logits_full[:, :, :zz, :yy, :xx]


def _gaussian_patch_weight(
    patch_size: tuple[int, int, int],
    *,
    device,
    dtype,
):
    """Importance mask peaking at patch center (reduces stitching seams)."""
    import torch

    pz, py, px = patch_size
    def _1d(n: int):
        if n <= 1:
            return torch.ones(n, device=device, dtype=dtype)
        t = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
        # sigma ~ 0.125 of half-width in normalized coords (MONAI-like).
        return torch.exp(-0.5 * (t / 0.125) ** 2)

    wz = _1d(pz)[:, None, None]
    wy = _1d(py)[None, :, None]
    wx = _1d(px)[None, None, :]
    w = wz * wy * wx
    w = w / w.max().clamp_min(1e-8)
    return w[None, None]


def _batch_metrics(
    logits,
    y,
    *,
    spacing_zyx: tuple[float, float, float] | None = None,
    compute_surface: bool = False,
) -> dict[str, float]:
    import torch

    # Generic_UNet with deep supervision can return a list; we disable DS, but
    # still accept list/tuple and take the full-resolution head.
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    pred = torch.argmax(logits, dim=1).detach().cpu().numpy() > 0
    # y is (B,1,Z,Y,X) float mask from DiceCELoss setup
    target = y.detach().cpu().numpy() > 0.5
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    # Aggregate per-volume in the batch, then average.
    rows: list[dict[str, float]] = []
    for i in range(pred.shape[0]):
        rows.append(
            binary_seg_metrics(
                pred[i],
                target[i],
                spacing_zyx=spacing_zyx,
                compute_surface=compute_surface,
            )
        )
    return average_metrics(rows)


def _log_dual_eval_comparison(
    baseline_both: dict[str, dict[str, float]],
    final_both: dict[str, dict[str, float]],
) -> None:
    """Print baseline vs best-ckpt for fg_centered and volume_center protocols."""
    from agentic_radiogen.imaging.tumor_models.finetune.metrics import _LOG_KEYS

    # Full overlap / volume / surface suite (not dice-only).
    keys = tuple(k for k in _LOG_KEYS if k != "loss") + ("loss",)
    log("[finetune] === Final comparison: catalog baseline vs best fine-tuned ckpt ===", enabled=True)
    log(
        "[finetune] protocols: fg_centered = patch on GT nodule; "
        "full_volume = sliding-window inference on the whole scan",
        enabled=True,
    )
    header = (
        f"{'method':<16} {'metric':<12} {'baseline':>10} {'finetuned':>10} {'delta':>10}"
    )
    log(f"[finetune] {header}", enabled=True)
    log(f"[finetune] {'-' * len(header)}", enabled=True)
    for method in ("fg_centered", "full_volume"):
        b = baseline_both.get(method) or {}
        f = final_both.get(method) or {}
        for k in keys:
            bv = float(b.get(k, float("nan")))
            fv = float(f.get(k, float("nan")))
            if np.isfinite(bv) and np.isfinite(fv):
                delta = fv - bv
                log(
                    f"[finetune] {method:<16} {k:<12} {bv:10.4f} {fv:10.4f} {delta:+10.4f}",
                    enabled=True,
                )
            else:
                b_s = f"{bv:.4f}" if np.isfinite(bv) else "nan"
                f_s = f"{fv:.4f}" if np.isfinite(fv) else "nan"
                log(
                    f"[finetune] {method:<16} {k:<12} {b_s:>10} {f_s:>10} {'n/a':>10}",
                    enabled=True,
                )


def _build_dice_ce_loss(monai_mod) -> Any:
    """Prefer MONAI DiceCELoss; fall back to a torch-only Dice+CE for nnU-Net runs."""
    if monai_mod is not None:
        from monai.losses import DiceCELoss

        return DiceCELoss(to_onehot_y=True, softmax=True)
    return _TorchDiceCELoss()


class _TorchDiceCELoss:
    """Minimal Dice + CE when MONAI is not installed (nnU-Net backend)."""

    def __call__(self, logits, target):
        import torch
        import torch.nn.functional as F

        # logits: (B,C,…)  target: (B,1,…) float {0,1}
        if target.ndim == logits.ndim and target.shape[1] == 1:
            tgt = target[:, 0].long()
        else:
            tgt = target.long()
        ce = F.cross_entropy(logits, tgt)
        probs = torch.softmax(logits, dim=1)
        # one-hot for dice on foreground class 1
        n_class = logits.shape[1]
        tgt_oh = F.one_hot(tgt.clamp(0, n_class - 1), num_classes=n_class)
        # (B,Z,Y,X,C) → (B,C,Z,Y,X)
        tgt_oh = tgt_oh.permute(0, -1, *range(1, tgt_oh.ndim - 1)).float()
        dims = tuple(range(2, probs.ndim))
        inter = (probs * tgt_oh).sum(dim=dims)
        denom = probs.sum(dim=dims) + tgt_oh.sum(dim=dims)
        dice = 1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5)).mean()
        return ce + dice


def soft_load(model, ckpt_path: Path, *, map_location) -> None:
    import torch

    try:
        blob = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(str(ckpt_path), map_location=map_location)
    except Exception as exc:
        log(
            f"[finetune] Could not load checkpoint {ckpt_path} ({exc}); "
            "training from scratch",
            enabled=True,
        )
        return
    if isinstance(blob, dict):
        for key in ("state_dict", "model", "network", "model_state_dict"):
            if key in blob and isinstance(blob[key], dict):
                blob = blob[key]
                break
    if not isinstance(blob, dict):
        log(
            f"[finetune] Could not interpret checkpoint {ckpt_path}; training from scratch",
            enabled=True,
        )
        return
    cleaned = {}
    for k, v in blob.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        cleaned[nk] = v
    model_sd = model.state_dict()
    shape_ok = {
        k: v
        for k, v in cleaned.items()
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    n_model = len(model_sd)
    n_matched = len(shape_ok)
    frac = float(n_matched) / float(max(n_model, 1))
    log(
        f"[finetune] Soft-loaded {ckpt_path} into {type(model).__name__}: "
        f"shape_matched={n_matched}/{n_model} ({frac:.1%}) "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        enabled=True,
    )
    if frac < 0.5:
        log(
            "[finetune] WARNING: pretrained overlap <50% — architecture mismatch "
            "(e.g. nnU-Net weights into MONAI UNet). Baseline metrics are NOT the "
            "pretrained model; use backend=nnunet so Generic_UNet is rebuilt from plans.pkl.",
            enabled=True,
        )


def _save_ckpt(
    out: Path,
    *,
    model,
    cfg: FineTuneConfig,
    epoch: int,
    pretrained: Path,
    train_metrics: dict[str, float] | None,
    eval_metrics: dict[str, float] | None,
    baseline: dict[str, float] | None,
    mode: str,
    backend_name: str,
) -> None:
    import torch

    torch.save(
        {
            "model": cfg.model,
            "dataset": cfg.dataset,
            "mode": mode,
            "backend": backend_name,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "pretrained": str(pretrained),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "baseline_metrics": baseline,
        },
        out,
    )


def _work_dir(cfg: FineTuneConfig) -> Path:
    if cfg.work_dir is not None:
        return Path(cfg.work_dir).expanduser().resolve()
    return (
        Path("outputs") / "finetune" / f"{cfg.model}_{cfg.dataset}_{normalize_mode(cfg.mode)}"
    ).resolve()


class NiftiSegDataset:
    """NIfTI dataset with nnU-Net-style foreground-aware patch sampling."""

    def __init__(
        self,
        cases: list[CasePair],
        *,
        patch_size: tuple[int, int, int],
        training: bool = True,
        intensity: dict[str, float] | None = None,
        fg_oversample: float = 0.67,
    ) -> None:
        self.cases = cases
        self.patch_size = patch_size
        self.training = training
        self.intensity = intensity
        self.fg_oversample = float(fg_oversample)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        from agentic_radiogen.imaging.tumor_models.finetune.nnunet_weights import (
            normalize_ct_volume,
        )

        case = self.cases[idx]
        image, spacing = _load_nifti_with_spacing(case.image)
        label, _ = _load_nifti_with_spacing(case.label)
        if label.shape != image.shape:
            raise ValueError(
                f"Shape mismatch for {case.case_id}: "
                f"image {image.shape} vs label {label.shape}"
            )
        label = (label > 0).astype(np.float32)
        if self.intensity is not None:
            image = normalize_ct_volume(image, self.intensity)
        else:
            image = _zscore(image)
        image_t, label_t = _crop_or_pad(
            image,
            label,
            self.patch_size,
            training=self.training,
            fg_oversample=self.fg_oversample,
        )
        return {
            "image": torch.from_numpy(image_t[None]).float(),
            "label": torch.from_numpy(label_t[None]).float(),
            "spacing_zyx": torch.tensor(spacing, dtype=torch.float32),
        }


def _load_nifti(path: Path) -> np.ndarray:
    arr, _ = _load_nifti_with_spacing(path)
    return arr


def _load_nifti_with_spacing(
    path: Path,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        import nibabel as nib

        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        # Saved as (x,y,z); convert to (z,y,x).
        arr = np.squeeze(np.asarray(data, dtype=np.float32))
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 1, 0))
        zooms = img.header.get_zooms()[:3]
        # zooms follow (x,y,z) → spacing_zyx
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
        return arr, spacing
    except Exception:
        pass
    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        arr = np.asarray(sitk.GetArrayFromImage(img), dtype=np.float32)
        sp = img.GetSpacing()  # (dx, dy, dz)
        spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
        return np.squeeze(arr), spacing
    except Exception as exc:
        raise RuntimeError(f"Failed to load {path}: {exc}") from exc

def _zscore(vol: np.ndarray) -> np.ndarray:
    mu = float(vol.mean())
    sd = float(vol.std())
    if sd < 1e-6:
        return vol - mu
    return (vol - mu) / sd


def _crop_or_pad(
    image: np.ndarray,
    label: np.ndarray,
    patch: tuple[int, int, int],
    *,
    training: bool,
    fg_oversample: float = 0.67,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop/pad a patch; prefer crops that contain tumor (LIDC nodules are tiny)."""
    z, y, x = image.shape
    pz, py, px = patch
    out_i = np.zeros(patch, dtype=np.float32)
    out_l = np.zeros(patch, dtype=np.float32)

    fg = np.argwhere(label > 0.5)
    use_fg = fg.shape[0] > 0 and (
        (not training and fg_oversample >= 1.0 - 1e-6)
        or (training and float(np.random.rand()) < fg_oversample)
        or (not training and float(np.random.rand()) < fg_oversample)
    )

    if use_fg:
        if training:
            # Place a random FG voxel uniformly inside the patch window.
            pick = fg[int(np.random.randint(0, fg.shape[0]))]
            z0 = _fg_window_start(int(pick[0]), pz, z, jitter=True)
            y0 = _fg_window_start(int(pick[1]), py, y, jitter=True)
            x0 = _fg_window_start(int(pick[2]), px, x, jitter=True)
        else:
            # Eval: center on FG centroid so metrics reflect real nodules.
            center = fg.mean(axis=0)
            z0 = _fg_window_start(int(round(center[0])), pz, z, jitter=False)
            y0 = _fg_window_start(int(round(center[1])), py, y, jitter=False)
            x0 = _fg_window_start(int(round(center[2])), px, x, jitter=False)
    else:

        def _start(size: int, win: int) -> int:
            if size <= win:
                return 0
            if training:
                return int(np.random.randint(0, size - win + 1))
            return max(0, (size - win) // 2)

        z0, y0, x0 = _start(z, pz), _start(y, py), _start(x, px)

    z1, y1, x1 = min(z0 + pz, z), min(y0 + py, y), min(x0 + px, x)
    sl_z, sl_y, sl_x = z1 - z0, y1 - y0, x1 - x0
    out_i[:sl_z, :sl_y, :sl_x] = image[z0:z1, y0:y1, x0:x1]
    out_l[:sl_z, :sl_y, :sl_x] = label[z0:z1, y0:y1, x0:x1]
    return out_i, out_l


def _fg_window_start(coord: int, win: int, size: int, *, jitter: bool) -> int:
    """Start index so ``coord`` lies inside ``[start, start+win)`` (clipped)."""
    if size <= win:
        return 0
    # Valid starts that keep coord in the window: [coord-win+1, coord]
    lo = max(0, coord - win + 1)
    hi = min(coord, size - win)
    if hi < lo:
        return int(np.clip(coord - win // 2, 0, size - win))
    if jitter:
        return int(np.random.randint(lo, hi + 1))
    return int(np.clip(coord - win // 2, 0, size - win))
