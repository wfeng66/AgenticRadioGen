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
) -> FineTuneResult:
    """Fine-tune a 3D UNet with soft-loaded ``pretrained`` weights."""
    mode = normalize_mode(cfg.mode)
    train_cases, eval_cases = split_for_mode(
        cases, mode=mode, val_frac=cfg.val_frac, seed=cfg.seed
    )
    work = _work_dir(cfg)
    work.mkdir(parents=True, exist_ok=True)

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
            "pretrained": str(pretrained),
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
    if backend_l in {"nnunet", "nnunetv1", "nnunet-v1"}:
        from agentic_radiogen.imaging.tumor_models.finetune.nnunet_weights import (
            load_nnunet_v1_network,
        )

        model, net_meta = load_nnunet_v1_network(
            pretrained, device=device, verbose=True
        )
        patch_size = tuple(net_meta.get("patch_size") or _PATCH)
        intensity = net_meta.get("intensity")
        log(
            f"[finetune] nnU-Net patch_size={patch_size} "
            f"(native Generic_UNet; not MONAI soft-load)",
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
        soft_load(model, pretrained, map_location=device)

    loss_fn = _build_dice_ce_loss(monai)
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    train_loader = DataLoader(
        NiftiSegDataset(
            train_cases,
            patch_size=patch_size,
            training=True,
            intensity=intensity,
        ),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        num_workers=max(0, int(cfg.num_workers)),
    )
    eval_loader = None
    if eval_cases:
        eval_loader = DataLoader(
            NiftiSegDataset(
                eval_cases,
                patch_size=patch_size,
                training=False,
                intensity=intensity,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    history: list[dict[str, Any]] = []
    baseline: dict[str, float] | None = None
    if mode == "evl" and eval_loader is not None:
        baseline = evaluate_loader(model, eval_loader, loss_fn, device)
        log(format_metrics("[finetune] baseline (before fine-tune) eval", baseline), enabled=True)
        history.append({"epoch": 0, "phase": "baseline", "eval": baseline})

    out = Path(cfg.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    last_train: dict[str, float] = {}
    last_eval: dict[str, float] | None = None

    for epoch in range(1, int(cfg.epochs) + 1):
        train_m = train_one_epoch(model, train_loader, loss_fn, optim, device)
        last_train = train_m
        row: dict[str, Any] = {"epoch": epoch, "train": train_m}

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
            score = float(eval_m.get("dice", 0.0))
            if score >= best_score:
                best_score = score
                best_epoch = epoch
                _save_ckpt(
                    out,
                    model=model,
                    cfg=cfg,
                    epoch=epoch,
                    pretrained=pretrained,
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
                pretrained=pretrained,
                train_metrics=train_m,
                eval_metrics=None,
                baseline=None,
                mode=mode,
                backend_name=backend_name,
            )
            best_epoch = epoch
            best_score = float(train_m.get("dice", 0.0))

        history.append(row)

    (work / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not out.is_file():
        _save_ckpt(
            out,
            model=model,
            cfg=cfg,
            epoch=int(cfg.epochs),
            pretrained=pretrained,
            train_metrics=last_train,
            eval_metrics=last_eval,
            baseline=baseline,
            mode=mode,
            backend_name=backend_name,
        )

    notes = f"mode={mode}; best_epoch={best_epoch}"
    if mode == "evl":
        notes += f"; best_eval_dice={best_score:.6f}"
        if baseline is not None:
            notes += f"; baseline_eval_dice={baseline.get('dice', 0.0):.6f}"
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
        baseline_metrics=baseline,
        best_metrics=last_eval if mode == "evl" else last_train,
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
    """Minimal random-crop NIfTI dataset (torch Dataset duck-type)."""

    def __init__(
        self,
        cases: list[CasePair],
        *,
        patch_size: tuple[int, int, int],
        training: bool = True,
        intensity: dict[str, float] | None = None,
    ) -> None:
        self.cases = cases
        self.patch_size = patch_size
        self.training = training
        self.intensity = intensity

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
            image, label, self.patch_size, random_crop=self.training
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
    random_crop: bool,
) -> tuple[np.ndarray, np.ndarray]:
    z, y, x = image.shape
    pz, py, px = patch
    out_i = np.zeros(patch, dtype=np.float32)
    out_l = np.zeros(patch, dtype=np.float32)

    def _start(size: int, win: int) -> int:
        if size <= win:
            return 0
        if random_crop:
            return int(np.random.randint(0, size - win + 1))
        return max(0, (size - win) // 2)

    z0, y0, x0 = _start(z, pz), _start(y, py), _start(x, px)
    z1, y1, x1 = min(z0 + pz, z), min(y0 + py, y), min(x0 + px, x)
    sl_z, sl_y, sl_x = z1 - z0, y1 - y0, x1 - x0
    out_i[:sl_z, :sl_y, :sl_x] = image[z0:z1, y0:y1, x0:x1]
    out_l[:sl_z, :sl_y, :sl_x] = label[z0:z1, y0:y1, x0:x1]
    return out_i, out_l
