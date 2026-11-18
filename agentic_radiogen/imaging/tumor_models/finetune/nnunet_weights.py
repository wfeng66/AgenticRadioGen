"""Load a real nnU-Net v1 ``Generic_UNet`` from Task plans + checkpoint.

Soft-loading nnU-Net weights into a MONAI ``UNet`` is architecture-incompatible
(typically almost all keys miss / mismatch), so baseline metrics look like a
random network. This helper rebuilds the native graph from ``plans.pkl``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from agentic_radiogen.util.progress import log


def find_plans_pkl(ckpt_path: Path) -> Path | None:
    """Locate ``plans.pkl`` next to a fold checkpoint."""
    ckpt_path = Path(ckpt_path)
    candidates = [
        ckpt_path.parent.parent / "plans.pkl",  # .../fold_0/model_*.model → trainer/
        ckpt_path.parent / "plans.pkl",
        ckpt_path.with_suffix(".pkl"),  # model_final_checkpoint.model.pkl is meta, not plans
    ]
    for p in candidates:
        if p.is_file() and p.name == "plans.pkl":
            return p
    # Walk up a few levels (3d_fullres/Task…/trainer/fold_0/…).
    cur = ckpt_path.parent
    for _ in range(5):
        hit = cur / "plans.pkl"
        if hit.is_file():
            return hit
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def load_nnunet_v1_network(
    ckpt_path: Path,
    *,
    device,
    verbose: bool = True,
):
    """Build Generic_UNet from plans and load ``model_final_checkpoint.model``.

    Returns ``(network, meta)`` where ``meta`` includes ``patch_size`` (z,y,x)
    and optional CT intensity stats for preprocessing.
    """
    import torch
    import torch.nn as nn

    ckpt_path = Path(ckpt_path)
    plans_path = find_plans_pkl(ckpt_path)
    if plans_path is None:
        raise FileNotFoundError(
            f"No plans.pkl found near {ckpt_path}; cannot rebuild nnU-Net architecture"
        )

    with open(plans_path, "rb") as f:
        plans = pickle.load(f)

    stage_plans = plans.get("plans_per_stage") or {}
    if not stage_plans:
        raise ValueError(f"plans.pkl missing plans_per_stage: {plans_path}")
    stage = max(stage_plans.keys())
    stage_cfg = stage_plans[stage]

    try:
        from nnunet.network_architecture.generic_UNet import Generic_UNet
        from nnunet.network_architecture.initialization import InitWeights_He
    except ImportError as exc:
        raise RuntimeError(
            "nnU-Net v1 package required to load Task006 weights "
            "(pip install nnunet). Soft-loading into MONAI UNet is not valid."
        ) from exc

    # Match nnUNetTrainer.process_plans / initialize_network.
    num_input = int(plans.get("num_modalities", 1))
    base_features = int(plans.get("base_num_features", 32))
    # plans['num_classes'] is foreground count; network adds background.
    num_classes = int(plans.get("num_classes", 1)) + 1
    conv_per_stage = int(plans.get("conv_per_stage", 2))
    pool_op_kernel_sizes = stage_cfg.get("pool_op_kernel_sizes") or stage_cfg.get(
        "net_num_pool_op_kernel_sizes"
    )
    conv_kernel_sizes = stage_cfg.get("conv_kernel_sizes") or stage_cfg.get(
        "net_conv_kernel_sizes"
    )
    if not pool_op_kernel_sizes or not conv_kernel_sizes:
        raise ValueError(f"Incomplete stage plans in {plans_path}: {stage_cfg.keys()}")

    net_numpool = len(pool_op_kernel_sizes)
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = {"eps": 1e-5, "affine": True}
    dropout_op_kwargs = {"p": 0, "inplace": True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}

    network = Generic_UNet(
        num_input,
        base_features,
        num_classes,
        net_numpool,
        conv_per_stage,
        2,
        conv_op,
        norm_op,
        norm_op_kwargs,
        dropout_op,
        dropout_op_kwargs,
        net_nonlin,
        net_nonlin_kwargs,
        True,  # deep_supervision (disabled below for single-logit training)
        False,
        lambda x: x,
        InitWeights_He(1e-2),
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        False,
        True,
        True,
    )
    # Single full-res logit map for DiceCELoss / argmax metrics.
    network.do_ds = False
    try:
        from nnunet.utilities.nd_softmax import softmax_helper

        network.inference_apply_nonlin = softmax_helper
    except Exception:
        network.inference_apply_nonlin = lambda x: torch.softmax(x, dim=1)

    state = _load_state_dict(ckpt_path, map_location="cpu")
    missing, unexpected = network.load_state_dict(state, strict=False)
    n_model = len(network.state_dict())
    n_loaded = n_model - len(missing)
    frac = float(n_loaded) / float(max(n_model, 1))
    if frac < 0.8:
        raise RuntimeError(
            f"nnU-Net weight load incomplete for {ckpt_path}: "
            f"loaded={n_loaded}/{n_model} ({frac:.1%}), "
            f"missing={len(missing)} unexpected={len(unexpected)}. "
            "Refusing to report this as a pretrained baseline."
        )
    log(
        f"[finetune] Loaded nnU-Net Generic_UNet from {ckpt_path.name} "
        f"(plans={plans_path.name} stage={stage} "
        f"loaded={n_loaded}/{n_model} missing={len(missing)} unexpected={len(unexpected)})",
        enabled=verbose,
    )

    patch = stage_cfg.get("patch_size")
    patch_zyx = _as_zyx_patch(patch)
    intensity = _ct_intensity_from_plans(plans)
    meta: dict[str, Any] = {
        "plans_path": str(plans_path),
        "stage": stage,
        "patch_size": patch_zyx,
        "num_classes": num_classes,
        "intensity": intensity,
        "loaded_frac": frac,
    }
    network = network.to(device)
    network.train()
    return network, meta


def _load_state_dict(ckpt_path: Path, *, map_location) -> dict:
    import torch

    # PyTorch 2.6+ may default weights_only=True; nnU-Net pickles need False.
    try:
        blob = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(str(ckpt_path), map_location=map_location)
    if isinstance(blob, dict):
        for key in ("state_dict", "model_state_dict", "network", "model"):
            if key in blob and isinstance(blob[key], dict):
                blob = blob[key]
                break
    if not isinstance(blob, dict):
        raise ValueError(f"Checkpoint is not a state dict: {ckpt_path}")
    cleaned: dict = {}
    for k, v in blob.items():
        nk = k[len("module.") :] if isinstance(k, str) and k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def _as_zyx_patch(patch: Any) -> tuple[int, int, int]:
    if patch is None:
        return (96, 96, 96)
    vals = [int(v) for v in list(patch)]
    if len(vals) != 3:
        return (96, 96, 96)
    # nnU-Net plans store patch as (x, y, z) in older code paths, or (z,y,x).
    # Task006 3d_fullres is typically [80, 192, 160] style with smallest often z.
    # We keep plan order as network spatial axes (D,H,W) matching our (z,y,x) tensors
    # when the first dim is the smallest / slice axis; otherwise transpose to zyx.
    a, b, c = vals
    if a <= b and a <= c:
        return (a, b, c)
    # Likely (x,y,z) → (z,y,x)
    return (c, b, a)


def _ct_intensity_from_plans(plans: dict) -> dict[str, float] | None:
    props = plans.get("intensityproperties") or plans.get("dataset_properties", {}).get(
        "intensityproperties"
    )
    if not props:
        return None
    # modality 0
    entry = props.get(0) or props.get("0") or (next(iter(props.values())) if props else None)
    if not isinstance(entry, dict):
        return None
    try:
        return {
            "mean": float(entry.get("mean")),
            "std": float(entry.get("sd") or entry.get("std")),
            "p_low": float(entry.get("percentile_00_5", entry.get("percentile_0_5", -1024))),
            "p_high": float(entry.get("percentile_99_5", 1024)),
        }
    except Exception:
        return None


def normalize_ct_volume(vol, intensity: dict[str, float] | None):
    """Clip + z-score like nnU-Net CT preprocessing when stats are available."""
    import numpy as np

    x = np.asarray(vol, dtype=np.float32)
    if intensity is None:
        mu = float(x.mean())
        sd = float(x.std())
        return x - mu if sd < 1e-6 else (x - mu) / sd
    low, high = intensity["p_low"], intensity["p_high"]
    x = np.clip(x, low, high)
    mu, sd = intensity["mean"], intensity["std"]
    if sd is None or abs(sd) < 1e-6:
        return x - float(mu)
    return (x - float(mu)) / float(sd)
