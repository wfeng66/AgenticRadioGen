"""Binary tumor-segmentation metrics.

Includes overlap (Dice, IoU), classification (precision/recall/specificity),
volume (RVD), and surface distances (HD, HD95, ASSD) when a 3D mask + spacing
are available.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Display order for logs (subset shown; full dict always returned).
_LOG_KEYS = (
    "loss",
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "rvd",
    "hd95",
    "hd",
    "assd",
)


def binary_seg_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float] | None = None,
    eps: float = 1e-6,
    compute_surface: bool = True,
) -> dict[str, float]:
    """Compute metrics for boolean / {0,1} masks.

    Parameters
    ----------
    pred, target:
        Same-shaped arrays (any rank). Surface metrics use the full array as a
        volume with ``spacing_zyx`` (defaults to 1 mm isotropic).
    """
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch pred={p.shape} target={t.shape}")

    p_flat = p.ravel()
    t_flat = t.ravel()
    tp = float(np.logical_and(p_flat, t_flat).sum())
    fp = float(np.logical_and(p_flat, np.logical_not(t_flat)).sum())
    fn = float(np.logical_and(np.logical_not(p_flat), t_flat).sum())
    tn = float(np.logical_and(np.logical_not(p_flat), np.logical_not(t_flat)).sum())

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)  # sensitivity / TPR
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    f2 = (5.0 * tp + eps) / (5.0 * tp + 4.0 * fn + fp + eps)

    vp = float(p_flat.sum())
    vt = float(t_flat.sum())
    # Relative volume difference (signed and absolute); GT volume in denominator.
    rvd_signed = (vp - vt) / (vt + eps)
    rvd = abs(rvd_signed)
    vol_sim = 1.0 - min(rvd, 1.0)  # 1 = perfect volume match (capped)

    out: dict[str, float] = {
        "dice": float(dice),
        "iou": float(iou),
        "jaccard": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "f2": float(f2),
        "rvd": float(rvd),
        "rvd_signed": float(rvd_signed),
        "volume_similarity": float(vol_sim),
        "pred_voxels": vp,
        "gt_voxels": vt,
    }

    if compute_surface and p.ndim >= 2:
        spacing = spacing_zyx or tuple(1.0 for _ in range(p.ndim))
        if len(spacing) < p.ndim:
            spacing = tuple(spacing) + (1.0,) * (p.ndim - len(spacing))
        surf = _surface_metrics(p, t, spacing_zyx=spacing[: p.ndim], eps=eps)
        out.update(surf)
    else:
        out.update({"hd": float("nan"), "hd95": float("nan"), "assd": float("nan")})
    return out


def _surface_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    spacing_zyx: tuple[float, ...],
    eps: float,
) -> dict[str, float]:
    """Hausdorff (max / 95%) and average symmetric surface distance in mm."""
    if not pred.any() and not target.any():
        return {"hd": 0.0, "hd95": 0.0, "assd": 0.0}
    if not pred.any() or not target.any():
        # One empty → treat as very large / undefined; use a large finite sentinel.
        return {"hd": 1e6, "hd95": 1e6, "assd": 1e6}

    try:
        from scipy import ndimage
    except ImportError:
        return {"hd": float("nan"), "hd95": float("nan"), "assd": float("nan")}

    spacing = np.asarray(spacing_zyx[: pred.ndim], dtype=float)
    spacing = np.where(spacing > 0, spacing, 1.0)

    pred_surf = _binary_surface(pred)
    gt_surf = _binary_surface(target)
    if not pred_surf.any() or not gt_surf.any():
        return {"hd": 1e6, "hd95": 1e6, "assd": 1e6}

    # Distance from every voxel to the *other* surface (edt on inverted surface).
    dt_gt = ndimage.distance_transform_edt(~gt_surf, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_surf, sampling=spacing)
    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]
    if d_pred_to_gt.size == 0 or d_gt_to_pred.size == 0:
        return {"hd": 1e6, "hd95": 1e6, "assd": 1e6}

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    hd = float(all_d.max())
    hd95 = float(np.percentile(all_d, 95))
    assd = float(0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean()))
    return {"hd": hd, "hd95": hd95, "assd": assd}


def _binary_surface(mask: np.ndarray) -> np.ndarray:
    """Surface voxels = mask AND (dilate(background) neighbor)."""
    from scipy import ndimage

    mask = mask.astype(bool)
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    # Voxels in mask but not in erosion ≈ boundary (also keep thin structures).
    surface = np.logical_xor(mask, eroded)
    if not surface.any() and mask.any():
        return mask
    return surface


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "rvd": 0.0,
            "hd": 0.0,
            "hd95": 0.0,
            "assd": 0.0,
            "loss": 0.0,
        }
    keys = sorted({k for row in rows for k in row})
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(row[k]) for row in rows if k in row and np.isfinite(row[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def format_metrics(prefix: str, metrics: dict[str, Any]) -> str:
    bits = [prefix]
    for key in _LOG_KEYS:
        if key in metrics and metrics[key] is not None and np.isfinite(float(metrics[key])):
            bits.append(f"{key}={float(metrics[key]):.4f}")
    return " ".join(bits)
