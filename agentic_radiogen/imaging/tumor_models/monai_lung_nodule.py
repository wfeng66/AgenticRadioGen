"""MONAI lung_nodule_ct_detection bundle → lesion mask (boxes rasterized).

Downloads ``MONAI/lung_nodule_ct_detection`` and runs RetinaNet detection, then
converts high-scoring boxes into a boolean ROI for radiomics.

On first use, auto-installs ``monai`` + ``huggingface_hub`` if missing (same
pattern as Task006 nnU-Net v1).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from agentic_radiogen.util.progress import log

_DEFAULT_CACHE = Path.cwd() / "data_cache" / "seg_models" / "monai_lung_nodule"
_HF_REPO = "MONAI/lung_nodule_ct_detection"
_BUNDLE_NAME = "lung_nodule_ct_detection"
_MARKER = "READY"
_MONAI_READY = False


def cache_root() -> Path:
    env = (os.environ.get("AGENTIC_RADIOGEN_MONAI_LUNG_CACHE") or "").strip()
    if env:
        return Path(env).expanduser()
    default = _DEFAULT_CACHE
    try:
        resolved = str(default.resolve()).replace("\\", "/")
        if resolved.startswith("/mnt/"):
            return Path.home() / ".cache" / "agentic_radiogen" / "monai_lung_nodule"
    except Exception:
        pass
    return default


def monai_bundle_dir() -> Path | None:
    env = (os.environ.get("AGENTIC_RADIOGEN_MONAI_LUNG_BUNDLE") or "").strip()
    if env:
        p = Path(env).expanduser()
        if _bundle_ready(p):
            return p
    root = cache_root()
    for cand in (root / _BUNDLE_NAME, root):
        if _bundle_ready(cand):
            return cand
    return None


def ensure_monai_runtime(*, verbose: bool = True, allow_install: bool = True) -> None:
    """Import ``monai``; pip-install on first use if missing."""
    global _MONAI_READY
    if _MONAI_READY:
        return
    try:
        import monai  # noqa: F401

        _MONAI_READY = True
        return
    except ImportError:
        pass
    if not allow_install:
        raise RuntimeError(
            "MONAI is not installed and auto-install is disabled "
            "(pip install monai huggingface_hub)"
        )
    log(
        "[seg-model] Installing monai + huggingface_hub (first use for "
        "lung_nodule_ct_detection)…",
        enabled=verbose,
    )
    _pip_install_monai()
    try:
        import monai
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import monai after pip install. "
            "Try: pip install 'monai[nibabel]' huggingface_hub"
        ) from exc
    _MONAI_READY = True
    log(
        f"[seg-model] MONAI ready (version={getattr(monai, '__version__', '?')})",
        enabled=verbose,
    )


def ensure_monai_lung_nodule(*, verbose: bool = True, allow_download: bool = True) -> Path:
    ensure_monai_runtime(verbose=verbose, allow_install=allow_download)
    existing = monai_bundle_dir()
    if existing is not None:
        log(f"[seg-model] Reusing MONAI lung nodule bundle → {existing}", enabled=verbose)
        return existing
    if not allow_download:
        raise FileNotFoundError(
            "MONAI lung_nodule_ct_detection missing and download disabled"
        )

    dest = cache_root()
    dest.mkdir(parents=True, exist_ok=True)
    log(
        f"[seg-model] Downloading MONAI bundle '{_BUNDLE_NAME}' "
        f"(HuggingFace {_HF_REPO})…",
        enabled=verbose,
    )
    bundle = _download_bundle(dest, verbose=verbose)
    (dest / _MARKER).write_text(_HF_REPO + "\n", encoding="utf-8")
    log(f"[seg-model] Installed MONAI lung nodule → {bundle}", enabled=verbose)
    return bundle


def run_monai_lung_nodule_mask(
    volume: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float] | None = None,
    allow_download: bool = True,
    verbose: bool = True,
    score_thresh: float | None = None,
) -> np.ndarray:
    """Detect pulmonary nodules and rasterize boxes to a boolean mask (z,y,x)."""
    vol = np.squeeze(np.asarray(volume, dtype=np.float32))
    if vol.ndim != 3:
        raise ValueError(f"MONAI lung nodule expects 3D volume; got {vol.shape}")
    min_slices = int(os.environ.get("AGENTIC_RADIOGEN_NNUNET_MIN_SLICES") or 8)
    if int(vol.shape[0]) < min_slices:
        raise RuntimeError(
            f"too few slices for MONAI lung nodule ({vol.shape[0]} < {min_slices}); "
            "unsuitable for 3D tumor model on this case"
        )

    ensure_monai_runtime(verbose=verbose, allow_install=allow_download)
    bundle = ensure_monai_lung_nodule(verbose=verbose, allow_download=allow_download)
    spacing = spacing_zyx or (1.0, 1.0, 1.0)
    thr = score_thresh
    if thr is None:
        thr = float(os.environ.get("AGENTIC_RADIOGEN_MONAI_SCORE_THRESH") or 0.3)

    try:
        mask = _infer_programmatic(
            vol, spacing_zyx=spacing, bundle=bundle, score_thresh=thr, verbose=verbose
        )
    except Exception as exc:
        log(
            f"[seg-model] MONAI programmatic infer failed ({exc}); "
            f"trying bundle CLI…",
            enabled=verbose,
        )
        mask = _infer_via_bundle_cli(
            vol, spacing_zyx=spacing, bundle=bundle, score_thresh=thr, verbose=verbose
        )

    if not mask.any():
        raise RuntimeError("MONAI lung_nodule_ct_detection returned an empty mask")
    return mask.astype(bool)


def _pip_install_monai() -> None:
    packages = ["monai", "huggingface_hub", "nibabel"]
    cmd = [sys.executable, "-m", "pip", "install", *packages, "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to install monai for lung_nodule_ct_detection. "
            f"pip stderr: {(proc.stderr or '')[:800]}"
        )


def _bundle_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "models" / "model.pt").is_file()


def _download_bundle(dest: Path, *, verbose: bool = True) -> Path:
    # Prefer monai.bundle.download; fall back to huggingface_hub.
    try:
        from monai.bundle import download

        download(
            name=_BUNDLE_NAME,
            bundle_dir=str(dest),
            source="monaihosting",
        )
        hit = dest / _BUNDLE_NAME
        if _bundle_ready(hit):
            return hit
    except Exception as exc:
        log(f"[seg-model] monai.bundle.download failed ({exc}); trying HuggingFace…", enabled=verbose)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _pip_install_monai()
        from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=_HF_REPO,
        local_dir=str(dest / _BUNDLE_NAME),
    )
    hit = Path(local)
    if not _bundle_ready(hit):
        raise FileNotFoundError(f"Downloaded {_HF_REPO} but models/model.pt missing")
    return hit


def _infer_programmatic(
    volume: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    bundle: Path,
    score_thresh: float,
    verbose: bool,
) -> np.ndarray:
    """Load RetinaNet from the bundle and run sliding-window detection."""
    try:
        import torch
        from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
        from monai.apps.detection.networks.retinanet_network import (
            retinanet_resnet50_fpn_feature_extractor,
        )
        from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
        from monai.data import MetaTensor
        from monai.transforms import (
            EnsureChannelFirst,
            ScaleIntensityRange,
            Spacing,
        )
    except ImportError as exc:
        raise RuntimeError(
            "MONAI detection extras required "
            "(pip install 'monai[nibabel]' with detection support)"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = bundle / "models" / "model.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing {ckpt}")

    # Input is (z,y,x); MONAI detection typically uses (x,y,z) channel-first after
    # Orientationd RAS. Convert zyx → xyz for network, then map boxes back.
    vol_xyz = np.transpose(volume, (2, 1, 0))  # x,y,z
    spacing_xyz = (float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0]))

    t = torch.from_numpy(vol_xyz.astype(np.float32))
    meta = MetaTensor(t, affine=_affine_from_spacing(spacing_xyz))
    img = EnsureChannelFirst()(meta)
    img = Spacing(pixdim=(0.703125, 0.703125, 1.25), mode="bilinear")(img)
    img = ScaleIntensityRange(
        a_min=-1024.0, a_max=300.0, b_min=0.0, b_max=1.0, clip=True
    )(img)
    img = img.to(device)

    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=(1, 2, 4),
        base_anchor_shapes=((6, 8, 4), (8, 6, 5), (10, 10, 6)),
    )
    feature_extractor = retinanet_resnet50_fpn_feature_extractor(
        spatial_dims=3, n_input_channels=1
    )
    from monai.apps.detection.networks.retinanet_network import RetinaNet

    network = RetinaNet(
        spatial_dims=3,
        num_classes=1,
        num_anchors=3,
        feature_extractor=feature_extractor,
        size_divisible=(16, 16, 8),
    ).to(device)
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    network.load_state_dict(state, strict=False)
    network.eval()

    detector = RetinaNetDetector(
        network=network,
        anchor_generator=anchor_generator,
        debug=False,
        spatial_dims=3,
        num_classes=1,
        size_divisible=(16, 16, 8),
    )
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=float(score_thresh),
        topk_candidates_per_level=1000,
        nms_thresh=0.22,
        detections_per_img=100,
    )
    # Smaller ROI on CPU; full size when CUDA available.
    roi = (512, 512, 192) if device.type == "cuda" else (256, 256, 128)
    detector.set_sliding_window_inferer(
        roi_size=roi,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device=str(device),
    )

    log(
        f"[seg-model] MONAI RetinaNet infer device={device} score_thresh={score_thresh}",
        enabled=verbose,
    )
    with torch.no_grad():
        pred = detector([img], use_inferer=True)

    if not pred:
        return np.zeros(volume.shape, dtype=bool)
    first = pred[0]
    boxes = first.get("box")
    scores = first.get("label_scores")
    if boxes is None:
        return np.zeros(volume.shape, dtype=bool)
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    else:
        boxes = np.asarray(boxes)
    if scores is not None and hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    elif scores is not None:
        scores = np.asarray(scores)
    else:
        scores = np.ones((len(boxes),), dtype=np.float32)

    # Boxes are in the resampled image voxel space (xyzxyz). Map to original zyx
    # via relative positions in the resampled grid.
    resampled_shape_xyz = tuple(int(x) for x in img.shape[-3:])
    return _boxes_xyzxyz_to_mask_zyx(
        volume.shape,
        boxes,
        scores,
        score_thresh=score_thresh,
        src_shape_xyz=resampled_shape_xyz,
    )


def _infer_via_bundle_cli(
    volume: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    bundle: Path,
    score_thresh: float,
    verbose: bool,
) -> np.ndarray:
    """Fallback: write NIfTI + datalist, run ``python -m monai.bundle run``."""
    import subprocess
    import sys

    from agentic_radiogen.imaging.segment import _nifti_from_volume_zyx
    import nibabel as nib

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        nifti_path = tmp_path / "case.nii.gz"
        img, _ = _nifti_from_volume_zyx(volume, spacing_zyx=spacing_zyx)
        nib.save(img, str(nifti_path))
        datalist = tmp_path / "datalist.json"
        datalist.write_text(
            json.dumps({"validation": [{"image": "case.nii.gz"}]}),
            encoding="utf-8",
        )
        out_dir = tmp_path / "eval"
        out_dir.mkdir()
        cmd = [
            sys.executable,
            "-m",
            "monai.bundle",
            "run",
            "--config_file",
            str(bundle / "configs" / "inference.json"),
            "--bundle_root",
            str(bundle),
            "--dataset_dir",
            str(tmp_path),
            "--data_list_file_path",
            str(datalist),
            "--output_dir",
            str(out_dir),
            "--output_filename",
            "pred.json",
            "--whether_raw_luna16",
            "true",
        ]
        # Raise score threshold via env-documented default; full JSON override is brittle.
        log(f"[seg-model] Running: {' '.join(cmd[:6])} …", enabled=verbose)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(bundle))
        if proc.returncode != 0:
            raise RuntimeError(
                "monai.bundle run failed: "
                + (proc.stderr or proc.stdout or "")[:800]
            )
        pred_path = out_dir / "pred.json"
        if not pred_path.is_file():
            # DetectionSaver may nest differently.
            found = list(out_dir.rglob("*.json"))
            if not found:
                raise RuntimeError("MONAI bundle produced no prediction JSON")
            pred_path = found[0]
        raw = json.loads(pred_path.read_text(encoding="utf-8"))
        return _parse_detection_json_to_mask(volume.shape, raw, score_thresh=score_thresh)


def _parse_detection_json_to_mask(
    shape_zyx: tuple[int, ...],
    raw: object,
    *,
    score_thresh: float,
) -> np.ndarray:
    """Best-effort parse of DetectionSaver JSON into a zyx mask."""
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    entries = raw if isinstance(raw, list) else [raw]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        box = entry.get("box") or entry.get("boxes")
        sc = entry.get("label_scores") or entry.get("score") or entry.get("scores")
        if box is None:
            continue
        arr = np.asarray(box, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        boxes.append(arr)
        if sc is None:
            scores.extend([1.0] * len(arr))
        else:
            sc_arr = np.asarray(sc, dtype=np.float32).reshape(-1)
            scores.extend(float(x) for x in sc_arr)
    if not boxes:
        return np.zeros(shape_zyx, dtype=bool)
    all_boxes = np.concatenate(boxes, axis=0)
    # World-coordinate cccwhd → approximate voxel box using volume extent.
    return _boxes_cccwhd_world_to_mask_zyx(
        shape_zyx, all_boxes, np.asarray(scores), score_thresh=score_thresh
    )


def _boxes_xyzxyz_to_mask_zyx(
    shape_zyx: tuple[int, ...],
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    score_thresh: float,
    src_shape_xyz: tuple[int, int, int],
) -> np.ndarray:
    """Rasterize xyzxyz boxes from a resampled grid onto original zyx volume."""
    mask = np.zeros(shape_zyx, dtype=bool)
    if boxes.size == 0:
        return mask
    sx, sy, sz = (max(1, int(x)) for x in src_shape_xyz)
    dz, dy, dx = shape_zyx
    scores = np.asarray(scores).reshape(-1)
    for i, box in enumerate(boxes):
        sc = float(scores[i]) if i < len(scores) else 1.0
        if sc < score_thresh:
            continue
        b = np.asarray(box, dtype=np.float32).reshape(-1)
        if b.size == 6:
            # xyzxyz
            x0, y0, z0, x1, y1, z1 = b.tolist()
        elif b.size >= 6:
            x0, y0, z0, x1, y1, z1 = b[:6].tolist()
        else:
            continue
        # Map resampled xyz → original zyx indices.
        X0 = int(np.clip(round(x0 / sx * dx), 0, dx - 1))
        X1 = int(np.clip(round(x1 / sx * dx), 0, dx))
        Y0 = int(np.clip(round(y0 / sy * dy), 0, dy - 1))
        Y1 = int(np.clip(round(y1 / sy * dy), 0, dy))
        Z0 = int(np.clip(round(z0 / sz * dz), 0, dz - 1))
        Z1 = int(np.clip(round(z1 / sz * dz), 0, dz))
        if X1 <= X0:
            X1 = min(dx, X0 + 1)
        if Y1 <= Y0:
            Y1 = min(dy, Y0 + 1)
        if Z1 <= Z0:
            Z1 = min(dz, Z0 + 1)
        # Ellipsoid fill in zyx.
        _fill_ellipsoid(mask, Z0, Z1, Y0, Y1, X0, X1)
    return mask


def _boxes_cccwhd_world_to_mask_zyx(
    shape_zyx: tuple[int, ...],
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    score_thresh: float,
) -> np.ndarray:
    """Approximate world cccwhd boxes as relative ellipsoids in the volume."""
    mask = np.zeros(shape_zyx, dtype=bool)
    if boxes.size == 0:
        return mask
    dz, dy, dx = shape_zyx
    # Normalize centers/sizes by robust range of centers (fallback: volume mid).
    centers = boxes[:, :3] if boxes.shape[-1] >= 6 else boxes
    lo = centers.min(axis=0)
    hi = centers.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    for i, box in enumerate(boxes):
        sc = float(scores[i]) if i < len(scores) else 1.0
        if sc < score_thresh:
            continue
        b = np.asarray(box, dtype=np.float32).reshape(-1)
        if b.size < 6:
            continue
        cx, cy, cz, w, h, d = b[:6].tolist()
        # Map world-ish center into [0,1] relative to detected span, then to voxels.
        fx = float(np.clip((cx - lo[0]) / span[0], 0, 1))
        fy = float(np.clip((cy - lo[1]) / span[1], 0, 1))
        fz = float(np.clip((cz - lo[2]) / span[2], 0, 1))
        # Sizes: relative to span.
        rx = max(1, int(round(0.5 * abs(w) / span[0] * dx)))
        ry = max(1, int(round(0.5 * abs(h) / span[1] * dy)))
        rz = max(1, int(round(0.5 * abs(d) / span[2] * dz)))
        xi = int(round(fx * (dx - 1)))
        yi = int(round(fy * (dy - 1)))
        zi = int(round(fz * (dz - 1)))
        _fill_ellipsoid(
            mask,
            max(0, zi - rz),
            min(dz, zi + rz + 1),
            max(0, yi - ry),
            min(dy, yi + ry + 1),
            max(0, xi - rx),
            min(dx, xi + rx + 1),
        )
    return mask


def _fill_ellipsoid(
    mask: np.ndarray, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int
) -> None:
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    zc = 0.5 * (z0 + z1 - 1)
    yc = 0.5 * (y0 + y1 - 1)
    xc = 0.5 * (x0 + x1 - 1)
    rz = max(0.5, 0.5 * (z1 - z0))
    ry = max(0.5, 0.5 * (y1 - y0))
    rx = max(0.5, 0.5 * (x1 - x0))
    ell = ((zz - zc) / rz) ** 2 + ((yy - yc) / ry) ** 2 + ((xx - xc) / rx) ** 2 <= 1.0
    mask[z0:z1, y0:y1, x0:x1] |= ell


def _affine_from_spacing(spacing_xyz: tuple[float, float, float]):
    import torch

    sp = spacing_xyz
    return torch.tensor(
        [
            [sp[0], 0.0, 0.0, 0.0],
            [0.0, sp[1], 0.0, 0.0],
            [0.0, 0.0, sp[2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
