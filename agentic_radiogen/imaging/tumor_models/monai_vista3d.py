"""NVIDIA / MONAI VISTA-3D bundle → lesion mask (label_prompt class segmentation).

Downloads ``MONAI/vista3d`` (or ``python -m monai.bundle download vista3d``) and
runs automatic class-prompt inference. For lung cancer / LUAD / LUSC the default
class is **lung tumor** (label index 23).

Manual setup (optional — first use can auto-download):

    pip install 'monai[nibabel,fire]' huggingface_hub nibabel
    python -m monai.bundle download vista3d \\
      --bundle_dir ~/.cache/agentic_radiogen/monai_vista3d
    # or point at an existing bundle:
    export AGENTIC_RADIOGEN_VISTA3D_BUNDLE=/path/to/vista3d

Force in the pipeline:

    python -m agentic_radiogen ... --seg-model monai_vista3d --store-segm
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from agentic_radiogen.util.progress import log

_DEFAULT_CACHE = Path.cwd() / "data_cache" / "seg_models" / "monai_vista3d"
_HF_REPO = "MONAI/vista3d"
_BUNDLE_NAME = "vista3d"
_MARKER = "READY"
_MONAI_READY = False

# VISTA-3D class ids (docs/labels.json). Tumor classes are NOT in "segment everything".
_LABEL_LUNG_TUMOR = 23
_LABEL_PANCREATIC_TUMOR = 24
_LABEL_HEPATIC_TUMOR = 26
_LABEL_COLON_CANCER = 27


def cache_root() -> Path:
    env = (os.environ.get("AGENTIC_RADIOGEN_VISTA3D_CACHE") or "").strip()
    if env:
        return Path(env).expanduser()
    default = _DEFAULT_CACHE
    try:
        resolved = str(default.resolve()).replace("\\", "/")
        if resolved.startswith("/mnt/"):
            return Path.home() / ".cache" / "agentic_radiogen" / "monai_vista3d"
    except Exception:
        pass
    return default


def vista3d_bundle_dir() -> Path | None:
    env = (os.environ.get("AGENTIC_RADIOGEN_VISTA3D_BUNDLE") or "").strip()
    if env:
        p = Path(env).expanduser()
        if _bundle_ready(p):
            return p
    root = cache_root()
    for cand in (root / _BUNDLE_NAME, root):
        if _bundle_ready(cand):
            return cand
    return None


def label_prompt_for_disease(disease: str | None) -> list[int]:
    """Pick VISTA-3D class index(es) from disease text (default: lung tumor)."""
    env = (os.environ.get("AGENTIC_RADIOGEN_VISTA3D_LABEL") or "").strip()
    if env:
        try:
            return [int(x) for x in re.split(r"[,\s]+", env) if x.strip()]
        except ValueError:
            pass
    d = (disease or "").lower()
    if any(t in d for t in ("hepatic", "liver tumor", "hcc", "hepatocellular")):
        return [_LABEL_HEPATIC_TUMOR]
    if any(t in d for t in ("pancrea", "pdac")):
        return [_LABEL_PANCREATIC_TUMOR]
    if any(t in d for t in ("colon", "colorectal", "crc")):
        return [_LABEL_COLON_CANCER]
    return [_LABEL_LUNG_TUMOR]


def ensure_monai_runtime(*, verbose: bool = True, allow_install: bool = True) -> None:
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
            "MONAI is not installed and auto-install is disabled. "
            "Setup: pip install 'monai[nibabel,fire]' huggingface_hub"
        )
    log(
        "[seg-model] Installing monai + huggingface_hub (first use for VISTA-3D)…",
        enabled=verbose,
    )
    _pip_install_monai()
    try:
        import monai  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import monai after pip install. "
            "Try: pip install 'monai[nibabel,fire]' huggingface_hub"
        ) from exc
    _MONAI_READY = True
    log(
        f"[seg-model] MONAI ready (version={getattr(monai, '__version__', '?')})",
        enabled=verbose,
    )


def ensure_monai_vista3d(*, verbose: bool = True, allow_download: bool = True) -> Path:
    ensure_monai_runtime(verbose=verbose, allow_install=allow_download)
    existing = vista3d_bundle_dir()
    if existing is not None:
        log(f"[seg-model] Reusing MONAI VISTA-3D bundle → {existing}", enabled=verbose)
        return existing
    if not allow_download:
        raise FileNotFoundError(
            "MONAI VISTA-3D bundle missing and download disabled. "
            "Download with: python -m monai.bundle download vista3d "
            f"--bundle_dir {cache_root()} "
            "or set AGENTIC_RADIOGEN_VISTA3D_BUNDLE=/path/to/vista3d"
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
    log(f"[seg-model] Installed MONAI VISTA-3D → {bundle}", enabled=verbose)
    return bundle


def run_monai_vista3d_mask(
    volume: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float] | None = None,
    allow_download: bool = True,
    verbose: bool = True,
    disease: str | None = None,
    label_prompt: list[int] | None = None,
) -> np.ndarray:
    """Segment with VISTA-3D class prompt; return boolean mask (z,y,x)."""
    vol = np.squeeze(np.asarray(volume, dtype=np.float32))
    if vol.ndim != 3:
        raise ValueError(f"VISTA-3D expects 3D volume; got {vol.shape}")
    min_slices = int(os.environ.get("AGENTIC_RADIOGEN_NNUNET_MIN_SLICES") or 8)
    if int(vol.shape[0]) < min_slices:
        raise RuntimeError(
            f"too few slices for VISTA-3D ({vol.shape[0]} < {min_slices}); "
            "unsuitable for 3D tumor model on this case"
        )

    bundle = ensure_monai_vista3d(verbose=verbose, allow_download=allow_download)
    spacing = spacing_zyx or (1.0, 1.0, 1.0)
    labels = label_prompt or label_prompt_for_disease(disease)
    log(
        f"[seg-model] VISTA-3D infer label_prompt={labels} "
        f"(lung tumor=23; override via AGENTIC_RADIOGEN_VISTA3D_LABEL)",
        enabled=verbose,
    )
    mask = _infer_via_bundle_cli(
        vol,
        spacing_zyx=spacing,
        bundle=bundle,
        label_prompt=labels,
        verbose=verbose,
    )
    if not mask.any():
        raise RuntimeError(
            f"MONAI VISTA-3D returned an empty mask (label_prompt={labels})"
        )
    return mask.astype(bool)


def _pip_install_monai() -> None:
    packages = ["monai[nibabel,fire]", "huggingface_hub", "nibabel"]
    cmd = [sys.executable, "-m", "pip", "install", *packages, "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to install monai for VISTA-3D. "
            f"pip stderr: {(proc.stderr or '')[:800]}"
        )


def _bundle_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    # HF / monaihosting layouts vary slightly.
    if (path / "configs" / "inference.json").is_file() and (path / "models").is_dir():
        return True
    if (path / "models" / "model.pt").is_file():
        return True
    return False


def _download_bundle(dest: Path, *, verbose: bool = True) -> Path:
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
        log(
            f"[seg-model] monai.bundle.download failed ({exc}); trying HuggingFace…",
            enabled=verbose,
        )

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
        raise FileNotFoundError(
            f"Downloaded {_HF_REPO} but configs/inference.json or models/ missing"
        )
    return hit


def _infer_via_bundle_cli(
    volume: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    bundle: Path,
    label_prompt: list[int],
    verbose: bool,
) -> np.ndarray:
    from agentic_radiogen.imaging.segment import _mask_xyz_to_zyx, _nifti_from_volume_zyx

    with tempfile.TemporaryDirectory(prefix="vista3d_") as tmp:
        tmp_path = Path(tmp)
        inp = tmp_path / "case.nii.gz"
        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        img, shape_zyx = _nifti_from_volume_zyx(volume, spacing_zyx=spacing_zyx)
        import nibabel as nib

        nib.save(img, str(inp))

        cfg = bundle / "configs" / "inference.json"
        if not cfg.is_file():
            raise FileNotFoundError(f"VISTA-3D inference.json missing under {bundle}")

        # Absolute POSIX-ish path for JSON (WSL/Linux friendly).
        image_path = str(inp.resolve()).replace("\\", "/")
        input_dict = {"image": image_path, "label_prompt": list(label_prompt)}
        # monai.bundle.run accepts override kwargs; keep input_dict as JSON string.
        cmd = [
            sys.executable,
            "-m",
            "monai.bundle",
            "run",
            "--config_file",
            str(cfg.resolve()),
            "--bundle_root",
            str(bundle.resolve()),
            "--input_dict",
            json.dumps([input_dict]),
            "--output_dir",
            str(out_dir.resolve()),
        ]
        log(
            f"[seg-model] VISTA-3D bundle CLI → {bundle.name}",
            enabled=verbose,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(bundle))
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-1200:]
            raise RuntimeError(f"VISTA-3D bundle run failed: {err}")

        seg_files = sorted(out_dir.rglob("*.nii.gz"))
        if not seg_files:
            # Some configs write next to input or under bundle/eval.
            seg_files = sorted(tmp_path.rglob("*trans*.nii.gz")) + sorted(
                tmp_path.rglob("*seg*.nii.gz")
            )
        if not seg_files:
            raise RuntimeError(
                f"VISTA-3D produced no NIfTI under {out_dir}. "
                f"stdout/stderr tail: {(proc.stdout or '')[-400:]} {(proc.stderr or '')[-400:]}"
            )
        # Prefer newest / largest output.
        seg_path = max(seg_files, key=lambda p: p.stat().st_mtime)
        data = np.asanyarray(nib.load(str(seg_path)).dataobj)
        arr = _mask_xyz_to_zyx(data, shape_zyx)
        # Output may be multi-label; keep requested classes (or any >0).
        wanted = set(int(x) for x in label_prompt)
        if wanted and np.unique(arr).size > 2:
            mask = np.isin(arr, list(wanted))
        else:
            mask = arr > 0
        return np.asarray(mask, dtype=bool)
