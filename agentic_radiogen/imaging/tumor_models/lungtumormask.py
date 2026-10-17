"""LungTumorMask (VemundFredriksen) — pip CLI for CT lung-tumor segmentation.

Official wheel pins ``torch<=1.11`` / ``numpy<=1.23`` (Python 3.8–3.10 era),
which conflicts with modern stacks (torch 2.x). We install the wheel with
``--no-deps`` and reuse the env's existing torch / monai / nibabel.

The wheel also imports APIs removed in MONAI ≥1.0 (``AddChanneld``,
``monai.utils.alias`` / ``export``). Our CLI entry
(``python -m agentic_radiogen.imaging.tumor_models.lungtumormask``) patches
those before invoking the upstream main.

Manual install (WSL / Linux)::

  pip install --no-deps https://github.com/VemundFredriksen/LungTumorMask/releases/download/v1.3.1/lungtumormask-1.3.1-py2.py3-none-any.whl
  pip install nibabel scikit-image lungmask
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from agentic_radiogen.util.progress import log

_WHEEL_URL = (
    "https://github.com/VemundFredriksen/LungTumorMask/releases/download/"
    "v1.3.1/lungtumormask-1.3.1-py2.py3-none-any.whl"
)
_SOFT_DEPS = ("nibabel", "scikit-image", "lungmask")


def lungtumormask_cli() -> str | None:
    """Return argv prefix for the CLI: path to script, or ``python -m lungtumormask``."""
    found = shutil.which("lungtumormask")
    if found:
        return found
    # Scripts/ may be missing from PATH after a fresh pip install in the same process.
    try:
        import lungtumormask  # noqa: F401

        return "module"
    except Exception:
        return None


def _noop_monai_decorator(*args, **kwargs):
    """Stand-in for removed ``monai.utils.alias`` / ``export`` decorators."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def _wrap(obj):
        return obj

    return _wrap


def apply_monai_addchanneld_shim() -> None:
    """Patch APIs broken between LungTumorMask 1.3.1 and modern deps.

    MONAI:
    - ``AddChanneld`` → ``EnsureChannelFirstd``
    - ``monai.utils.alias`` / ``export`` → no-op decorators (imported unused)
    - ``LoadImaged`` defaults to ``image_only=False`` so ``*_meta_dict`` keys exist

    lungmask ≥0.2:
    - ``get_model('unet', 'R231')`` (legacy) → ``get_model('R231')``
    """
    import monai.transforms as mt

    try:
        from monai.transforms import AddChanneld  # noqa: F401
    except ImportError:
        from monai.transforms import EnsureChannelFirstd

        mt.AddChanneld = EnsureChannelFirstd  # type: ignore[attr-defined, assignment]

    import monai.utils as mu

    if not hasattr(mu, "alias"):
        mu.alias = _noop_monai_decorator  # type: ignore[attr-defined]
    if not hasattr(mu, "export"):
        mu.export = _noop_monai_decorator  # type: ignore[attr-defined]

    # MONAI ≥1.0 defaults LoadImaged(image_only=True), which drops image_meta_dict
    # that LungTumorMask reads for pixdim/affine.
    if not getattr(mt.LoadImaged, "_agentic_meta_dict_compat", False):
        _BaseLoadImaged = mt.LoadImaged

        class LoadImaged(_BaseLoadImaged):  # type: ignore[valid-type,misc]
            _agentic_meta_dict_compat = True

            def __init__(self, *args, **kwargs):
                if "image_only" not in kwargs:
                    kwargs["image_only"] = False
                super().__init__(*args, **kwargs)

        mt.LoadImaged = LoadImaged  # type: ignore[misc,assignment]

    _apply_lungmask_get_model_compat()


def _apply_lungmask_get_model_compat() -> None:
    """LungTumorMask vs lungmask ≥0.2 API gaps.

    - ``get_model('unet', 'R231')`` → ``get_model('R231')``
    - restore ``utils.LungLabelsDS_inf`` (removed; used by LungTumorMask mask_lung)
    """
    try:
        import lungmask.mask as lm
        import lungmask.utils as lu
    except ImportError:
        return

    if not getattr(lm.get_model, "_agentic_legacy_compat", False):
        _orig = lm.get_model
        urls = getattr(lm, "MODEL_URLS", {}) or {}

        def get_model(modelname, modelpath=None, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Legacy: get_model('unet'|'resunet', 'R231')
            if (
                isinstance(modelname, str)
                and modelname.lower() in {"unet", "resunet"}
                and isinstance(modelpath, str)
                and modelpath in urls
            ):
                return _orig(modelpath)
            return _orig(modelname, modelpath, *args, **kwargs)

        get_model._agentic_legacy_compat = True  # type: ignore[attr-defined]
        lm.get_model = get_model  # type: ignore[assignment]

    if not hasattr(lu, "LungLabelsDS_inf"):
        import torch
        from torch.utils.data import Dataset

        class LungLabelsDS_inf(Dataset):
            """Minimal stand-in for the dataset class removed from lungmask."""

            def __init__(self, img: np.ndarray):
                self.img = np.asarray(img)

            def __len__(self) -> int:
                return int(self.img.shape[0])

            def __getitem__(self, idx: int):
                sl = self.img[idx]
                if sl.ndim == 2:
                    sl = sl[None, ...]
                return torch.from_numpy(np.ascontiguousarray(sl)).float()

        lu.LungLabelsDS_inf = LungLabelsDS_inf  # type: ignore[attr-defined]


def _cli_argv(cli: str) -> list[str]:
    # Always enter via this module so the AddChanneld shim is applied in-process
    # before LungTumorMask imports monai.transforms (subprocess would miss a
    # parent-process monkeypatch).
    _ = cli
    return [sys.executable, "-m", "agentic_radiogen.imaging.tumor_models.lungtumormask"]


def ensure_lungtumormask(*, verbose: bool = True, allow_install: bool = True) -> str:
    """Return CLI handle; pip-install wheel with ``--no-deps`` on first use."""
    found = lungtumormask_cli()
    if found:
        return found
    if not allow_install:
        raise RuntimeError(
            "lungtumormask not found and install disabled. "
            f"Install with: pip install --no-deps {_WHEEL_URL} && "
            f"pip install {' '.join(_SOFT_DEPS)}"
        )
    log(
        "[seg-model] Installing LungTumorMask (wheel v1.3.1, --no-deps; "
        "official torch<=1.11 pin is skipped for modern envs)…",
        enabled=verbose,
    )
    # Soft deps first (no ancient torch pin), then the wheel without deps.
    soft = [sys.executable, "-m", "pip", "install", *_SOFT_DEPS, "--quiet"]
    proc_soft = subprocess.run(soft, capture_output=True, text=True)
    if proc_soft.returncode != 0:
        log(
            f"[seg-model] Soft-deps install warning: {(proc_soft.stderr or '')[-400:]}",
            enabled=verbose,
        )
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", _WHEEL_URL, "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to install LungTumorMask (--no-deps). "
            f"Try manually:\n"
            f"  pip install --no-deps {_WHEEL_URL}\n"
            f"  pip install {' '.join(_SOFT_DEPS)}\n"
            f"{(proc.stderr or proc.stdout or '')[-800:]}"
        )
    found = lungtumormask_cli()
    if not found:
        raise RuntimeError(
            "lungtumormask installed but import/CLI not found. "
            "Re-open the shell or: python -c \"import lungtumormask\""
        )
    return found


def run_lungtumormask(
    volume: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float] | None = None,
    allow_download: bool = True,
    verbose: bool = True,
    threshold: float | None = None,
    lung_filter: bool = True,
) -> np.ndarray:
    """Run LungTumorMask CLI on a (z,y,x) CT volume → boolean mask."""
    from agentic_radiogen.imaging.segment import (
        _mask_xyz_to_zyx,
        _nifti_from_volume_zyx,
        _prepare_volume_zyx,
    )

    cli = ensure_lungtumormask(verbose=verbose, allow_install=allow_download)
    vol = _prepare_volume_zyx(volume)
    spacing = spacing_zyx or (1.0, 1.0, 1.0)
    thr = threshold
    if thr is None:
        thr = float(os.environ.get("AGENTIC_RADIOGEN_LUNGTUMORMASK_THRESHOLD") or 0.3)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        inp = tmp_path / "case.nii.gz"
        out = tmp_path / "mask.nii.gz"
        img, vol_shape = _nifti_from_volume_zyx(vol, spacing_zyx=spacing)
        import nibabel as nib

        nib.save(img, str(inp))

        cmd = [
            *_cli_argv(cli),
            str(inp),
            str(out),
            "--threshold",
            str(thr),
            "--batch-size",
            os.environ.get("AGENTIC_RADIOGEN_LUNGTUMORMASK_BATCH") or "4",
        ]
        if lung_filter:
            cmd.append("--lung-filter")
        # Prefer CPU if no CUDA to avoid hard crashes on CPU-only boxes.
        try:
            import torch

            if not torch.cuda.is_available():
                cmd.append("--cpu")
        except Exception:
            cmd.append("--cpu")

        log(
            f"[seg-model] LungTumorMask → {cli} (threshold={thr}, lung_filter={lung_filter})",
            enabled=verbose,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(
                "LungTumorMask failed: "
                + (proc.stderr or proc.stdout or "no output")[-2000:]
            )
        seg = np.asanyarray(nib.load(str(out)).dataobj)
        try:
            mask = _mask_xyz_to_zyx(seg > 0, vol_shape).astype(bool)
        except Exception:
            # CLI may already write (z,y,x)-ordered arrays depending on version.
            arr = np.squeeze(seg)
            if arr.shape == vol_shape:
                mask = (arr > 0).astype(bool)
            elif arr.shape == (vol_shape[2], vol_shape[1], vol_shape[0]):
                mask = np.transpose(arr > 0, (2, 1, 0))
            else:
                raise RuntimeError(
                    f"LungTumorMask mask shape {getattr(arr, 'shape', None)} "
                    f"incompatible with volume {vol_shape}"
                )
        if not mask.any():
            raise RuntimeError("LungTumorMask returned an empty mask")
        return mask


def main(argv: list[str] | None = None) -> None:
    """CLI entry: apply MONAI shim, then delegate to LungTumorMask."""
    apply_monai_addchanneld_shim()
    if argv is None:
        argv = sys.argv[1:]
    sys.argv = ["lungtumormask", *argv]
    from lungtumormask.__main__ import main as _ltm_main

    raise SystemExit(_ltm_main())


if __name__ == "__main__":
    main()
