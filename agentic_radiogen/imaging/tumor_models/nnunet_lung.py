"""nnU-Net MSD lung tumor (Task006): auto-download Zenodo weights + run inference.

Zenodo Task006_Lung.zip is an **nnU-Net v1** pretrained package (not usable by
nnU-Net v2). The SegmentationAgent downloads it on first use into
``data_cache/seg_models/nnunet_msd_lung/`` and runs ``nnunet`` (v1) predict.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from agentic_radiogen.util.progress import log

_DEFAULT_CACHE = Path.cwd() / "data_cache" / "seg_models" / "nnunet_msd_lung"
_ZENODO_URL = (
    "https://zenodo.org/records/4003545/files/Task006_Lung.zip?download=1"
)
_MARKER = "READY"


def cache_root() -> Path:
    env = (os.environ.get("AGENTIC_RADIOGEN_NNUNET_LUNG_CACHE") or "").strip()
    if env:
        return Path(env).expanduser()
    # Prefer Linux-native home cache when the project lives on /mnt/* (WSL→NTFS),
    # because atomic rename of multi-GB files on DrvFs often fails.
    default = _DEFAULT_CACHE
    try:
        resolved = str(default.resolve()).replace("\\", "/")
        if resolved.startswith("/mnt/"):
            return Path.home() / ".cache" / "agentic_radiogen" / "nnunet_msd_lung"
    except Exception:
        pass
    return default


def download_url() -> str:
    return (
        os.environ.get("AGENTIC_RADIOGEN_NNUNET_LUNG_URL") or ""
    ).strip() or _ZENODO_URL


def nnunet_lung_model_dir() -> Path | None:
    """Return trainer folder (v1 or v2) if already installed."""
    env = (os.environ.get("AGENTIC_RADIOGEN_NNUNET_LUNG_MODEL") or "").strip()
    if env:
        p = Path(env).expanduser()
        if _trainer_ready(p):
            return p

    roots: list[Path] = [cache_root(), _DEFAULT_CACHE]
    for key in ("nnUNet_results", "NNUNET_RESULTS", "RESULTS_FOLDER"):
        val = (os.environ.get(key) or "").strip()
        if val:
            roots.append(Path(val).expanduser())
    roots.append(Path.home() / "nnUNet_results")
    roots.append(Path.home() / ".cache" / "agentic_radiogen" / "nnunet_msd_lung")

    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        hit = _find_trainer(root)
        if hit is not None:
            return hit
    return None


def nnunet_lung_available() -> bool:
    """True when weights exist *or* can be auto-downloaded."""
    return nnunet_lung_model_dir() is not None or bool(download_url())


def ensure_nnunet_msd_lung(*, verbose: bool = True, allow_download: bool = True) -> Path:
    """Locate or download+install Task006 trainer folder; return path."""
    existing = nnunet_lung_model_dir()
    if existing is not None:
        log(
            f"[seg-model] Reusing nnU-Net MSD lung → {existing}",
            enabled=verbose,
        )
        return existing

    if not allow_download:
        raise FileNotFoundError(
            "nnU-Net MSD lung weights missing and download disabled"
        )

    dest = cache_root()
    dest.mkdir(parents=True, exist_ok=True)
    url = download_url()
    zip_path = dest / "Task006_Lung.zip"
    log(
        f"[seg-model] Downloading nnU-Net Task006_Lung (~5GB) from Zenodo "
        f"(first use only)…\n  → {url}",
        enabled=verbose,
    )
    _download_file(url, zip_path, verbose=verbose)
    log(f"[seg-model] Extracting {zip_path.name} → {dest}", enabled=verbose)
    _extract_zip(zip_path, dest)
    # Zip is large; keep it so re-extract is possible, but mark ready.
    (dest / _MARKER).write_text(url + "\n", encoding="utf-8")

    hit = _find_trainer(dest)
    if hit is None:
        raise FileNotFoundError(
            f"Downloaded Task006_Lung but no trainer folder found under {dest}. "
            "Expected plans.pkl (v1) or plans.json (v2) with fold checkpoints."
        )
    log(f"[seg-model] Installed nnU-Net MSD lung → {hit}", enabled=verbose)
    return hit


def run_nnunet_msd_lung_mask(
    volume: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float] | None = None,
    allow_download: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Segment lung tumor mask; auto-download Task006 on first call."""
    from agentic_radiogen.imaging.segment import _nifti_from_volume_zyx

    model_dir = ensure_nnunet_msd_lung(
        verbose=verbose, allow_download=allow_download
    )
    layout = _detect_layout(model_dir)
    vol = _prepare_zyx(volume)
    # Task006 was trained on multi-slice CTs; 2–3 slice series produce empty /
    # nonsense ROIs after resampling (see before crop: (1, 2, 512, 512)).
    min_slices = int(os.environ.get("AGENTIC_RADIOGEN_NNUNET_MIN_SLICES") or 8)
    if int(vol.shape[0]) < min_slices:
        raise RuntimeError(
            f"too few slices for nnU-Net MSD lung ({vol.shape[0]} < {min_slices}); "
            "unsuitable for 3D tumor model on this case"
        )
    vol_shape = vol.shape
    spacing = spacing_zyx or (1.0, 1.0, 1.0)
    os.environ.setdefault("nnUNet_n_proc_DA", "0")
    os.environ.setdefault("nnUNet_def_n_proc", "1")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        inp = tmp_path / "input"
        out = tmp_path / "output"
        inp.mkdir()
        out.mkdir()
        # nnU-Net expects modality suffix _0000
        case = inp / "case_0000.nii.gz"
        img, _ = _nifti_from_volume_zyx(vol, spacing_zyx=spacing)
        import nibabel as nib

        nib.save(img, str(case))

        if layout == "v2":
            _predict_v2(model_dir, inp, out)
        else:
            _predict_v1(model_dir, inp, out, verbose=verbose)

        seg_files = sorted(out.glob("*.nii.gz"))
        if not seg_files:
            raise RuntimeError(f"nnU-Net produced no output in {out}")
        seg_data = np.asanyarray(nib.load(str(seg_files[0])).dataobj)
        mask = _seg_to_zyx(seg_data, vol_shape)
        if not mask.any():
            # Common when no nodule is detected — fall back per patient, do not
            # disable the model for the whole cohort.
            raise RuntimeError("nnU-Net MSD lung returned an empty mask")
        return mask


def _predict_v2(model_dir: Path, inp: Path, out: Path) -> None:
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as exc:
        raise RuntimeError("nnU-Net v2 is not installed (pip install nnunetv2)") from exc

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case = sorted(inp.glob("*.nii.gz"))[0]
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    folds = _available_folds(model_dir)
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=folds or (0,),
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.predict_from_files(
        [[str(case)]],
        str(out),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )


def _predict_v1(
    model_dir: Path, inp: Path, out: Path, *, verbose: bool = True
) -> None:
    """Zenodo Task006 is nnU-Net v1; require ``nnunet`` package."""
    # Silence path warnings; inference only needs the trainer folder we pass in.
    results = str(model_dir.parents[2] if len(model_dir.parents) >= 3 else model_dir)
    os.environ.setdefault("RESULTS_FOLDER", results)
    os.environ.setdefault("nnUNet_raw_data_base", str(Path(results) / "raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(Path(results) / "preprocessed"))

    # PyTorch 2.6+ defaults torch.load(weights_only=True), which breaks nnU-Net v1
    # checkpoints (numpy scalars in the pickle). Zenodo MIC-DKFZ weights are trusted.
    _allow_legacy_torch_load()

    try:
        from nnunet.inference.predict import predict_from_folder  # type: ignore
    except ImportError:
        log(
            "[seg-model] Installing nnunet (v1) for Task006 Zenodo weights…",
            enabled=verbose,
        )
        _pip_install_nnunet_v1()
        from nnunet.inference.predict import predict_from_folder  # type: ignore

    # Fold 0 only: faster batch runs; ensemble of 5 folds is optional later.
    folds = [0]
    if not (model_dir / "fold_0").is_dir():
        folds = list(_available_folds(model_dir) or (0,))[:1]

    log(
        f"[seg-model] nnU-Net v1 predict folds={folds} model={model_dir.name}",
        enabled=verbose,
    )
    predict_from_folder(
        str(model_dir),
        str(inp),
        str(out),
        folds,
        False,  # save_npz
        1,  # num_threads_preprocessing
        1,  # num_threads_nifti_save
        None,  # lowres_segmentations
        0,  # part_id
        1,  # num_parts
        False,  # tta
        overwrite_existing=True,
        mode="fastest",
        overwrite_all_in_gpu=None,
        mixed_precision=True,
        step_size=0.5,
        checkpoint_name="model_final_checkpoint",
    )


def _allow_legacy_torch_load() -> None:
    """Patch torch.load so nnU-Net v1 checkpoints load under PyTorch 2.6+."""
    import torch

    if getattr(torch.load, "_agentic_radiogen_patched", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._agentic_radiogen_patched = True  # type: ignore[attr-defined]
    torch.load = _load  # type: ignore[assignment]


def _pip_install_nnunet_v1() -> None:
    # Official v1 package name on PyPI historically: nnunet
    cmd = [sys.executable, "-m", "pip", "install", "nnunet", "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to install nnunet (v1) required for Zenodo Task006_Lung. "
            f"pip stderr: {(proc.stderr or '')[:500]}"
        )


def _download_file(url: str, dest: Path, *, verbose: bool = True) -> None:
    """Stream download (Task006 is ~5GB — do not load into RAM)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.parent / (dest.name + ".partial")
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        log(f"[seg-model] Using existing zip {dest}", enabled=verbose)
        return
    # Resume-ish: if a large partial already exists from a failed rename, reuse it.
    if (
        partial.is_file()
        and partial.stat().st_size > 1_000_000_000
        and not dest.is_file()
    ):
        log(
            f"[seg-model] Finalizing existing partial download "
            f"({partial.stat().st_size // (1024**2)} MB)…",
            enabled=verbose,
        )
        _finalize_download(partial, dest)
        return

    req = urllib.request.Request(url, headers={"User-Agent": "agentic-radiogen/0.1"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        total = resp.headers.get("Content-Length")
        total_i = int(total) if total and total.isdigit() else None
        done = 0
        chunk = 1024 * 1024
        last_pct = -1
        with open(partial, "wb") as fh:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if verbose and total_i:
                    pct = int(100 * done / total_i)
                    if pct >= last_pct + 5:
                        log(
                            f"[seg-model] Download {pct}% ({done // (1024**2)} MB)",
                            enabled=True,
                        )
                        last_pct = pct
            fh.flush()
            os.fsync(fh.fileno())
    _finalize_download(partial, dest)


def _finalize_download(partial: Path, dest: Path) -> None:
    """Move partial → final. Avoid Path.replace on WSL/DrvFS (often fails)."""
    import shutil

    if dest.is_file():
        try:
            partial.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            if partial.is_file():
                partial.unlink()
        return
    try:
        # Same-filesystem rename when possible.
        os.replace(str(partial), str(dest))
        return
    except OSError:
        pass
    # Cross-device / DrvFS: copy then delete.
    shutil.copy2(str(partial), str(dest))
    try:
        partial.unlink()
    except OSError:
        pass


def _extract_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    # Zenodo zip often nests under nnUNet/; flatten one level if needed.
    nested = dest / "nnUNet"
    if nested.is_dir() and not (dest / "3d_fullres").is_dir():
        for item in nested.iterdir():
            target = dest / item.name
            if not target.exists():
                item.rename(target)


def _find_trainer(root: Path) -> Path | None:
    """Prefer 3d_fullres Task006 trainer over 2d / lowres / cascade."""
    candidates: list[Path] = []
    for plans in list(root.rglob("plans.json")) + list(root.rglob("plans.pkl")):
        parent = plans.parent
        if _trainer_ready(parent):
            candidates.append(parent)
    if not candidates:
        return None

    def _rank(p: Path) -> tuple:
        s = str(p).replace("\\", "/").lower()
        # Lower is better.
        pref = 50
        if "3d_fullres" in s and "cascade" not in s:
            pref = 0
        elif "3d_lowres" in s:
            pref = 1
        elif "2d" in s:
            pref = 2
        elif "cascade" in s:
            pref = 3
        lung = 0 if "task006" in s or "lung" in s else 1
        return (pref, lung, -len(p.parts), str(p))

    candidates.sort(key=_rank)
    return candidates[0]


def _trainer_ready(folder: Path) -> bool:
    has_plans = (folder / "plans.json").is_file() or (folder / "plans.pkl").is_file()
    if not has_plans:
        return False
    for fold in folder.glob("fold_*"):
        if (fold / "checkpoint_final.pth").is_file():
            return True
        if (fold / "model_final_checkpoint.model").is_file():
            return True
        if (fold / "model_final_checkpoint.model.pkl").is_file():
            return True
    return (folder / "checkpoint_final.pth").is_file()


def _detect_layout(model_dir: Path) -> str:
    if (model_dir / "plans.json").is_file():
        return "v2"
    return "v1"


def _available_folds(model_dir: Path) -> tuple[int, ...] | None:
    folds: list[int] = []
    for fold_dir in sorted(model_dir.glob("fold_*")):
        name = fold_dir.name
        try:
            folds.append(int(name.split("_", 1)[1]))
        except Exception:
            continue
    return tuple(folds) if folds else None


def _prepare_zyx(volume: np.ndarray) -> np.ndarray:
    vol = np.squeeze(np.asarray(volume, dtype=np.float32))
    if vol.ndim != 3:
        raise ValueError(f"nnU-Net expects 3D volume; got {vol.shape}")
    return vol


def _seg_to_zyx(data: np.ndarray, volume_shape_zyx: tuple[int, ...]) -> np.ndarray:
    from agentic_radiogen.imaging.segment import _mask_xyz_to_zyx

    arr = np.squeeze(np.asarray(data))
    if arr.ndim > 3:
        arr = arr[..., 0]
    return _mask_xyz_to_zyx(arr > 0, volume_shape_zyx).astype(bool)
