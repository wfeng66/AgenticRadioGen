"""AutoPET nnU-Net (lab-midas / AutoPET II): metabolic lesion segmentation.

Expects dual-channel PET+CT (``_0000``=CT, ``_0001``=PET/SUV). CT-only volumes
are skipped with a patient-local error so the agent can try the next tumor model.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from agentic_radiogen.util.progress import log

_DEFAULT_CACHE = Path.cwd() / "data_cache" / "seg_models" / "nnunet_autopet"
_ZENODO_URL = (
    "https://zenodo.org/records/8362371/files/"
    "nnUNetTrainer__nnUNetPlans__3d_fullres_resenc_bs80_exported.zip?download=1"
)
_GDRIVE_ID = "1G0HGHzQMXzslGDxFSNs5fq3RCeAu7M6l"
_MARKER = "READY"


def cache_root() -> Path:
    env = (os.environ.get("AGENTIC_RADIOGEN_AUTOPET_CACHE") or "").strip()
    if env:
        return Path(env).expanduser()
    default = _DEFAULT_CACHE
    try:
        resolved = str(default.resolve()).replace("\\", "/")
        if resolved.startswith("/mnt/"):
            return Path.home() / ".cache" / "agentic_radiogen" / "nnunet_autopet"
    except Exception:
        pass
    return default


def download_url() -> str:
    return (
        os.environ.get("AGENTIC_RADIOGEN_AUTOPET_URL") or ""
    ).strip() or _ZENODO_URL


def autopet_model_dir() -> Path | None:
    env = (os.environ.get("AGENTIC_RADIOGEN_AUTOPET_MODEL") or "").strip()
    if env:
        p = Path(env).expanduser()
        if _trainer_ready(p):
            return p
    roots = [cache_root(), _DEFAULT_CACHE]
    for key in ("nnUNet_results", "NNUNET_RESULTS", "RESULTS_FOLDER"):
        val = (os.environ.get(key) or "").strip()
        if val:
            roots.append(Path(val).expanduser())
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


def ensure_nnunet_autopet(*, verbose: bool = True, allow_download: bool = True) -> Path:
    existing = autopet_model_dir()
    if existing is not None:
        log(f"[seg-model] Reusing AutoPET nnU-Net → {existing}", enabled=verbose)
        return existing
    if not allow_download:
        raise FileNotFoundError("AutoPET nnU-Net weights missing and download disabled")

    dest = cache_root()
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "autopet_weights.zip"

    # Prefer lab-midas Google Drive baseline (fold_0) when gdown works; else Zenodo.
    gdrive_id = (
        os.environ.get("AGENTIC_RADIOGEN_AUTOPET_GDRIVE_ID") or ""
    ).strip() or _GDRIVE_ID
    prefer_gdrive = (
        os.environ.get("AGENTIC_RADIOGEN_AUTOPET_PREFER_GDRIVE") or "1"
    ).strip().lower() not in {"0", "false", "no"}

    if prefer_gdrive and not (zip_path.is_file() and zip_path.stat().st_size > 1_000_000):
        try:
            log(
                f"[seg-model] Downloading AutoPET baseline (lab-midas) from Google Drive "
                f"id={gdrive_id}…",
                enabled=verbose,
            )
            _download_gdrive(gdrive_id, zip_path, verbose=verbose)
        except Exception as exc:
            log(
                f"[seg-model] Google Drive AutoPET download failed ({exc}); "
                f"falling back to Zenodo…",
                enabled=verbose,
            )

    if not (zip_path.is_file() and zip_path.stat().st_size > 1_000_000):
        url = download_url()
        log(
            f"[seg-model] Downloading AutoPET II nnU-Net (~3.8GB) from Zenodo…\n  → {url}",
            enabled=verbose,
        )
        from agentic_radiogen.imaging.tumor_models.nnunet_lung import _download_file

        _download_file(url, zip_path, verbose=verbose)

    log(f"[seg-model] Extracting {zip_path.name} → {dest}", enabled=verbose)
    _extract_zip(zip_path, dest)
    (dest / _MARKER).write_text("autopet\n", encoding="utf-8")

    hit = _find_trainer(dest)
    if hit is None:
        raise FileNotFoundError(
            f"Downloaded AutoPET weights but no trainer folder under {dest}"
        )
    log(f"[seg-model] Installed AutoPET nnU-Net → {hit}", enabled=verbose)
    return hit


def run_nnunet_autopet_mask(
    volume: np.ndarray,
    *,
    pet_volume: np.ndarray | None = None,
    spacing_zyx: tuple[float, float, float] | None = None,
    modality: str = "CT",
    allow_download: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Segment metabolic lesions. Requires PET+CT (or modality=PT with PET vol)."""
    from agentic_radiogen.imaging.segment import _nifti_from_volume_zyx
    from agentic_radiogen.imaging.tumor_models.nnunet_lung import (
        _prepare_zyx,
        _seg_to_zyx,
    )

    mod = (modality or "CT").strip().upper()
    if mod == "MRI":
        mod = "MR"
    ct = _prepare_zyx(volume)
    pet = None if pet_volume is None else _prepare_zyx(pet_volume)

    # AutoPET is trained on FDG PET + CT. CT-only → patient-local skip.
    if pet is None and mod not in {"PT", "PET"}:
        if _looks_like_ct_hu(ct):
            raise RuntimeError(
                "AutoPET requires PET+CT (metabolic); unsuitable for CT-only case"
            )
        # Volume may already be SUV (PT series passed as sole channel).
        pet = ct
        ct = ct.copy()
    elif pet is None:
        # Single-channel PET: duplicate as CT proxy (degraded).
        pet = ct
        log(
            "[seg-model] AutoPET: PET-only input — using PET as CT channel proxy",
            enabled=verbose,
        )

    if pet.shape != ct.shape:
        raise RuntimeError(
            f"AutoPET PET/CT shape mismatch: CT {ct.shape} vs PET {pet.shape}"
        )

    min_slices = int(os.environ.get("AGENTIC_RADIOGEN_NNUNET_MIN_SLICES") or 8)
    if int(ct.shape[0]) < min_slices:
        raise RuntimeError(
            f"too few slices for AutoPET ({ct.shape[0]} < {min_slices}); "
            "unsuitable for 3D tumor model on this case"
        )

    model_dir = ensure_nnunet_autopet(verbose=verbose, allow_download=allow_download)
    spacing = spacing_zyx or (1.0, 1.0, 1.0)
    vol_shape = ct.shape
    os.environ.setdefault("nnUNet_n_proc_DA", "0")
    os.environ.setdefault("nnUNet_def_n_proc", "1")

    import nibabel as nib

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        inp = tmp_path / "input"
        out = tmp_path / "output"
        inp.mkdir()
        out.mkdir()
        # Dataset221: _0000 = CT, _0001 = PET/SUV
        ct_img, _ = _nifti_from_volume_zyx(ct, spacing_zyx=spacing)
        pet_img, _ = _nifti_from_volume_zyx(pet, spacing_zyx=spacing)
        nib.save(ct_img, str(inp / "case_0000.nii.gz"))
        nib.save(pet_img, str(inp / "case_0001.nii.gz"))

        _predict_v2(model_dir, inp, out, verbose=verbose)

        seg_files = sorted(out.glob("*.nii.gz"))
        if not seg_files:
            raise RuntimeError(f"AutoPET nnU-Net produced no output in {out}")
        seg_data = np.asanyarray(nib.load(str(seg_files[0])).dataobj)
        mask = _seg_to_zyx(seg_data, vol_shape)
        if not mask.any():
            raise RuntimeError("AutoPET nnU-Net returned an empty mask")
        return mask


def _looks_like_ct_hu(volume: np.ndarray) -> bool:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return True
    return float(np.percentile(finite, 5)) < -100.0


def _predict_v2(model_dir: Path, inp: Path, out: Path, *, verbose: bool = True) -> None:
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        log("[seg-model] Installing nnunetv2 for AutoPET…", enabled=verbose)
        _pip_install_nnunetv2()
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    import torch

    from agentic_radiogen.imaging.tumor_models.nnunet_lung import _available_folds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ct = inp / "case_0000.nii.gz"
    pet = inp / "case_0001.nii.gz"
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
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
        [[str(ct), str(pet)]],
        str(out),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )


def _pip_install_nnunetv2() -> None:
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "pip", "install", "nnunetv2", "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to install nnunetv2 required for AutoPET. "
            f"pip stderr: {(proc.stderr or '')[:500]}"
        )


def _download_gdrive(file_id: str, dest: Path, *, verbose: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown", "--quiet"],
            check=False,
            capture_output=True,
        )
        import gdown  # type: ignore

    url = f"https://drive.google.com/uc?id={file_id}"
    log(f"[seg-model] gdown {url} → {dest}", enabled=verbose)
    out = gdown.download(url=url, output=str(dest), quiet=not verbose)
    if not out or not dest.is_file():
        raise RuntimeError("gdown failed to download AutoPET weights")


def _extract_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    # Flatten common nestings.
    for nested_name in ("nnUNet_results", "nnUNet"):
        nested = dest / nested_name
        if nested.is_dir():
            for item in nested.iterdir():
                target = dest / item.name
                if not target.exists():
                    item.rename(target)


def _find_trainer(root: Path) -> Path | None:
    candidates: list[Path] = []
    for plans in list(root.rglob("plans.json")) + list(root.rglob("plans.pkl")):
        parent = plans.parent
        if _trainer_ready(parent):
            candidates.append(parent)
    if not candidates:
        return None

    def _rank(p: Path) -> tuple:
        s = str(p).replace("\\", "/").lower()
        pref = 50
        if "3d_fullres" in s and "cascade" not in s:
            pref = 0
        elif "3d_lowres" in s:
            pref = 1
        elif "2d" in s:
            pref = 2
        autopet = 0 if ("autopet" in s or "dataset998" in s or "dataset221" in s) else 1
        return (pref, autopet, -len(p.parts), str(p))

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
    return (folder / "checkpoint_final.pth").is_file()
