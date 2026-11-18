"""Dataset discovery for segmentation fine-tuning.

Supported layouts (any may be used under ``--data-path``):

1. nnU-Net style::

       <root>/imagesTr/<id>_0000.nii.gz
       <root>/labelsTr/<id>.nii.gz

2. Flat paired NIfTI::

       <root>/images/<id>.nii.gz
       <root>/labels/<id>.nii.gz

3. Case folders::

       <root>/<case_id>/image.nii.gz
       <root>/<case_id>/label.nii.gz

4. LIDC-IDRI TCIA DICOM (+ XML in-tree or sibling ``manifest-*/LIDC-IDRI``)::

       <root>/LIDC-IDRI-0001/<study>/<series>/*.dcm
       # XML often at: ../manifest-*/LIDC-IDRI/LIDC-IDRI-0001/.../*.xml

``--dataset ntcia`` / ``lidc`` auto-detects the LIDC-IDRI tree and caches NIfTI
pairs under ``<root>/_finetune_cache/``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from agentic_radiogen.imaging.tumor_models.finetune.base import CasePair
from agentic_radiogen.imaging.tumor_models.finetune.lidc_idri import (
    discover_lidc_cases,
    looks_like_lidc_root,
    resolve_data_path,
)
from agentic_radiogen.util.progress import log

_IMAGE_NAMES = (
    "image.nii.gz",
    "image.nii",
    "ct.nii.gz",
    "ct.nii",
    "volume.nii.gz",
    "img.nii.gz",
)
_LABEL_NAMES = (
    "label.nii.gz",
    "label.nii",
    "segmentation.nii.gz",
    "segmentation.nii",
    "mask.nii.gz",
    "mask.nii",
    "seg.nii.gz",
    "gt.nii.gz",
)

_NIFTI_RE = re.compile(r"\.(nii(\.gz)?)$", re.IGNORECASE)


def list_datasets() -> dict[str, str]:
    return {
        "ntcia": (
            "NSCLC / lung TCIA collections. Also auto-detects LIDC-IDRI DICOM+XML "
            "under --data-path (caches NIfTI to <root>/_finetune_cache)."
        ),
        "lidc": "LIDC-IDRI TCIA DICOM (+ XML annotations); same as ntcia LIDC path.",
        "lidc-idri": "Alias for lidc.",
        "generic": "Generic image/label NIfTI pairs.",
        "nnunet": "nnU-Net raw layout (imagesTr/ + labelsTr/).",
    }


def discover_cases(
    dataset: str,
    data_path: Path,
    *,
    verbose: bool = True,
    cache_dir: Path | None = None,
) -> list[CasePair]:
    """Return paired image/label cases for ``dataset`` under ``data_path``."""
    root = resolve_data_path(Path(data_path).expanduser())
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    name = (dataset or "generic").strip().lower().replace("_", "-")
    search_roots = _search_roots(name, root)

    cases: list[CasePair] = []
    seen: set[str] = set()

    # Prefer LIDC DICOM+XML when the tree matches (or dataset explicitly asks).
    want_lidc = name in {"ntcia", "lidc", "lidc-idri", "lidcidri"} or any(
        looks_like_lidc_root(base) for base in search_roots
    )
    if want_lidc:
        max_cases = _max_cases_env()
        # Prefer a single real DICOM root; keep one shared cache under ``root``.
        lidc_bases = [b for b in search_roots if looks_like_lidc_root(b)]
        shared_cache = cache_dir or (root / "_finetune_cache")
        for base in lidc_bases:
            for case in discover_lidc_cases(
                base,
                cache_dir=shared_cache,
                max_cases=max_cases,
                verbose=verbose,
            ):
                if case.case_id in seen:
                    continue
                seen.add(case.case_id)
                cases.append(case)
        # If DICOM pairing failed but a prior conversion cache exists, use it.
        if not cases:
            cached = _from_lidc_nifti_cache(shared_cache)
            if cached and verbose:
                log(
                    f"[finetune] DICOM/XML pairing yielded 0 cases; "
                    f"reusing {len(cached)} NIfTI pair(s) from {shared_cache}",
                    enabled=True,
                )
            for case in cached:
                if case.case_id in seen:
                    continue
                seen.add(case.case_id)
                cases.append(case)

    if not cases:
        for base in search_roots:
            if base.name.lower() == "_finetune_cache":
                continue
            for finder in (_from_nnunet_raw, _from_flat_pairs, _from_case_folders):
                for case in finder(base):
                    if case.case_id in seen:
                        continue
                    seen.add(case.case_id)
                    cases.append(case)

    cases.sort(key=lambda c: c.case_id)
    if not cases:
        raise FileNotFoundError(
            f"No image/label pairs found under {root} for dataset='{name}'. "
            "Supported: imagesTr/labelsTr, images/labels, <case>/{image,label}.nii.gz, "
            "or LIDC-IDRI DICOM with XML (in-tree or sibling manifest-*/LIDC-IDRI)."
        )
    log(
        f"[finetune] Discovered {len(cases)} labeled cases "
        f"(dataset={name}, root={root})",
        enabled=verbose,
    )
    return cases


def _max_cases_env() -> int | None:
    raw = (os.environ.get("AGENTIC_RADIOGEN_LIDC_MAX_CASES") or "").strip()
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _search_roots(dataset: str, root: Path) -> list[Path]:
    roots = [root]
    # Parent pointing at LIDC-IDRI/ with lidc_idri/ child.
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if not child.is_dir():
            continue
        low = child.name.lower().replace("-", "_")
        # Never treat conversion cache as a second LIDC data root (caused
        # nested ``_finetune_cache/_finetune_cache`` and 100% false skips).
        if low == "_finetune_cache" or child.name.lower() == "_finetune_cache":
            continue
        if low in {"lidc_idri", "lidc-idri", "lidcidri"} or looks_like_lidc_root(child):
            roots.append(child)
        if dataset in {"ntcia", "nsclc", "nsclc-radiogenomics", "nsclc-radiomics"}:
            if low.startswith("nsclc") or "radiogenomic" in low or "radiomics" in low:
                roots.append(child)
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            key = str(r.resolve())
        except OSError:
            key = str(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _stem_key(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        name = name[:-7]
    elif name.lower().endswith(".nii"):
        name = name[:-4]
    if re.search(r"_0000$", name):
        name = name[:-5]
    return name


def _is_nifti(path: Path) -> bool:
    return bool(_NIFTI_RE.search(path.name))


def _from_nnunet_raw(root: Path) -> list[CasePair]:
    images = root / "imagesTr"
    labels = root / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        return []
    label_map = {_stem_key(p): p for p in labels.iterdir() if p.is_file() and _is_nifti(p)}
    pairs: list[CasePair] = []
    for img in sorted(images.iterdir()):
        if not img.is_file() or not _is_nifti(img):
            continue
        key = _stem_key(img)
        lab = label_map.get(key)
        if lab is None:
            continue
        pairs.append(CasePair(case_id=key, image=img, label=lab))
    return pairs


def _from_flat_pairs(root: Path) -> list[CasePair]:
    for img_dir_name, lab_dir_name in (
        ("images", "labels"),
        ("image", "label"),
        ("imgs", "masks"),
        ("ct", "seg"),
    ):
        images = root / img_dir_name
        labels = root / lab_dir_name
        if not images.is_dir() or not labels.is_dir():
            continue
        label_map = {
            _stem_key(p): p for p in labels.iterdir() if p.is_file() and _is_nifti(p)
        }
        pairs: list[CasePair] = []
        for img in sorted(images.iterdir()):
            if not img.is_file() or not _is_nifti(img):
                continue
            key = _stem_key(img)
            lab = label_map.get(key)
            if lab is None:
                continue
            pairs.append(CasePair(case_id=key, image=img, label=lab))
        if pairs:
            return pairs
    return []


def _from_case_folders(root: Path) -> list[CasePair]:
    pairs: list[CasePair] = []
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        if case_dir.name.lower() in {
            "images",
            "labels",
            "imagestr",
            "labelstr",
            "imagests",
            "labelsts",
            "_finetune_cache",
        }:
            continue
        if _PATIENT_RE_SKIP.match(case_dir.name):
            # LIDC patients without conversion should not be treated as nifti folders.
            continue
        img = _first_existing(case_dir, _IMAGE_NAMES)
        lab = _first_existing(case_dir, _LABEL_NAMES)
        if img is None or lab is None:
            nifti = [p for p in case_dir.iterdir() if p.is_file() and _is_nifti(p)]
            if len(nifti) == 2:
                a, b = nifti
                a_is_lab = bool(re.search(r"(seg|label|mask|gt)", a.name, re.I))
                b_is_lab = bool(re.search(r"(seg|label|mask|gt)", b.name, re.I))
                if a_is_lab and not b_is_lab:
                    img, lab = b, a
                elif b_is_lab and not a_is_lab:
                    img, lab = a, b
                else:
                    continue
            else:
                continue
        pairs.append(CasePair(case_id=case_dir.name, image=img, label=lab))
    return pairs


_PATIENT_RE_SKIP = re.compile(r"^LIDC-IDRI-\d+$", re.IGNORECASE)


def _from_lidc_nifti_cache(cache: Path) -> list[CasePair]:
    """Load previously converted ``_finetune_cache/LIDC-IDRI-*/{image,label}.nii.gz``.

    Skips nested ``_finetune_cache`` children and empty/tiny label stubs.
    """
    cache = Path(cache)
    if not cache.is_dir():
        return []
    pairs: list[CasePair] = []
    for case_dir in sorted(cache.iterdir()):
        if not case_dir.is_dir() or not _PATIENT_RE_SKIP.match(case_dir.name):
            continue
        img = case_dir / "image.nii.gz"
        lab = case_dir / "label.nii.gz"
        if not (img.is_file() and lab.is_file()):
            continue
        # Reject clearly empty / stub labels (header-only or tiny).
        try:
            if lab.stat().st_size < 200 or img.stat().st_size < 200:
                continue
        except OSError:
            continue
        pairs.append(CasePair(case_id=case_dir.name, image=img, label=lab))
    return pairs


def _first_existing(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = folder / name
        if p.is_file():
            return p
    return None
