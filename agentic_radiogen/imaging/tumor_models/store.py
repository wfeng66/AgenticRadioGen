"""Local cache + download for tumor segmentation model packages."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable

from agentic_radiogen.data.http_json import get_bytes
from agentic_radiogen.imaging.tumor_models.registry import (
    TumorModelSpec,
    model_download_url,
)
from agentic_radiogen.util.progress import log

_DEFAULT_ROOT = Path.cwd() / "data_cache" / "seg_models"


class ModelStore:
    """Ensure disease-specific segmentation weights exist under data_cache/seg_models/."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        downloader: Callable[..., bytes] | None = None,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root) if root else _DEFAULT_ROOT
        self._download = downloader or get_bytes
        self.verbose = verbose

    def model_dir(self, spec: TumorModelSpec) -> Path:
        return self.root / spec.model_id

    def is_ready(self, spec: TumorModelSpec) -> bool:
        """True when the model can run without a network download."""
        if spec.runner in {
            "threshold_proxy",
            "totalsegmentator_organ",
            "totalsegmentator_task",
            "nnunet",
            "nnunet_autopet",
            "monai_bundle",
        }:
            # These runners manage their own cache / auto-download.
            return True
        if not spec.weight_filename:
            return True
        weight = self.model_dir(spec) / spec.weight_filename
        return weight.is_file()

    def ensure_local(self, spec: TumorModelSpec) -> Path:
        """Return local model directory; download zip package if missing."""
        dest = self.model_dir(spec)
        if self.is_ready(spec):
            log(
                f"[seg-model] Reusing local model '{spec.model_id}' → {dest}",
                enabled=self.verbose and bool(spec.weight_filename),
            )
            return dest

        url = model_download_url(spec)
        if not url:
            raise FileNotFoundError(
                f"Tumor model '{spec.model_id}' is not installed under {dest} and "
                f"no download URL is configured"
                + (
                    f" (set {spec.download_url_env}=https://.../model.zip)"
                    if spec.download_url_env
                    else ""
                )
            )

        log(
            f"[seg-model] Downloading '{spec.model_id}' from {url} ...",
            enabled=self.verbose,
        )
        raw = self._download(url, timeout=600)
        dest.mkdir(parents=True, exist_ok=True)
        self._extract_package(raw, dest)
        marker = dest / "manifest.json"
        if not marker.is_file():
            marker.write_text(
                json.dumps(
                    {
                        "model_id": spec.model_id,
                        "runner": spec.runner,
                        "weight_filename": spec.weight_filename,
                        "is_tumor_model": spec.is_tumor_model,
                        "source_url": url,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if not self.is_ready(spec):
            raise FileNotFoundError(
                f"Downloaded package for '{spec.model_id}' but missing "
                f"{spec.weight_filename} in {dest}"
            )
        log(f"[seg-model] Installed '{spec.model_id}' → {dest}", enabled=self.verbose)
        return dest

    @staticmethod
    def _extract_package(raw: bytes, dest: Path) -> None:
        # Zip package (preferred)
        if raw[:2] == b"PK":
            with zipfile.ZipFile(BytesIO(raw)) as zf:
                zf.extractall(dest)
            # Flatten single top-level directory if needed
            children = [p for p in dest.iterdir() if p.name != "__MACOSX"]
            if len(children) == 1 and children[0].is_dir():
                nested = children[0]
                for item in nested.iterdir():
                    target = dest / item.name
                    if not target.exists():
                        item.rename(target)
            return
        # Raw weight blob
        weight_name = "model.ts"
        (dest / weight_name).write_bytes(raw)
