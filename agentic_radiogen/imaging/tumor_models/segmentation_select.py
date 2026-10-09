"""Which catalog models the SegmentationAgent may select (cache / LLM / rules)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agentic_radiogen.imaging.tumor_models.registry import TumorModelSpec

_DEFAULT_BLOCK = Path.cwd() / "data_cache" / "seg_plans" / "blocked_models.json"


def ts_lung_nodules_enabled() -> bool:
    return (os.environ.get("AGENTIC_RADIOGEN_ENABLE_TS_LUNG_NODULES") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def load_runtime_blocked() -> set[str]:
    """Models blocked for this project (failed in prior runs)."""
    blocked: set[str] = set()
    if not ts_lung_nodules_enabled():
        blocked.add("ts_lung_nodules")
    if not _DEFAULT_BLOCK.is_file():
        return blocked
    try:
        raw = json.loads(_DEFAULT_BLOCK.read_text(encoding="utf-8"))
        ids = raw.get("model_ids") or raw.get("blocked") or []
        blocked |= {str(x) for x in ids if str(x).strip()}
        return blocked
    except Exception:
        return blocked


# Persist only hard failures (e.g. broken TS crop). Do NOT persist
# nnunet_msd_lung for "download disabled" — that is a test/session setting;
# the agent still auto-downloads Task006 in normal runs.
_PERSIST_BLOCK = frozenset({"ts_lung_nodules"})


def persist_blocked(model_id: str, *, reason: str = "") -> None:
    if model_id not in _PERSIST_BLOCK:
        return
    reason_l = (reason or "").lower()
    if "download disabled" in reason_l:
        return
    blocked = load_runtime_blocked()
    if model_id in blocked:
        return
    blocked.add(model_id)
    _DEFAULT_BLOCK.parent.mkdir(parents=True, exist_ok=True)
    _DEFAULT_BLOCK.write_text(
        json.dumps(
            {
                "model_ids": sorted(blocked),
                "reasons": {model_id: reason[:500]} if reason else {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def is_selectable(spec: TumorModelSpec) -> bool:
    if spec.model_id == "ts_lung_nodules" and not ts_lung_nodules_enabled():
        return False
    if spec.model_id in load_runtime_blocked():
        return False
    if spec.operational_type == "fine_tune":
        return False
    return True
