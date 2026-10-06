from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress_enabled() -> bool:
    return os.environ.get("AGENTIC_RADIOGEN_QUIET", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }


def log(message: str, *, enabled: bool | None = None) -> None:
    """Print a progress / intermediate-result line to stderr (keeps stdout JSON clean)."""
    if enabled is None:
        enabled = progress_enabled()
    if not enabled:
        return
    print(message, file=sys.stderr, flush=True)


def progress_iter(
    items: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    enabled: bool | None = None,
) -> Iterator[T]:
    """Yield items with a progress bar when possible; otherwise plain iteration + heartbeat logs."""
    if enabled is None:
        enabled = progress_enabled()
    sequence = list(items) if total is None else items
    n_total = total if total is not None else len(sequence)  # type: ignore[arg-type]
    if not enabled or n_total <= 0:
        yield from sequence  # type: ignore[misc]
        return
    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(sequence, total=n_total, desc=desc or "progress", unit=unit, file=sys.stderr)
        return
    except Exception:
        pass
    # Fallback ASCII bar when tqdm is unavailable.
    done = 0
    width = 28
    for item in sequence:  # type: ignore[misc]
        yield item
        done += 1
        filled = int(width * done / max(1, n_total))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r{desc or 'progress'}: |{bar}| {done}/{n_total} {unit}",
            end="",
            file=sys.stderr,
            flush=True,
        )
    print(file=sys.stderr, flush=True)
