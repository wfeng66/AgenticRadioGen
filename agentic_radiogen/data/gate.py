from __future__ import annotations

from typing import Protocol

from agentic_radiogen.schemas.contracts import MatchPreview


class DownloadGate(Protocol):
    def approve(self, preview: MatchPreview) -> bool: ...


class AlwaysAllowGate:
    def approve(self, preview: MatchPreview) -> bool:
        return True


class AlwaysDenyGate:
    def approve(self, preview: MatchPreview) -> bool:
        return False


class FlagGate:
    """CLI human gate: fetch only when the user passed --approve-download."""

    def __init__(self, approved: bool) -> None:
        self.approved = approved

    def approve(self, preview: MatchPreview) -> bool:
        return self.approved
