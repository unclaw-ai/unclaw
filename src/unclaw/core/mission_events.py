"""Compact mission-progress events for the local agent kernel."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


type MissionEventCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class MissionEvent:
    """One user-facing mission progress event."""

    scope: str
    detail: str

    def render(self) -> str:
        return f"[{self.scope}] {self.detail}"


def emit_mission_event(
    callback: MissionEventCallback | None,
    *,
    scope: str,
    detail: str,
) -> None:
    """Emit one formatted mission-progress event when a callback is present."""

    if callback is None:
        return
    callback(MissionEvent(scope=scope, detail=detail).render())
