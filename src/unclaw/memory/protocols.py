"""Explicit memory contracts used by runtime and channel boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from unclaw.memory.manager import SessionMemoryState


@runtime_checkable
class SessionMemoryContextProvider(Protocol):
    """Build a bounded memory note for model context injection."""

    def build_context_note(self, session_id: str | None = None) -> str | None:
        """Return one compact context note for a session, if available."""


@runtime_checkable
class SessionMemorySummaryRefresher(Protocol):
    """Refresh the persisted deterministic session summary."""

    def build_or_refresh_session_summary(self, session_id: str | None = None) -> str:
        """Rebuild and persist the summary for a session."""


class SessionMemoryCommandInterface(SessionMemoryContextProvider, Protocol):
    """Memory features exposed through slash commands and runtime context."""

    def get_session_summary(self, session_id: str | None = None) -> str:
        """Return the current session summary."""

    def get_session_state(
        self,
        session_id: str | None = None,
        *,
        recent_limit: int | None = None,
    ) -> SessionMemoryState:
        """Return the compact memory state for one session."""


class SessionMemoryChannelInterface(
    SessionMemoryCommandInterface,
    SessionMemorySummaryRefresher,
    Protocol,
):
    """Combined memory contract required by interactive channels."""
