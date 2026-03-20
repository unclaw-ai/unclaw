"""Memory helpers for unclaw."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from unclaw.memory.summarizer import (
    SessionMemoryFinding,
    SessionMemoryStats,
    StructuredSessionMemory,
    build_structured_session_memory,
    parse_persisted_session_memory,
    serialize_structured_session_memory,
    summarize_session_messages,
)

if TYPE_CHECKING:
    from unclaw.memory.manager import MemoryManager, SessionMemoryState

__all__ = [
    "MemoryManager",
    "SessionMemoryState",
    "SessionMemoryFinding",
    "SessionMemoryStats",
    "StructuredSessionMemory",
    "build_structured_session_memory",
    "parse_persisted_session_memory",
    "serialize_structured_session_memory",
    "summarize_session_messages",
]


def __getattr__(name: str) -> Any:
    if name in {"MemoryManager", "SessionMemoryState"}:
        from unclaw.memory.manager import MemoryManager, SessionMemoryState

        exports = {
            "MemoryManager": MemoryManager,
            "SessionMemoryState": SessionMemoryState,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
