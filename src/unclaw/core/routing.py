"""Public routing helpers and minimal compatibility wrappers.

The runtime keeps only structural legacy retry hints plus search dedup here.
Deterministic semantic routing and entity recentering are intentionally
disabled.
"""

from __future__ import annotations

from dataclasses import dataclass

from unclaw.core.routing_legacy import (
    _build_request_routing_note,
)
from unclaw.core.session_manager import SessionManager
from unclaw.tools.contracts import ToolCall

_SEARCH_TOOL_NAMES_FOR_DEDUP: frozenset[str] = frozenset(
    {"fast_web_search", "search_web"}
)


@dataclass(frozen=True, slots=True)
class _EntityAnchor:
    surface: str
    corrected: bool = False
    from_current_turn: bool = False


def _looks_like_deep_search_request(user_input: str) -> bool:
    """Compatibility stub: semantic deep-search classification is disabled."""
    del user_input
    return False


def _resolve_entity_anchor_for_turn(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
) -> _EntityAnchor | None:
    del session_manager, session_id, user_input
    return None


def _deduplicate_search_tool_calls(
    tool_calls: tuple[ToolCall, ...],
) -> tuple[ToolCall, ...]:
    """Drop duplicate search calls with identical (tool, query) in a single batch.

    When entity-routing goes wrong and the same query is emitted for all
    planned searches (e.g. 'Michou' three times instead of Michou/Cyprien/
    Squeezie), executing them all wastes cycles and produces misleading output.
    This guard eliminates the duplicates before execution.  Non-search tool
    calls are always kept.
    """
    seen_signatures: set[tuple[str, str]] = set()
    deduped: list[ToolCall] = []
    for tc in tool_calls:
        if tc.tool_name in _SEARCH_TOOL_NAMES_FOR_DEDUP:
            query = str(tc.arguments.get("query", "")).casefold().strip()
            sig = (tc.tool_name, query)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
        deduped.append(tc)
    return tuple(deduped)
