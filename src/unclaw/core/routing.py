"""Legacy routing compatibility helpers.

Active runtime execution is model-first and does not use deterministic routing
or entity recentering. This module keeps inactive compatibility wrappers for
legacy unit tests and bounded helper calls outside the runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def _build_request_routing_note(*, user_input: str, capability_summary: object) -> str | None:
    """Compatibility wrapper for the inactive legacy routing-note builder."""
    from unclaw.core.routing_legacy import _build_request_routing_note as _legacy_builder

    return _legacy_builder(
        user_input=user_input,
        capability_summary=capability_summary,
    )


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
