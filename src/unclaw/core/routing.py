"""Public routing helpers and compatibility wrappers.

Deterministic request-routing classifiers now live in `routing_legacy.py`
because they only exist to support the bounded legacy fallback retry. This
module keeps the compatibility surface plus the still-active entity-anchor and
search-dedup helpers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from unclaw.core.routing_legacy import (
    _build_request_routing_note,
    _looks_like_deep_search_request,
    _looks_like_explicit_entity_correction,
    _looks_like_follow_up_entity_request,
    _looks_like_identity_request,
    _looks_like_joint_entity_request,
    _looks_like_multi_entity_request,
    _normalize_runtime_routing_text,
)
from unclaw.core.session_manager import SessionManager
from unclaw.schemas.chat import ChatMessage, MessageRole
from unclaw.tools.contracts import ToolCall

_SEARCH_TOOL_NAMES_FOR_DEDUP: frozenset[str] = frozenset({"fast_web_search", "search_web"})


@dataclass(frozen=True, slots=True)
class _EntityAnchor:
    surface: str
    corrected: bool = False
    # True when the entity surface was extracted from the CURRENT user turn.
    # False when it came from a history traversal (follow-up fallback).
    # The entity guard is only applied when from_current_turn is True OR
    # the anchor is an explicit correction — never for stale history fallbacks.
    from_current_turn: bool = False


def _find_recent_entity_anchor(
    history: Sequence[ChatMessage],
) -> _EntityAnchor | None:
    from unclaw.tools.web_entity_guard import extract_user_entity_surface

    fallback_anchor: _EntityAnchor | None = None
    user_messages_seen = 0
    # When a correction utterance is found but the entity can't be extracted
    # from it (e.g. the sentence is too long: "Non, je parle bien du duo
    # YouTube francais McFly et Carlito"), carry this flag forward so the
    # next entity-bearing message is still treated as the corrected target.
    correction_context_active = False

    for message in reversed(history):
        if message.role is not MessageRole.USER:
            continue
        user_messages_seen += 1
        if user_messages_seen > 8:
            break

        message_is_correction = _looks_like_explicit_entity_correction(message.content)
        if message_is_correction:
            correction_context_active = True

        surface = extract_user_entity_surface(message.content)
        if not surface:
            continue
        normalized_message = _normalize_runtime_routing_text(message.content)
        if (
            _looks_like_follow_up_entity_request(
                user_input=message.content,
                normalized_user_input=normalized_message,
            )
            and _normalize_runtime_routing_text(surface).strip(" .!?")
            == normalized_message.strip(" .!?")
        ):
            continue

        anchor = _EntityAnchor(
            surface=surface,
            # Corrected either when this message is itself a correction, or when
            # a correction utterance appeared more recently in history (and its
            # entity could not be extracted, but it signals the same topic).
            corrected=message_is_correction or correction_context_active,
        )
        if anchor.corrected:
            return anchor
        if fallback_anchor is None:
            fallback_anchor = anchor

    return fallback_anchor


def _resolve_entity_anchor_for_turn(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
) -> _EntityAnchor | None:
    from unclaw.tools.web_entity_guard import extract_user_entity_surface

    explicit_surface = extract_user_entity_surface(user_input)
    normalized_user_input = _normalize_runtime_routing_text(user_input)
    normalized_explicit_surface = _normalize_runtime_routing_text(explicit_surface)
    explicit_surface_is_generic_follow_up = (
        explicit_surface
        and _looks_like_follow_up_entity_request(
            user_input=user_input,
            normalized_user_input=normalized_user_input,
        )
        and normalized_explicit_surface.strip(" .!?") == normalized_user_input.strip(" .!?")
    )
    if explicit_surface and not explicit_surface_is_generic_follow_up:
        # Multi-entity sequential requests (e.g. "Michou puis Cyprien puis Squeezie")
        # cannot be reduced to one enforcement surface — disable the guard so each
        # tool call keeps the entity the model naturally picks per sub-request.
        if _looks_like_multi_entity_request(user_input):
            return None
        return _EntityAnchor(
            surface=explicit_surface,
            corrected=_looks_like_explicit_entity_correction(user_input),
            from_current_turn=True,
        )

    if not _looks_like_follow_up_entity_request(
        user_input=user_input,
        normalized_user_input=normalized_user_input,
    ):
        return None

    # Follow-up: walk history to find the anchor entity.  History anchors always
    # have from_current_turn=False so the entity guard stays inactive for them
    # (only the recentering note nudges the model — no hard enforcement).
    return _find_recent_entity_anchor(session_manager.list_messages(session_id))


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
