"""Pre-tool literal entity guard for web search tools.

When a small model substitutes a more-famous entity name before calling a
search tool (e.g. 'Marine Leleu' → 'Marine Le Pen'), this guard detects the
drift and restores the literal surface form from the user's original request.

Design constraints
------------------
- NOT a truth engine: does not verify whether an entity is correct.
- NOT a hallucination filter: only acts on query-level drift.
- First-lookup safeguard: the first search uses the user's literal entity.
- Later broadening is allowed: the model gets evidence back and adjusts.
- Side-effect free: returns new ToolCall instances, never mutates inputs.
- Multilingual-friendly: uses the same entity extraction as web_search.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace as _dataclass_replace

from unclaw.tools.contracts import ToolCall
from unclaw.tools.web_search import (
    _analyze_query_discipline,
    _entity_match_tokens,
    _extract_entity_surface,
)
from unclaw.tools.web_text import _fold_for_match

_GUARDED_SEARCH_TOOL_NAMES: frozenset[str] = frozenset({"search_web", "fast_web_search"})


def extract_user_entity_surface(user_input: str) -> str:
    """Extract the literal entity surface from the user's raw message.

    Uses the same multi-strategy extraction as the query discipline layer
    (quoted entity → prefixed entity → leading capitalized span → short token
    sequence).  Returns empty string when no entity can be confidently
    identified — in that case the guard is a no-op.
    """
    return _extract_entity_surface(user_input.strip()).strip()


def apply_entity_guard_to_tool_calls(
    tool_calls: Sequence[ToolCall],
    user_entity_surface: str,
) -> tuple[ToolCall, ...]:
    """Restore literal entity surface in search queries that have drifted.

    For ``search_web`` and ``fast_web_search`` calls, if the model's query
    entity differs from the user's literal entity surface, the query is
    corrected so the first lookup preserves user intent.

    No-op when:
    - ``user_entity_surface`` is empty (no identifiable entity in user input)
    - the model's query already contains the user entity
    - the model's query has no recognisable entity (no substitution occurred)

    Args:
        tool_calls: Model-generated tool calls for one agentic step.
        user_entity_surface: Literal entity surface extracted from user input.

    Returns:
        A tuple of tool calls with any drifted search queries restored.
    """
    if not user_entity_surface:
        return tuple(tool_calls)

    result: list[ToolCall] = []
    for tc in tool_calls:
        if tc.tool_name in _GUARDED_SEARCH_TOOL_NAMES:
            tc = _guard_search_tool_call(tc, user_entity_surface)
        result.append(tc)
    return tuple(result)


def _guard_search_tool_call(
    tool_call: ToolCall,
    user_entity_surface: str,
) -> ToolCall:
    query = tool_call.arguments.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return tool_call

    if not _entity_drift_detected(query, user_entity_surface):
        return tool_call

    corrected_query = _build_corrected_query(query, user_entity_surface)
    return _dataclass_replace(
        tool_call,
        arguments={**tool_call.arguments, "query": corrected_query},
    )


def _entity_drift_detected(model_query: str, user_entity_surface: str) -> bool:
    """Return True when the model's query has silently substituted the user entity.

    Drift conditions (all must hold):
    1. User entity string is NOT contained in the model query.
    2. Not all user entity tokens are present in the model query tokens.
    3. The model query has a recognisable entity surface different from the
       user's entity — i.e. genuine substitution, not just a keyword query.
    """
    if not user_entity_surface or not model_query:
        return False

    user_entity_folded = _fold_for_match(user_entity_surface)
    model_query_folded = _fold_for_match(model_query)

    # No drift: user entity string present in model query.
    if user_entity_folded in model_query_folded:
        return False

    user_entity_tokens = _entity_match_tokens(user_entity_surface)
    if not user_entity_tokens:
        return False

    # Only treat as drift if the model query has a proper-name-like entity.
    # The entity extraction fallback returns the entire query for 1-4 token
    # keyword queries (all lowercase).  Those are not entity substitutions —
    # the model is just doing a keyword search.
    model_discipline = _analyze_query_discipline(model_query)
    if not model_discipline.entity_surface:
        return False

    # Require at least one capitalised alphabetic token in the entity surface.
    # This distinguishes proper names ("Marine Le Pen") from keyword phrases
    # ("french athlete fitness") that the fallback extractor returns verbatim.
    has_proper_noun = any(
        tok[0].isupper()
        for tok in model_discipline.entity_surface.split()
        if tok and tok[0].isalpha()
    )
    if not has_proper_noun and '"' not in model_discipline.entity_surface:
        return False

    model_entity_folded = model_discipline.normalized_entity
    if model_entity_folded == user_entity_folded:
        return False

    model_query_token_set = set(model_query_folded.split())
    if all(tok in model_query_token_set for tok in user_entity_tokens):
        extra_proper_name_token_present = any(
            raw_token
            and raw_token[0].isupper()
            and _fold_for_match(raw_token) not in set(user_entity_tokens)
            for raw_token in model_discipline.entity_surface.split()
        )
        if not extra_proper_name_token_present:
            return False

    # Narrow noise tolerance: if the user entity ends with a single isolated
    # uppercase letter (e.g. "Inoxtag N" — likely a stray keypress), and the
    # model's entity matches the entity surface without that trailing token,
    # treat it as no drift.  This is intentionally narrow: requires ≥2 tokens
    # AND the trailing token is exactly one uppercase letter.
    user_entity_parts = user_entity_surface.split()
    if (
        len(user_entity_parts) >= 2
        and len(user_entity_parts[-1]) == 1
        and user_entity_parts[-1][0].isupper()
    ):
        trimmed_user_entity_folded = _fold_for_match(" ".join(user_entity_parts[:-1]))
        if model_entity_folded == trimmed_user_entity_folded:
            return False  # Model used entity without the trailing noise token — OK

    return model_entity_folded != user_entity_folded


def _build_corrected_query(model_query: str, user_entity_surface: str) -> str:
    """Build a corrected query that uses the user's literal entity.

    Preserves useful context tokens from the model's original query so
    the search is still scoped (e.g. 'Marine Leleu biographie' rather
    than just 'Marine Leleu').
    """
    model_discipline = _analyze_query_discipline(model_query)
    context_tokens = model_discipline.context_tokens
    if context_tokens:
        return user_entity_surface + " " + " ".join(context_tokens[:3])
    return user_entity_surface


__all__ = [
    "apply_entity_guard_to_tool_calls",
    "extract_user_entity_surface",
]
