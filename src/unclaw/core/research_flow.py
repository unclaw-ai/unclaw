"""Shared /search command helpers and tool-history formatting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

from unclaw.core.search_payload_helpers import (
    append_compact_search_sources,
    read_search_display_sources,
    read_search_string_items,
)
from unclaw.core.search_grounding import (
    build_search_tool_history_summary,
    parse_search_tool_history,
    shape_reply_with_grounding,
)
from unclaw.core.runtime import run_user_turn
from unclaw.llm.base import LLMContentCallback
from unclaw.schemas.chat import ChatMessage, MessageRole
from unclaw.tools.contracts import SearchWebPayload, ToolCall, ToolResult
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

if TYPE_CHECKING:
    from unclaw.tools.registry import ToolRegistry

_NO_FINDINGS_REPLY = "No clear findings were extracted from the retrieved sources."


@dataclass(frozen=True, slots=True)
class ResearchTurnResult:
    """Completed /search turn metadata returned to interactive channels."""

    assistant_reply: str
    tool_result: ToolResult | None = None


def is_search_tool_call(call: ToolCall) -> bool:
    """Return whether one tool call targets the search_web tool."""
    return call.tool_name == SEARCH_WEB_DEFINITION.name


def persist_tool_result(
    *,
    session_manager: Any,
    session_id: str,
    result: ToolResult,
    tool_call: ToolCall | None = None,
) -> None:
    """Persist one tool result into conversation history when possible."""
    add_message = getattr(session_manager, "add_message", None)
    if not callable(add_message):
        return

    content = build_tool_history_content(result, tool_call=tool_call)
    if not content.strip():
        return

    add_message(
        MessageRole.TOOL,
        content,
        session_id=session_id,
    )


def build_tool_history_content(
    result: ToolResult,
    *,
    tool_call: ToolCall | None = None,
) -> str:
    """Build compact history content for one tool result."""
    if result.tool_name == SEARCH_WEB_DEFINITION.name:
        return _build_search_tool_history_content(result, tool_call=tool_call)

    outcome = "success" if result.success else "error"
    body = result.output_text.strip()
    if not body:
        return ""

    return "\n".join(
        [
            f"Tool: {result.tool_name}",
            f"Outcome: {outcome}",
            "",
            body,
        ]
    )


def run_search_command(
    *,
    session_manager: Any,
    command_handler: Any,
    tracer: Any,
    tool_call: ToolCall,
    stream_output_func: LLMContentCallback | None = None,
    tool_registry: ToolRegistry | None = None,
) -> ResearchTurnResult:
    """Route /search through the shared runtime path without pre-executing.

    The search query is shaped into a user-intent request and fed into
    ``run_user_turn()``. For profiles without native tool calling, the runtime
    executes the explicit ``search_web`` request itself before the shared model
    answer step. Native tool-calling profiles keep the common agent loop path.
    """
    query = _read_tool_call_query(tool_call)
    session = session_manager.ensure_current_session()
    shaped_request = _build_search_user_request(query)
    session_manager.add_message(
        MessageRole.USER,
        shaped_request,
        session_id=session.id,
    )

    # Capture the session message count after adding the current turn's USER
    # message but before any search tool executes.  The grounding transform
    # uses this floor so it only inspects messages added in this turn,
    # preventing stale sources from an earlier turn leaking into the reply.
    _turn_messages = session_manager.list_messages(session.id)
    _turn_start_count = len(_turn_messages)

    # The grounding transform runs after the agent loop completes (tool
    # results are already persisted) but before the assistant reply is
    # stored, so it can safely inspect session history.
    def _grounding_transform(reply: str) -> str:
        return apply_search_grounding_from_history(
            reply,
            query=query,
            session_manager=session_manager,
            session_id=session.id,
            turn_start_message_count=_turn_start_count,
        )

    assistant_reply = run_user_turn(
        session_manager=session_manager,
        command_handler=command_handler,
        user_input=shaped_request,
        tracer=tracer,
        tool_registry=tool_registry,
        stream_output_func=stream_output_func,
        explicit_tool_call=tool_call,
        assistant_reply_transform=_grounding_transform,
    )
    return ResearchTurnResult(assistant_reply=assistant_reply)


def append_search_sources_section(
    reply_text: str,
    *,
    payload: SearchWebPayload | Mapping[str, Any] | None,
) -> str:
    """Append a compact sources section to a natural-language reply."""
    sources = read_search_display_sources(payload)
    return append_compact_search_sources(reply_text, sources=sources)


def _build_search_tool_history_content(
    result: ToolResult,
    *,
    tool_call: ToolCall | None,
) -> str:
    payload = result.payload if isinstance(result.payload, dict) else {}
    query = _read_search_query(payload, tool_call=tool_call)
    lines = [
        f"Tool: {result.tool_name}",
        f"Outcome: {'success' if result.success else 'error'}",
    ]

    body_lines: list[str] = []
    if result.success:
        grounded_lines = build_search_tool_history_summary(
            payload=payload,
            query=query,
            current_date=date.today(),
        )
        if grounded_lines:
            body_lines.extend(["", *grounded_lines])
        else:
            summary_points = read_search_string_items(payload.get("summary_points"))
            body_lines.extend(
                [
                    "",
                    "Findings:",
                ]
            )
            if summary_points:
                body_lines.extend(f"- {point}" for point in summary_points)
            else:
                body_lines.append(f"- {_NO_FINDINGS_REPLY}")

            sources = read_search_display_sources(payload)
            if sources:
                body_lines.extend(
                    [
                        "",
                        "Sources:",
                    ]
                )
                for title, url in sources:
                    if title:
                        body_lines.append(f"- {title}: {url}")
                    else:
                        body_lines.append(f"- {url}")
    else:
        error_text = result.error or result.output_text.strip()
        if query:
            body_lines.append(f"Search request: {query}")
        if error_text:
            if body_lines:
                body_lines.append("")
            body_lines.append(f"Error: {error_text}")

    if not body_lines:
        fallback_body = result.output_text.strip()
        if not fallback_body:
            return ""
        body_lines.append(fallback_body)

    return "\n".join([*lines, "", *body_lines])


def _read_search_query(
    payload: Mapping[str, Any],
    *,
    tool_call: ToolCall | None,
) -> str:
    query = payload.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()
    if tool_call is None:
        return ""
    return _read_tool_call_query(tool_call)


def _read_tool_call_query(tool_call: ToolCall) -> str:
    query = tool_call.arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search tool calls must include a non-empty query string.")
    return query.strip()


def _build_search_user_request(query: str) -> str:
    """Shape a raw search query into an explicit user-intent request.

    The shaped request works for both runtime-executed explicit searches and
    native tool-calling profiles. When search output is already present in the
    conversation, the model should answer from that grounded context instead of
    pretending it still needs to search.
    """
    return (
        f"The user explicitly invoked /search for: {query}\n\n"
        "If search results are not already present in this conversation, use "
        "the search_web tool to find relevant, up-to-date information. Then "
        "answer from the retrieved evidence and include compact sources."
    )


def build_web_search_route_note(
    *,
    query: str,
    search_results_ready: bool,
) -> str:
    """Build a small system note for normal turns routed into web-backed mode."""
    lines = [
        "Route requirement: this turn needs web-backed grounding.",
        f"Ground this request: {query}",
    ]
    if search_results_ready:
        lines.append(
            "Grounded search results should already be present in this "
            "conversation. Do not answer from unsupported memory."
        )
    else:
        lines.append(
            "If grounded search results are not already present in this "
            "conversation, call the search_web tool with a focused query."
        )
    lines.append("Answer from retrieved evidence and include compact sources.")
    return "\n".join(lines)


def apply_search_grounding_from_history(
    reply: str,
    *,
    query: str,
    session_manager: Any,
    session_id: str,
    turn_start_message_count: int = 0,
) -> str:
    """Apply grounding rewrite and sources using tool results from session history.

    Called as an ``assistant_reply_transform`` after the agent loop has
    completed and tool results are already persisted.

    ``turn_start_message_count`` restricts the scan to messages added during
    the current turn only (index >= turn_start_message_count).  When zero
    (default), all history is scanned — preserving backward-compatible
    behaviour for callers that do not supply a turn boundary.
    """
    grounding = _find_latest_search_grounding(
        session_manager, session_id, after_count=turn_start_message_count
    )
    if grounding is None:
        return reply

    shaped = shape_reply_with_grounding(reply, grounding=grounding, query=query)
    return append_compact_search_sources(shaped, sources=grounding.display_sources)


def _find_latest_search_grounding(
    session_manager: Any,
    session_id: str,
    after_count: int = 0,
) -> Any:
    """Find the most recent search grounding context from session history.

    ``after_count`` limits the scan to messages at index >= after_count so
    that stale grounding from earlier turns cannot contaminate a newer reply.
    """
    list_messages = getattr(session_manager, "list_messages", None)
    if not callable(list_messages):
        return None

    messages: Sequence[ChatMessage] = list_messages(session_id)
    candidates = messages[after_count:] if after_count > 0 else messages
    for message in reversed(candidates):
        if message.role is not MessageRole.TOOL:
            continue
        grounding = parse_search_tool_history(message.content)
        if grounding is not None:
            return grounding
    return None
