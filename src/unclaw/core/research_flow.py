"""Shared /search command helpers and tool-history formatting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

from unclaw.core.search_grounding import (
    build_search_tool_history_summary,
    parse_search_tool_history,
    shape_reply_with_grounding,
    shape_search_backed_reply,
)
from unclaw.constants import EMPTY_RESPONSE_REPLY, RUNTIME_ERROR_REPLY
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
    ``run_user_turn()``.  If the active model supports native tool calling,
    the agent loop will execute ``search_web`` autonomously and the grounding
    transform will apply post-processing from the tool result in session
    history.  For models without native tool calling (e.g. json_plan), the
    model answers from its own knowledge — this is an honest temporary
    limitation until those profiles are upgraded.
    """
    query = _read_tool_call_query(tool_call)
    session = session_manager.ensure_current_session()
    shaped_request = _build_search_user_request(query)
    session_manager.add_message(
        MessageRole.USER,
        shaped_request,
        session_id=session.id,
    )

    # The grounding transform runs after the agent loop completes (tool
    # results are already persisted) but before the assistant reply is
    # stored, so it can safely inspect session history.
    def _grounding_transform(reply: str) -> str:
        return _apply_search_grounding_from_history(
            reply,
            query=query,
            session_manager=session_manager,
            session_id=session.id,
        )

    assistant_reply = run_user_turn(
        session_manager=session_manager,
        command_handler=command_handler,
        user_input=shaped_request,
        tracer=tracer,
        tool_registry=tool_registry,
        stream_output_func=stream_output_func,
        assistant_reply_transform=_grounding_transform,
    )
    return ResearchTurnResult(assistant_reply=assistant_reply)


def append_search_sources_section(
    reply_text: str,
    *,
    payload: SearchWebPayload | Mapping[str, Any] | None,
) -> str:
    """Append a compact sources section to a natural-language reply."""
    if reply_text in {RUNTIME_ERROR_REPLY, EMPTY_RESPONSE_REPLY}:
        return reply_text

    sources = _extract_search_sources(payload)
    if not sources:
        return reply_text

    lines = [reply_text.rstrip(), "", "Sources:"]
    for title, url in sources:
        if title:
            lines.append(f"- {title}: {url}")
        else:
            lines.append(f"- {url}")
    return "\n".join(lines)


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
            summary_points = _read_string_list(payload.get("summary_points"))
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

            sources = _extract_search_sources(payload)
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


def _extract_search_sources(
    payload: SearchWebPayload | Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, Mapping):
        return ()

    raw_sources = payload.get("display_sources")
    if not isinstance(raw_sources, list):
        raw_sources = payload.get("results")
        if not isinstance(raw_sources, list):
            return ()

    sources: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for entry in raw_sources:
        if not isinstance(entry, Mapping):
            continue
        title = entry.get("title")
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        normalized_url = url.strip()
        if normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        resolved_title = title.strip() if isinstance(title, str) else ""
        sources.append((resolved_title, normalized_url))

    return tuple(sources)


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

    The shaped request tells the model that the user explicitly asked for a
    web search, so models with native tool calling will invoke ``search_web``
    through the agent loop.
    """
    return (
        f"Search the web for: {query}\n\n"
        "Use the search_web tool to find relevant, up-to-date information "
        "and provide a detailed, grounded answer with sources."
    )


def _apply_search_grounding_from_history(
    reply: str,
    *,
    query: str,
    session_manager: Any,
    session_id: str,
) -> str:
    """Apply grounding rewrite and sources using tool results from session history.

    Called as an ``assistant_reply_transform`` after the agent loop has
    completed and tool results are already persisted.
    """
    grounding = _find_latest_search_grounding(session_manager, session_id)
    if grounding is None:
        return reply

    shaped = shape_reply_with_grounding(reply, grounding=grounding, query=query)
    return _append_sources_from_grounding(shaped, sources=grounding.display_sources)


def _find_latest_search_grounding(
    session_manager: Any,
    session_id: str,
) -> Any:
    """Find the most recent search grounding context from session history."""
    list_messages = getattr(session_manager, "list_messages", None)
    if not callable(list_messages):
        return None

    messages: Sequence[ChatMessage] = list_messages(session_id)
    for message in reversed(messages):
        if message.role is not MessageRole.TOOL:
            continue
        grounding = parse_search_tool_history(message.content)
        if grounding is not None:
            return grounding
    return None


def _append_sources_from_grounding(
    reply: str,
    *,
    sources: tuple[tuple[str, str], ...],
) -> str:
    """Append a compact sources section using parsed grounding display sources."""
    if reply in {RUNTIME_ERROR_REPLY, EMPTY_RESPONSE_REPLY}:
        return reply
    if not sources:
        return reply

    lines = [reply.rstrip(), "", "Sources:"]
    for title, url in sources:
        if title:
            lines.append(f"- {title}: {url}")
        else:
            lines.append(f"- {url}")
    return "\n".join(lines)


def _read_string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )
