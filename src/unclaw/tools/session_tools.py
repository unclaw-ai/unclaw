"""Read-only session history recall tool for the Unclaw runtime.

One tool: inspect_session_history.

Reads the current session's persisted chat history from the dedicated
thread-safe ChatMemoryStore (JSONL files under data/memory/chats/).
The tool never touches the thread-bound SQLite connection — it reads
from JSONL files, which are safe to open from any execution thread.

Supports filtering by role, nth-message lookup, and listing with a cap.
Read-only. No mutation. No deletion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unclaw.memory.chat_store import ChatMemoryRecord, ChatMemoryStore
from unclaw.tools.contracts import (
    ToolArgumentSpec,
    ToolCall,
    ToolDefinition,
    ToolHandler,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from unclaw.core.session_manager import SessionManager

_DEFAULT_LIST_LIMIT = 50
_CONTENT_TRUNCATE_CHARS = 500

INSPECT_SESSION_HISTORY_DEFINITION = ToolDefinition(
    name="inspect_session_history",
    description=(
        "Return an exact, deterministic view of the current session's persisted "
        "message history. Use this tool to answer exact questions about prior "
        "prompts, their order, or message counts — never guess from memory. "
        "Supports filtering by role (user/assistant/tool), selecting the nth "
        "message (1-indexed), and limiting result size."
    ),
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={
        "filter_role": ToolArgumentSpec(
            description=(
                'Optional role filter: "user", "assistant", or "tool". '
                "Omit or leave empty to include all roles."
            ),
        ),
        "nth": ToolArgumentSpec(
            description=(
                "Optional 1-indexed position. Return only the nth message matching "
                "the filter. Omit or set to 0 to return all messages."
            ),
            value_type="integer",
        ),
        "limit": ToolArgumentSpec(
            description=(
                f"Maximum number of messages to return when listing all. "
                f"Default: {_DEFAULT_LIST_LIMIT}."
            ),
            value_type="integer",
        ),
    },
)


def register_session_tools(
    registry: ToolRegistry,
    *,
    session_manager: SessionManager,
) -> None:
    """Register the session history recall tool on the provided registry.

    Requires session_manager.chat_store to be a ChatMemoryStore instance.
    If chat_store is None or absent, registration is skipped gracefully.
    """
    chat_store = getattr(session_manager, "chat_store", None)
    if not isinstance(chat_store, ChatMemoryStore):
        return

    registry.register(
        INSPECT_SESSION_HISTORY_DEFINITION,
        _make_inspect_session_history_handler(session_manager, chat_store),
    )


def _make_inspect_session_history_handler(
    session_manager: SessionManager,
    chat_store: ChatMemoryStore,
) -> ToolHandler:
    def _handle(call: ToolCall) -> ToolResult:
        return _inspect_session_history(call, session_manager, chat_store)

    return _handle


def _inspect_session_history(
    call: ToolCall,
    session_manager: SessionManager,
    chat_store: ChatMemoryStore,
) -> ToolResult:
    """Execute the inspect_session_history tool call.

    Reads from chat_store (JSONL files) — not from the SQLite connection.
    Thread-safe: chat_store.read_messages() opens a fresh file handle.
    """
    session_id = session_manager.current_session_id
    if session_id is None:
        return ToolResult.failure(
            tool_name=INSPECT_SESSION_HISTORY_DEFINITION.name,
            error="No active session.",
        )

    all_records: list[ChatMemoryRecord] = chat_store.read_messages(session_id)

    user_count = sum(1 for r in all_records if r.role == "user")
    assistant_count = sum(1 for r in all_records if r.role == "assistant")
    tool_count = sum(1 for r in all_records if r.role == "tool")

    header = (
        f"Session history — {len(all_records)} messages total "
        f"({user_count} user, {assistant_count} assistant, {tool_count} tool)"
    )

    filter_role_raw = str(call.arguments.get("filter_role", "")).strip().lower()
    if filter_role_raw == "user":
        filtered = [r for r in all_records if r.role == "user"]
        role_label = "user"
    elif filter_role_raw == "assistant":
        filtered = [r for r in all_records if r.role == "assistant"]
        role_label = "assistant"
    elif filter_role_raw == "tool":
        filtered = [r for r in all_records if r.role == "tool"]
        role_label = "tool"
    else:
        filtered = list(all_records)
        role_label = "all"

    nth_raw = call.arguments.get("nth")
    try:
        nth = int(nth_raw) if nth_raw is not None else 0
    except (TypeError, ValueError):
        nth = 0

    limit_raw = call.arguments.get("limit")
    try:
        limit = int(limit_raw) if limit_raw is not None else _DEFAULT_LIST_LIMIT
    except (TypeError, ValueError):
        limit = _DEFAULT_LIST_LIMIT

    if nth > 0:
        if nth > len(filtered):
            return ToolResult.failure(
                tool_name=INSPECT_SESSION_HISTORY_DEFINITION.name,
                error=(
                    f"Requested message #{nth} ({role_label}) but only "
                    f"{len(filtered)} {role_label} messages exist in this session."
                ),
            )
        record = filtered[nth - 1]
        output = (
            f"{header}\n\n"
            f"{record.role.capitalize()} message #{nth}:\n"
            f"{record.content}"
        )
        return ToolResult.ok(
            tool_name=INSPECT_SESSION_HISTORY_DEFINITION.name,
            output_text=output,
        )

    shown = filtered[:limit] if limit > 0 else filtered
    lines = [header, "", f"Messages ({role_label}, {len(shown)} shown of {len(filtered)}):"]
    for index, record in enumerate(shown, start=1):
        content = record.content
        if len(content) > _CONTENT_TRUNCATE_CHARS:
            content = content[:_CONTENT_TRUNCATE_CHARS] + "...[truncated]"
        lines.append(f"[{index}] ({record.role}) {content}")

    return ToolResult.ok(
        tool_name=INSPECT_SESSION_HISTORY_DEFINITION.name,
        output_text="\n".join(lines),
    )


__all__ = [
    "INSPECT_SESSION_HISTORY_DEFINITION",
    "register_session_tools",
]
