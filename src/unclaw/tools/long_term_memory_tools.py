"""Long-term cross-session user memory tools.

Four tools:
  remember_long_term_memory  — Store a persistent memory item.
  search_long_term_memory    — Search memories by text.
  list_long_term_memory      — List all memories (with optional filter).
  forget_long_term_memory    — Delete a memory by id.

All backed by LongTermStore (local SQLite at data_dir/memory/long_term.db).

Design rules:
- Long-term memory is NOT injected into context by default.
- The model retrieves it when needed, just like web_search.
- Store conservatively: only on explicit user request ("remember this").
- All tool output is structured and language-neutral — the model composes
  the final answer in the user's language.
- Safe to call from worker threads: LongTermStore opens a fresh connection
  per read; writes are serialised by an internal lock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unclaw.memory.long_term_store import LongTermMemoryRecord, LongTermStore
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
    pass

_DEFAULT_LIST_LIMIT = 50
_DEFAULT_SEARCH_LIMIT = 20

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

REMEMBER_LONG_TERM_MEMORY_DEFINITION = ToolDefinition(
    name="remember_long_term_memory",
    description=(
        "Store a persistent memory item that will survive across sessions. "
        "Use this only when the user explicitly asks to remember something, "
        "or when storing a clearly important user preference or fact for the future. "
        "Do NOT auto-store random facts. "
        "key is a short title (e.g. 'preferred language'); "
        "value is the full content to remember. "
        "category and tags are optional for organisation."
    ),
    permission_level=ToolPermissionLevel.LOCAL_WRITE,
    arguments={
        "key": ToolArgumentSpec(
            description="Short title or name for this memory (required).",
        ),
        "value": ToolArgumentSpec(
            description="Full content of the memory to store (required).",
        ),
        "category": ToolArgumentSpec(
            description=(
                "Optional category label (e.g. 'preference', 'fact', 'goal'). "
                "Omit if not relevant."
            ),
        ),
        "tags": ToolArgumentSpec(
            description=(
                "Optional comma-separated tags for later search (e.g. 'language,ui'). "
                "Omit if not relevant."
            ),
        ),
        "source_session_id": ToolArgumentSpec(
            description="Optional: the session id where this memory originated.",
        ),
    },
)

SEARCH_LONG_TERM_MEMORY_DEFINITION = ToolDefinition(
    name="search_long_term_memory",
    description=(
        "Search the user's long-term memory store for items matching a query. "
        "Searches across key, value, and tags using substring matching. "
        "Returns matching memory items in structured form. "
        "Use this when the user asks about past preferences, facts, or goals "
        "that may have been remembered in a prior session."
    ),
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={
        "query": ToolArgumentSpec(
            description="Text to search for in key, value, and tags (required).",
        ),
        "category": ToolArgumentSpec(
            description=(
                "Optional category filter. Only return memories in this category."
            ),
        ),
        "limit": ToolArgumentSpec(
            description=(
                f"Maximum number of results to return. Default: {_DEFAULT_SEARCH_LIMIT}."
            ),
            value_type="integer",
        ),
    },
)

LIST_LONG_TERM_MEMORY_DEFINITION = ToolDefinition(
    name="list_long_term_memory",
    description=(
        "List all stored long-term memory items, optionally filtered by category. "
        "Returns items in structured form, newest first. "
        "Use this to show the user what has been remembered, or to browse "
        "all stored preferences and facts."
    ),
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={
        "category": ToolArgumentSpec(
            description=(
                "Optional category filter. Omit to list all memories regardless of category."
            ),
        ),
        "limit": ToolArgumentSpec(
            description=(
                f"Maximum number of items to return. Default: {_DEFAULT_LIST_LIMIT}."
            ),
            value_type="integer",
        ),
    },
)

FORGET_LONG_TERM_MEMORY_DEFINITION = ToolDefinition(
    name="forget_long_term_memory",
    description=(
        "Delete a specific memory item by its id. "
        "Use this only when the user explicitly asks to forget or remove something. "
        "To find the id of a memory, use list_long_term_memory or "
        "search_long_term_memory first."
    ),
    permission_level=ToolPermissionLevel.LOCAL_WRITE,
    arguments={
        "id": ToolArgumentSpec(
            description="UUID id of the memory to delete (required).",
        ),
    },
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_long_term_memory_tools(
    registry: ToolRegistry,
    *,
    long_term_store: LongTermStore,
) -> None:
    """Register all four long-term memory tools on the provided registry."""
    registry.register(
        REMEMBER_LONG_TERM_MEMORY_DEFINITION,
        _make_remember_handler(long_term_store),
    )
    registry.register(
        SEARCH_LONG_TERM_MEMORY_DEFINITION,
        _make_search_handler(long_term_store),
    )
    registry.register(
        LIST_LONG_TERM_MEMORY_DEFINITION,
        _make_list_handler(long_term_store),
    )
    registry.register(
        FORGET_LONG_TERM_MEMORY_DEFINITION,
        _make_forget_handler(long_term_store),
    )


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------

def _make_remember_handler(store: LongTermStore) -> ToolHandler:
    def _handle(call: ToolCall) -> ToolResult:
        key = str(call.arguments.get("key", "")).strip()
        value = str(call.arguments.get("value", "")).strip()
        if not key:
            return ToolResult.failure(
                tool_name=REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
                error="key is required and must not be empty.",
            )
        if not value:
            return ToolResult.failure(
                tool_name=REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
                error="value is required and must not be empty.",
            )
        category = str(call.arguments.get("category", "")).strip()
        tags = str(call.arguments.get("tags", "")).strip()
        source_session_id = str(call.arguments.get("source_session_id", "")).strip()
        mem_id = store.store(
            key=key,
            value=value,
            category=category,
            tags=tags,
            source_session_id=source_session_id,
        )
        return ToolResult.ok(
            tool_name=REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
            output_text=(
                f"Memory stored.\n"
                f"id: {mem_id}\n"
                f"key: {key}\n"
                f"value: {value}"
                + (f"\ncategory: {category}" if category else "")
                + (f"\ntags: {tags}" if tags else "")
            ),
        )

    return _handle


def _make_search_handler(store: LongTermStore) -> ToolHandler:
    def _handle(call: ToolCall) -> ToolResult:
        query = str(call.arguments.get("query", "")).strip()
        if not query:
            return ToolResult.failure(
                tool_name=SEARCH_LONG_TERM_MEMORY_DEFINITION.name,
                error="query is required and must not be empty.",
            )
        category = str(call.arguments.get("category", "")).strip()
        limit_raw = call.arguments.get("limit")
        try:
            limit = int(limit_raw) if limit_raw is not None else _DEFAULT_SEARCH_LIMIT
        except (TypeError, ValueError):
            limit = _DEFAULT_SEARCH_LIMIT

        results = store.search(query=query, category=category, limit=limit)
        return ToolResult.ok(
            tool_name=SEARCH_LONG_TERM_MEMORY_DEFINITION.name,
            output_text=_format_records(results, f"Search results for '{query}'"),
        )

    return _handle


def _make_list_handler(store: LongTermStore) -> ToolHandler:
    def _handle(call: ToolCall) -> ToolResult:
        category = str(call.arguments.get("category", "")).strip()
        limit_raw = call.arguments.get("limit")
        try:
            limit = int(limit_raw) if limit_raw is not None else _DEFAULT_LIST_LIMIT
        except (TypeError, ValueError):
            limit = _DEFAULT_LIST_LIMIT

        results = store.list_all(category=category, limit=limit)
        label = f"Long-term memories{f' (category: {category})' if category else ''}"
        return ToolResult.ok(
            tool_name=LIST_LONG_TERM_MEMORY_DEFINITION.name,
            output_text=_format_records(results, label),
        )

    return _handle


def _make_forget_handler(store: LongTermStore) -> ToolHandler:
    def _handle(call: ToolCall) -> ToolResult:
        mem_id = str(call.arguments.get("id", "")).strip()
        if not mem_id:
            return ToolResult.failure(
                tool_name=FORGET_LONG_TERM_MEMORY_DEFINITION.name,
                error="id is required.",
            )
        deleted = store.forget(mem_id)
        if deleted:
            return ToolResult.ok(
                tool_name=FORGET_LONG_TERM_MEMORY_DEFINITION.name,
                output_text=f"Memory deleted.\nid: {mem_id}",
            )
        return ToolResult.failure(
            tool_name=FORGET_LONG_TERM_MEMORY_DEFINITION.name,
            error=f"No memory found with id '{mem_id}'.",
        )

    return _handle


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_records(records: list[LongTermMemoryRecord], header: str) -> str:
    if not records:
        return f"{header}\n\n(No memories found.)"
    lines = [f"{header} — {len(records)} item(s):", ""]
    for record in records:
        lines.append(f"[id: {record.id}]")
        lines.append(f"  key: {record.key}")
        lines.append(f"  value: {record.value}")
        if record.category:
            lines.append(f"  category: {record.category}")
        if record.tags:
            lines.append(f"  tags: {record.tags}")
        lines.append(f"  created: {record.created_at}")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "FORGET_LONG_TERM_MEMORY_DEFINITION",
    "LIST_LONG_TERM_MEMORY_DEFINITION",
    "REMEMBER_LONG_TERM_MEMORY_DEFINITION",
    "SEARCH_LONG_TERM_MEMORY_DEFINITION",
    "register_long_term_memory_tools",
]
