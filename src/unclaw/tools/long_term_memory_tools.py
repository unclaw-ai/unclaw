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
        "Store or update a persistent fact or preference that must survive across "
        "sessions. Call this tool when the user explicitly requests storage of a "
        "stable personal fact, hardware detail, identity information, or preference "
        "— regardless of the language used. "
        "Positive examples (any phrasing that means 'store this for later'): "
        "'remember that my GPU is an RTX 4080', "
        "'souviens-toi que je m'appelle Vincent', "
        "'enregistre que ma langue préférée est le français', "
        "'remember my name is Alice', "
        "'keep in mind that I prefer dark mode'. "
        "Also call this tool for corrections: "
        "'my name is actually Vincent' (not just Alice), "
        "'correction: my first name is only Vincent', "
        "'my main GPU is actually an RTX 4090 now'. "
        "For corrections, use the same key and category as the original fact — "
        "the existing record will be updated in-place, not duplicated. "
        "Negative examples (do NOT store): general chat, questions, fetched URLs, "
        "web search results, or facts not explicitly flagged for long-term retention. "
        "key is a short semantic title (e.g. 'user name', 'main GPU', "
        "'preferred language'); value is the full content to remember. "
        "category (e.g. 'identity', 'hardware', 'preference') and tags are optional "
        "but strongly recommended — they improve later retrieval."
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
        "Search the user's long-term memory for stable facts or preferences stored "
        "in a prior session. "
        "Call this tool when the user asks about remembered facts, identity, hardware "
        "details, or preferences — regardless of the language used. "
        "Positive examples: "
        "'what do you remember about my hardware?', "
        "'do you know my name?', "
        "'que sais-tu de mon setup?', "
        "'je m'appelle comment?', "
        "'mon prénom ?', "
        "'what GPU do I have?', "
        "'what personal info do you have about me?', "
        "'sur mon matériel informatique, tu as quoi ?'. "
        "Pass a concise semantic English query term, not verbatim user phrasing. "
        "Examples: "
        "for 'je m'appelle comment?' pass query='name'; "
        "for 'mon prénom ?' pass query='name' with category='identity'; "
        "for 'what do you remember about my hardware?' pass query='hardware' with "
        "category='hardware'; for 'do you know my name?' pass query='name'. "
        "IMPORTANT: use this tool (not system_info) when the user asks what you "
        "*remember* or *know* about their hardware, setup, or identity. "
        "Use system_info only when the user asks about the *current* machine state "
        "(OS version, CPU usage, running processes). "
        "Searches across key, value, tags, and category fields. "
        "Do NOT use this tool for questions about the current session's message "
        "history or message order — use inspect_session_history for that."
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
        "Returns items newest-first in structured form. "
        "Use this when the user asks broadly what has been stored, without a specific "
        "topic: 'what do you know about me?', 'what have you remembered?', "
        "'que sais-tu de moi?', 'list everything you know about me'. "
        "For targeted recall of a specific topic, prefer search_long_term_memory."
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
        # upsert: if same (category, key) exists, update it — prevents stale
        # duplicate records after user corrections ('my name is actually X').
        mem_id, created = store.upsert(
            key=key,
            value=value,
            category=category,
            tags=tags,
            source_session_id=source_session_id,
        )
        action = "Memory stored" if created else "Memory updated"
        return ToolResult.ok(
            tool_name=REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
            output_text=(
                f"{action}.\n"
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
