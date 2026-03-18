"""Tests for inspect_session_history tool — deterministic session recall.

Covers:
1. Exact first/second/nth user prompt lookup
2. Chronological listing of user prompts
3. Exact counts by role
4. Works after a long session with tool calls interspersed
5. Integrates correctly into the default tool registry via executor
6. Session tools read from ChatMemoryStore (JSONL), not from the thread-bound
   SQLite connection — no SQLite thread-affinity errors from worker threads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.memory.chat_store import ChatMemoryStore
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.session_tools import (
    INSPECT_SESSION_HISTORY_DEFINITION,
    _inspect_session_history,
    register_session_tools,
)
from unclaw.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Minimal stub SessionManager — only current_session_id + chat_store needed.
# ---------------------------------------------------------------------------


class _StubSessionManager:
    """Minimal stub providing only what session_tools needs."""

    def __init__(self, session_id: str, chat_store: ChatMemoryStore) -> None:
        self.current_session_id = session_id
        self.chat_store = chat_store


def _populate(
    store: ChatMemoryStore,
    session_id: str,
    messages: list[tuple[str, str]],
) -> None:
    """Append (role, content) pairs to the store."""
    for i, (role, content) in enumerate(messages):
        store.append_message(
            session_id=session_id,
            role=role,
            content=content,
            created_at=f"2026-03-18T10:00:{i:02d}Z",
        )


def _call(arguments: dict) -> ToolCall:
    return ToolCall(tool_name=INSPECT_SESSION_HISTORY_DEFINITION.name, arguments=arguments)


# ---------------------------------------------------------------------------
# Tests: nth-message exact lookup
# ---------------------------------------------------------------------------


def test_inspect_session_history_first_user_prompt(tmp_path: Path) -> None:
    session_id = "sess_001"
    store = ChatMemoryStore(tmp_path / "chats")
    _populate(store, session_id, [
        ("user", "Hello world"),
        ("assistant", "Hi!"),
        ("user", "What is 2+2?"),
        ("assistant", "4"),
    ])
    stub = _StubSessionManager(session_id, store)

    result = _inspect_session_history(
        _call({"filter_role": "user", "nth": 1}),
        stub,
        store,
    )

    assert result.success
    assert "User message #1:" in result.output_text
    assert "Hello world" in result.output_text


def test_inspect_session_history_second_user_prompt(tmp_path: Path) -> None:
    session_id = "sess_002"
    store = ChatMemoryStore(tmp_path / "chats")
    _populate(store, session_id, [
        ("user", "First prompt"),
        ("assistant", "Reply 1"),
        ("user", "Second prompt"),
        ("assistant", "Reply 2"),
    ])
    stub = _StubSessionManager(session_id, store)

    result = _inspect_session_history(
        _call({"filter_role": "user", "nth": 2}),
        stub,
        store,
    )

    assert result.success
    assert "User message #2:" in result.output_text
    assert "Second prompt" in result.output_text
    assert "First prompt" not in result.output_text


def test_inspect_session_history_nth_out_of_bounds_returns_failure(tmp_path: Path) -> None:
    session_id = "sess_003"
    store = ChatMemoryStore(tmp_path / "chats")
    _populate(store, session_id, [("user", "Only prompt")])
    stub = _StubSessionManager(session_id, store)

    result = _inspect_session_history(
        _call({"filter_role": "user", "nth": 3}),
        stub,
        store,
    )

    assert not result.success
    assert "3" in result.error
    assert "1" in result.error  # "only 1 user messages"


# ---------------------------------------------------------------------------
# Tests: listing in chronological order
# ---------------------------------------------------------------------------


def test_inspect_session_history_all_user_prompts_in_order(tmp_path: Path) -> None:
    session_id = "sess_004"
    store = ChatMemoryStore(tmp_path / "chats")
    prompts = ["Alpha", "Beta", "Gamma", "Delta"]
    pairs: list[tuple[str, str]] = []
    for prompt in prompts:
        pairs.append(("user", prompt))
        pairs.append(("assistant", f"Answer to {prompt}"))
    _populate(store, session_id, pairs)
    stub = _StubSessionManager(session_id, store)

    result = _inspect_session_history(
        _call({"filter_role": "user"}),
        stub,
        store,
    )

    assert result.success
    text = result.output_text
    for prompt in prompts:
        assert prompt in text
    # Chronological: index markers appear in order
    assert text.index("[1]") < text.index("[2]") < text.index("[3]") < text.index("[4]")
    assert text.index("Alpha") < text.index("Beta") < text.index("Gamma") < text.index("Delta")


def test_inspect_session_history_listing_excludes_non_matching_roles(tmp_path: Path) -> None:
    session_id = "sess_005"
    store = ChatMemoryStore(tmp_path / "chats")
    _populate(store, session_id, [
        ("user", "User Q"),
        ("tool", "Tool output"),
        ("assistant", "Asst A"),
    ])
    stub = _StubSessionManager(session_id, store)

    result = _inspect_session_history(
        _call({"filter_role": "user"}),
        stub,
        store,
    )

    assert result.success
    assert "User Q" in result.output_text
    assert "Tool output" not in result.output_text
    assert "Asst A" not in result.output_text


# ---------------------------------------------------------------------------
# Tests: counts by role
# ---------------------------------------------------------------------------


def test_inspect_session_history_counts_by_role_are_exact(tmp_path: Path) -> None:
    session_id = "sess_006"
    store = ChatMemoryStore(tmp_path / "chats")
    _populate(store, session_id, [
        ("user", "U1"),
        ("tool", "T1"),
        ("assistant", "A1"),
        ("user", "U2"),
        ("assistant", "A2"),
        ("tool", "T2"),
        ("tool", "T3"),
    ])
    stub = _StubSessionManager(session_id, store)

    result = _inspect_session_history(_call({}), stub, store)

    assert result.success
    assert "7 messages total" in result.output_text
    assert "2 user" in result.output_text
    assert "2 assistant" in result.output_text
    assert "3 tool" in result.output_text


def test_inspect_session_history_empty_session_returns_zero_counts(tmp_path: Path) -> None:
    session_id = "sess_007"
    store = ChatMemoryStore(tmp_path / "chats")
    stub = _StubSessionManager(session_id, store)

    result = _inspect_session_history(_call({}), stub, store)

    assert result.success
    assert "0 messages total" in result.output_text
    assert "0 user" in result.output_text


# ---------------------------------------------------------------------------
# Tests: works beyond context window (long session with tool calls)
# ---------------------------------------------------------------------------


def test_inspect_session_history_works_after_long_session_with_tool_calls(
    tmp_path: Path,
) -> None:
    """Simulate a session much longer than the context injection window (20 msgs).

    The tool reads from the JSONL ChatMemoryStore, not from the context window,
    so all turns are accessible regardless of conversation length.
    """
    session_id = "sess_008"
    store = ChatMemoryStore(tmp_path / "chats")
    pairs: list[tuple[str, str]] = []
    for i in range(40):
        pairs.append(("user", f"Prompt number {i + 1}"))
        pairs.append(("assistant", f"Answer {i + 1}"))
    # Intersperse a tool message
    pairs.insert(5, ("tool", "Search result"))
    _populate(store, session_id, pairs)
    stub = _StubSessionManager(session_id, store)

    # First prompt — far outside what would be in context
    first_result = _inspect_session_history(
        _call({"filter_role": "user", "nth": 1}),
        stub,
        store,
    )
    assert first_result.success
    assert "Prompt number 1" in first_result.output_text

    # Second prompt
    second_result = _inspect_session_history(
        _call({"filter_role": "user", "nth": 2}),
        stub,
        store,
    )
    assert second_result.success
    assert "Prompt number 2" in second_result.output_text

    # Listing — all 40 user prompts reachable (default limit 50)
    list_result = _inspect_session_history(
        _call({"filter_role": "user", "limit": 50}),
        stub,
        store,
    )
    assert list_result.success
    assert "40 user" in list_result.output_text
    assert "Prompt number 1" in list_result.output_text
    assert "Prompt number 40" in list_result.output_text


# ---------------------------------------------------------------------------
# Tests: no active session
# ---------------------------------------------------------------------------


def test_inspect_session_history_no_active_session_returns_failure(tmp_path: Path) -> None:
    class _NoSessionManager:
        current_session_id = None
        chat_store = ChatMemoryStore(tmp_path / "chats")

    store = ChatMemoryStore(tmp_path / "chats")
    result = _inspect_session_history(_call({}), _NoSessionManager(), store)

    assert not result.success
    assert "No active session" in result.error


# ---------------------------------------------------------------------------
# Tests: registry registration
# ---------------------------------------------------------------------------


def test_register_session_tools_adds_inspect_session_history_to_registry(
    tmp_path: Path,
) -> None:
    session_id = "sess_reg"
    store = ChatMemoryStore(tmp_path / "chats")
    stub = _StubSessionManager(session_id, store)
    registry = ToolRegistry()

    register_session_tools(registry, session_manager=stub)

    registered = registry.get(INSPECT_SESSION_HISTORY_DEFINITION.name)
    assert registered is not None
    assert registered.definition.name == "inspect_session_history"


def test_register_session_tools_handler_is_callable(tmp_path: Path) -> None:
    session_id = "sess_callable"
    store = ChatMemoryStore(tmp_path / "chats")
    _populate(store, session_id, [("user", "Registered call")])
    stub = _StubSessionManager(session_id, store)
    registry = ToolRegistry()
    register_session_tools(registry, session_manager=stub)

    registered = registry.get(INSPECT_SESSION_HISTORY_DEFINITION.name)
    assert registered is not None

    result = registered.handler(_call({"filter_role": "user", "nth": 1}))
    assert result.success
    assert "Registered call" in result.output_text


def test_register_session_tools_skipped_when_chat_store_is_none() -> None:
    """If session_manager has no chat_store, registration is skipped gracefully."""

    class _NoStoreManager:
        current_session_id = "sess_noop"
        chat_store = None

    registry = ToolRegistry()
    register_session_tools(registry, session_manager=_NoStoreManager())

    assert registry.get(INSPECT_SESSION_HISTORY_DEFINITION.name) is None


# ---------------------------------------------------------------------------
# Tests: integration with create_default_tool_registry
# ---------------------------------------------------------------------------


def test_create_default_tool_registry_without_session_manager_excludes_session_tool(
    make_temp_project,
) -> None:
    from unclaw.core.executor import create_default_tool_registry
    from unclaw.settings import load_settings

    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    registry = create_default_tool_registry(settings)

    assert registry.get("inspect_session_history") is None


def test_create_default_tool_registry_with_session_manager_includes_session_tool(
    make_temp_project,
) -> None:
    from unclaw.core.executor import create_default_tool_registry
    from unclaw.core.session_manager import SessionManager
    from unclaw.settings import load_settings

    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        registry = create_default_tool_registry(settings, session_manager=session_manager)
        assert registry.get("inspect_session_history") is not None
    finally:
        session_manager.close()


# ---------------------------------------------------------------------------
# Tests: capability context guidance
# ---------------------------------------------------------------------------


def test_capability_context_includes_session_history_guidance_when_available() -> None:
    from unclaw.core.capabilities import (
        RuntimeCapabilitySummary,
        build_runtime_capability_context,
    )

    summary = RuntimeCapabilitySummary(
        available_builtin_tool_names=("inspect_session_history",),
        local_file_read_available=False,
        local_directory_listing_available=False,
        url_fetch_available=False,
        web_search_available=False,
        system_info_available=False,
        memory_summary_available=False,
        model_can_call_tools=True,
        session_history_recall_available=True,
    )
    context = build_runtime_capability_context(summary)

    assert "inspect_session_history" in context
    assert "prior prompts" in context
    assert "Do not guess from memory" in context


def test_capability_context_omits_session_history_guidance_when_unavailable() -> None:
    from unclaw.core.capabilities import (
        RuntimeCapabilitySummary,
        build_runtime_capability_context,
    )

    summary = RuntimeCapabilitySummary(
        available_builtin_tool_names=(),
        local_file_read_available=False,
        local_directory_listing_available=False,
        url_fetch_available=False,
        web_search_available=False,
        system_info_available=False,
        memory_summary_available=False,
        model_can_call_tools=True,
        session_history_recall_available=False,
    )
    context = build_runtime_capability_context(summary)

    assert "inspect_session_history" not in context
