"""Tests for the ChatMemoryStore — thread-safe JSONL persistent chat memory.

Covers:
1. Basic write / read round-trip
2. Thread-safe concurrent appends from multiple threads
3. Reads from a worker thread — proves no SQLite thread-affinity error
4. Integration: SessionManager.from_settings writes to the store; tool reads
   from a worker thread — the full manual-target scenario exercised in process
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.memory.chat_store import ChatMemoryRecord, ChatMemoryStore


# ---------------------------------------------------------------------------
# Tests: basic JSONL round-trip
# ---------------------------------------------------------------------------


def test_chat_memory_store_append_and_read_round_trip(tmp_path: Path) -> None:
    store = ChatMemoryStore(tmp_path / "chats")
    session_id = "sess_rtrip"

    store.append_message(session_id=session_id, role="user", content="Hello", created_at="2026-01-01T00:00:00Z")
    store.append_message(session_id=session_id, role="assistant", content="Hi!", created_at="2026-01-01T00:00:01Z")

    records = store.read_messages(session_id)

    assert len(records) == 2
    assert records[0] == ChatMemoryRecord(seq=1, role="user", content="Hello", created_at="2026-01-01T00:00:00Z")
    assert records[1] == ChatMemoryRecord(seq=2, role="assistant", content="Hi!", created_at="2026-01-01T00:00:01Z")


def test_chat_memory_store_empty_session_returns_empty_list(tmp_path: Path) -> None:
    store = ChatMemoryStore(tmp_path / "chats")
    assert store.read_messages("sess_empty") == []


def test_chat_memory_store_creates_base_directory_lazily(tmp_path: Path) -> None:
    chats_dir = tmp_path / "deep" / "memory" / "chats"
    assert not chats_dir.exists()
    store = ChatMemoryStore(chats_dir)
    store.append_message(session_id="s", role="user", content="x", created_at="ts")
    assert chats_dir.exists()


def test_chat_memory_store_unicode_content_roundtrips_correctly(tmp_path: Path) -> None:
    store = ChatMemoryStore(tmp_path / "chats")
    session_id = "sess_unicode"
    content = "café / こんにちは / مرحبًا / 😀"

    store.append_message(session_id=session_id, role="user", content=content, created_at="ts")
    records = store.read_messages(session_id)

    assert len(records) == 1
    assert records[0].content == content


def test_chat_memory_store_multiple_sessions_are_independent(tmp_path: Path) -> None:
    store = ChatMemoryStore(tmp_path / "chats")

    store.append_message(session_id="s1", role="user", content="From s1", created_at="ts1")
    store.append_message(session_id="s2", role="user", content="From s2", created_at="ts2")

    s1 = store.read_messages("s1")
    s2 = store.read_messages("s2")

    assert len(s1) == 1
    assert s1[0].content == "From s1"
    assert len(s2) == 1
    assert s2[0].content == "From s2"


# ---------------------------------------------------------------------------
# Tests: thread-safe concurrent appends
# ---------------------------------------------------------------------------


def test_chat_memory_store_concurrent_appends_from_multiple_threads(tmp_path: Path) -> None:
    """Multiple threads appending to the same session must produce all entries."""
    store = ChatMemoryStore(tmp_path / "chats")
    session_id = "sess_concurrent"
    message_count = 20
    errors: list[BaseException] = []

    def _writer(prefix: str) -> None:
        try:
            for i in range(message_count):
                store.append_message(
                    session_id=session_id,
                    role="user",
                    content=f"{prefix}-{i}",
                    created_at=f"ts-{i}",
                )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(f"t{n}",)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    records = store.read_messages(session_id)
    assert len(records) == message_count * 4
    contents = {r.content for r in records}
    for n in range(4):
        for i in range(message_count):
            assert f"t{n}-{i}" in contents


# ---------------------------------------------------------------------------
# Tests: reads from a worker thread — no SQLite thread-affinity error
# ---------------------------------------------------------------------------


def test_chat_memory_store_read_from_worker_thread_succeeds(tmp_path: Path) -> None:
    """Reading from a worker thread must not raise any thread-affinity error.

    This directly validates requirement 6: no SQLite thread-affinity failure.
    ChatMemoryStore.read_messages() uses Path.read_text() (no SQLite) so it
    is safe to call from any thread.
    """
    store = ChatMemoryStore(tmp_path / "chats")
    session_id = "sess_worker"
    store.append_message(session_id=session_id, role="user", content="First prompt", created_at="ts")
    store.append_message(session_id=session_id, role="assistant", content="Reply", created_at="ts")

    result_holder: list[list[ChatMemoryRecord]] = []
    errors: list[BaseException] = []

    def _reader() -> None:
        try:
            result_holder.append(store.read_messages(session_id))
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=_reader)
    t.start()
    t.join(timeout=5)

    assert not errors
    assert len(result_holder) == 1
    records = result_holder[0]
    assert len(records) == 2
    assert records[0].role == "user"
    assert records[0].content == "First prompt"


# ---------------------------------------------------------------------------
# Tests: integration — SessionManager.from_settings writes, tool reads from thread
# ---------------------------------------------------------------------------


def test_session_manager_writes_to_chat_store_on_add_message(
    make_temp_project,
) -> None:
    """SessionManager.from_settings creates a ChatMemoryStore.
    add_message writes to both SQLite and the store.
    """
    from unclaw.core.session_manager import SessionManager
    from unclaw.settings import load_settings

    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message("user", "First question", session_id=session.id)
        session_manager.add_message("assistant", "First answer", session_id=session.id)
        session_manager.add_message("user", "Second question", session_id=session.id)

        assert session_manager.chat_store is not None
        records = session_manager.chat_store.read_messages(session.id)
        assert len(records) == 3
        assert records[0].role == "user"
        assert records[0].content == "First question"
        assert records[1].role == "assistant"
        assert records[1].content == "First answer"
        assert records[2].role == "user"
        assert records[2].content == "Second question"
    finally:
        session_manager.close()


def test_inspect_session_history_dispatched_from_worker_thread_no_sqlite_error(
    make_temp_project,
) -> None:
    """End-to-end: messages added via SessionManager, then inspect_session_history
    dispatched from a worker thread — must succeed without any SQLite thread error.

    This exercises the full manual-target scenario:
      - greeting, file-read, web-search, math question, definition question
      - then: 'what was my first prompt?', 'list all my prompts in order'

    The SQLite connection is owned by the main thread.
    The tool reads from the JSONL ChatMemoryStore — no SQLite involved.
    """
    from unclaw.core.executor import create_default_tool_registry
    from unclaw.core.session_manager import SessionManager
    from unclaw.settings import load_settings
    from unclaw.tools.contracts import ToolCall
    from unclaw.tools.dispatcher import ToolDispatcher

    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        session = session_manager.ensure_current_session()

        # Simulate the manual-target conversation flow
        for role, content in [
            ("user", "Hello, how are you?"),
            ("assistant", "I'm doing great, thanks!"),
            ("user", "/read notes/research.md"),
            ("tool", "File content: ..."),
            ("assistant", "Here is the file content."),
            ("user", "/search latest AI news"),
            ("tool", "Search results: ..."),
            ("assistant", "Based on the search results..."),
            ("user", "What is 7 times 8?"),
            ("assistant", "56"),
            ("user", "What is entropy?"),
            ("assistant", "Entropy is a measure of disorder..."),
        ]:
            session_manager.add_message(role, content, session_id=session.id)

        registry = create_default_tool_registry(settings, session_manager=session_manager)
        dispatcher = ToolDispatcher(registry)

        errors: list[BaseException] = []
        results: list = []

        def _dispatch_from_thread(arguments: dict) -> None:
            try:
                call = ToolCall(
                    tool_name="inspect_session_history",
                    arguments=arguments,
                )
                results.append(dispatcher.dispatch(call))
            except BaseException as exc:
                errors.append(exc)

        # Dispatch from worker threads — simulates tool execution in runtime
        threads = [
            threading.Thread(target=_dispatch_from_thread, args=({"filter_role": "user", "nth": 1},)),
            threading.Thread(target=_dispatch_from_thread, args=({"filter_role": "user", "nth": 2},)),
            threading.Thread(target=_dispatch_from_thread, args=({"filter_role": "user"},)),
            threading.Thread(target=_dispatch_from_thread, args=({},)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 4

        # All 4 dispatches must succeed — no SQLite thread-affinity error
        for r in results:
            assert r.success, f"Tool call failed: {r.error}"

        # Verify header counts are consistent across results
        # Conversation: 5 user, 5 assistant, 2 tool = 12 total
        for r in results:
            assert "12 messages total" in r.output_text
            assert "5 user" in r.output_text
            assert "5 assistant" in r.output_text
            assert "2 tool" in r.output_text

        # First user message (#1) = "Hello, how are you?"
        first_result = next(
            r for r in results
            if "User message #1:" in r.output_text
        )
        assert "Hello, how are you?" in first_result.output_text

        # Second user message (#2) = "/read notes/research.md"
        second_result = next(
            r for r in results
            if "User message #2:" in r.output_text
        )
        assert "/read notes/research.md" in second_result.output_text

        # Listing all user messages should include all 5 user prompts
        list_result = next(
            r for r in results
            if "Messages (user," in r.output_text
        )
        assert "Hello, how are you?" in list_result.output_text
        assert "/search latest AI news" in list_result.output_text
        assert "What is entropy?" in list_result.output_text

    finally:
        session_manager.close()
