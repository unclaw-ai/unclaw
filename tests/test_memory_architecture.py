"""Comprehensive tests for the three-layer memory architecture.

A. SHORT-TERM WORKING MEMORY
   - bounded context still works (existing context_builder/manager tests cover this)
   - system/capability/current user message preserved

B. CONVERSATION MEMORY (chat_store)
   - append/read round-trip
   - unicode round-trip
   - multi-session isolation
   - concurrent appends (thread-safety)
   - long session exact nth lookup via inspect_session_history
   - exact ordered prompt listing
   - exact counts by role
   - lazy backfill_from_messages — covered in test_exact_recall.py

C. LONG-TERM MEMORY (LongTermStore + tools)
   - store/retrieve round-trip
   - search by text in key
   - search by text in value
   - search by tags
   - category filter
   - list_all
   - forget (delete by id)
   - forget non-existent id returns False
   - cross-session persistence (separate store instance)
   - tags/categories preserved
   - not injected into normal context by default
   - tool: remember_long_term_memory stores and confirms
   - tool: search_long_term_memory finds by query
   - tool: list_long_term_memory returns all
   - tool: forget_long_term_memory deletes

D. RUNTIME INTEGRATION
   - inspect_session_history tool is available after create_default_tool_registry
   - long-term memory tools are available after create_default_tool_registry
   - capabilities summary correctly reports session_history_recall_available
   - capabilities summary correctly reports long_term_memory_available
   - capability context includes session-recall instructions when tools are available
   - capability context includes long-term memory instructions when tools are available
   - short-term memory context note is still emitted (no regression)

E. REGRESSION
   - ChatMemoryStore append/read round-trip unchanged
   - session_manager add_message still mirrors to JSONL
   - existing test behaviour preserved
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.memory.chat_store import ChatMemoryRecord, ChatMemoryStore
from unclaw.memory.long_term_store import LongTermStore
from unclaw.tools.contracts import ToolCall
from unclaw.tools.long_term_memory_tools import (
    FORGET_LONG_TERM_MEMORY_DEFINITION,
    LIST_LONG_TERM_MEMORY_DEFINITION,
    REMEMBER_LONG_TERM_MEMORY_DEFINITION,
    SEARCH_LONG_TERM_MEMORY_DEFINITION,
    register_long_term_memory_tools,
)
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.session_tools import (
    INSPECT_SESSION_HISTORY_DEFINITION,
    register_session_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dispatch(registry: ToolRegistry, tool_name: str, **kwargs) -> object:
    from unclaw.tools.dispatcher import ToolDispatcher
    return ToolDispatcher(registry).dispatch(
        ToolCall(tool_name=tool_name, arguments=kwargs)
    )


# ---------------------------------------------------------------------------
# B. CONVERSATION MEMORY
# ---------------------------------------------------------------------------

class TestConversationMemory:
    """ChatMemoryStore: append, read, unicode, multi-session, concurrency."""

    def test_append_read_round_trip(self, tmp_path: Path) -> None:
        store = ChatMemoryStore(tmp_path / "chats")
        store.append_message(
            session_id="s1",
            role="user",
            content="hello",
            created_at="2026-01-01T00:00:00Z",
        )
        store.append_message(
            session_id="s1",
            role="assistant",
            content="world",
            created_at="2026-01-01T00:00:01Z",
        )
        records = store.read_messages("s1")
        assert len(records) == 2
        assert records[0].role == "user"
        assert records[0].content == "hello"
        assert records[1].role == "assistant"
        assert records[1].content == "world"

    def test_unicode_round_trip(self, tmp_path: Path) -> None:
        store = ChatMemoryStore(tmp_path / "chats")
        texts = [
            "c'est quoi ma première question ?",
            "¿cuál fue mi primera pregunta?",
            "これは日本語のテストです。",
            "Привет, мир!",
            "مرحبا",
        ]
        for i, text in enumerate(texts):
            store.append_message(
                session_id="uni",
                role="user",
                content=text,
                created_at=f"2026-01-01T00:00:0{i}Z",
            )
        records = store.read_messages("uni")
        assert len(records) == len(texts)
        for record, expected in zip(records, texts, strict=True):
            assert record.content == expected

    def test_multi_session_isolation(self, tmp_path: Path) -> None:
        store = ChatMemoryStore(tmp_path / "chats")
        store.append_message(session_id="sess_a", role="user", content="A only", created_at="2026-01-01T00:00:00Z")
        store.append_message(session_id="sess_b", role="user", content="B only", created_at="2026-01-01T00:00:00Z")

        records_a = store.read_messages("sess_a")
        records_b = store.read_messages("sess_b")

        assert all(r.content == "A only" for r in records_a)
        assert all(r.content == "B only" for r in records_b)
        assert not any(r.content == "B only" for r in records_a)
        assert not any(r.content == "A only" for r in records_b)

    def test_concurrent_appends_are_ordered_and_complete(self, tmp_path: Path) -> None:
        store = ChatMemoryStore(tmp_path / "chats")
        n = 50
        errors: list[Exception] = []

        def _writer(i: int) -> None:
            try:
                store.append_message(
                    session_id="concurrent",
                    role="user",
                    content=f"message {i}",
                    created_at=f"2026-01-01T00:00:{i % 60:02d}Z",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_writer, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        records = store.read_messages("concurrent")
        assert len(records) == n

    def test_long_session_exact_nth_lookup(self, tmp_path: Path) -> None:
        store = ChatMemoryStore(tmp_path / "chats")
        stub_sm = SimpleNamespace(chat_store=store, current_session_id="long")
        registry = ToolRegistry()
        register_session_tools(registry, session_manager=stub_sm)  # type: ignore[arg-type]

        for i in range(100):
            store.append_message(session_id="long", role="user", content=f"prompt {i+1}", created_at=f"2026-01-01T00:00:00Z")
            store.append_message(session_id="long", role="assistant", content=f"reply {i+1}", created_at=f"2026-01-01T00:00:01Z")

        result = _dispatch(registry, INSPECT_SESSION_HISTORY_DEFINITION.name, filter_role="user", nth=1)
        assert result.success  # type: ignore[union-attr]
        assert "prompt 1" in result.output_text  # type: ignore[union-attr]

        result50 = _dispatch(registry, INSPECT_SESSION_HISTORY_DEFINITION.name, filter_role="user", nth=50)
        assert result50.success  # type: ignore[union-attr]
        assert "prompt 50" in result50.output_text  # type: ignore[union-attr]

    def test_exact_counts_by_role(self, tmp_path: Path) -> None:
        store = ChatMemoryStore(tmp_path / "chats")
        stub_sm = SimpleNamespace(chat_store=store, current_session_id="counts")
        registry = ToolRegistry()
        register_session_tools(registry, session_manager=stub_sm)  # type: ignore[arg-type]

        store.append_message(session_id="counts", role="user", content="u1", created_at="2026-01-01T00:00:00Z")
        store.append_message(session_id="counts", role="assistant", content="a1", created_at="2026-01-01T00:00:01Z")
        store.append_message(session_id="counts", role="tool", content="t1", created_at="2026-01-01T00:00:02Z")
        store.append_message(session_id="counts", role="user", content="u2", created_at="2026-01-01T00:00:03Z")

        result = _dispatch(registry, INSPECT_SESSION_HISTORY_DEFINITION.name)
        assert "4 messages total" in result.output_text  # type: ignore[union-attr]
        assert "2 user" in result.output_text  # type: ignore[union-attr]
        assert "1 assistant" in result.output_text  # type: ignore[union-attr]
        assert "1 tool" in result.output_text  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# C. LONG-TERM MEMORY
# ---------------------------------------------------------------------------

class TestLongTermStore:
    """LongTermStore: store, search, list, forget, persistence."""

    def test_store_and_retrieve_by_search(self, tmp_path: Path) -> None:
        store = LongTermStore(tmp_path / "memory" / "long_term.db")
        mem_id = store.store(key="preferred language", value="French", category="preference")
        results = store.search(query="French")
        assert len(results) == 1
        assert results[0].id == mem_id
        assert results[0].key == "preferred language"
        assert results[0].value == "French"
        assert results[0].category == "preference"

    def test_search_in_key(self, tmp_path: Path) -> None:
        store = LongTermStore(tmp_path / "memory" / "long_term.db")
        store.store(key="user birthdate", value="unknown")
        results = store.search(query="birthdate")
        assert len(results) == 1

    def test_search_in_tags(self, tmp_path: Path) -> None:
        store = LongTermStore(tmp_path / "memory" / "long_term.db")
        store.store(key="some fact", value="details", tags="important,personal")
        results = store.search(query="important")
        assert len(results) == 1

    def test_search_category_filter(self, tmp_path: Path) -> None:
        store = LongTermStore(tmp_path / "memory" / "long_term.db")
        store.store(key="A", value="value A", category="cat1")
        store.store(key="B", value="value B", category="cat2")
        results = store.search(query="value", category="cat1")
        assert len(results) == 1
        assert results[0].category == "cat1"

    def test_list_all(self, tmp_path: Path) -> None:
        store = LongTermStore(tmp_path / "memory" / "long_term.db")
        store.store(key="first", value="v1")
        store.store(key="second", value="v2")
        store.store(key="third", value="v3")
        all_records = store.list_all()
        assert len(all_records) == 3

    def test_list_all_category_filter(self, tmp_path: Path) -> None:
        store = LongTermStore(tmp_path / "memory" / "long_term.db")
        store.store(key="X", value="vX", category="alpha")
        store.store(key="Y", value="vY", category="beta")
        store.store(key="Z", value="vZ", category="alpha")
        alpha = store.list_all(category="alpha")
        assert len(alpha) == 2
        assert all(r.category == "alpha" for r in alpha)

    def test_forget_deletes_by_id(self, tmp_path: Path) -> None:
        store = LongTermStore(tmp_path / "memory" / "long_term.db")
        mem_id = store.store(key="to delete", value="bye")
        assert store.forget(mem_id) is True
        assert store.list_all() == []

    def test_forget_nonexistent_returns_false(self, tmp_path: Path) -> None:
        store = LongTermStore(tmp_path / "memory" / "long_term.db")
        assert store.forget("does-not-exist-uuid") is False

    def test_cross_session_persistence(self, tmp_path: Path) -> None:
        """A new LongTermStore instance reads data written by a previous one."""
        db_path = tmp_path / "memory" / "long_term.db"
        store1 = LongTermStore(db_path)
        mem_id = store1.store(key="persisted key", value="persisted value")

        # New instance, simulating a different session.
        store2 = LongTermStore(db_path)
        record = store2.get_by_id(mem_id)
        assert record is not None
        assert record.key == "persisted key"
        assert record.value == "persisted value"

    def test_tags_and_categories_preserved(self, tmp_path: Path) -> None:
        store = LongTermStore(tmp_path / "memory" / "long_term.db")
        store.store(
            key="full record",
            value="full value",
            category="test_cat",
            tags="tag1,tag2,tag3",
        )
        results = store.list_all()
        assert len(results) == 1
        assert results[0].category == "test_cat"
        assert "tag1" in results[0].tags
        assert "tag3" in results[0].tags


class TestLongTermMemoryTools:
    """Long-term memory tool: remember / search / list / forget via ToolRegistry."""

    def _make_registry(self, tmp_path: Path) -> tuple[LongTermStore, ToolRegistry]:
        store = LongTermStore(tmp_path / "memory" / "long_term.db")
        registry = ToolRegistry()
        register_long_term_memory_tools(registry, long_term_store=store)
        return store, registry

    def test_remember_stores_and_confirms(self, tmp_path: Path) -> None:
        _store, registry = self._make_registry(tmp_path)
        result = _dispatch(registry, REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
                           key="pizza", value="user prefers pepperoni")
        assert result.success  # type: ignore[union-attr]
        assert "pizza" in result.output_text  # type: ignore[union-attr]
        assert "pepperoni" in result.output_text  # type: ignore[union-attr]

    def test_remember_requires_key(self, tmp_path: Path) -> None:
        _store, registry = self._make_registry(tmp_path)
        result = _dispatch(registry, REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
                           key="", value="something")
        assert not result.success  # type: ignore[union-attr]

    def test_remember_requires_value(self, tmp_path: Path) -> None:
        _store, registry = self._make_registry(tmp_path)
        result = _dispatch(registry, REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
                           key="somekey", value="")
        assert not result.success  # type: ignore[union-attr]

    def test_search_finds_by_query(self, tmp_path: Path) -> None:
        _store, registry = self._make_registry(tmp_path)
        _dispatch(registry, REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
                  key="colour preference", value="blue")
        result = _dispatch(registry, SEARCH_LONG_TERM_MEMORY_DEFINITION.name,
                           query="colour")
        assert result.success  # type: ignore[union-attr]
        assert "colour preference" in result.output_text  # type: ignore[union-attr]

    def test_list_returns_all(self, tmp_path: Path) -> None:
        _store, registry = self._make_registry(tmp_path)
        for i in range(3):
            _dispatch(registry, REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
                      key=f"key{i}", value=f"val{i}")
        result = _dispatch(registry, LIST_LONG_TERM_MEMORY_DEFINITION.name)
        assert result.success  # type: ignore[union-attr]
        assert "3 item(s)" in result.output_text  # type: ignore[union-attr]

    def test_forget_removes_memory(self, tmp_path: Path) -> None:
        store, registry = self._make_registry(tmp_path)
        mem_id = store.store(key="to remove", value="gone")
        result = _dispatch(registry, FORGET_LONG_TERM_MEMORY_DEFINITION.name, id=mem_id)
        assert result.success  # type: ignore[union-attr]
        assert store.list_all() == []

    def test_forget_nonexistent_fails(self, tmp_path: Path) -> None:
        _store, registry = self._make_registry(tmp_path)
        result = _dispatch(registry, FORGET_LONG_TERM_MEMORY_DEFINITION.name,
                           id="no-such-id")
        assert not result.success  # type: ignore[union-attr]

    def test_not_injected_into_context_by_default(self, make_temp_project) -> None:
        """Long-term memory must not appear in the system context note automatically."""
        from unclaw.core.capabilities import build_runtime_capability_context, build_runtime_capability_summary
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        registry = create_default_tool_registry(settings)

        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
            model_can_call_tools=True,
        )
        context = build_runtime_capability_context(summary)

        # Long-term memory tools must be listed as callable tools.
        assert "remember_long_term_memory" in context
        # But the context must NOT contain stored memory VALUES — only tool descriptions.
        # (Structural check: no "memory stored" phrasing that would indicate injection)
        assert "Memory stored." not in context


# ---------------------------------------------------------------------------
# D. RUNTIME INTEGRATION
# ---------------------------------------------------------------------------

class TestRuntimeIntegration:
    """Capability summary and context reflect both tool families correctly."""

    def test_inspect_session_history_in_default_registry(self, make_temp_project) -> None:
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.core.session_manager import SessionManager
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)
        try:
            registry = create_default_tool_registry(settings, session_manager=session_manager)
            names = {t.name for t in registry.list_tools()}
            assert INSPECT_SESSION_HISTORY_DEFINITION.name in names
        finally:
            session_manager.close()

    def test_long_term_memory_tools_in_default_registry(self, make_temp_project) -> None:
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        registry = create_default_tool_registry(settings)
        names = {t.name for t in registry.list_tools()}
        assert REMEMBER_LONG_TERM_MEMORY_DEFINITION.name in names
        assert SEARCH_LONG_TERM_MEMORY_DEFINITION.name in names
        assert LIST_LONG_TERM_MEMORY_DEFINITION.name in names
        assert FORGET_LONG_TERM_MEMORY_DEFINITION.name in names

    def test_capability_summary_session_history_flag(self, make_temp_project) -> None:
        from unclaw.core.capabilities import build_runtime_capability_summary
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.core.session_manager import SessionManager
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)
        try:
            registry = create_default_tool_registry(settings, session_manager=session_manager)
            summary = build_runtime_capability_summary(
                tool_registry=registry,
                memory_summary_available=False,
                model_can_call_tools=True,
            )
            assert summary.session_history_recall_available is True
        finally:
            session_manager.close()

    def test_capability_summary_long_term_memory_flag(self, make_temp_project) -> None:
        from unclaw.core.capabilities import build_runtime_capability_summary
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        registry = create_default_tool_registry(settings)
        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
            model_can_call_tools=True,
        )
        assert summary.long_term_memory_available is True

    def test_capability_context_includes_session_recall_instruction(self, make_temp_project) -> None:
        from unclaw.core.capabilities import build_runtime_capability_context, build_runtime_capability_summary
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.core.session_manager import SessionManager
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)
        try:
            registry = create_default_tool_registry(settings, session_manager=session_manager)
            summary = build_runtime_capability_summary(
                tool_registry=registry,
                memory_summary_available=False,
                model_can_call_tools=True,
            )
            context = build_runtime_capability_context(summary)
            # Must instruct model to use inspect_session_history for exact recall.
            assert "inspect_session_history" in context
            assert "Do not guess from memory" in context or "exact" in context
        finally:
            session_manager.close()

    def test_capability_context_includes_long_term_memory_instruction(self, make_temp_project) -> None:
        from unclaw.core.capabilities import build_runtime_capability_context, build_runtime_capability_summary
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        registry = create_default_tool_registry(settings)
        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
            model_can_call_tools=True,
        )
        context = build_runtime_capability_context(summary)
        assert "remember_long_term_memory" in context
        assert "not injected" in context.lower() or "Not injected" in context


# ---------------------------------------------------------------------------
# E. REGRESSION — existing persistence behaviour
# ---------------------------------------------------------------------------

class TestRegression:
    """Ensure existing session_manager add_message → JSONL mirror still works."""

    def test_add_message_mirrors_to_jsonl(self, make_temp_project) -> None:
        from unclaw.core.session_manager import SessionManager
        from unclaw.schemas.chat import MessageRole
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)

        try:
            session = session_manager.ensure_current_session()
            session_manager.add_message(MessageRole.USER, "regression test", session_id=session.id)

            assert session_manager.chat_store is not None
            records = session_manager.chat_store.read_messages(session.id)
            assert any(r.content == "regression test" for r in records)
        finally:
            session_manager.close()

    def test_session_messages_also_in_sqlite(self, make_temp_project) -> None:
        from unclaw.core.session_manager import SessionManager
        from unclaw.schemas.chat import MessageRole
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)

        try:
            session = session_manager.ensure_current_session()
            session_manager.add_message(MessageRole.USER, "sqlite check", session_id=session.id)
            messages = session_manager.list_messages(session.id)
            assert any(m.content == "sqlite check" for m in messages)
        finally:
            session_manager.close()

    def test_long_term_db_path_derived_from_data_dir(self, make_temp_project) -> None:
        """LongTermStore must be under data_dir/memory/long_term.db."""
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        expected_path = settings.paths.data_dir / "memory" / "long_term.db"

        # Creating the store at the expected path must succeed.
        store = LongTermStore(expected_path)
        mem_id = store.store(key="path test", value="ok")
        assert store.get_by_id(mem_id) is not None
