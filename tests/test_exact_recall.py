"""Tests verifying that the deterministic exact-recall shortcut has been removed
and that session recall now flows through the normal model/tool path.

A. No deterministic recall shortcut
   - exact_recall module does not exist
   - runtime has no _try_exact_recall_shortcut function
   - run_user_turn always reaches the model for all query types

B. inspect_session_history tool (agentic recall)
   - tool is registered when session_manager has a chat_store
   - tool returns correct nth user message
   - tool returns correct counts
   - tool returns ordered list
   - tool handles out-of-range nth gracefully
   - tool is thread-safe (callable from worker threads)

C. Lazy JSONL backfill from SQLite
   - sessions that have SQLite messages but no JSONL are backfilled on switch
   - inspect_session_history reads backfilled data correctly

D. Runtime: model is always called (no deterministic bypass)
   - recall-style queries ("première question", "first prompt") reach the model
   - multilingual queries (FR / EN / ES style) reach the model
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.memory.chat_store import ChatMemoryRecord, ChatMemoryStore
from unclaw.tools.contracts import ToolCall
from unclaw.tools.session_tools import (
    INSPECT_SESSION_HISTORY_DEFINITION,
    register_session_tools,
)
from unclaw.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# A. No deterministic recall shortcut
# ---------------------------------------------------------------------------

class TestNoExactRecallShortcut:
    """The deterministic exact_recall module and shortcut must not exist."""

    def test_exact_recall_module_absent(self) -> None:
        """unclaw.core.exact_recall must not be importable."""
        # Remove any stale cached import first.
        sys.modules.pop("unclaw.core.exact_recall", None)
        with pytest.raises(ImportError):
            importlib.import_module("unclaw.core.exact_recall")

    def test_runtime_has_no_try_exact_recall_shortcut(self) -> None:
        """runtime.py must not expose _try_exact_recall_shortcut."""
        import unclaw.core.runtime as runtime_module
        assert not hasattr(runtime_module, "_try_exact_recall_shortcut"), (
            "_try_exact_recall_shortcut still present in runtime — "
            "the deterministic shortcut was not removed"
        )

    def test_runtime_model_called_for_recall_style_query(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_temp_project,
    ) -> None:
        """For any recall-style query the model must be called (no bypass)."""
        from types import SimpleNamespace as NS

        from unclaw.core.command_handler import CommandHandler
        from unclaw.core.runtime import run_user_turn
        from unclaw.core.session_manager import SessionManager
        from unclaw.llm.base import LLMResponse
        from unclaw.logs.event_bus import EventBus
        from unclaw.logs.tracer import Tracer
        from unclaw.schemas.chat import MessageRole
        from unclaw.settings import load_settings
        from unclaw.tools.registry import ToolRegistry

        model_call_count = 0

        class _CountingProvider:
            provider_name = "ollama"

            def __init__(self, *, base_url: str = "", default_timeout_seconds: float = 60.0) -> None:
                pass

            def chat(self, profile, messages, **kwargs) -> LLMResponse:  # type: ignore[no-untyped-def]
                nonlocal model_call_count
                model_call_count += 1
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="Votre première question était : salut.",
                    created_at="2026-03-18T10:00:00Z",
                    finish_reason="stop",
                )

        monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", _CountingProvider)

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)
        tracer = Tracer(event_bus=EventBus(), event_repository=session_manager.event_repository)
        command_handler = CommandHandler(
            settings=settings,
            session_manager=session_manager,
            memory_manager=NS(),
        )

        try:
            session = session_manager.ensure_current_session()
            session_manager.add_message(MessageRole.USER, "salut", session_id=session.id)
            session_manager.add_message(MessageRole.ASSISTANT, "Bonjour!", session_id=session.id)

            # These recall-style queries must ALL reach the model now.
            recall_queries = [
                "première question ?",
                "first prompt ?",
                "¿cuál fue mi primera pregunta?",
                "et la 3eme ?",
                "combien de messages ?",
                "liste mes prompts dans l'ordre",
            ]
            for query in recall_queries:
                session_manager.add_message(MessageRole.USER, query, session_id=session.id)
                run_user_turn(
                    session_manager=session_manager,
                    command_handler=command_handler,
                    user_input=query,
                    tracer=tracer,
                    tool_registry=ToolRegistry(),
                )
            assert model_call_count == len(recall_queries), (
                f"Expected model to be called {len(recall_queries)} times, "
                f"got {model_call_count}. Deterministic bypass still active?"
            )
        finally:
            session_manager.close()


# ---------------------------------------------------------------------------
# B. inspect_session_history tool
# ---------------------------------------------------------------------------

def _records(*pairs: tuple[str, str]) -> list[ChatMemoryRecord]:
    return [
        ChatMemoryRecord(
            seq=i + 1,
            role=role,
            content=content,
            created_at=f"2026-01-01T00:00:{i:02d}Z",
        )
        for i, (role, content) in enumerate(pairs)
    ]


class TestInspectSessionHistoryTool:
    def _make_store_and_registry(self, tmp_path: Path, session_id: str) -> tuple[ChatMemoryStore, ToolRegistry]:
        store = ChatMemoryStore(tmp_path / "chats")
        stub_sm = SimpleNamespace(
            chat_store=store,
            current_session_id=session_id,
        )
        registry = ToolRegistry()
        register_session_tools(registry, session_manager=stub_sm)  # type: ignore[arg-type]
        return store, registry

    def _dispatch(self, registry: ToolRegistry, **kwargs) -> str:
        from unclaw.tools.dispatcher import ToolDispatcher
        result = ToolDispatcher(registry).dispatch(
            ToolCall(tool_name=INSPECT_SESSION_HISTORY_DEFINITION.name, arguments=kwargs)
        )
        return result.output_text

    def test_tool_registered(self, tmp_path: Path) -> None:
        store, registry = self._make_store_and_registry(tmp_path, "sess1")
        names = {t.name for t in registry.list_tools()}
        assert INSPECT_SESSION_HISTORY_DEFINITION.name in names

    def test_nth_user_message(self, tmp_path: Path) -> None:
        store, registry = self._make_store_and_registry(tmp_path, "sess2")
        store.append_message(session_id="sess2", role="user", content="salut", created_at="2026-01-01T00:00:00Z")
        store.append_message(session_id="sess2", role="assistant", content="Bonjour!", created_at="2026-01-01T00:00:01Z")
        store.append_message(session_id="sess2", role="user", content="read file.txt", created_at="2026-01-01T00:00:02Z")

        output = self._dispatch(registry, filter_role="user", nth=1)
        assert "salut" in output
        assert "read file.txt" not in output

        output2 = self._dispatch(registry, filter_role="user", nth=2)
        assert "read file.txt" in output2

    def test_count_in_header(self, tmp_path: Path) -> None:
        store, registry = self._make_store_and_registry(tmp_path, "sess3")
        for i in range(5):
            store.append_message(session_id="sess3", role="user", content=f"prompt {i}", created_at=f"2026-01-01T00:00:0{i}Z")
            store.append_message(session_id="sess3", role="assistant", content=f"reply {i}", created_at=f"2026-01-01T00:01:0{i}Z")

        output = self._dispatch(registry)
        assert "10 messages total" in output  # 5 user + 5 assistant
        assert "5 user" in output

    def test_ordered_list(self, tmp_path: Path) -> None:
        store, registry = self._make_store_and_registry(tmp_path, "sess4")
        prompts = ["alpha", "beta", "gamma"]
        for p in prompts:
            store.append_message(session_id="sess4", role="user", content=p, created_at="2026-01-01T00:00:00Z")

        output = self._dispatch(registry, filter_role="user")
        assert output.index("alpha") < output.index("beta") < output.index("gamma")

    def test_out_of_range_nth(self, tmp_path: Path) -> None:
        store, registry = self._make_store_and_registry(tmp_path, "sess5")
        store.append_message(session_id="sess5", role="user", content="only one", created_at="2026-01-01T00:00:00Z")
        from unclaw.tools.dispatcher import ToolDispatcher
        result = ToolDispatcher(registry).dispatch(
            ToolCall(
                tool_name=INSPECT_SESSION_HISTORY_DEFINITION.name,
                arguments={"filter_role": "user", "nth": 99},
            )
        )
        assert not result.success
        assert "99" in (result.error or "")

    def test_empty_session(self, tmp_path: Path) -> None:
        _store, registry = self._make_store_and_registry(tmp_path, "empty_sess")
        output = self._dispatch(registry, filter_role="user")
        assert "0 messages total" in output

    def test_worker_thread_safe(self, tmp_path: Path) -> None:
        """Tool output is identical whether called from main thread or worker thread."""
        import threading

        store, registry = self._make_store_and_registry(tmp_path, "thread_sess")
        store.append_message(session_id="thread_sess", role="user", content="from main", created_at="2026-01-01T00:00:00Z")

        results: list[str] = []
        def _worker() -> None:
            results.append(self._dispatch(registry, filter_role="user"))

        t = threading.Thread(target=_worker)
        t.start()
        t.join(timeout=5)
        assert len(results) == 1
        assert "from main" in results[0]

    def test_unicode_content(self, tmp_path: Path) -> None:
        store, registry = self._make_store_and_registry(tmp_path, "uni_sess")
        store.append_message(
            session_id="uni_sess",
            role="user",
            content="c'est quoi ma première question ?",
            created_at="2026-01-01T00:00:00Z",
        )
        output = self._dispatch(registry, filter_role="user", nth=1)
        assert "première" in output


# ---------------------------------------------------------------------------
# C. Lazy JSONL backfill from SQLite
# ---------------------------------------------------------------------------

class TestLazyJsonlBackfill:
    """When a session has SQLite messages but no JSONL, switch_session triggers backfill."""

    def test_backfill_on_switch_session(self, make_temp_project) -> None:
        from unclaw.core.session_manager import SessionManager
        from unclaw.schemas.chat import MessageRole
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)

        try:
            session = session_manager.ensure_current_session()
            session_manager.add_message(MessageRole.USER, "hello backfill", session_id=session.id)
            session_manager.add_message(MessageRole.ASSISTANT, "Hi!", session_id=session.id)

            # Remove the JSONL to simulate a pre-JSONL session.
            assert session_manager.chat_store is not None
            jsonl_path = session_manager.chat_store._session_path(session.id)  # type: ignore[union-attr]
            if jsonl_path.exists():
                jsonl_path.unlink()

            # Trigger backfill via switch_session.
            session_manager.switch_session(session.id)

            # JSONL must now exist and contain the messages.
            records = session_manager.chat_store.read_messages(session.id)  # type: ignore[union-attr]
            contents = [r.content for r in records]
            assert "hello backfill" in contents
            assert "Hi!" in contents
        finally:
            session_manager.close()

    def test_backfill_does_not_overwrite_existing_jsonl(self, make_temp_project) -> None:
        from unclaw.core.session_manager import SessionManager
        from unclaw.schemas.chat import MessageRole
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)

        try:
            session = session_manager.ensure_current_session()
            session_manager.add_message(MessageRole.USER, "original", session_id=session.id)

            # Read once to confirm content.
            records_before = session_manager.chat_store.read_messages(session.id)  # type: ignore[union-attr]
            assert any(r.content == "original" for r in records_before)
            count_before = len(records_before)

            # Calling backfill again must not duplicate records.
            session_manager.switch_session(session.id)
            records_after = session_manager.chat_store.read_messages(session.id)  # type: ignore[union-attr]
            assert len(records_after) == count_before
        finally:
            session_manager.close()

    def test_inspect_tool_works_on_backfilled_session(self, make_temp_project) -> None:
        """After backfill, inspect_session_history returns complete history."""
        from unclaw.core.session_manager import SessionManager
        from unclaw.schemas.chat import MessageRole
        from unclaw.settings import load_settings
        from unclaw.tools.dispatcher import ToolDispatcher

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)

        try:
            session = session_manager.ensure_current_session()
            session_manager.add_message(MessageRole.USER, "pre-jsonl prompt", session_id=session.id)
            session_manager.add_message(MessageRole.ASSISTANT, "pre-jsonl reply", session_id=session.id)

            # Remove JSONL, trigger backfill.
            jsonl_path = session_manager.chat_store._session_path(session.id)  # type: ignore[union-attr]
            if jsonl_path.exists():
                jsonl_path.unlink()
            session_manager.switch_session(session.id)

            # Register and call inspect_session_history.
            registry = ToolRegistry()
            register_session_tools(registry, session_manager=session_manager)
            dispatcher = ToolDispatcher(registry)
            result = dispatcher.dispatch(
                ToolCall(
                    tool_name=INSPECT_SESSION_HISTORY_DEFINITION.name,
                    arguments={"filter_role": "user"},
                )
            )
            assert result.success
            assert "pre-jsonl prompt" in result.output_text
        finally:
            session_manager.close()


# ---------------------------------------------------------------------------
# D. Multi-session isolation
# ---------------------------------------------------------------------------

class TestMultiSessionIsolation:
    def test_sessions_isolated(self, tmp_path: Path) -> None:
        store_a = ChatMemoryStore(tmp_path / "chats")
        store_b = ChatMemoryStore(tmp_path / "chats")

        store_a.append_message(session_id="sess_a", role="user", content="only in A", created_at="2026-01-01T00:00:00Z")
        store_b.append_message(session_id="sess_b", role="user", content="only in B", created_at="2026-01-01T00:00:00Z")

        records_a = store_a.read_messages("sess_a")
        records_b = store_b.read_messages("sess_b")

        assert any(r.content == "only in A" for r in records_a)
        assert not any(r.content == "only in A" for r in records_b)
        assert any(r.content == "only in B" for r in records_b)
        assert not any(r.content == "only in B" for r in records_a)
