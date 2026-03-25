from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.core.command_handler import CommandHandler
from unclaw.core.runtime import run_user_turn
from unclaw.core.runtime_support import _build_session_progress_ledger_context_note
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.file_tools import WRITE_TEXT_FILE_DEFINITION
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.system_tools import SYSTEM_INFO_DEFINITION

pytestmark = pytest.mark.integration


def _build_native_runtime(project_root, set_profile_tool_mode):
    settings = load_settings(project_root=project_root)
    set_profile_tool_mode(settings, "main", tool_mode="native")
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )
    return settings, session_manager, tracer, command_handler


def _tool_call_response(tool_name: str, arguments: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        provider="ollama",
        model_name="fake-model",
        content="",
        created_at="2026-03-25T09:00:00Z",
        finish_reason="stop",
        tool_calls=(ToolCall(tool_name=tool_name, arguments=arguments),),
        raw_payload={
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_name,
                            "arguments": arguments,
                        }
                    }
                ],
            }
        },
    )


def test_session_progress_ledger_persists_bounded_tail_across_session_manager_reloads(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    session_id: str | None = None
    ledger = ()

    try:
        session = session_manager.ensure_current_session()
        session_id = session.id
        session_manager.persist_session_progress_entry(
            session_id=session.id,
            status="active",
            step="fast_web_search",
            detail="tool succeeded",
        )
        session_manager.persist_session_progress_entry(
            session_id=session.id,
            status="blocked",
            step="search_web",
            detail="timed out after 15 seconds.",
        )
        session_manager.persist_session_progress_entry(
            session_id=session.id,
            status="active",
            step="write_text_file",
            detail="file write succeeded",
        )
        ledger = session_manager.persist_session_progress_entry(
            session_id=session.id,
            status="active",
            step="read_text_file",
            detail="tool succeeded",
        )

        assert len(ledger) == 3
        assert [(entry.status, entry.step, entry.detail) for entry in ledger] == [
            ("blocked", "search_web", "timed out after 15 seconds."),
            ("active", "write_text_file", "file write succeeded"),
            ("active", "read_text_file", "tool succeeded"),
        ]
        assert all(entry.updated_at for entry in ledger)
        note = _build_session_progress_ledger_context_note(
            session_manager=session_manager,
            session_id=session.id,
        )
        assert note is not None
        assert note.startswith("Session progress ledger:")
        assert "\n" not in note
        assert 'step="fast_web_search"' not in note
        event_types = [
            event.event_type
            for event in session_manager.event_repository.list_recent_events(
                session.id,
                limit=10,
            )
        ]
        assert "session.progress.ledger.updated" in event_types
    finally:
        session_manager.close()

    assert session_id is not None
    reloaded_manager = SessionManager.from_settings(settings)
    try:
        reloaded_session = reloaded_manager.ensure_current_session()
        assert reloaded_session.id == session_id
        assert reloaded_manager.get_session_progress_ledger(session_id) == ledger
    finally:
        reloaded_manager.close()


def test_runtime_progress_ledger_tracks_last_tool_fact_and_injects_compact_note(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    build_scripted_ollama_provider,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    output_path = project_root / "data" / "files" / "progress-note.txt"
    captured_messages: list[list[object]] = []
    tool_registry = ToolRegistry()

    def _write_tool(call: ToolCall) -> ToolResult:
        path = Path(str(call.arguments["path"]))
        content = str(call.arguments["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text=f"Wrote text file: {path}",
            payload={"path": str(path), "size": len(content)},
        )

    tool_registry.register(
        WRITE_TEXT_FILE_DEFINITION,
        _write_tool,
    )
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response(
            "write_text_file",
            {"path": str(output_path), "content": "progress"},
        ),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="I saved the note locally.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        ),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="You just wrote a local note file.",
            created_at="2026-03-25T09:00:02Z",
            finish_reason="stop",
        ),
        captured_messages=captured_messages,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    first_prompt = "Write a short local note file."
    second_prompt = "What progress do we have in this session?"

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            first_prompt,
            session_id=session.id,
        )
        first_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=first_prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )
        ledger = session_manager.get_session_progress_ledger(session.id)

        session_manager.add_message(
            MessageRole.USER,
            second_prompt,
            session_id=session.id,
        )
        second_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=second_prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert first_reply == "I saved the note locally."
        assert second_reply == "You just wrote a local note file."
        assert ledger
        assert len(ledger) == 1
        assert ledger[0].status == "active"
        assert ledger[0].step == "write_text_file"
        assert ledger[0].detail == "file write succeeded"
        assert output_path.read_text(encoding="utf-8") == "progress"
        assert fake_provider.call_count() == 3

        first_turn_system_messages = [
            message.content
            for message in captured_messages[0]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        assert all(
            not message.startswith("Session progress ledger:")
            for message in first_turn_system_messages
        )

        second_turn_system_messages = [
            message.content
            for message in captured_messages[2]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        progress_notes = [
            message
            for message in second_turn_system_messages
            if message.startswith("Session progress ledger:")
        ]
        assert len(progress_notes) == 1
        assert 'status="active"' in progress_notes[0]
        assert 'step="write_text_file"' in progress_notes[0]
        assert 'detail="file write succeeded"' in progress_notes[0]
        assert progress_notes[0].count("[") == 1
        assert "\n" not in progress_notes[0]
    finally:
        session_manager.close()


def test_runtime_progress_ledger_tracks_blocked_tool_timeout(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    build_scripted_ollama_provider,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    tool_registry = ToolRegistry()
    tool_registry.register(
        SYSTEM_INFO_DEFINITION,
        lambda call: ToolResult.failure(
            tool_name=call.tool_name,
            error="Tool 'system_info' timed out after 5 seconds.",
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response("system_info", {}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="Please try again.",
            created_at="2026-03-25T09:20:00Z",
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    prompt = "Check the machine and tell me what failed."

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            prompt,
            session_id=session.id,
        )
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        ledger = session_manager.get_session_progress_ledger(session.id)
        assert reply == (
            "The tool step timed out, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
        assert len(ledger) == 1
        assert ledger[0].status == "blocked"
        assert ledger[0].step == "system_info"
        assert ledger[0].detail == "timed out after 5 seconds."
    finally:
        session_manager.close()


def test_chat_only_sessions_remain_unchanged_without_progress_ledger(
    monkeypatch,
    make_temp_project,
    build_scripted_ollama_provider,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )
    captured_messages: list[list[object]] = []
    fake_provider = build_scripted_ollama_provider(
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="Just a normal chat reply.",
            created_at="2026-03-25T09:10:00Z",
            finish_reason="stop",
        ),
        captured_messages=captured_messages,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    prompt = "Hello there."

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            prompt,
            session_id=session.id,
        )
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=ToolRegistry(),
        )

        assert reply == "Just a normal chat reply."
        assert fake_provider.call_count() == 1
        assert session_manager.get_session_progress_ledger(session.id) == ()

        system_messages = [
            message.content
            for message in captured_messages[0]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        assert all(
            not message.startswith("Session progress ledger:")
            for message in system_messages
        )
        event_types = [
            event.event_type
            for event in session_manager.event_repository.list_recent_events(
                session.id,
                limit=10,
            )
        ]
        assert "session.progress.ledger.updated" not in event_types
    finally:
        session_manager.close()
