from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.core.command_handler import CommandHandler
from unclaw.core.runtime import run_user_turn
from unclaw.core.runtime_support import _build_session_task_continuity_note
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.file_tools import READ_TEXT_FILE_DEFINITION, WRITE_TEXT_FILE_DEFINITION
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.system_tools import SYSTEM_INFO_DEFINITION
from unclaw.tools.web_tools import FAST_WEB_SEARCH_DEFINITION, SEARCH_WEB_DEFINITION

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
        session_manager.persist_session_goal_state(
            session_id=session.id,
            goal="Write a local progress note.",
            status="active",
            current_step="write_text_file",
        )
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
        note = _build_session_task_continuity_note(
            session_manager=session_manager,
            session_id=session.id,
        )
        assert note is not None
        assert note.startswith("Session task continuity:")
        assert "\n" not in note
        assert 'goal="Write a local progress note."' in note
        assert 'status="active"' in note
        assert 'current_step="write_text_file"' in note
        assert (
            'latest_progress=[status="active"; step="read_text_file"; '
            'detail="tool succeeded"]'
            in note
        )
        assert 'step="fast_web_search"' not in note
        assert "last_blocker=" not in note
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


def test_later_turn_injects_compact_active_task_continuity_note_without_extra_model_call(
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
            content="You are still writing the local note file.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        ),
        captured_messages=captured_messages,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    persisted_goal = "Write a short local note file."
    second_prompt = "Where does this task stand?"

    try:
        session = session_manager.ensure_current_session()
        session_manager.persist_session_goal_state(
            session_id=session.id,
            goal=persisted_goal,
            status="active",
            current_step="write_text_file",
        )
        session_manager.persist_session_progress_entry(
            session_id=session.id,
            status="active",
            step="write_text_file",
            detail="file write succeeded",
        )
        session_manager.add_message(
            MessageRole.USER,
            second_prompt,
            session_id=session.id,
        )
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=second_prompt,
            tracer=tracer,
            tool_registry=ToolRegistry(),
        )

        assert reply == "You are still writing the local note file."
        assert fake_provider.call_count() == 1

        system_messages = [
            message.content
            for message in captured_messages[0]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        continuity_notes = [
            message
            for message in system_messages
            if message.startswith("Session task continuity:")
        ]
        assert len(continuity_notes) == 1
        assert all(
            not message.startswith("Session goal state:")
            and not message.startswith("Session progress ledger:")
            for message in system_messages
        )
        assert 'goal="Write a short local note file."' in continuity_notes[0]
        assert 'status="active"' in continuity_notes[0]
        assert 'current_step="write_text_file"' in continuity_notes[0]
        assert (
            'latest_progress=[status="active"; step="write_text_file"; '
            'detail="file write succeeded"]'
            in continuity_notes[0]
        )
        assert "last_blocker=" not in continuity_notes[0]
        assert "\n" not in continuity_notes[0]
    finally:
        session_manager.close()


def test_one_shot_turn_without_goal_state_does_not_inject_task_continuity_note_later(
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
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="OS: ExampleOS 1.0\nLocal datetime: 2026-03-25 10:00:00 CET",
        ),
    )
    captured_messages: list[list[object]] = []
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response("system_info", {}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="The machine is running ExampleOS 1.0.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        ),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="There is no persisted task progress for this session.",
            created_at="2026-03-25T09:00:02Z",
            finish_reason="stop",
        ),
        captured_messages=captured_messages,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    first_prompt = "quel est mon OS ?"
    second_prompt = "Where does this task stand?"

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
        assert first_reply == "The machine is running ExampleOS 1.0."
        assert session_manager.get_session_goal_state(session.id) is None
        assert session_manager.get_session_progress_ledger(session.id) == ()

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

        assert second_reply == "There is no persisted task progress for this session."
        assert fake_provider.call_count() == 3
        second_turn_system_messages = [
            message.content
            for message in captured_messages[2]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        assert all(
            not message.startswith("Session task continuity:")
            for message in second_turn_system_messages
        )
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


def test_blocked_multi_tool_turn_still_creates_goal_state_and_progress_ledger(
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
    missing_path = project_root / "data" / "files" / "missing.txt"
    tool_registry = ToolRegistry()
    tool_registry.register(
        SYSTEM_INFO_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="OS: ExampleOS 1.0",
        ),
    )
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: ToolResult.failure(
            tool_name=call.tool_name,
            error=f"File not found: {call.arguments['path']}",
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response("system_info", {}),
        _tool_call_response("read_text_file", {"path": str(missing_path)}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="Please try again.",
            created_at="2026-03-25T09:20:00Z",
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    prompt = "Inspect the machine, then read the missing note file."

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

        goal_state = session_manager.get_session_goal_state(session.id)
        ledger = session_manager.get_session_progress_ledger(session.id)
        assert reply == "Please try again."
        assert fake_provider.call_count() == 3
        assert goal_state is not None
        assert goal_state.goal == prompt
        assert goal_state.status == "blocked"
        assert goal_state.current_step == "read_text_file"
        assert goal_state.last_blocker == f"File not found: {missing_path}"
        assert len(ledger) == 1
        assert ledger[0].status == "blocked"
        assert ledger[0].step == "read_text_file"
        assert ledger[0].detail == f"File not found: {missing_path}"
    finally:
        session_manager.close()


def test_search_web_failure_then_successful_write_text_file_turn_persists_blocked_ledger(
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
    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.failure(
            tool_name=call.tool_name,
            error="Tool 'search_web' timed out after 15 seconds.",
        ),
    )

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

    tool_registry.register(WRITE_TEXT_FILE_DEFINITION, _write_tool)
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response("search_web", {"query": "progress note"}),
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
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    prompt = "Research the topic, then write a short local note file."

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
        assert reply == "I saved the note locally."
        assert fake_provider.call_count() == 3
        assert len(ledger) == 1
        assert [(entry.status, entry.step, entry.detail) for entry in ledger] == [
            ("blocked", "search_web", "timed out after 15 seconds."),
        ]
        assert output_path.read_text(encoding="utf-8") == "progress"
    finally:
        session_manager.close()


def test_existing_goal_state_session_preserves_progress_ledger_across_trivial_system_info_turn(
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

    tool_registry.register(WRITE_TEXT_FILE_DEFINITION, _write_tool)
    tool_registry.register(
        SYSTEM_INFO_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="OS: ExampleOS 1.0",
        ),
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
        _tool_call_response("system_info", {}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="The machine is running ExampleOS 1.0.",
            created_at="2026-03-25T09:00:02Z",
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    first_prompt = "Write a short local note file."
    second_prompt = "What OS is this machine running?"

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

        goal_state = session_manager.get_session_goal_state(session.id)
        ledger = session_manager.get_session_progress_ledger(session.id)
        assert first_reply == "I saved the note locally."
        assert second_reply == "The machine is running ExampleOS 1.0."
        assert fake_provider.call_count() == 4
        assert goal_state is not None
        assert goal_state.goal == first_prompt
        assert goal_state.status == "completed"
        assert goal_state.current_step == "write_text_file"
        assert len(ledger) == 1
        assert [(entry.status, entry.step, entry.detail) for entry in ledger] == [
            ("active", "write_text_file", "file write succeeded"),
        ]
    finally:
        session_manager.close()


def test_failed_write_text_file_then_successful_write_text_file_turn_keeps_successful_write_ledger(
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
    tool_registry = ToolRegistry()
    write_attempt_count = 0

    def _write_tool(call: ToolCall) -> ToolResult:
        nonlocal write_attempt_count
        write_attempt_count += 1
        if write_attempt_count == 1:
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=f"File already exists: {call.arguments['path']}",
            )

        path = Path(str(call.arguments["path"]))
        content = str(call.arguments["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text=f"Wrote text file: {path}",
            payload={"path": str(path), "size": len(content)},
        )

    tool_registry.register(WRITE_TEXT_FILE_DEFINITION, _write_tool)
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response(
            "write_text_file",
            {"path": str(output_path), "content": "draft"},
        ),
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
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    prompt = "Write a short local note file."

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
        assert reply == "I saved the note locally."
        assert fake_provider.call_count() == 3
        assert len(ledger) == 1
        assert [(entry.status, entry.step, entry.detail) for entry in ledger] == [
            ("active", "write_text_file", "file write succeeded"),
        ]
        assert output_path.read_text(encoding="utf-8") == "progress"
    finally:
        session_manager.close()


def test_existing_goal_state_session_preserves_progress_ledger_across_trivial_fast_web_search_turn(
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

    tool_registry.register(WRITE_TEXT_FILE_DEFINITION, _write_tool)
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Inoxtag\n- Inoxtag is a French content creator.",
            payload={
                "query": call.arguments["query"],
                "result_count": 1,
                "grounding_note": (
                    "Inoxtag\n- Inoxtag is a French content creator."
                ),
            },
        ),
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
        _tool_call_response("fast_web_search", {"query": "Inoxtag"}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="Inoxtag is a French content creator.",
            created_at="2026-03-25T09:00:02Z",
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    first_prompt = "Write a short local note file."
    second_prompt = "Who is Inoxtag?"

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

        goal_state = session_manager.get_session_goal_state(session.id)
        ledger = session_manager.get_session_progress_ledger(session.id)
        assert first_reply == "I saved the note locally."
        assert second_reply == "Inoxtag is a French content creator."
        assert fake_provider.call_count() == 4
        assert goal_state is not None
        assert goal_state.goal == first_prompt
        assert goal_state.status == "completed"
        assert goal_state.current_step == "write_text_file"
        assert len(ledger) == 1
        assert [(entry.status, entry.step, entry.detail) for entry in ledger] == [
            ("active", "write_text_file", "file write succeeded"),
        ]
    finally:
        session_manager.close()


def test_existing_goal_state_session_updates_on_blocked_multi_tool_turn(
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
    missing_path = project_root / "data" / "files" / "missing.txt"
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

    tool_registry.register(WRITE_TEXT_FILE_DEFINITION, _write_tool)
    tool_registry.register(
        SYSTEM_INFO_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="OS: ExampleOS 1.0",
        ),
    )
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: ToolResult.failure(
            tool_name=call.tool_name,
            error=f"File not found: {call.arguments['path']}",
        ),
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
        _tool_call_response("system_info", {}),
        _tool_call_response("read_text_file", {"path": str(missing_path)}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="Please try again.",
            created_at="2026-03-25T09:00:02Z",
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    first_prompt = "Write a short local note file."
    second_prompt = "Inspect the machine, then read the missing note file."

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

        goal_state = session_manager.get_session_goal_state(session.id)
        ledger = session_manager.get_session_progress_ledger(session.id)
        assert first_reply == "I saved the note locally."
        assert second_reply == "Please try again."
        assert fake_provider.call_count() == 5
        assert goal_state is not None
        assert goal_state.goal == second_prompt
        assert goal_state.status == "blocked"
        assert goal_state.current_step == "read_text_file"
        assert goal_state.last_blocker == f"File not found: {missing_path}"
        assert len(ledger) == 2
        assert [(entry.status, entry.step, entry.detail) for entry in ledger] == [
            ("active", "write_text_file", "file write succeeded"),
            ("blocked", "read_text_file", f"File not found: {missing_path}"),
        ]
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
            not message.startswith("Session task continuity:")
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
