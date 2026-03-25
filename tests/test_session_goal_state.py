from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.core.command_handler import CommandHandler
from unclaw.core.runtime import run_user_turn
from unclaw.core.runtime_support import _build_session_goal_state_context_note
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
from unclaw.tools.web_tools import FAST_WEB_SEARCH_DEFINITION

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


def test_one_shot_system_info_turn_does_not_create_session_goal_state(
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
            output_text="Local datetime: 2026-03-25 10:00:00 CET",
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response("system_info", {}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="It is 10:00 CET.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    prompt = "What is the local time on this machine?"

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

        assert reply == "It is 10:00 CET."
        assert fake_provider.call_count() == 2
        assert session_manager.get_session_goal_state(session.id) is None
        event_types = [
            event.event_type
            for event in session_manager.event_repository.list_recent_events(
                session.id,
                limit=10,
            )
        ]
        assert "session.goal_state.updated" not in event_types
    finally:
        session_manager.close()


def test_one_shot_fast_web_search_turn_does_not_create_session_goal_state(
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
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Inoxtag\n- Inoxtag is a French content creator.",
            payload={
                "query": call.arguments["query"],
                "result_count": 2,
                "grounding_note": (
                    "Inoxtag\n"
                    "- Inoxtag is a French content creator.\n"
                    "- He is known for online videos."
                ),
            },
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response("fast_web_search", {"query": "Inoxtag"}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="Inoxtag is a French content creator.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    prompt = "Qui est Inoxtag ?"

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

        assert reply == "Inoxtag is a French content creator."
        assert fake_provider.call_count() == 2
        assert session_manager.get_session_goal_state(session.id) is None
        event_types = [
            event.event_type
            for event in session_manager.event_repository.list_recent_events(
                session.id,
                limit=10,
            )
        ]
        assert "session.goal_state.updated" not in event_types
    finally:
        session_manager.close()


def test_existing_goal_state_is_preserved_after_later_one_shot_system_info_turn(
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
    output_path = project_root / "data" / "files" / "session-goal-note.txt"
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
            output_text="Local datetime: 2026-03-25 10:00:00 CET",
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response(
            "write_text_file",
            {"path": str(output_path), "content": "session note"},
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
            content="It is 10:00 CET.",
            created_at="2026-03-25T09:00:02Z",
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    first_prompt = "Write a short local note file."
    second_prompt = "What is the local time on this machine?"

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
        assert first_reply == "I saved the note locally."
        assert second_reply == "It is 10:00 CET."
        assert fake_provider.call_count() == 4
        assert goal_state is not None
        assert goal_state.goal == first_prompt
        assert goal_state.status == "active"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "session note"
    finally:
        session_manager.close()


def test_existing_goal_state_is_preserved_after_later_one_shot_fast_web_search_turn(
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
    output_path = project_root / "data" / "files" / "session-goal-note.txt"
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
            {"path": str(output_path), "content": "session note"},
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
        assert first_reply == "I saved the note locally."
        assert second_reply == "Inoxtag is a French content creator."
        assert fake_provider.call_count() == 4
        assert goal_state is not None
        assert goal_state.goal == first_prompt
        assert goal_state.status == "active"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "session note"
    finally:
        session_manager.close()


def test_later_turn_injects_compact_goal_state_note_without_extra_model_call(
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
    output_path = project_root / "data" / "files" / "session-goal-note.txt"
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
    captured_messages: list[list[object]] = []
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response(
            "write_text_file",
            {"path": str(output_path), "content": "session note"},
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
            content="You were writing a local note file.",
            created_at="2026-03-25T09:00:02Z",
            finish_reason="stop",
        ),
        captured_messages=captured_messages,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    first_prompt = "Write a short local note file."
    second_prompt = "What are we working on in this session?"

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
        assert first_reply == "I saved the note locally."
        assert second_reply == "You were writing a local note file."
        assert fake_provider.call_count() == 3
        assert goal_state is not None
        assert goal_state.goal == first_prompt
        assert goal_state.status == "active"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "session note"

        first_turn_system_messages = [
            message.content
            for message in captured_messages[0]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        assert all(
            not message.startswith("Session goal state:")
            for message in first_turn_system_messages
        )

        second_turn_system_messages = [
            message.content
            for message in captured_messages[2]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        goal_notes = [
            message
            for message in second_turn_system_messages
            if message.startswith("Session goal state:")
        ]
        assert len(goal_notes) == 1
        assert 'goal="Write a short local note file."' in goal_notes[0]
        assert 'status="active"' in goal_notes[0]
        assert 'current_step="write_text_file"' in goal_notes[0]
        assert "last_blocker=none" in goal_notes[0]
    finally:
        session_manager.close()


def test_chat_only_sessions_remain_unchanged_without_goal_state(
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
        assert session_manager.get_session_goal_state(session.id) is None

        system_messages = [
            message.content
            for message in captured_messages[0]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        assert all(
            not message.startswith("Session goal state:")
            for message in system_messages
        )
        event_types = [
            event.event_type
            for event in session_manager.event_repository.list_recent_events(
                session.id,
                limit=10,
            )
        ]
        assert "session.goal_state.updated" not in event_types
    finally:
        session_manager.close()


def test_session_goal_state_stays_bounded_and_tracks_runtime_blockers(
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

    long_prompt = (
        "   Please inspect the machine and tell me what failed.\n"
        + ("Very long detail segment. " * 40)
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Earlier conversation fragment that must not be copied into the goal note.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Earlier assistant fragment.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.USER,
            long_prompt,
            session_id=session.id,
        )
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=long_prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert reply == (
            "The tool step timed out, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
        goal_state = session_manager.get_session_goal_state(session.id)
        assert goal_state is not None
        assert goal_state.status == "blocked"
        assert goal_state.current_step == "system_info"
        assert goal_state.last_blocker == "Tool 'system_info' timed out after 5 seconds."
        assert "\n" not in goal_state.goal
        assert len(goal_state.goal) <= 240
        assert len(goal_state.goal) < len(long_prompt)

        goal_note = _build_session_goal_state_context_note(
            session_manager=session_manager,
            session_id=session.id,
        )
        assert goal_note is not None
        assert len(goal_note) < 700
        assert "Earlier conversation fragment" not in goal_note
        assert 'status="blocked"' in goal_note
        assert (
            'last_blocker="Tool \'system_info\' timed out after 5 seconds."'
            in goal_note
        )
    finally:
        session_manager.close()
