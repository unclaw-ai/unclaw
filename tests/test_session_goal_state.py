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


def test_one_shot_write_text_file_turn_creates_completed_session_goal_state(
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

        goal_state = session_manager.get_session_goal_state(session.id)
        assert reply == "I saved the note locally."
        assert fake_provider.call_count() == 2
        assert goal_state is not None
        assert goal_state.goal == prompt
        assert goal_state.status == "completed"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "session note"
    finally:
        session_manager.close()


def test_search_web_failure_then_successful_write_text_file_turn_stays_blocked(
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
        _tool_call_response("search_web", {"query": "session goal note"}),
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

        goal_state = session_manager.get_session_goal_state(session.id)
        assert reply == "I saved the note locally."
        assert fake_provider.call_count() == 3
        assert goal_state is not None
        assert goal_state.goal == prompt
        assert goal_state.status == "blocked"
        assert goal_state.current_step == "search_web"
        assert goal_state.last_blocker == "Tool 'search_web' timed out after 15 seconds."
        assert output_path.read_text(encoding="utf-8") == "session note"
    finally:
        session_manager.close()


def test_fast_web_search_mismatch_then_successful_write_text_file_turn_stays_blocked(
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
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="No exact top match; different entity found.",
            payload={
                "query": call.arguments["query"],
                "result_count": 1,
                "grounding_note": "",
            },
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
        _tool_call_response("fast_web_search", {"query": "Marine Leleu"}),
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
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    prompt = "Research Marine Leleu, then write a short local note file."

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
        assert reply == "I saved the note locally."
        assert fake_provider.call_count() == 3
        assert goal_state is not None
        assert goal_state.goal == prompt
        assert goal_state.status == "blocked"
        assert goal_state.current_step == "fast_web_search"
        assert goal_state.last_blocker == (
            "Quick web grounding matched a different entity or found no exact "
            "usable match."
        )
        assert output_path.read_text(encoding="utf-8") == "session note"
    finally:
        session_manager.close()


def test_search_web_thin_then_successful_write_text_file_turn_stays_blocked(
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
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                f"Search query: {call.arguments['query']}\n"
                "Sources fetched: 1 of 1 attempted\n"
                "Evidence kept: 1\n"
            ),
            payload={
                "query": call.arguments["query"],
                "summary_points": ["One thin supported point."],
                "display_sources": [
                    {
                        "title": "Example source",
                        "url": "https://example.com/source",
                    }
                ],
                "evidence_count": 1,
                "finding_count": 1,
            },
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
        _tool_call_response("search_web", {"query": "Marine Leleu biography"}),
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
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    prompt = "Research Marine Leleu, then write a short local note file."

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
        assert reply.startswith("I saved the note locally.")
        assert "Sources:" in reply
        assert "https://example.com/source" in reply
        assert fake_provider.call_count() == 3
        assert goal_state is not None
        assert goal_state.goal == prompt
        assert goal_state.status == "blocked"
        assert goal_state.current_step == "search_web"
        assert (
            goal_state.last_blocker
            == "Web evidence was too thin to confirm requested details."
        )
        assert output_path.read_text(encoding="utf-8") == "session note"
    finally:
        session_manager.close()


def test_failed_write_text_file_turn_persists_blocked_session_goal_state(
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
    tool_registry.register(
        WRITE_TEXT_FILE_DEFINITION,
        lambda call: ToolResult.failure(
            tool_name=call.tool_name,
            error=f"Permission denied: {call.arguments['path']}",
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
            content="Please try again.",
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

        run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        goal_state = session_manager.get_session_goal_state(session.id)
        assert fake_provider.call_count() == 2
        assert goal_state is not None
        assert goal_state.goal == prompt
        assert goal_state.status == "blocked"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker == f"Permission denied: {output_path}"
        assert not output_path.exists()
    finally:
        session_manager.close()


def test_failed_write_text_file_then_successful_write_text_file_turn_can_complete(
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
            {"path": str(output_path), "content": "first draft"},
        ),
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

        goal_state = session_manager.get_session_goal_state(session.id)
        assert reply == "I saved the note locally."
        assert fake_provider.call_count() == 3
        assert goal_state is not None
        assert goal_state.goal == prompt
        assert goal_state.status == "completed"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "session note"
    finally:
        session_manager.close()


def test_search_web_then_successful_write_text_file_turn_can_complete_with_strong_evidence(
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
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                f"Search query: {call.arguments['query']}\n"
                "Sources fetched: 2 of 2 attempted\n"
                "Evidence kept: 4\n"
            ),
            payload={
                "query": call.arguments["query"],
                "summary_points": [
                    "Marine Leleu is a French endurance athlete and creator."
                ],
                "display_sources": [
                    {
                        "title": "Marine Leleu",
                        "url": "https://example.com/marine-leleu",
                    },
                    {
                        "title": "Athlete profile",
                        "url": "https://example.com/athlete-profile",
                    },
                ],
                "evidence_count": 4,
                "finding_count": 2,
            },
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
        _tool_call_response("search_web", {"query": "Marine Leleu biography"}),
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
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    prompt = "Research Marine Leleu, then write a short local note file."

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
        assert reply.startswith("I saved the note locally.")
        assert "Sources:" in reply
        assert "https://example.com/marine-leleu" in reply
        assert "https://example.com/athlete-profile" in reply
        assert fake_provider.call_count() == 3
        assert goal_state is not None
        assert goal_state.goal == prompt
        assert goal_state.status == "completed"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "session note"
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
        assert goal_state.status == "completed"
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
        assert goal_state.status == "completed"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "session note"
    finally:
        session_manager.close()


def test_later_turn_injects_compact_completed_task_continuity_note_without_extra_model_call(
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
        assert goal_state.status == "completed"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "session note"

        first_turn_system_messages = [
            message.content
            for message in captured_messages[0]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        assert all(
            not message.startswith("Session task continuity:")
            for message in first_turn_system_messages
        )

        second_turn_system_messages = [
            message.content
            for message in captured_messages[2]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        continuity_notes = [
            message
            for message in second_turn_system_messages
            if message.startswith("Session task continuity:")
        ]
        assert len(continuity_notes) == 1
        assert all(
            not message.startswith("Session goal state:")
            and not message.startswith("Session progress ledger:")
            for message in second_turn_system_messages
        )
        assert 'goal="Write a short local note file."' in continuity_notes[0]
        assert 'status="completed"' in continuity_notes[0]
        assert 'current_step="write_text_file"' in continuity_notes[0]
        assert "last_blocker=" not in continuity_notes[0]
        assert "latest_progress=" not in continuity_notes[0]
        assert "\n" not in continuity_notes[0]
    finally:
        session_manager.close()


def test_blocked_task_compact_continuation_turn_keeps_original_goal_text_and_can_complete(
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
    output_path = project_root / "data" / "files" / "resumed-session-goal-note.txt"
    source_path = project_root / "data" / "files" / "resume-source.txt"
    tool_registry = ToolRegistry()
    captured_messages: list[list[object]] = []

    def _read_tool(call: ToolCall) -> ToolResult:
        path = Path(str(call.arguments["path"]))
        if not path.exists():
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=f"File not found: {path}",
            )

        content = path.read_text(encoding="utf-8")
        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text=content,
            payload={"path": str(path), "size": len(content)},
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

    tool_registry.register(READ_TEXT_FILE_DEFINITION, _read_tool)
    tool_registry.register(WRITE_TEXT_FILE_DEFINITION, _write_tool)
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response("read_text_file", {"path": str(source_path)}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="Please try again.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        ),
        _tool_call_response("read_text_file", {"path": str(source_path)}),
        _tool_call_response(
            "write_text_file",
            {"path": str(output_path), "content": "session note"},
        ),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="I saved the note locally.",
            created_at="2026-03-25T09:00:02Z",
            finish_reason="stop",
        ),
        captured_messages=captured_messages,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    first_prompt = "Read the local source note, then write a short local note file."
    second_prompt = "ok"

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
        blocked_goal_state = session_manager.get_session_goal_state(session.id)
        assert blocked_goal_state is not None
        assert blocked_goal_state.goal == first_prompt
        assert blocked_goal_state.status == "blocked"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("Recovered local source note.", encoding="utf-8")
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
        assert first_reply == (
            "The tool step failed, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
        assert second_reply == "I saved the note locally."
        assert fake_provider.call_count() == 5
        assert goal_state is not None
        assert goal_state.goal == first_prompt
        assert goal_state.status == "completed"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "session note"

        second_turn_system_messages = [
            message.content
            for message in captured_messages[2]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        continuity_notes = [
            message
            for message in second_turn_system_messages
            if message.startswith("Session task continuity:")
        ]
        assert len(continuity_notes) == 1
        assert 'goal="Read the local source note, then write a short local note file."' in (
            continuity_notes[0]
        )
        assert 'status="blocked"' in continuity_notes[0]
        assert 'current_step="read_text_file"' in continuity_notes[0]
        assert f'last_blocker="File not found: {source_path}"' in (
            continuity_notes[0]
        )
    finally:
        session_manager.close()


def test_active_task_resume_turn_keeps_original_goal_text_and_can_complete(
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
    output_path = project_root / "data" / "files" / "active-session-goal-note.txt"
    source_path = project_root / "data" / "files" / "active-source.txt"
    tool_registry = ToolRegistry()
    captured_messages: list[list[object]] = []
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("Local source note for the active task.", encoding="utf-8")
    tool_registry.register(
        SYSTEM_INFO_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="OS: ExampleOS 1.0",
        ),
    )
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=Path(str(call.arguments["path"])).read_text(encoding="utf-8"),
            payload={
                "path": str(call.arguments["path"]),
                "size": len(
                    Path(str(call.arguments["path"])).read_text(encoding="utf-8")
                ),
            },
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
        _tool_call_response("system_info", {}),
        _tool_call_response("read_text_file", {"path": str(source_path)}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="I gathered the local context for the note.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        ),
        _tool_call_response(
            "write_text_file",
            {"path": str(output_path), "content": "session note"},
        ),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="I saved the note locally.",
            created_at="2026-03-25T09:00:02Z",
            finish_reason="stop",
        ),
        captured_messages=captured_messages,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    first_prompt = (
        "Inspect the machine, read the local source note, then write a short local note file."
    )
    second_prompt = "Finish the same task."

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
        assert first_reply == "I gathered the local context for the note."
        assert second_reply == "I saved the note locally."
        assert fake_provider.call_count() == 5
        assert goal_state is not None
        assert goal_state.goal == first_prompt
        assert goal_state.status == "completed"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "session note"

        second_turn_system_messages = [
            message.content
            for message in captured_messages[3]
            if getattr(message, "role", None) is LLMRole.SYSTEM
        ]
        continuity_notes = [
            message
            for message in second_turn_system_messages
            if message.startswith("Session task continuity:")
        ]
        assert len(continuity_notes) == 1
        assert (
            'goal="Inspect the machine, read the local source note, then write a short local note file."'
            in continuity_notes[0]
        )
        assert 'status="active"' in continuity_notes[0]
        assert 'current_step="read_text_file"' in continuity_notes[0]
        assert (
            'latest_progress=[status="active"; step="read_text_file"; '
            'detail="tool succeeded"]'
            in continuity_notes[0]
        )
    finally:
        session_manager.close()


def test_blocked_task_retry_turn_keeps_original_goal_text_and_blocked_state(
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
    missing_path = project_root / "data" / "files" / "retry-source.txt"
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: ToolResult.failure(
            tool_name=call.tool_name,
            error=f"File not found: {call.arguments['path']}",
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response("read_text_file", {"path": str(missing_path)}),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="Please try again.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        ),
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

    first_prompt = "Read the local source note, then write a short local note file."
    second_prompt = "Try the same task again."

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
        blocked_goal_state = session_manager.get_session_goal_state(session.id)
        assert blocked_goal_state is not None
        assert blocked_goal_state.goal == first_prompt
        assert blocked_goal_state.status == "blocked"
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
        assert first_reply == (
            "The tool step failed, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
        assert second_reply == first_reply
        assert fake_provider.call_count() == 4
        assert goal_state is not None
        assert goal_state.goal == first_prompt
        assert goal_state.status == "blocked"
        assert goal_state.current_step == "read_text_file"
        assert goal_state.last_blocker == f"File not found: {missing_path}"
    finally:
        session_manager.close()


def test_blocked_task_can_be_replaced_by_substantive_new_successful_task_with_same_generic_tools(
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
    output_path = project_root / "data" / "files" / "new-session-goal-note.txt"
    tool_registry = ToolRegistry()

    def _search_tool(call: ToolCall) -> ToolResult:
        if call.arguments["query"] == "topic one note":
            return ToolResult.failure(
                tool_name=call.tool_name,
                error="Tool 'search_web' timed out after 15 seconds.",
            )

        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                f"Search query: {call.arguments['query']}\n"
                "Sources fetched: 2 of 2 attempted\n"
                "Evidence kept: 4\n"
            ),
            payload={
                "query": call.arguments["query"],
                "summary_points": [
                    "Topic two has enough grounded evidence for a short note."
                ],
                "display_sources": [
                    {
                        "title": "Topic two profile",
                        "url": "https://example.com/topic-two-profile",
                    },
                    {
                        "title": "Topic two recap",
                        "url": "https://example.com/topic-two-recap",
                    },
                ],
                "evidence_count": 4,
                "finding_count": 2,
            },
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

    tool_registry.register(SEARCH_WEB_DEFINITION, _search_tool)
    tool_registry.register(WRITE_TEXT_FILE_DEFINITION, _write_tool)
    fake_provider = build_scripted_ollama_provider(
        _tool_call_response("search_web", {"query": "topic one note"}),
        _tool_call_response(
            "write_text_file",
            {"path": str(output_path), "content": "first note"},
        ),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="I saved the first note locally.",
            created_at="2026-03-25T09:00:02Z",
            finish_reason="stop",
        ),
        _tool_call_response("search_web", {"query": "topic two note"}),
        _tool_call_response(
            "write_text_file",
            {"path": str(output_path), "content": "second note"},
        ),
        LLMResponse(
            provider="ollama",
            model_name="fake-model",
            content="I saved the note locally.",
            created_at="2026-03-25T09:00:03Z",
            finish_reason="stop",
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    first_prompt = "Research topic one, then write a short local note file."
    second_prompt = "Research topic two and save a fresh local note file."

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
        assert first_reply == "I saved the first note locally."
        assert second_reply == "I saved the note locally."
        assert fake_provider.call_count() == 6
        assert goal_state is not None
        assert goal_state.goal == second_prompt
        assert goal_state.status == "completed"
        assert goal_state.current_step == "write_text_file"
        assert goal_state.last_blocker is None
        assert output_path.read_text(encoding="utf-8") == "second note"
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

        continuity_note = _build_session_task_continuity_note(
            session_manager=session_manager,
            session_id=session.id,
        )
        assert continuity_note is not None
        assert continuity_note.startswith("Session task continuity:")
        assert len(continuity_note) < 600
        assert "Earlier conversation fragment" not in continuity_note
        assert 'status="blocked"' in continuity_note
        assert 'current_step="system_info"' in continuity_note
        assert (
            'last_blocker="Tool \'system_info\' timed out after 5 seconds."'
            in continuity_note
        )
        assert "latest_progress=" not in continuity_note
    finally:
        session_manager.close()
