from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from unclaw.core.command_handler import CommandHandler
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolResult
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


def _action_response(action: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        provider="ollama",
        model_name="fake-model",
        content=json.dumps(action, ensure_ascii=False),
        created_at="2026-03-27T09:00:00Z",
        finish_reason="stop",
    )


def test_completed_mission_projects_to_legacy_goal_state(
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
            output_text="OS: Linux 6.8",
            payload={"os": "Linux"},
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "start_new",
                "mission_goal": "Answer the local OS question.",
                "reasoning_summary": "One-step local fact mission.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Answer the OS question",
                        "kind": "mixed",
                        "status": "active",
                        "depends_on": [],
                        "required_evidence": ["tool:system_info", "reply_emitted"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    }
                ],
                "active_task_id": "t1",
                "tool_calls": [
                    {"task_id": "t1", "tool_name": "system_info", "arguments": {}}
                ],
                "reply_to_user": "Your OS is Linux.",
                "completion_claim": True,
                "blocker": None,
                "next_expected_evidence": None,
            }
        )
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "What OS am I running?",
            session_id=session.id,
        )
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="What OS am I running?",
            tracer=tracer,
            tool_registry=tool_registry,
        )
        goal_state = session_manager.get_session_goal_state(session.id)

        assert reply == "Your OS is Linux."
        assert goal_state is not None
        assert goal_state.goal == "Answer the local OS question."
        assert goal_state.status == "completed"
        assert goal_state.current_step == "Answer the OS question"
        assert goal_state.last_blocker is None
    finally:
        session_manager.close()


def test_blocked_mission_projects_latest_blocker_to_legacy_goal_state(
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
    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "start_new",
                "mission_goal": "Write a protected file.",
                "reasoning_summary": "This mission is blocked.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Write the file",
                        "kind": "file_write",
                        "status": "blocked",
                        "depends_on": [],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [],
                        "latest_error": "permission denied",
                        "repair_count": 0,
                    }
                ],
                "active_task_id": "t1",
                "tool_calls": [],
                "reply_to_user": None,
                "completion_claim": False,
                "blocker": "permission denied",
                "next_expected_evidence": None,
            }
        )
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Write a protected file.",
            session_id=session.id,
        )
        run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Write a protected file.",
            tracer=tracer,
            tool_registry=tool_registry,
        )
        goal_state = session_manager.get_session_goal_state(session.id)

        assert goal_state is not None
        assert goal_state.goal == "Write a protected file."
        assert goal_state.status == "blocked"
        assert goal_state.current_step == "Write the file"
        assert goal_state.last_blocker == "permission denied"
    finally:
        session_manager.close()
