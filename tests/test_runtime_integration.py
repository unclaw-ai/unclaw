from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from unclaw.core.command_handler import CommandHandler
from unclaw.core.mission_state import MissionState, MissionTaskState
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolResult
from unclaw.tools.file_tools import (
    READ_TEXT_FILE_DEFINITION,
    WRITE_TEXT_FILE_DEFINITION,
    read_text_file,
    write_text_file,
)
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.system_tools import SYSTEM_INFO_DEFINITION
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

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


def _run_turn(
    *,
    session_manager: SessionManager,
    command_handler: CommandHandler,
    tracer: Tracer,
    tool_registry: ToolRegistry,
    prompt: str,
) -> str:
    session = session_manager.ensure_current_session()
    session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
    return run_user_turn(
        session_manager=session_manager,
        command_handler=command_handler,
        user_input=prompt,
        tracer=tracer,
        tool_registry=tool_registry,
    )


def test_os_question_is_a_one_step_mission_that_completes_cleanly(
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
    observed_tool_calls: list[str] = []
    tool_registry = ToolRegistry()
    tool_registry.register(
        SYSTEM_INFO_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or ToolResult.ok(
                tool_name=call.tool_name,
                output_text="OS: Linux 6.8 (x86_64)",
                payload={"os": "Linux", "os_release": "6.8"},
            )
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "start_new",
                "mission_goal": "Answer what OS the user is running.",
                "reasoning_summary": "Use system_info and answer in one step.",
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
                "reply_to_user": "Ton OS est Linux.",
                "completion_claim": True,
                "blocker": None,
                "next_expected_evidence": None,
            }
        )
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        reply = _run_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_registry=tool_registry,
            prompt="quel est mon OS ?",
        )
        mission_state = session_manager.get_current_mission_state()

        assert reply == "Ton OS est Linux."
        assert fake_provider.call_count() == 1
        assert observed_tool_calls == ["system_info"]
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert len(mission_state.tasks) == 1
        assert mission_state.tasks[0].satisfied_evidence == (
            "tool:system_info",
            "reply_emitted",
        )
    finally:
        session_manager.close()


def test_kernel_requests_one_more_agent_step_when_a_status_summary_is_not_a_real_final_reply(
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
    observed_tool_calls: list[str] = []
    tool_registry = ToolRegistry()
    tool_registry.register(
        SYSTEM_INFO_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or ToolResult.ok(
                tool_name=call.tool_name,
                output_text="OS: Linux 6.8 (x86_64)",
                payload={"os": "Linux", "os_release": "6.8"},
            )
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "start_new",
                "mission_goal": "Answer what OS the user is running.",
                "reasoning_summary": "Get the OS first.",
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
                "reply_to_user": (
                    "Mission goal: Answer what OS the user is running.\n"
                    "Current active task: none\n"
                    "Completed tasks: Answer the OS question"
                ),
                "completion_claim": True,
                "blocker": None,
                "next_expected_evidence": None,
            }
        ),
        _action_response(
            {
                "mission_action": "continue_existing",
                "mission_goal": "Answer what OS the user is running.",
                "reasoning_summary": "Now answer directly from the proven system_info result.",
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
                "tool_calls": [],
                "reply_to_user": "Ton OS est Linux 6.8.",
                "completion_claim": True,
                "blocker": None,
                "next_expected_evidence": None,
            }
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        reply = _run_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_registry=tool_registry,
            prompt="quel est mon OS ?",
        )
        mission_state = session_manager.get_current_mission_state()

        assert reply == "Ton OS est Linux 6.8."
        assert fake_provider.call_count() == 2
        assert observed_tool_calls == ["system_info"]
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.tasks[0].satisfied_evidence == (
            "tool:system_info",
            "reply_emitted",
        )
    finally:
        session_manager.close()


def test_kernel_ignores_progress_narration_until_it_gets_a_real_final_reply(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    build_scripted_ollama_provider,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    output_path = settings.paths.files_dir / "smoke-note.txt"
    tool_registry = ToolRegistry()
    tool_registry.register(
        WRITE_TEXT_FILE_DEFINITION,
        lambda call: write_text_file(
            call,
            allowed_roots=(settings.paths.project_root,),
            default_write_dir=settings.paths.files_dir,
        ),
    )
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: read_text_file(
            call,
            read_allowed_roots=(settings.paths.project_root,),
            default_read_dir=settings.paths.files_dir,
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "start_new",
                "mission_goal": "Write and verify a local note file.",
                "reasoning_summary": "Write the file, read it back, then confirm.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Write and verify the file",
                        "kind": "file_write",
                        "status": "active",
                        "depends_on": [],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [str(output_path)],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t2",
                        "title": "Confirm the result to the user",
                        "kind": "reply",
                        "status": "pending",
                        "depends_on": ["t1"],
                        "required_evidence": ["reply_emitted"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                ],
                "active_task_id": "t1",
                "tool_calls": [
                    {
                        "task_id": "t1",
                        "tool_name": "write_text_file",
                        "arguments": {"path": str(output_path), "content": "bonjour"},
                    },
                    {
                        "task_id": "t1",
                        "tool_name": "read_text_file",
                        "arguments": {"path": str(output_path)},
                    },
                ],
                "reply_to_user": "I am now reading the file to verify its content.",
                "completion_claim": True,
                "blocker": None,
                "next_expected_evidence": None,
            }
        ),
        _action_response(
            {
                "mission_action": "continue_existing",
                "mission_goal": "Write and verify a local note file.",
                "reasoning_summary": "The file is already proven, so confirm cleanly now.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Write and verify the file",
                        "kind": "file_write",
                        "status": "completed",
                        "depends_on": [],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [str(output_path)],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t2",
                        "title": "Confirm the result to the user",
                        "kind": "reply",
                        "status": "active",
                        "depends_on": ["t1"],
                        "required_evidence": ["reply_emitted"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                ],
                "active_task_id": "t2",
                "tool_calls": [],
                "reply_to_user": "Le fichier smoke-note.txt est écrit et vérifié.",
                "completion_claim": True,
                "blocker": None,
                "next_expected_evidence": None,
            }
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        reply = _run_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_registry=tool_registry,
            prompt="ecris bonjour dans un fichier texte appele smoke-note.txt puis confirme qu'il est verifie",
        )
        mission_state = session_manager.get_current_mission_state()

        assert reply == "Le fichier smoke-note.txt est écrit et vérifié."
        assert fake_provider.call_count() == 2
        assert output_path.read_text(encoding="utf-8") == "bonjour"
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.get_task("t1").satisfied_evidence == (
            "artifact_write",
            "artifact_readback",
        )
        assert mission_state.get_task("t2").satisfied_evidence == ("reply_emitted",)
    finally:
        session_manager.close()


def test_new_file_mission_is_isolated_from_an_older_blocked_research_mission(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    build_scripted_ollama_provider,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    output_path = settings.paths.files_dir / "fresh-note.txt"
    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.failure(
            tool_name=call.tool_name,
            error="previous timeout",
            failure_kind="timeout",
        ),
    )
    tool_registry.register(
        WRITE_TEXT_FILE_DEFINITION,
        lambda call: write_text_file(
            call,
            allowed_roots=(settings.paths.project_root,),
            default_write_dir=settings.paths.files_dir,
        ),
    )
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: read_text_file(
            call,
            read_allowed_roots=(settings.paths.project_root,),
            default_read_dir=settings.paths.files_dir,
        ),
    )

    session = session_manager.ensure_current_session()
    session_manager.persist_mission_state(
        MissionState(
            mission_id="mission-blocked",
            mission_goal="Blocked research mission",
            status="blocked",
            tasks=(
                MissionTaskState(
                    id="t1",
                    title="Research Macron",
                    kind="web_research",
                    status="blocked",
                    required_evidence=("full_web_research",),
                    latest_error="search timeout",
                ),
            ),
            active_task_id=None,
            updated_at="2026-03-27T09:00:00Z",
            blocker="search timeout",
            last_user_input="continue",
            executor_state="blocked",
            last_blocker="search timeout",
        ),
        session_id=session.id,
    )

    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "start_new",
                "mission_goal": "Write a local note file.",
                "reasoning_summary": "The user asked for a different local file task, so start a new mission.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Write and verify the file",
                        "kind": "file_write",
                        "status": "active",
                        "depends_on": [],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [str(output_path)],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t2",
                        "title": "Reply to the user",
                        "kind": "reply",
                        "status": "pending",
                        "depends_on": ["t1"],
                        "required_evidence": ["reply_emitted"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                ],
                "active_task_id": "t1",
                "tool_calls": [
                    {
                        "task_id": "t1",
                        "tool_name": "write_text_file",
                        "arguments": {"path": str(output_path), "content": "new isolated note"},
                    },
                    {
                        "task_id": "t1",
                        "tool_name": "read_text_file",
                        "arguments": {"path": str(output_path)},
                    },
                ],
                "reply_to_user": "La nouvelle mission fichier est terminée.",
                "completion_claim": True,
                "blocker": None,
                "next_expected_evidence": None,
            }
        )
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        reply = _run_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_registry=tool_registry,
            prompt="Écris une note locale.",
        )
        mission_state = session_manager.get_current_mission_state(session.id)

        assert reply == "La nouvelle mission fichier est terminée."
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.mission_goal == "Write a local note file."
        assert "isolated note" in output_path.read_text(encoding="utf-8")
    finally:
        session_manager.close()


def test_status_request_is_rendered_from_persisted_mission_state_only(
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
    session = session_manager.ensure_current_session()
    session_manager.persist_mission_state(
        MissionState(
            mission_id="mission-status",
            mission_goal="Write a report and send a joke.",
            status="active",
            tasks=(
                MissionTaskState(
                    id="t1",
                    title="Research the topic",
                    kind="web_research",
                    status="completed",
                    required_evidence=("full_web_research",),
                    satisfied_evidence=("full_web_research",),
                ),
                MissionTaskState(
                    id="t2",
                    title="Write the file",
                    kind="file_write",
                    status="active",
                    depends_on=("t1",),
                    required_evidence=("artifact_write", "artifact_readback"),
                    artifact_paths=("/tmp/report.txt",),
                ),
            ),
            active_task_id="t2",
            updated_at="2026-03-27T09:00:00Z",
            reasoning_summary="persisted mission",
            next_expected_evidence="artifact_write, artifact_readback",
            last_user_input="continue",
        ),
        session_id=session.id,
    )
    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "continue_existing",
                "mission_goal": "Write a report and send a joke.",
                "reasoning_summary": "Status only.",
                "tasks": [
                    {
                        "id": "t2",
                        "title": "Write the file",
                        "kind": "file_write",
                        "status": "active",
                        "depends_on": ["t1"],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": ["/tmp/report.txt"],
                        "latest_error": None,
                        "repair_count": 0,
                    }
                ],
                "active_task_id": "t2",
                "tool_calls": [],
                "reply_to_user": None,
                "completion_claim": False,
                "blocker": None,
                "next_expected_evidence": "artifact_write, artifact_readback",
            }
        )
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        reply = _run_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_registry=ToolRegistry(),
            prompt="où en est cette mission ?",
        )

        assert "Mission goal: Write a report and send a joke." in reply
        assert "Current active task: Write the file" in reply
        assert "Completed tasks: Research the topic" in reply
        assert "Next expected action or evidence: artifact_write, artifact_readback" in reply
    finally:
        session_manager.close()
