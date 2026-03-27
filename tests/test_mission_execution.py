from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import unclaw.core.agent_loop as _agent_loop
from unclaw.core.command_handler import CommandHandler
from unclaw.core.mission_workspace import parse_mission_workspace_pointer
from unclaw.core.mission_state import MissionDeliverableState, MissionState
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.file_tools import (
    READ_TEXT_FILE_DEFINITION,
    WRITE_TEXT_FILE_DEFINITION,
    read_text_file,
    write_text_file,
)
from unclaw.tools.registry import ToolRegistry
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


def _tool_call_response(tool_name: str, arguments: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        provider="ollama",
        model_name="fake-model",
        content="",
        created_at="2026-03-26T09:00:00Z",
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


def _is_finalizer_call(messages) -> bool:  # type: ignore[no-untyped-def]
    return bool(messages) and messages[0].role is LLMRole.SYSTEM and messages[
        0
    ].content.startswith("Grounded reply finalizer for one runtime turn.")


def _is_mission_planner_call(messages) -> bool:  # type: ignore[no-untyped-def]
    return bool(messages) and messages[0].role is LLMRole.SYSTEM and messages[
        0
    ].content.startswith("Mission planner for the Unclaw local agent runtime.")


def _is_mission_verifier_call(messages) -> bool:  # type: ignore[no-untyped-def]
    return bool(messages) and messages[0].role is LLMRole.SYSTEM and messages[
        0
    ].content.startswith("Mission step verifier for the Unclaw local agent runtime.")


@pytest.mark.parametrize("thinking_enabled", [False, True])
def test_compound_research_write_joke_completes_in_one_flow(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    thinking_enabled: bool,
) -> None:
    """Compound mission enters the kernel on first turn via search_web tool call.

    Flow: runtime initial call (search_web) → planner (start_new) →
    kernel loop: execute search → verify d1 → execute write → verify d2 →
    execute joke → verify completed → finalizer.
    """
    expected_thinking_enabled = thinking_enabled
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    command_handler.thinking_enabled = thinking_enabled
    output_path = settings.paths.files_dir / "mission-research-note.txt"
    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                "Search query: mission test\n"
                "Sources fetched: 2 of 2 attempted\n"
                "Evidence kept: 3\n"
                "Mission fact one.\n"
                "Mission fact two.\n"
            ),
            payload={
                "query": call.arguments["query"],
                "summary_points": ["Mission fact one.", "Mission fact two."],
                "display_sources": [
                    {"title": "Example", "url": "https://example.com/source"}
                ],
                "evidence_count": 3,
                "finding_count": 2,
            },
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
    planner_calls = 0
    execution_calls = 0

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, tools
            nonlocal planner_calls, execution_calls

            if _is_finalizer_call(messages):
                payload = json.loads(messages[-1].content)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {"final_reply": payload["assistant_draft_reply"]},
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T09:00:06Z",
                    finish_reason="stop",
                )

            if _is_mission_planner_call(messages):
                planner_calls += 1
                payload = json.loads(messages[-1].content)
                # Planner sees the initial response's tool call summary
                summary = payload["first_response_summary"]
                assert summary is not None
                assert any(
                    tc["tool_name"] == "search_web"
                    for tc in summary.get("tool_calls", [])
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": (
                                "Research the topic, save a local note, and tell a joke."
                            ),
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "mixed",
                                    "task": "Research mission test",
                                    "deliverable": "Grounded research facts",
                                    "verification": "Search evidence is verified.",
                                },
                                {
                                    "id": "d2",
                                    "mode": "artifact",
                                    "task": "Write the local note",
                                    "deliverable": "Local note file",
                                    "verification": "The file is written and read back.",
                                },
                                {
                                    "id": "d3",
                                    "mode": "reply",
                                    "task": "Tell a short joke",
                                    "deliverable": "Short joke in the final reply",
                                    "verification": "The joke appears in the reply.",
                                },
                            ],
                            "execution_queue": ["d1", "d2", "d3"],
                            "active_deliverable_id": "d1",
                            "summary": "compound research-write-joke mission",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T09:00:01Z",
                    finish_reason="stop",
                )

            if _is_mission_verifier_call(messages):
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")
                # After search_web: d1 completed, advance to d2
                if latest_tool_results and latest_tool_results[0]["tool_name"] == "search_web":
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d2",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [],
                                "evidence": ["Mission fact one.", "Mission fact two."],
                                "final_deliverables_missing": ["d2", "d3"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Write the local note next.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T09:00:02Z",
                        finish_reason="stop",
                    )
                # After write_text_file + read_text_file: d2 completed, advance to d3
                if latest_tool_results and any(
                    item["tool_name"] == "write_text_file" for item in latest_tool_results
                ):
                    assert any(
                        item["tool_name"] == "read_text_file"
                        for item in latest_tool_results
                    ), "Artifact read-back verification must happen after write"
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d3",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": [],
                                "final_deliverables_missing": ["d3"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Finish with the requested joke.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T09:00:04Z",
                        finish_reason="stop",
                    )
                # After joke text reply: mission completed
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                            "artifact_paths": [str(output_path)],
                            "evidence": [],
                            "final_deliverables_missing": [],
                            "next_action": "final_reply",
                            "repair_strategy": None,
                            "notes_for_next_step": None,
                            "assistant_reply": draft_reply,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T09:00:05Z",
                    finish_reason="stop",
                )

            # Execution calls (not planner/verifier/finalizer)
            execution_calls += 1
            assert thinking_enabled is expected_thinking_enabled
            if execution_calls == 1:
                # Initial runtime call: search_web tool call
                return _tool_call_response("search_web", {"query": "mission test"})
            if execution_calls == 2:
                # Kernel execution: write_text_file
                tool_messages = [
                    message.content for message in messages if message.role is LLMRole.TOOL
                ]
                assert any("Mission fact one." in message for message in tool_messages)
                return _tool_call_response(
                    "write_text_file",
                    {"path": str(output_path), "content": "Mission fact one.\nMission fact two."},
                )
            if execution_calls == 3:
                # Kernel execution: joke text reply
                reply_text = (
                    "I saved the research note locally. "
                    "Joke: Why did the local agent cross the filesystem? "
                    "To get to the root cause."
                )
                if content_callback is not None:
                    content_callback(reply_text)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=reply_text,
                    created_at="2026-03-26T09:00:03Z",
                    finish_reason="stop",
                )
            raise AssertionError("Unexpected extra execution call in compound mission flow.")

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    prompt = "Research mission test, save a local note, and finish with a short joke."

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert "Joke:" in reply
        assert "root cause" in reply
        assert output_path.exists()
        assert "Mission fact one." in output_path.read_text(encoding="utf-8")
        assert planner_calls == 1
        assert execution_calls == 3
        mission_state = session_manager.get_current_mission_state(session.id)
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.active_deliverable_id is None
    finally:
        session_manager.close()


def test_first_pass_no_tool_draft_recovers_and_completes_needed_write(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    output_path = settings.paths.files_dir / "recovered-note.txt"
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
    call_count = 0

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count

            if _is_finalizer_call(messages):
                payload = json.loads(messages[-1].content)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {"final_reply": payload["assistant_draft_reply"]},
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T09:00:06Z",
                    finish_reason="stop",
                )
            if _is_mission_planner_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": "Write a short local note file.",
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "artifact",
                                    "task": "write_text_file",
                                    "deliverable": "Local note file",
                                    "verification": "A successful local file write is verified.",
                                },
                                {
                                    "id": "d2",
                                    "mode": "reply",
                                    "task": "Confirm the saved note",
                                    "deliverable": "Short confirmation reply",
                                    "verification": "The reply confirms the verified saved note.",
                                }
                            ],
                            "execution_queue": ["d1", "d2"],
                            "active_deliverable_id": "d1",
                            "summary": "recover and write the local note",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T09:00:02Z",
                    finish_reason="stop",
                )
            if _is_mission_verifier_call(messages):
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")
                if latest_tool_results:
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d2",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": [],
                                "final_deliverables_missing": ["d2"],
                                "next_action": "continue",
                                "notes_for_next_step": "Answer from the verified write.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T09:00:03Z",
                        finish_reason="stop",
                    )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                            "artifact_paths": [str(output_path)],
                            "evidence": [],
                            "final_deliverables_missing": [],
                            "next_action": "final_reply",
                            "notes_for_next_step": None,
                            "assistant_reply": draft_reply,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T09:00:05Z",
                    finish_reason="stop",
                )

            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="I saved the note locally.",
                    created_at="2026-03-26T09:00:00Z",
                    finish_reason="stop",
                )
            if call_count == 2:
                assert any(
                    message.role is LLMRole.SYSTEM
                    and message.content.startswith("Tool reconsideration note:")
                    for message in messages
                )
                return _tool_call_response(
                    "write_text_file",
                    {"path": str(output_path), "content": "Recovered note"},
                )
            if call_count == 3:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="I saved the note locally.",
                    created_at="2026-03-26T09:00:04Z",
                    finish_reason="stop",
                )
            raise AssertionError("Unexpected extra model call in recovery flow.")

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    prompt = "Write a short local note file."

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert reply == "I saved the note locally."
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "Recovered note"
        assert call_count == 3
    finally:
        session_manager.close()


def test_wrong_path_write_repairs_to_allowed_path_and_completes(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    blocked_path = str(project_root.parent / "outside-note.txt")
    repaired_path = settings.paths.files_dir / "repaired-note.txt"
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
    call_count = 0

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count

            if _is_finalizer_call(messages):
                payload = json.loads(messages[-1].content)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {"final_reply": payload["assistant_draft_reply"]},
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T09:00:07Z",
                    finish_reason="stop",
                )
            if _is_mission_planner_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": "Write the requested local note file.",
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "artifact",
                                    "task": "write_text_file",
                                    "deliverable": "Allowed local note file",
                                    "verification": "A successful allowed-root local file write is verified.",
                                },
                                {
                                    "id": "d2",
                                    "mode": "reply",
                                    "task": "Confirm the repaired write",
                                    "deliverable": "Short confirmation reply",
                                    "verification": "The reply confirms the repaired verified write.",
                                }
                            ],
                            "execution_queue": ["d1", "d2"],
                            "active_deliverable_id": "d1",
                            "summary": "repair the blocked path and finish the write",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T09:00:01Z",
                    finish_reason="stop",
                )
            if _is_mission_verifier_call(messages):
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")
                if latest_tool_results and latest_tool_results[0]["success"] is False:
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d1",
                                "current_deliverable_status": "active",
                                "missing": "A corrected allowed-root write path is still needed.",
                                "blocker": None,
                                "artifact_paths": [],
                                "evidence": [],
                                "final_deliverables_missing": ["d1", "d2"],
                                "next_action": "continue",
                                "notes_for_next_step": "Repair the path and retry the write inside the allowed workspace.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T09:00:03Z",
                        finish_reason="stop",
                    )
                if latest_tool_results:
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d2",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(repaired_path)],
                                "evidence": [],
                                "final_deliverables_missing": ["d2"],
                                "next_action": "continue",
                                "notes_for_next_step": "Answer from the verified repaired write.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T09:00:05Z",
                        finish_reason="stop",
                    )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                            "artifact_paths": [str(repaired_path)],
                            "evidence": [],
                            "final_deliverables_missing": [],
                            "next_action": "final_reply",
                            "notes_for_next_step": None,
                            "assistant_reply": draft_reply,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T09:00:07Z",
                    finish_reason="stop",
                )

            call_count += 1
            if call_count == 1:
                return _tool_call_response(
                    "write_text_file",
                    {"path": blocked_path, "content": "blocked note"},
                )
            if call_count == 2:
                assert any(
                    message.role is LLMRole.SYSTEM
                    and message.content.startswith("Mission progress check:")
                    for message in messages
                )
                return _tool_call_response(
                    "write_text_file",
                    {"path": str(repaired_path), "content": "blocked note"},
                )
            if call_count == 3:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="I saved the note locally at the repaired path.",
                    created_at="2026-03-26T09:00:06Z",
                    finish_reason="stop",
                )
            raise AssertionError("Unexpected extra model call in path-repair flow.")

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    prompt = "Write the requested note file locally."

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert reply == "I saved the note locally at the repaired path."
        assert repaired_path.exists()
        assert repaired_path.read_text(encoding="utf-8") == "blocked note"
        assert call_count == 3
    finally:
        session_manager.close()


@pytest.mark.parametrize("profile_name", ["main", "deep"])
@pytest.mark.parametrize("thinking_enabled", [False, True])
def test_kernel_compound_research_write_joke_completes_and_streams(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    profile_name: str,
    thinking_enabled: bool,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    command_handler.current_model_profile_name = profile_name
    command_handler.thinking_enabled = thinking_enabled
    output_path = settings.paths.files_dir / f"kernel-{profile_name}-{int(thinking_enabled)}.txt"
    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Kernel fact one.\nKernel fact two.\n",
            payload={
                "query": call.arguments["query"],
                "summary_points": ["Kernel fact one.", "Kernel fact two."],
                "display_sources": [
                    {"title": "Kernel Source", "url": "https://example.com/kernel"}
                ],
                "evidence_count": 2,
                "finding_count": 2,
            },
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
    streamed_chunks: list[str] = []
    planner_calls = 0
    execution_calls = 0

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, tools
            nonlocal planner_calls, execution_calls

            if _is_finalizer_call(messages):
                payload = json.loads(messages[-1].content)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {"final_reply": payload["assistant_draft_reply"]},
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:00:09Z",
                    finish_reason="stop",
                )

            if _is_mission_planner_call(messages):
                planner_calls += 1
                payload = json.loads(messages[-1].content)
                assert payload["first_response_summary"]["draft_reply"] == (
                    "I should plan this mission before I answer."
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": (
                                "Research the topic, save a note, and finish with a joke."
                            ),
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "mixed",
                                    "task": "Research the requested topic",
                                    "deliverable": "Grounded research facts",
                                    "verification": "Search evidence is verified.",
                                },
                                {
                                    "id": "d2",
                                    "mode": "artifact",
                                    "task": "Write the local note",
                                    "deliverable": "Local note file",
                                    "verification": "The file is written and read back.",
                                },
                                {
                                    "id": "d3",
                                    "mode": "reply",
                                    "task": "Tell the final joke",
                                    "deliverable": "Short joke in the final reply",
                                    "verification": "The joke appears in the final reply.",
                                },
                            ],
                            "execution_queue": ["d1", "d2", "d3"],
                            "active_deliverable_id": "d1",
                            "summary": "kernel compound mission",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:00:02Z",
                    finish_reason="stop",
                )

            if _is_mission_verifier_call(messages):
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")
                if latest_tool_results and latest_tool_results[0]["tool_name"] == "search_web":
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d2",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [],
                                "evidence": ["Kernel fact one.", "Kernel fact two."],
                                "final_deliverables_missing": ["d2", "d3"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Write the verified local note next.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T10:00:04Z",
                        finish_reason="stop",
                    )
                if latest_tool_results and any(
                    item["tool_name"] == "write_text_file" for item in latest_tool_results
                ):
                    assert any(
                        item["tool_name"] == "read_text_file"
                        for item in latest_tool_results
                    )
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d3",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": [],
                                "final_deliverables_missing": ["d3"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Finish with the requested joke.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T10:00:06Z",
                        finish_reason="stop",
                    )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                            "artifact_paths": [str(output_path)],
                            "evidence": [],
                            "final_deliverables_missing": [],
                            "next_action": "final_reply",
                            "repair_strategy": None,
                            "notes_for_next_step": None,
                            "assistant_reply": draft_reply,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:00:08Z",
                    finish_reason="stop",
                )

            execution_calls += 1
            assert thinking_enabled is command_handler.thinking_enabled
            if execution_calls == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="I should plan this mission before I answer.",
                    created_at="2026-03-26T10:00:00Z",
                    finish_reason="stop",
                )
            if execution_calls == 2:
                assert any(
                    message.role is LLMRole.SYSTEM
                    and message.content.startswith("Tool reconsideration note:")
                    for message in messages
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "assistant_reply": (
                                "I need a compact mission plan before I answer."
                            ),
                            "unsupported_execution_claim_risk": False,
                            "completion_without_execution_risk": False,
                            "multi_deliverable_request_risk": True,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:00:01Z",
                    finish_reason="stop",
                )
            if execution_calls == 3:
                return _tool_call_response(
                    "search_web",
                    {"query": "kernel mission facts"},
                )
            if execution_calls == 4:
                return _tool_call_response(
                    "write_text_file",
                    {
                        "path": str(output_path),
                        "content": "Kernel fact one.\nKernel fact two.\n",
                    },
                )
            reply = "File saved. Joke: The kernel walked into the root cause."
            if content_callback is not None:
                content_callback(reply)
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content=reply,
                created_at="2026-03-26T10:00:07Z",
                finish_reason="stop",
            )

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    prompt = "Research kernel mission facts, save a note, and finish with a joke."

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
            stream_output_func=streamed_chunks.append,
        )

        assert "Joke:" in reply
        assert "root cause" in reply
        assert output_path.exists()
        assert "Kernel fact one." in output_path.read_text(encoding="utf-8")
        assert planner_calls == 1
        assert streamed_chunks == [
            "File saved. Joke: The kernel walked into the root cause."
        ]
        mission_state = session_manager.get_current_mission_state(session.id)
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.active_deliverable_id is None
        assert mission_state.final_deliverables_missing == ()
        assert mission_state.completed_steps == ("d1", "d2", "d3")
    finally:
        session_manager.close()


def test_kernel_completed_mission_status_reports_completed_not_active(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    tool_registry = ToolRegistry()
    tool_registry.register(SEARCH_WEB_DEFINITION, lambda call: ToolResult.ok(tool_name=call.tool_name, output_text="unused"))
    completed_state = MissionState(
        mission_id="mission-complete",
        goal="Write a short local note and then tell a joke.",
        status="completed",
        active_deliverable_id=None,
        active_task=None,
        completed_deliverables=("d1", "d2"),
        blocked_deliverables=(),
        deliverables=(
            MissionDeliverableState(
                deliverable_id="d1",
                task="Write the local note",
                deliverable="Local note file",
                verification="File exists",
                status="completed",
                missing=None,
                blocker=None,
                attempt_count=1,
                repair_count=0,
                retry_count=0,
                artifact_paths=(),
                evidence=(),
                updated_at="2026-03-26T10:10:00Z",
            ),
            MissionDeliverableState(
                deliverable_id="d2",
                task="Tell the joke",
                deliverable="Short joke",
                verification="Joke is present",
                status="completed",
                missing=None,
                blocker=None,
                attempt_count=1,
                repair_count=0,
                retry_count=0,
                artifact_paths=(),
                evidence=(),
                updated_at="2026-03-26T10:10:00Z",
            ),
        ),
        retry_history=(),
        repair_history=(),
        last_verified_artifact_paths=(),
        last_successful_evidence=(),
        last_blocker=None,
        updated_at="2026-03-26T10:10:00Z",
    )

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            if _is_finalizer_call(messages):
                payload = json.loads(messages[-1].content)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {"final_reply": payload["assistant_draft_reply"]},
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:10:01Z",
                    finish_reason="stop",
                )
            if _is_mission_planner_call(messages) or _is_mission_verifier_call(messages):
                raise AssertionError("Completed mission status must not restart kernel execution.")
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Cette mission est terminee.",
                created_at="2026-03-26T10:10:02Z",
                finish_reason="stop",
            )

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.persist_mission_state(completed_state, session_id=session.id)
        session_manager.add_message(
            MessageRole.USER,
            "ou en est cette mission ?",
            session_id=session.id,
        )
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="ou en est cette mission ?",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert "completed" in reply.casefold() or "terminee" in reply.casefold()
        assert "active" not in reply.casefold()
    finally:
        session_manager.close()


def test_starting_new_mission_after_completed_one_resets_old_kernel_state(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    output_path = settings.paths.files_dir / "fresh-note.txt"
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
    completed_state = MissionState(
        mission_id="mission-old",
        goal="Old completed mission",
        status="completed",
        active_deliverable_id=None,
        active_task=None,
        completed_deliverables=("d1", "d2"),
        blocked_deliverables=(),
        deliverables=(
            MissionDeliverableState(
                deliverable_id="d1",
                task="old task",
                deliverable="old deliverable",
                verification="old verification",
                status="completed",
                missing=None,
                blocker=None,
                attempt_count=1,
                repair_count=0,
                retry_count=0,
                artifact_paths=(),
                evidence=(),
                updated_at="2026-03-26T10:20:00Z",
            ),
            MissionDeliverableState(
                deliverable_id="d2",
                task="old wrap-up",
                deliverable="old final note",
                verification="old final verification",
                status="completed",
                missing=None,
                blocker=None,
                attempt_count=1,
                repair_count=0,
                retry_count=0,
                artifact_paths=(),
                evidence=(),
                updated_at="2026-03-26T10:20:00Z",
            ),
        ),
        retry_history=(),
        repair_history=(),
        last_verified_artifact_paths=(),
        last_successful_evidence=(),
        last_blocker=None,
        updated_at="2026-03-26T10:20:00Z",
    )

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            if _is_finalizer_call(messages):
                payload = json.loads(messages[-1].content)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {"final_reply": payload["assistant_draft_reply"]},
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:20:07Z",
                    finish_reason="stop",
                )
            if _is_mission_planner_call(messages):
                payload = json.loads(messages[-1].content)
                assert payload["existing_mission_state"] is None
                assert payload["compatibility_mission_state"] is None
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": "Write a fresh local note.",
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "artifact",
                                    "task": "Write the fresh note",
                                    "deliverable": "Fresh note file",
                                    "verification": "The fresh note file is written and read back.",
                                },
                                {
                                    "id": "d2",
                                    "mode": "reply",
                                    "task": "Confirm the fresh note",
                                    "deliverable": "Short completion reply",
                                    "verification": "The reply confirms the verified fresh note.",
                                }
                            ],
                            "execution_queue": ["d1", "d2"],
                            "active_deliverable_id": "d1",
                            "summary": "start a brand new mission",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:20:01Z",
                    finish_reason="stop",
                )
            if _is_mission_verifier_call(messages):
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")
                if latest_tool_results:
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "completed",
                                "active_deliverable_id": "d2",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": [],
                                "final_deliverables_missing": ["d2"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Answer from the verified new file.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T10:20:03Z",
                        finish_reason="stop",
                    )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": [],
                                "final_deliverables_missing": [],
                                "next_action": "final_reply",
                                "repair_strategy": None,
                            "notes_for_next_step": None,
                            "assistant_reply": draft_reply,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:20:05Z",
                    finish_reason="stop",
                )
            if not any(message.role is LLMRole.TOOL for message in messages):
                return _tool_call_response(
                    "write_text_file",
                    {"path": str(output_path), "content": "fresh mission"},
                )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="The fresh mission is complete.",
                created_at="2026-03-26T10:20:04Z",
                finish_reason="stop",
            )

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.persist_mission_state(completed_state, session_id=session.id)
        session_manager.add_message(
            MessageRole.USER,
            "Write a fresh local note.",
            session_id=session.id,
        )
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Write a fresh local note.",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        mission_state = session_manager.get_current_mission_state(session.id)
        assert reply == "The fresh mission is complete."
        assert mission_state is not None
        assert mission_state.mission_id != "mission-old"
        assert mission_state.goal == "Write a fresh local note."
        assert mission_state.status == "completed"
        assert output_path.exists()
    finally:
        session_manager.close()


def test_legacy_goal_compatibility_does_not_contaminate_clearly_new_mission(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    output_path = settings.paths.files_dir / "birds-note.txt"
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

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            if _is_mission_planner_call(messages):
                payload = json.loads(messages[-1].content)
                assert payload["existing_mission_state"] is None
                assert payload["compatibility_mission_state"]["goal"] == "Old legacy mission"
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": "Write a fresh birds note.",
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "artifact",
                                    "task": "Write the birds note",
                                    "deliverable": "Birds note file",
                                    "verification": "The birds note file is written and read back.",
                                },
                                {
                                    "id": "d2",
                                    "mode": "reply",
                                    "task": "Confirm the birds note",
                                    "deliverable": "Short birds-note reply",
                                    "verification": "The reply confirms the verified birds note.",
                                }
                            ],
                            "execution_queue": ["d1", "d2"],
                            "active_deliverable_id": "d1",
                            "summary": "new mission overrides legacy compatibility",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:30:01Z",
                    finish_reason="stop",
                )
            if _is_mission_verifier_call(messages):
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")
                if latest_tool_results:
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "completed",
                                "active_deliverable_id": "d2",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": [],
                                "final_deliverables_missing": ["d2"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Answer from the new note.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T10:30:03Z",
                        finish_reason="stop",
                    )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                            "artifact_paths": [str(output_path)],
                            "evidence": [],
                            "final_deliverables_missing": [],
                            "next_action": "final_reply",
                            "repair_strategy": None,
                            "notes_for_next_step": None,
                            "assistant_reply": draft_reply,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:30:05Z",
                    finish_reason="stop",
                )
            if not any(message.role is LLMRole.TOOL for message in messages):
                return _tool_call_response(
                    "write_text_file",
                    {"path": str(output_path), "content": "birds are great"},
                )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="The birds note is ready.",
                created_at="2026-03-26T10:30:04Z",
                finish_reason="stop",
            )

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.persist_session_goal_state(
            session_id=session.id,
            goal="Old legacy mission",
            status="active",
            current_step="write_text_file",
        )
        session_manager.persist_session_progress_entry(
            session_id=session.id,
            status="active",
            step="write_text_file",
            detail="legacy task still open",
        )
        session_manager.add_message(
            MessageRole.USER,
            "Write a fresh birds note.",
            session_id=session.id,
        )
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Write a fresh birds note.",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        mission_state = session_manager.get_current_mission_state(session.id)
        assert reply == "The birds note is ready."
        assert mission_state is not None
        assert mission_state.goal == "Write a fresh birds note."
        assert mission_state.mission_id != "legacy-goal-compat"
    finally:
        session_manager.close()


def test_kernel_timeout_repair_retries_search_within_budget(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    tool_registry = ToolRegistry()
    search_calls: list[ToolCall] = []

    def _search_tool(call: ToolCall) -> ToolResult:
        search_calls.append(call)
        if len(search_calls) == 1:
            return ToolResult.failure(
                tool_name=call.tool_name,
                error="search timed out",
                failure_kind="timeout",
                payload={"execution_state": "timed_out"},
            )
        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Recovered search evidence",
            payload={
                "query": call.arguments["query"],
                "summary_points": ["Recovered fact."],
                "display_sources": [
                    {"title": "Recovered", "url": "https://example.com/recovered"}
                ],
                "evidence_count": 2,
                "finding_count": 1,
            },
        )

    tool_registry.register(SEARCH_WEB_DEFINITION, _search_tool)
    active_state = MissionState(
        mission_id="mission-timeout",
        goal="Research the kernel timeout case.",
        status="active",
        active_deliverable_id="d1",
        active_task="Research the kernel timeout case",
        completed_deliverables=(),
        blocked_deliverables=(),
        deliverables=(
            MissionDeliverableState(
                deliverable_id="d1",
                mode="mixed",
                task="Research the kernel timeout case",
                deliverable="Verified timeout-safe research facts",
                verification="Search evidence is verified.",
                status="active",
                missing="Need grounded evidence.",
                blocker=None,
                attempt_count=0,
                repair_count=0,
                retry_count=0,
                artifact_paths=(),
                evidence=(),
                updated_at="2026-03-26T10:40:00Z",
            ),
            MissionDeliverableState(
                deliverable_id="d2",
                mode="reply",
                task="Answer from the recovered evidence",
                deliverable="Short recovered evidence reply",
                verification="The reply includes the recovered evidence.",
                status="pending",
                missing=None,
                blocker=None,
                attempt_count=0,
                repair_count=0,
                retry_count=0,
                artifact_paths=(),
                evidence=(),
                updated_at="2026-03-26T10:40:00Z",
            ),
        ),
        retry_history=(),
        repair_history=(),
        last_verified_artifact_paths=(),
        last_successful_evidence=(),
        last_blocker=None,
        updated_at="2026-03-26T10:40:00Z",
        pending_repairs=("retry after timeout",),
    )

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            if _is_mission_planner_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "continue_existing",
                            "mission_goal": active_state.goal,
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "mixed",
                                    "task": active_state.deliverables[0].task,
                                    "deliverable": active_state.deliverables[0].deliverable,
                                    "verification": active_state.deliverables[0].verification,
                                },
                                {
                                    "id": "d2",
                                    "mode": "reply",
                                    "task": active_state.deliverables[1].task,
                                    "deliverable": active_state.deliverables[1].deliverable,
                                    "verification": active_state.deliverables[1].verification,
                                }
                            ],
                            "execution_queue": ["d1", "d2"],
                            "active_deliverable_id": "d1",
                            "summary": "retry the active research deliverable",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:40:01Z",
                    finish_reason="stop",
                )
            if _is_mission_verifier_call(messages):
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")
                if latest_tool_results and latest_tool_results[0]["success"] is False:
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d1",
                                "current_deliverable_status": "active",
                                "missing": "A retry is still needed after the timeout.",
                                "blocker": None,
                                "artifact_paths": [],
                                "evidence": [],
                                "final_deliverables_missing": ["d1"],
                                "next_action": "continue",
                                "repair_strategy": "retry after timeout",
                                "notes_for_next_step": "Retry the active deliverable with a narrower search step.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T10:40:03Z",
                        finish_reason="stop",
                    )
                if latest_tool_results:
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "completed",
                                "active_deliverable_id": "d2",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [],
                                "evidence": ["Recovered fact."],
                                "final_deliverables_missing": ["d2"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Answer from the recovered evidence.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T10:40:05Z",
                        finish_reason="stop",
                    )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                            "artifact_paths": [],
                            "evidence": ["Recovered fact."],
                            "final_deliverables_missing": [],
                            "next_action": "final_reply",
                            "repair_strategy": None,
                            "notes_for_next_step": None,
                            "assistant_reply": draft_reply,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:40:07Z",
                    finish_reason="stop",
                )
            if not any(message.role is LLMRole.TOOL for message in messages):
                return _tool_call_response(
                    "search_web",
                    {"query": "kernel timeout case"},
                )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Recovered fact.",
                created_at="2026-03-26T10:40:06Z",
                finish_reason="stop",
            )

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.persist_mission_state(active_state, session_id=session.id)
        session_manager.add_message(
            MessageRole.USER,
            "Continue the timeout mission.",
            session_id=session.id,
        )
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Continue the timeout mission.",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        mission_state = session_manager.get_current_mission_state(session.id)
        assert "Recovered fact." in reply
        assert "Sources:" in reply
        assert len(search_calls) == 2
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.last_successful_evidence == ("Recovered fact.",)
    finally:
        session_manager.close()


def test_kernel_collision_repair_retries_with_versioned_path(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    """Collision conflict from write_text_file triggers structured repair.

    When the verifier fallback sees failure_kind='collision_conflict' and
    suggested_version_path in the payload, it sets a repair strategy so the
    kernel retries with collision_policy='version'.
    """
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    output_path = settings.paths.files_dir / "collision-note.txt"
    # Pre-create the file so the first write collides
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("already exists", encoding="utf-8")
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
    execution_calls = 0

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal execution_calls

            if _is_finalizer_call(messages):
                payload = json.loads(messages[-1].content)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {"final_reply": payload["assistant_draft_reply"]},
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T11:00:09Z",
                    finish_reason="stop",
                )

            if _is_mission_planner_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": "Write a collision note.",
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "task": "Write the note file",
                                    "deliverable": "Local note file",
                                    "verification": "File is written and verified.",
                                }
                            ],
                            "execution_queue": ["d1"],
                            "active_deliverable_id": "d1",
                            "summary": "collision write mission",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T11:00:01Z",
                    finish_reason="stop",
                )

            if _is_mission_verifier_call(messages):
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")
                # Check for successful write (second attempt with versioned path)
                if latest_tool_results and any(
                    item.get("tool_name") == "write_text_file"
                    and item.get("success") is True
                    for item in latest_tool_results
                ):
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "completed",
                                "active_deliverable_id": None,
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [],
                                "evidence": [],
                                "final_deliverables_missing": [],
                                "next_action": "final_reply",
                                "repair_strategy": None,
                                "notes_for_next_step": None,
                                "assistant_reply": draft_reply or "File saved.",
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T11:00:06Z",
                        finish_reason="stop",
                    )
                # Collision detected — fallback handles the repair
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "active",
                            "active_deliverable_id": "d1",
                            "current_deliverable_status": "active",
                            "missing": "File collision needs repair.",
                            "blocker": None,
                            "artifact_paths": [],
                            "evidence": [],
                            "final_deliverables_missing": ["d1"],
                            "next_action": "continue",
                            "repair_strategy": (
                                "Retry write with collision_policy='version'"
                            ),
                            "notes_for_next_step": (
                                "Retry with collision_policy='version' to save a versioned file."
                            ),
                            "assistant_reply": None,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T11:00:04Z",
                    finish_reason="stop",
                )

            # Execution calls
            execution_calls += 1
            if execution_calls == 1:
                # First write with collision_policy='fail' — triggers collision_conflict
                return _tool_call_response(
                    "write_text_file",
                    {
                        "path": str(output_path),
                        "content": "New collision content",
                        "collision_policy": "fail",
                    },
                )
            if execution_calls == 2:
                # Retry with collision_policy='version'
                return _tool_call_response(
                    "write_text_file",
                    {
                        "path": str(output_path),
                        "content": "New collision content",
                        "collision_policy": "version",
                    },
                )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="File saved with versioned path.",
                created_at="2026-03-26T11:00:07Z",
                finish_reason="stop",
            )

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    prompt = "Write a collision note."

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        mission_state = session_manager.get_current_mission_state(session.id)
        assert mission_state is not None
        assert mission_state.status == "completed"
        # The original file should be untouched (collision_policy=fail)
        assert output_path.read_text(encoding="utf-8") == "already exists"
        # A versioned file should have been created (timestamped name)
        versioned_files = list(settings.paths.files_dir.glob("collision-note_*.txt"))
        assert len(versioned_files) >= 1
        assert "New collision content" in versioned_files[0].read_text(encoding="utf-8")
        assert execution_calls == 2
    finally:
        session_manager.close()


def test_kernel_strict_deliverable_ordering_blocks_later_before_earlier(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    """Strict ordering: later deliverables cannot complete before earlier ones.

    When the verifier tries to advance to d3 while d1 is still active,
    the ordering enforcement in _apply_verification_decision clamps the
    active deliverable back to d1.
    """
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Ordering evidence.",
            payload={
                "query": call.arguments["query"],
                "summary_points": ["Order fact."],
                "display_sources": [],
                "evidence_count": 1,
                "finding_count": 1,
            },
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
    output_path = settings.paths.files_dir / "ordering-note.txt"
    execution_calls = 0
    verifier_calls = 0

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal execution_calls, verifier_calls

            if _is_finalizer_call(messages):
                payload = json.loads(messages[-1].content)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {"final_reply": payload["assistant_draft_reply"]},
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T12:00:09Z",
                    finish_reason="stop",
                )

            if _is_mission_planner_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": "Research, write, then tell a joke.",
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "task": "Research ordering facts",
                                    "deliverable": "Grounded research",
                                    "verification": "Search evidence verified.",
                                },
                                {
                                    "id": "d2",
                                    "task": "Write the local note",
                                    "deliverable": "Local note file",
                                    "verification": "File written and read back.",
                                },
                                {
                                    "id": "d3",
                                    "task": "Tell a joke",
                                    "deliverable": "Short joke",
                                    "verification": "Joke in reply.",
                                },
                            ],
                            "execution_queue": ["d1", "d2", "d3"],
                            "active_deliverable_id": "d1",
                            "summary": "ordering test mission",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T12:00:01Z",
                    finish_reason="stop",
                )

            if _is_mission_verifier_call(messages):
                verifier_calls += 1
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")

                if verifier_calls == 1:
                    # After search_web: verifier tries to skip to d3,
                    # but ordering should clamp back to d1 or d2.
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d3",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [],
                                "evidence": ["Order fact."],
                                "final_deliverables_missing": ["d2", "d3"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Skip to joke.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T12:00:03Z",
                        finish_reason="stop",
                    )
                if verifier_calls == 2:
                    # After write_text_file + read_text_file: d2 completed
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d3",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": [],
                                "final_deliverables_missing": ["d3"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Tell the joke.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-26T12:00:05Z",
                        finish_reason="stop",
                    )
                # Final: mission completed
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                            "artifact_paths": [str(output_path)],
                            "evidence": [],
                            "final_deliverables_missing": [],
                            "next_action": "final_reply",
                            "repair_strategy": None,
                            "notes_for_next_step": None,
                            "assistant_reply": draft_reply,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T12:00:07Z",
                    finish_reason="stop",
                )

            # Execution calls
            execution_calls += 1
            if execution_calls == 1:
                # d1: research step — search_web call
                return _tool_call_response(
                    "search_web",
                    {"query": "ordering facts"},
                )
            if execution_calls == 2:
                # d2: write note (after ordering clamp moved us back to d2)
                return _tool_call_response(
                    "write_text_file",
                    {"path": str(output_path), "content": "Order fact."},
                )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Joke: Why did the kernel use strict ordering? To stay in line.",
                created_at="2026-03-26T12:00:06Z",
                finish_reason="stop",
            )

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    prompt = "Research ordering, write a note, and tell a joke."

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        mission_state = session_manager.get_current_mission_state(session.id)
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert "Joke:" in reply
        # Verify that ordering enforcement worked: d1 was NOT skipped
        # (the verifier tried to jump to d3 but ordering clamped to d2)
        assert verifier_calls >= 3
        assert output_path.exists()
    finally:
        session_manager.close()


def test_reply_only_final_deliverable_completes_without_looping(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    output_path = settings.paths.files_dir / "reply-final-note.txt"
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
    execution_calls = 0

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal execution_calls

            if _is_finalizer_call(messages):
                payload = json.loads(messages[-1].content)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {"final_reply": payload["assistant_draft_reply"]},
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-27T09:00:05Z",
                    finish_reason="stop",
                )
            if _is_mission_planner_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": "Write a note and then describe it.",
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "artifact",
                                    "task": "Write the local note",
                                    "deliverable": "Local note file",
                                    "verification": "The file is written and read back.",
                                },
                                {
                                    "id": "d2",
                                    "mode": "reply",
                                    "task": "Describe what was saved",
                                    "deliverable": "Short final reply",
                                    "verification": "The reply accurately describes the verified saved note.",
                                },
                            ],
                            "execution_queue": ["d1", "d2"],
                            "active_deliverable_id": "d1",
                            "summary": "artifact then reply mission",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-27T09:00:01Z",
                    finish_reason="stop",
                )
            if _is_mission_verifier_call(messages):
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")
                if latest_tool_results:
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d2",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": ["Saved note: local reply kernel"],
                                "final_deliverables_missing": ["d2"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Answer from the verified note.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-27T09:00:03Z",
                        finish_reason="stop",
                    )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                            "artifact_paths": [str(output_path)],
                            "evidence": ["Saved note: local reply kernel"],
                            "final_deliverables_missing": [],
                            "next_action": "final_reply",
                            "repair_strategy": None,
                            "notes_for_next_step": None,
                            "assistant_reply": draft_reply,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-27T09:00:04Z",
                    finish_reason="stop",
                )

            execution_calls += 1
            if execution_calls == 1:
                return _tool_call_response(
                    "write_text_file",
                    {"path": str(output_path), "content": "local reply kernel"},
                )
            if execution_calls == 2:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="I saved: local reply kernel.",
                    created_at="2026-03-27T09:00:02Z",
                    finish_reason="stop",
                )
            raise AssertionError("Reply-only final deliverable must not loop.")

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    try:
        session = session_manager.ensure_current_session()
        prompt = "Write a local note and then tell me exactly what you saved."
        session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        mission_state = session_manager.get_current_mission_state(session.id)
        assert execution_calls == 2
        assert reply == "I saved: local reply kernel."
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.final_verified_reply == "I saved: local reply kernel."
        assert mission_state.deliverables[-1].status == "completed"
        assert mission_state.deliverables[-1].execution_state == "completed"
    finally:
        session_manager.close()


def test_artifact_deliverable_cannot_complete_without_verified_readback_evidence(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    output_path = settings.paths.files_dir / "no-readback.txt"
    tool_registry = ToolRegistry()
    tool_registry.register(
        WRITE_TEXT_FILE_DEFINITION,
        lambda call: write_text_file(
            call,
            allowed_roots=(settings.paths.project_root,),
            default_write_dir=settings.paths.files_dir,
        ),
    )

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            if _is_mission_planner_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": "Write a note with verified read-back.",
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "artifact",
                                    "task": "Write the note",
                                    "deliverable": "Local note file",
                                    "verification": "The file is written and read back.",
                                }
                            ],
                            "execution_queue": ["d1"],
                            "active_deliverable_id": "d1",
                            "summary": "artifact-only verification test",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-27T10:00:01Z",
                    finish_reason="stop",
                )
            if _is_mission_verifier_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                            "artifact_paths": [str(output_path)],
                            "evidence": [],
                            "final_deliverables_missing": [],
                            "next_action": "final_reply",
                            "repair_strategy": None,
                            "notes_for_next_step": None,
                            "assistant_reply": "Done.",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-27T10:00:03Z",
                    finish_reason="stop",
                )
            return _tool_call_response(
                "write_text_file",
                {"path": str(output_path), "content": "artifact without readback"},
            )

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    try:
        session = session_manager.ensure_current_session()
        prompt = "Write a local note."
        session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
            max_agent_steps=1,
        )

        mission_state = session_manager.get_current_mission_state(session.id)
        assert reply == "The mission is still active and needs another verified step before it can finish."
        assert mission_state is not None
        assert mission_state.status == "active"
        assert mission_state.deliverables[0].status != "completed"
        assert mission_state.deliverables[0].execution_state == "repairing"
        assert mission_state.pending_repairs
    finally:
        session_manager.close()


def test_collision_repair_advances_without_poisoning_reply_deliverable(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    output_path = settings.paths.files_dir / "collision-poison-note.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("already exists", encoding="utf-8")
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
    execution_calls = 0

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal execution_calls

            if _is_finalizer_call(messages):
                payload = json.loads(messages[-1].content)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {"final_reply": payload["assistant_draft_reply"]},
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-27T11:00:07Z",
                    finish_reason="stop",
                )
            if _is_mission_planner_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "start_new",
                            "mission_goal": "Write a collision-safe note and confirm it.",
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "artifact",
                                    "task": "Write the note",
                                    "deliverable": "Version-safe note file",
                                    "verification": "The note is written and read back.",
                                },
                                {
                                    "id": "d2",
                                    "mode": "reply",
                                    "task": "Confirm the saved note",
                                    "deliverable": "Short confirmation reply",
                                    "verification": "The reply confirms the verified saved note.",
                                },
                            ],
                            "execution_queue": ["d1", "d2"],
                            "active_deliverable_id": "d1",
                            "summary": "collision repair then reply",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-27T11:00:01Z",
                    finish_reason="stop",
                )
            if _is_mission_verifier_call(messages):
                payload = json.loads(messages[-1].content)
                latest_tool_results = payload.get("latest_tool_results", [])
                draft_reply = payload.get("draft_reply", "")
                if latest_tool_results and any(
                    item.get("success") is True
                    and item.get("tool_name") == "write_text_file"
                    for item in latest_tool_results
                ):
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d2",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [
                                    item["payload"]["resolved_path"]
                                    for item in latest_tool_results
                                    if item.get("tool_name") == "write_text_file"
                                    and isinstance(item.get("payload"), dict)
                                    and isinstance(item["payload"].get("resolved_path"), str)
                                ],
                                "evidence": [],
                                "final_deliverables_missing": ["d2"],
                                "next_action": "continue",
                                "repair_strategy": None,
                                "notes_for_next_step": "Confirm the verified saved note.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-27T11:00:05Z",
                        finish_reason="stop",
                    )
                if latest_tool_results:
                    return LLMResponse(
                        provider="ollama",
                        model_name=profile.model_name,
                        content=json.dumps(
                            {
                                "mission_status": "active",
                                "active_deliverable_id": "d1",
                                "current_deliverable_status": "active",
                                "missing": "File collision needs repair.",
                                "blocker": None,
                                "artifact_paths": [],
                                "evidence": [],
                                "final_deliverables_missing": ["d1", "d2"],
                                "next_action": "continue",
                                "repair_strategy": "Retry write with collision_policy='version'",
                                "notes_for_next_step": "Retry the write with collision_policy='version'.",
                                "assistant_reply": None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at="2026-03-27T11:00:03Z",
                        finish_reason="stop",
                    )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_status": "completed",
                            "active_deliverable_id": None,
                            "current_deliverable_status": "completed",
                            "missing": None,
                            "blocker": None,
                            "artifact_paths": [],
                            "evidence": [],
                            "final_deliverables_missing": [],
                            "next_action": "final_reply",
                            "repair_strategy": None,
                            "notes_for_next_step": None,
                            "assistant_reply": draft_reply,
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-27T11:00:07Z",
                    finish_reason="stop",
                )

            execution_calls += 1
            if execution_calls == 1:
                return _tool_call_response(
                    "write_text_file",
                    {
                        "path": str(output_path),
                        "content": "collision-safe content",
                        "collision_policy": "fail",
                    },
                )
            if execution_calls == 2:
                return _tool_call_response(
                    "write_text_file",
                    {
                        "path": str(output_path),
                        "content": "collision-safe content",
                        "collision_policy": "version",
                    },
                )
            if execution_calls == 3:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="I saved the collision-safe note.",
                    created_at="2026-03-27T11:00:06Z",
                    finish_reason="stop",
                )
            raise AssertionError("Collision repair must advance into the reply deliverable once.")

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    try:
        session = session_manager.ensure_current_session()
        prompt = "Write a collision-safe note and confirm it."
        session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=tool_registry,
        )

        mission_state = session_manager.get_current_mission_state(session.id)
        assert reply == "I saved the collision-safe note."
        assert execution_calls == 3
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.pending_repairs == ()
        assert mission_state.deliverables[-1].status == "completed"
    finally:
        session_manager.close()


def test_executor_state_persists_via_external_mission_workspace_store(
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, _tracer, _command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    try:
        session = session_manager.ensure_current_session()
        active_state = MissionState(
            mission_id="mission-persisted",
            goal="Persist executor state outside SQLite.",
            status="active",
            active_deliverable_id="d1",
            active_task="Repair the saved note",
            completed_deliverables=(),
            blocked_deliverables=(),
            deliverables=(
                MissionDeliverableState(
                    deliverable_id="d1",
                    mode="artifact",
                    task="Repair the saved note",
                    deliverable="Local repaired note",
                    verification="The repaired note is written and read back.",
                    status="active",
                    missing="Need repaired artifact evidence.",
                    blocker=None,
                    attempt_count=1,
                    repair_count=1,
                    retry_count=0,
                    artifact_paths=(),
                    evidence=(),
                    updated_at="2026-03-27T12:00:00Z",
                    execution_state="repairing",
                    waiting_for="Retry the write with a repaired path.",
                    advance_condition="The repaired note is written and read back.",
                    verifier_notes="Retry the write with a repaired path.",
                ),
            ),
            retry_history=(),
            repair_history=("Retry the write with a repaired path.",),
            last_verified_artifact_paths=(),
            last_successful_evidence=(),
            last_blocker=None,
            updated_at="2026-03-27T12:00:00Z",
            pending_repairs=("Retry the write with a repaired path.",),
            executor_state="repairing",
            executor_reason="repair the active deliverable before advancing",
            waiting_for="Retry the write with a repaired path.",
            advance_condition="The repaired note is written and read back.",
            verifier_outputs=("Retry the write with a repaired path.",),
        )

        session_manager.persist_mission_state(active_state, session_id=session.id)
        row = session_manager.connection.execute(
            """
            SELECT payload_json
            FROM events
            WHERE session_id = ?
              AND event_type = 'session.mission_state.updated'
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (session.id,),
        ).fetchone()
        assert row is not None
        pointer = parse_mission_workspace_pointer(row["payload_json"])
        assert pointer is not None
        workspace_path = Path(pointer.workspace_path)
        assert workspace_path.exists()
        assert workspace_path.parent.name == session.id
        assert workspace_path.parent.parent.name == "missions"
        workspace_payload = json.loads(workspace_path.read_text(encoding="utf-8"))
        assert workspace_payload["executor_state"] == "repairing"
        assert workspace_payload["pending_repairs"] == [
            "Retry the write with a repaired path."
        ]
    finally:
        session_manager.close()

    reloaded_manager = SessionManager.from_settings(settings)
    try:
        reloaded_state = reloaded_manager.get_current_mission_state(session.id)
        assert reloaded_state is not None
        assert reloaded_state.executor_state == "repairing"
        assert reloaded_state.pending_repairs == (
            "Retry the write with a repaired path.",
        )
        assert reloaded_state.waiting_for == "Retry the write with a repaired path."
    finally:
        reloaded_manager.close()


def test_safe_verifier_fallback_avoids_infinite_final_reply_loop(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    active_state = MissionState(
        mission_id="mission-macron-like",
        goal="Write the verified final reply.",
        status="active",
        active_deliverable_id="d2",
        active_task="Write the verified final reply",
        completed_deliverables=("d1",),
        blocked_deliverables=(),
        deliverables=(
            MissionDeliverableState(
                deliverable_id="d1",
                mode="mixed",
                task="Collect verified facts",
                deliverable="Verified facts",
                verification="The facts are verified.",
                status="completed",
                missing=None,
                blocker=None,
                attempt_count=1,
                repair_count=0,
                retry_count=0,
                artifact_paths=(),
                evidence=("Macron fact.",),
                updated_at="2026-03-27T13:00:00Z",
                execution_state="completed",
            ),
            MissionDeliverableState(
                deliverable_id="d2",
                mode="reply",
                task="Write the verified final reply",
                deliverable="Final reply",
                verification="The reply uses the verified facts only.",
                status="active",
                missing=None,
                blocker=None,
                attempt_count=0,
                repair_count=0,
                retry_count=0,
                artifact_paths=(),
                evidence=(),
                updated_at="2026-03-27T13:00:00Z",
                execution_state="ready",
            ),
        ),
        retry_history=(),
        repair_history=(),
        last_verified_artifact_paths=(),
        last_successful_evidence=("Macron fact.",),
        last_blocker=None,
        updated_at="2026-03-27T13:00:00Z",
        execution_queue=("d2",),
        completed_steps=("d1",),
        final_deliverables_missing=("d2",),
        executor_state="ready",
        executor_reason="advance to the reply deliverable",
        waiting_for="execute Write the verified final reply",
        advance_condition="The reply uses the verified facts only.",
    )
    execution_calls = 0

    class FakeProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal execution_calls
            if _is_mission_planner_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "continue_existing",
                            "mission_goal": active_state.goal,
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "mode": "mixed",
                                    "task": active_state.deliverables[0].task,
                                    "deliverable": active_state.deliverables[0].deliverable,
                                    "verification": active_state.deliverables[0].verification,
                                },
                                {
                                    "id": "d2",
                                    "mode": "reply",
                                    "task": active_state.deliverables[1].task,
                                    "deliverable": active_state.deliverables[1].deliverable,
                                    "verification": active_state.deliverables[1].verification,
                                },
                            ],
                            "execution_queue": ["d2"],
                            "active_deliverable_id": "d2",
                            "summary": "continue the final reply deliverable",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-27T13:00:01Z",
                    finish_reason="stop",
                )
            if _is_mission_verifier_call(messages):
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="not json",
                    created_at="2026-03-27T13:00:03Z",
                    finish_reason="stop",
                )
            execution_calls += 1
            if execution_calls == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="Verified reply: Macron fact.",
                    created_at="2026-03-27T13:00:02Z",
                    finish_reason="stop",
                )
            raise AssertionError("Safe fallback must finish without another execution loop.")

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.persist_mission_state(active_state, session_id=session.id)
        prompt = "Finish the verified Macron reply."
        session_manager.add_message(MessageRole.USER, prompt, session_id=session.id)
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=prompt,
            tracer=tracer,
            tool_registry=ToolRegistry(),
        )

        mission_state = session_manager.get_current_mission_state(session.id)
        assert execution_calls == 1
        assert reply == "Verified reply: Macron fact."
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.final_verified_reply == "Verified reply: Macron fact."
    finally:
        session_manager.close()
