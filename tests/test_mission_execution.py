from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import unclaw.core.agent_loop as _agent_loop
from unclaw.core.command_handler import CommandHandler
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.file_tools import WRITE_TEXT_FILE_DEFINITION, write_text_file
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
    _agent_loop._continuation_check_enabled = True
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
            del timeout_seconds, content_callback, tools
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

            call_count += 1
            assert thinking_enabled is expected_thinking_enabled
            if call_count == 1:
                return _tool_call_response("search_web", {"query": "mission test"})
            if call_count == 2:
                tool_messages = [
                    message.content for message in messages if message.role is LLMRole.TOOL
                ]
                assert any("Mission fact one." in message for message in tool_messages)
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="I found grounded details for the note.",
                    created_at="2026-03-26T09:00:01Z",
                    finish_reason="stop",
                )
            if call_count == 3:
                assert any(
                    message.role is LLMRole.SYSTEM
                    and message.content.startswith("Mission continuation check:")
                    for message in messages
                )
                return _tool_call_response(
                    "write_text_file",
                    {"path": str(output_path), "content": "Mission fact one.\nMission fact two."},
                )
            if call_count == 4:
                assert any(
                    message.role is LLMRole.SYSTEM
                    and message.content.startswith("Pre-write grounding check:")
                    for message in messages
                )
                return _tool_call_response(
                    "write_text_file",
                    {"path": str(output_path), "content": "Mission fact one.\nMission fact two."},
                )
            if call_count == 5:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=(
                        "I saved the research note locally. "
                        "Joke: Why did the local agent cross the filesystem? "
                        "To get to the root cause."
                    ),
                    created_at="2026-03-26T09:00:02Z",
                    finish_reason="stop",
                )
            if call_count == 6:
                assert any(
                    message.role is LLMRole.SYSTEM
                    and message.content.startswith("Mission continuation check:")
                    for message in messages
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=(
                        "I saved the research note locally. "
                        "Joke: Why did the local agent cross the filesystem? "
                        "To get to the root cause."
                    ),
                    created_at="2026-03-26T09:00:02Z",
                    finish_reason="stop",
                )
            raise AssertionError("Unexpected extra model call in compound mission flow.")

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
        assert call_count == 6
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
                                    "task": "write_text_file",
                                    "deliverable": "Local note file",
                                    "verification": "A successful local file write is verified.",
                                }
                            ],
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
                                "active_deliverable_id": "d1",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": [],
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
                                    "task": "write_text_file",
                                    "deliverable": "Allowed local note file",
                                    "verification": "A successful allowed-root local file write is verified.",
                                }
                            ],
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
                                "active_deliverable_id": "d1",
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(repaired_path)],
                                "evidence": [],
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
