from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import unclaw.core.agent_loop as _agent_loop
from unclaw.core.command_handler import CommandHandler
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
                                    "task": "Research the requested topic",
                                    "deliverable": "Grounded research facts",
                                    "verification": "Search evidence is verified.",
                                },
                                {
                                    "id": "d2",
                                    "task": "Write the local note",
                                    "deliverable": "Local note file",
                                    "verification": "The file is written and read back.",
                                },
                                {
                                    "id": "d3",
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
    planner_calls = 0

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
            nonlocal planner_calls
            if _is_mission_planner_call(messages):
                planner_calls += 1
                payload = json.loads(messages[-1].content)
                assert payload["existing_mission_state"]["status"] == "completed"
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=json.dumps(
                        {
                            "mission_action": "continue_existing",
                            "mission_goal": completed_state.goal,
                            "deliverables": [
                                {
                                    "id": "d1",
                                    "task": "Write the local note",
                                    "deliverable": "Local note file",
                                    "verification": "File exists",
                                },
                                {
                                    "id": "d2",
                                    "task": "Tell the joke",
                                    "deliverable": "Short joke",
                                    "verification": "Joke is present",
                                },
                            ],
                            "execution_queue": [],
                            "active_deliverable_id": None,
                            "summary": "completed mission status check",
                        },
                        ensure_ascii=False,
                    ),
                    created_at="2026-03-26T10:10:01Z",
                    finish_reason="stop",
                )
            if _is_mission_verifier_call(messages):
                draft_reply = json.loads(messages[-1].content)["draft_reply"]
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
                    created_at="2026-03-26T10:10:03Z",
                    finish_reason="stop",
                )
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

        assert planner_calls == 1
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
            if _is_mission_planner_call(messages):
                payload = json.loads(messages[-1].content)
                assert payload["existing_mission_state"]["goal"] == "Old completed mission"
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
                                    "task": "Write the fresh note",
                                    "deliverable": "Fresh note file",
                                    "verification": "The fresh note file is written and read back.",
                                }
                            ],
                            "execution_queue": ["d1"],
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
                                "active_deliverable_id": None,
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": [],
                                "final_deliverables_missing": [],
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
                                    "task": "Write the birds note",
                                    "deliverable": "Birds note file",
                                    "verification": "The birds note file is written and read back.",
                                }
                            ],
                            "execution_queue": ["d1"],
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
                                "active_deliverable_id": None,
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [str(output_path)],
                                "evidence": [],
                                "final_deliverables_missing": [],
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
                                    "task": active_state.deliverables[0].task,
                                    "deliverable": active_state.deliverables[0].deliverable,
                                    "verification": active_state.deliverables[0].verification,
                                }
                            ],
                            "execution_queue": ["d1"],
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
                                "active_deliverable_id": None,
                                "current_deliverable_status": "completed",
                                "missing": None,
                                "blocker": None,
                                "artifact_paths": [],
                                "evidence": ["Recovered fact."],
                                "final_deliverables_missing": [],
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
