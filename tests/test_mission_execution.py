from __future__ import annotations

import json
from pathlib import Path
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


def test_compound_research_write_and_joke_finishes_from_one_single_agent_action(
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
    output_path = settings.paths.files_dir / "inoxtag-bio.txt"
    observed_tool_calls: list[str] = []

    tool_registry = ToolRegistry()
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or ToolResult.ok(
                tool_name=call.tool_name,
                output_text="Marine safe grounding",
                payload={
                    "query": call.arguments["query"],
                    "grounding_note": "Inoxtag is a French creator and adventurer.",
                },
            )
        ),
    )
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or ToolResult.ok(
                tool_name=call.tool_name,
                output_text="Inoxtag research complete",
                payload={
                    "query": call.arguments["query"],
                    "summary_points": [
                        "Inoxtag is a French content creator.",
                        "He is known for ambitious challenge videos.",
                    ],
                    "display_sources": [
                        {"title": "Example", "url": "https://example.com/inoxtag"}
                    ],
                },
            )
        ),
    )
    tool_registry.register(
        WRITE_TEXT_FILE_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or write_text_file(
                call,
                allowed_roots=(settings.paths.project_root,),
                default_write_dir=settings.paths.files_dir,
            )
        ),
    )
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or read_text_file(
                call,
                read_allowed_roots=(settings.paths.project_root,),
                default_read_dir=settings.paths.files_dir,
            )
        ),
    )

    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "start_new",
                "mission_goal": (
                    "Research Inoxtag, write the result to a local text file, and tell a banana joke."
                ),
                "reasoning_summary": "Ground first, research fully, verify the file, then answer with the joke.",
                "tasks": [
                    {
                        "id": "t0",
                        "title": "Ground the target person",
                        "kind": "web_grounding",
                        "status": "active",
                        "depends_on": [],
                        "required_evidence": ["fast_grounding"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t1",
                        "title": "Do the full research",
                        "kind": "web_research",
                        "status": "pending",
                        "depends_on": ["t0"],
                        "required_evidence": ["full_web_research"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t2",
                        "title": "Write and verify the file",
                        "kind": "file_write",
                        "status": "pending",
                        "depends_on": ["t1"],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [str(output_path)],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t3",
                        "title": "Tell the banana joke",
                        "kind": "reply",
                        "status": "pending",
                        "depends_on": ["t2"],
                        "required_evidence": ["reply_emitted"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                ],
                "active_task_id": "t0",
                "tool_calls": [
                    {
                        "task_id": "t0",
                        "tool_name": "fast_web_search",
                        "arguments": {"query": "Inoxtag"},
                    },
                    {
                        "task_id": "t1",
                        "tool_name": "search_web",
                        "arguments": {"query": "Inoxtag"},
                    },
                    {
                        "task_id": "t2",
                        "tool_name": "write_text_file",
                        "arguments": {
                            "path": str(output_path),
                            "content": (
                                "Inoxtag is a French content creator known for ambitious challenge videos."
                            ),
                        },
                    },
                    {
                        "task_id": "t2",
                        "tool_name": "read_text_file",
                        "arguments": {"path": str(output_path)},
                    },
                ],
                "reply_to_user": "Le fichier est écrit. Et la blague: pourquoi les bananes ne sont jamais seules ? Parce qu'elles vont en régime groupé.",
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
            prompt=(
                "Qui est Inoxtag ? fais une recherche complète sur lui, écrit le résultat "
                "dans un fichier texte et fais moi une blague sur les bananes"
            ),
        )
        mission_state = session_manager.get_current_mission_state()

        assert "bananes" in reply
        assert fake_provider.call_count() == 1
        assert observed_tool_calls == [
            "fast_web_search",
            "search_web",
            "write_text_file",
            "read_text_file",
        ]
        assert output_path.read_text(encoding="utf-8").startswith("Inoxtag is a French")
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.completed_deliverables == ("t0", "t1", "t2", "t3")
        assert mission_state.get_task("t1").satisfied_evidence == ("full_web_research",)
        assert mission_state.get_task("t2").satisfied_evidence == (
            "artifact_write",
            "artifact_readback",
        )
    finally:
        session_manager.close()


def test_compound_mission_status_follow_up_reports_the_real_completed_state(
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
    output_path = settings.paths.files_dir / "inoxtag-status.txt"
    observed_tool_calls: list[str] = []

    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or ToolResult.ok(
                tool_name=call.tool_name,
                output_text="Inoxtag research complete",
                payload={
                    "query": call.arguments["query"],
                    "summary_points": [
                        "Inoxtag is a French content creator.",
                        "He is known for ambitious challenge videos.",
                    ],
                    "display_sources": [
                        {"title": "Example", "url": "https://example.com/inoxtag"}
                    ],
                },
            )
        ),
    )
    tool_registry.register(
        WRITE_TEXT_FILE_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or write_text_file(
                call,
                allowed_roots=(settings.paths.project_root,),
                default_write_dir=settings.paths.files_dir,
            )
        ),
    )
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or read_text_file(
                call,
                read_allowed_roots=(settings.paths.project_root,),
                default_read_dir=settings.paths.files_dir,
            )
        ),
    )

    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "start_new",
                "step_mode": "final_reply",
                "mission_goal": "Research Inoxtag, write a verified note, and tell a joke.",
                "reasoning_summary": "Research, write, verify, then answer with the joke.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Research Inoxtag",
                        "kind": "web_research",
                        "status": "active",
                        "depends_on": [],
                        "required_evidence": ["full_web_research"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t2",
                        "title": "Write and verify the note",
                        "kind": "file_write",
                        "status": "pending",
                        "depends_on": ["t1"],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [str(output_path)],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t3",
                        "title": "Tell the joke",
                        "kind": "reply",
                        "status": "pending",
                        "depends_on": ["t2"],
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
                        "tool_name": "search_web",
                        "arguments": {"query": "Inoxtag biography"},
                    },
                    {
                        "task_id": "t2",
                        "tool_name": "write_text_file",
                        "arguments": {
                            "path": str(output_path),
                            "content": "Inoxtag is a French content creator known for ambitious challenge videos.",
                        },
                    },
                    {
                        "task_id": "t2",
                        "tool_name": "read_text_file",
                        "arguments": {"path": str(output_path)},
                    },
                ],
                "reply_to_user": "Le fichier est prêt et vérifié. Et la blague: les bananes adorent les missions parce qu'elles avancent en régime groupé.",
                "completion_claim": True,
                "blocker": None,
                "next_expected_evidence": None,
            }
        ),
        _action_response(
            {
                "mission_action": "continue_existing",
                "step_mode": "final_reply",
                "mission_goal": "Research Inoxtag, write a verified note, and tell a joke.",
                "reasoning_summary": "Answer the status follow-up from the persisted completed mission.",
                "tasks": [],
                "active_task_id": None,
                "tool_calls": [],
                "reply_to_user": (
                    f"La mission est terminée. La recherche est faite, le fichier vérifié est {output_path}, "
                    "et la blague sur les bananes a déjà été livrée."
                ),
                "completion_claim": True,
                "blocker": None,
                "next_expected_evidence": None,
            }
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        first_reply = _run_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_registry=tool_registry,
            prompt=(
                "Qui est Inoxtag ? fais une recherche complète sur lui, écrit le résultat "
                "dans un fichier texte et fais moi une blague sur les bananes"
            ),
        )
        first_state = session_manager.get_current_mission_state()
        assert first_state is not None
        mission_id = first_state.mission_id

        second_reply = _run_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_registry=tool_registry,
            prompt="où en est cette mission ?",
        )
        second_state = session_manager.get_current_mission_state()

        assert "bananes" in first_reply
        assert observed_tool_calls == [
            "search_web",
            "write_text_file",
            "read_text_file",
        ]
        assert second_state is not None
        assert second_state.mission_id == mission_id
        assert second_state.status == "completed"
        assert str(output_path) in second_reply
        assert "mission est terminée" in second_reply.lower()
    finally:
        session_manager.close()


def test_marine_leleu_uses_grounding_without_confusing_her_with_marine_le_pen(
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
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or ToolResult.ok(
                tool_name=call.tool_name,
                output_text="Marine Leleu grounding",
                payload={
                    "query": call.arguments["query"],
                    "grounding_note": "Marine Leleu is a French endurance athlete and content creator.",
                },
            )
        ),
    )
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or ToolResult.ok(
                tool_name=call.tool_name,
                output_text="should not run",
            )
        ),
    )
    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "start_new",
                "mission_goal": "Identify who Marine Leleu is.",
                "reasoning_summary": "Ground the person first and answer from the grounding note.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Identify the correct person",
                        "kind": "mixed",
                        "status": "active",
                        "depends_on": [],
                        "required_evidence": ["fast_grounding", "reply_emitted"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    }
                ],
                "active_task_id": "t1",
                "tool_calls": [
                    {
                        "task_id": "t1",
                        "tool_name": "fast_web_search",
                        "arguments": {"query": "Marine Leleu"},
                    }
                ],
                "reply_to_user": "Marine Leleu est une sportive d'endurance et créatrice de contenu française.",
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
            prompt="Qui est Marine Leleu ?",
        )
        mission_state = session_manager.get_current_mission_state()

        assert "Marine Le Pen" not in reply
        assert observed_tool_calls == ["fast_web_search"]
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.get_task("t1").satisfied_evidence == (
            "fast_grounding",
            "reply_emitted",
        )
    finally:
        session_manager.close()


def test_search_timeout_triggers_bounded_repair_instead_of_nonsense_repetition(
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
    output_path = settings.paths.files_dir / "macron-bio.txt"
    observed_search_args: list[dict[str, object]] = []
    search_attempts = {"count": 0}

    def _search_handler(call):
        observed_search_args.append(dict(call.arguments))
        search_attempts["count"] += 1
        if search_attempts["count"] == 1:
            return ToolResult.failure(
                tool_name=call.tool_name,
                error="search_web timed out",
                failure_kind="timeout",
                payload={"execution_state": "timed_out"},
            )
        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Macron research",
            payload={
                "query": call.arguments["query"],
                "summary_points": [
                    "Emmanuel Macron is the President of France.",
                    "He was born in Amiens.",
                ],
                "display_sources": [
                    {"title": "Example", "url": "https://example.com/macron"}
                ],
            },
        )

    tool_registry = ToolRegistry()
    tool_registry.register(SEARCH_WEB_DEFINITION, _search_handler)
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
                "mission_goal": "Research Emmanuel Macron and write his bio with emojis.",
                "reasoning_summary": "Start with research.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Research Emmanuel Macron",
                        "kind": "web_research",
                        "status": "active",
                        "depends_on": [],
                        "required_evidence": ["full_web_research"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t2",
                        "title": "Write and verify the bio file",
                        "kind": "file_write",
                        "status": "pending",
                        "depends_on": ["t1"],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [str(output_path)],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t3",
                        "title": "Reply to the user",
                        "kind": "reply",
                        "status": "pending",
                        "depends_on": ["t2"],
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
                        "tool_name": "search_web",
                        "arguments": {"query": "Emmanuel Macron biography"},
                    }
                ],
                "reply_to_user": None,
                "completion_claim": False,
                "blocker": None,
                "next_expected_evidence": "full_web_research",
            }
        ),
        _action_response(
            {
                "mission_action": "continue_existing",
                "mission_goal": "Research Emmanuel Macron and write his bio with emojis.",
                "reasoning_summary": "Repair after timeout with a narrower search.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Research Emmanuel Macron",
                        "kind": "web_research",
                        "status": "repairing",
                        "depends_on": [],
                        "required_evidence": ["full_web_research"],
                        "artifact_paths": [],
                        "latest_error": "search_web timed out",
                        "repair_count": 1,
                    }
                ],
                "active_task_id": "t1",
                "tool_calls": [
                    {
                        "task_id": "t1",
                        "tool_name": "search_web",
                        "arguments": {
                            "query": "Emmanuel Macron biography",
                            "max_results": 2,
                        },
                    }
                ],
                "reply_to_user": None,
                "completion_claim": False,
                "blocker": None,
                "next_expected_evidence": "full_web_research",
            }
        ),
        _action_response(
            {
                "mission_action": "continue_existing",
                "mission_goal": "Research Emmanuel Macron and write his bio with emojis.",
                "reasoning_summary": "Write the verified bio and confirm completion.",
                "tasks": [
                    {
                        "id": "t2",
                        "title": "Write and verify the bio file",
                        "kind": "file_write",
                        "status": "active",
                        "depends_on": ["t1"],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [str(output_path)],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t3",
                        "title": "Reply to the user",
                        "kind": "reply",
                        "status": "pending",
                        "depends_on": ["t2"],
                        "required_evidence": ["reply_emitted"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                ],
                "active_task_id": "t2",
                "tool_calls": [
                    {
                        "task_id": "t2",
                        "tool_name": "write_text_file",
                        "arguments": {
                            "path": str(output_path),
                            "content": "Emmanuel Macron 😊\nPresident of France 🇫🇷\nBorn in Amiens.",
                        },
                    },
                    {
                        "task_id": "t2",
                        "tool_name": "read_text_file",
                        "arguments": {"path": str(output_path)},
                    },
                ],
                "reply_to_user": "La bio avec emojis est prête 😊🇫🇷",
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
            prompt="Fais une recherche sur Emmanuel Macron, puis écrit sa bio dans un fichier texte, avec emojis...",
        )
        mission_state = session_manager.get_current_mission_state()

        assert reply == "La bio avec emojis est prête 😊🇫🇷"
        assert fake_provider.call_count() == 3
        assert observed_search_args[0]["query"] == "Emmanuel Macron biography"
        assert "max_results" not in observed_search_args[0]
        assert observed_search_args[1]["query"] == "Emmanuel Macron biography"
        assert observed_search_args[1]["max_results"] in {2, 3}
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert mission_state.get_task("t1").repair_count in {0, 1}
        assert "President of France" in output_path.read_text(encoding="utf-8")
    finally:
        session_manager.close()


def test_completed_tasks_are_not_rerun_automatically(
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
    output_path = settings.paths.files_dir / "note.txt"
    observed_tool_calls: list[str] = []
    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or ToolResult.ok(
                tool_name=call.tool_name,
                output_text="Should never run",
            )
        ),
    )
    tool_registry.register(
        WRITE_TEXT_FILE_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or write_text_file(
                call,
                allowed_roots=(settings.paths.project_root,),
                default_write_dir=settings.paths.files_dir,
            )
        ),
    )
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or read_text_file(
                call,
                read_allowed_roots=(settings.paths.project_root,),
                default_read_dir=settings.paths.files_dir,
            )
        ),
    )

    session = session_manager.ensure_current_session()
    session_manager.persist_mission_state(
        MissionState(
            mission_id="mission-existing",
            mission_goal="Research then write a note.",
            status="active",
            tasks=(
                MissionTaskState(
                    id="t1",
                    title="Research the subject",
                    kind="web_research",
                    status="completed",
                    required_evidence=("full_web_research",),
                    satisfied_evidence=("full_web_research",),
                    evidence=("Verified fact.",),
                ),
                MissionTaskState(
                    id="t2",
                    title="Write and verify the note",
                    kind="file_write",
                    status="active",
                    depends_on=("t1",),
                    required_evidence=("artifact_write", "artifact_readback"),
                    artifact_paths=(str(output_path),),
                ),
                MissionTaskState(
                    id="t3",
                    title="Reply to the user",
                    kind="reply",
                    status="pending",
                    depends_on=("t2",),
                    required_evidence=("reply_emitted",),
                ),
            ),
            active_task_id="t2",
            updated_at="2026-03-27T09:00:00Z",
            reasoning_summary="existing mission",
            last_user_input="continue",
        ),
        session_id=session.id,
    )

    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "continue_existing",
                "mission_goal": "Research then write a note.",
                "reasoning_summary": "Do not rerun the completed research task.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Research the subject",
                        "kind": "web_research",
                        "status": "completed",
                        "depends_on": [],
                        "required_evidence": ["full_web_research"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t2",
                        "title": "Write and verify the note",
                        "kind": "file_write",
                        "status": "active",
                        "depends_on": ["t1"],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [str(output_path)],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t3",
                        "title": "Reply to the user",
                        "kind": "reply",
                        "status": "pending",
                        "depends_on": ["t2"],
                        "required_evidence": ["reply_emitted"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                ],
                "active_task_id": "t2",
                "tool_calls": [
                    {
                        "task_id": "t1",
                        "tool_name": "search_web",
                        "arguments": {"query": "should not rerun"},
                    },
                    {
                        "task_id": "t2",
                        "tool_name": "write_text_file",
                        "arguments": {"path": str(output_path), "content": "safe note"},
                    },
                    {
                        "task_id": "t2",
                        "tool_name": "read_text_file",
                        "arguments": {"path": str(output_path)},
                    },
                ],
                "reply_to_user": "The note is written.",
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
            prompt="continue",
        )
        mission_state = session_manager.get_current_mission_state(session.id)

        assert reply == "The note is written."
        assert observed_tool_calls == ["write_text_file", "read_text_file"]
        assert mission_state is not None
        assert mission_state.status == "completed"
        assert any(
            history.tool_name == "search_web"
            and history.executed is False
            and history.reason == "task already completed"
            for history in mission_state.tool_history
        )
    finally:
        session_manager.close()


def test_no_dependent_task_runs_before_prerequisite_proof_exists(
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
    output_path = settings.paths.files_dir / "premature-note.txt"
    observed_tool_calls: list[str] = []
    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or ToolResult.ok(
                tool_name=call.tool_name,
                output_text="Research success",
                payload={
                    "query": call.arguments["query"],
                    "summary_points": ["Verified research fact."],
                    "display_sources": [
                        {"title": "Example", "url": "https://example.com/research"}
                    ],
                },
            )
        ),
    )
    tool_registry.register(
        WRITE_TEXT_FILE_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or write_text_file(
                call,
                allowed_roots=(settings.paths.project_root,),
                default_write_dir=settings.paths.files_dir,
            )
        ),
    )
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: (
            observed_tool_calls.append(call.tool_name)
            or read_text_file(
                call,
                read_allowed_roots=(settings.paths.project_root,),
                default_read_dir=settings.paths.files_dir,
            )
        ),
    )

    fake_provider = build_scripted_ollama_provider(
        _action_response(
            {
                "mission_action": "start_new",
                "mission_goal": "Research first, then write a file.",
                "reasoning_summary": "The write step incorrectly appears first; the runtime must gate it.",
                "tasks": [
                    {
                        "id": "t1",
                        "title": "Research the topic",
                        "kind": "web_research",
                        "status": "active",
                        "depends_on": [],
                        "required_evidence": ["full_web_research"],
                        "artifact_paths": [],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                    {
                        "id": "t2",
                        "title": "Write and verify the file",
                        "kind": "file_write",
                        "status": "pending",
                        "depends_on": ["t1"],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [str(output_path)],
                        "latest_error": None,
                        "repair_count": 0,
                    },
                ],
                "active_task_id": "t1",
                "tool_calls": [
                    {
                        "task_id": "t2",
                        "tool_name": "write_text_file",
                        "arguments": {"path": str(output_path), "content": "too early"},
                    },
                    {
                        "task_id": "t2",
                        "tool_name": "read_text_file",
                        "arguments": {"path": str(output_path)},
                    },
                    {
                        "task_id": "t1",
                        "tool_name": "search_web",
                        "arguments": {"query": "ordered proof"},
                    },
                ],
                "reply_to_user": None,
                "completion_claim": False,
                "blocker": None,
                "next_expected_evidence": "full_web_research",
            }
        ),
        _action_response(
            {
                "mission_action": "continue_existing",
                "mission_goal": "Research first, then write a file.",
                "reasoning_summary": "Stop after reporting the persisted state.",
                "tasks": [
                    {
                        "id": "t2",
                        "title": "Write and verify the file",
                        "kind": "file_write",
                        "status": "pending",
                        "depends_on": ["t1"],
                        "required_evidence": ["artifact_write", "artifact_readback"],
                        "artifact_paths": [str(output_path)],
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
        ),
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        reply = _run_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_registry=tool_registry,
            prompt="Research, then write a file.",
        )
        mission_state = session_manager.get_current_mission_state()

        assert observed_tool_calls == ["search_web"]
        assert "Mission in progress: Research first, then write a file." in reply
        assert "Current task: Write and verify the file." in reply
        assert "Waiting for: artifact_write, artifact_readback." in reply
        assert mission_state is not None
        assert mission_state.status == "active"
        assert mission_state.get_task("t1").status == "completed"
        assert mission_state.get_task("t2").status != "completed"
        assert any(
            history.tool_name == "write_text_file"
            and history.executed is False
            and history.reason == "prerequisite proof missing"
            for history in mission_state.tool_history
        )
    finally:
        session_manager.close()
