from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from unclaw.core.capabilities import (
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.command_handler import CommandHandler
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.file_tools import LIST_DIRECTORY_DEFINITION, READ_TEXT_FILE_DEFINITION
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.system_tools import SYSTEM_INFO_DEFINITION
from unclaw.tools.web_tools import FAST_WEB_SEARCH_DEFINITION, SEARCH_WEB_DEFINITION


def _build_runtime_test_env(project_root: Path):
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
    command_handler.current_model_profile_name = "main"
    return settings, session_manager, tracer, command_handler


def test_capability_context_includes_phase1a_verify_before_guess_guidance() -> None:
    summary = build_runtime_capability_summary(
        tool_registry=create_default_tool_registry(),
        memory_summary_available=False,
        model_can_call_tools=True,
    )

    context = build_runtime_capability_context(summary)

    assert "prefer the tool over guessing" in context
    assert (
        "Do not ask for clarification when the current request already gives enough "
        "information for the first obvious tool call." in context
    )
    assert "look something up online" in context
    assert "For obvious local date, time, day, OS, hostname, or machine-spec questions" in context


def test_runtime_adds_system_info_routing_note_for_obvious_local_runtime_requests(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_runtime_test_env(
        project_root
    )
    observed_tool_calls: list[ToolCall] = []
    call_count = 0

    tool_registry = ToolRegistry()
    tool_registry.register(
        SYSTEM_INFO_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                "OS: ExampleOS 1.0 (x86_64)\n"
                "Local datetime: 2026-03-24 09:41:00 CET\n"
                "Locale: fr_FR/UTF-8"
            ),
        ),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            system_messages = [
                message.content for message in messages if message.role is LLMRole.SYSTEM
            ]
            if call_count == 1:
                assert all(
                    "Current request routing hint:" not in message
                    for message in system_messages
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T08:41:00Z",
                    finish_reason="stop",
                    tool_calls=(ToolCall(tool_name="system_info", arguments={}),),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "system_info",
                                        "arguments": {},
                                    }
                                }
                            ],
                        }
                    },
                )

            assert any(
                message.startswith("Post-tool grounding note:")
                for message in system_messages
            )
            assert any(
                message.role is LLMRole.TOOL
                and "Local datetime: 2026-03-24 09:41:00 CET" in message.content
                for message in messages
            )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="It is 09:41 CET on 2026-03-24.",
                created_at="2026-03-24T08:41:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "What is the local date and time on this machine?",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="What is the local date and time on this machine?",
            tracer=tracer,
            tool_registry=tool_registry,
            tool_call_callback=observed_tool_calls.append,
        )

        assert reply == "It is 09:41 CET on 2026-03-24."
        assert call_count == 2
        assert observed_tool_calls == [ToolCall(tool_name="system_info", arguments={})]
    finally:
        session_manager.close()


def test_runtime_identity_requests_prefer_fast_web_search_first(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_runtime_test_env(
        project_root
    )
    observed_tool_calls: list[ToolCall] = []
    call_count = 0

    tool_registry = ToolRegistry()
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                "Grounding note: Marine Leleu is a French endurance athlete and creator."
            ),
        ),
    )
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=f"Search query: {call.arguments['query']}",
        ),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            system_messages = [
                message.content for message in messages if message.role is LLMRole.SYSTEM
            ]
            if call_count == 1:
                assert all(
                    "Current request routing hint:" not in message
                    for message in system_messages
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T10:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fast_web_search",
                            arguments={"query": "Marine Leleu"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fast_web_search",
                                        "arguments": {"query": "Marine Leleu"},
                                    }
                                }
                            ],
                        }
                    },
                )

            assert any(
                message.startswith("Post-tool grounding note:")
                for message in system_messages
            )
            assert any(
                message.role is LLMRole.TOOL
                and "Marine Leleu is a French endurance athlete" in message.content
                for message in messages
            )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Marine Leleu is a French endurance athlete and creator.",
                created_at="2026-03-24T10:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Qui est Marine Leleu ?",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Qui est Marine Leleu ?",
            tracer=tracer,
            tool_registry=tool_registry,
            tool_call_callback=observed_tool_calls.append,
        )

        assert reply == "Marine Leleu is a French endurance athlete and creator."
        assert call_count == 3
        assert observed_tool_calls == [
            ToolCall(
                tool_name="fast_web_search",
                arguments={"query": "Marine Leleu"},
            )
        ]
    finally:
        session_manager.close()


def test_runtime_local_inspection_continues_to_second_tool_without_clarifying(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_runtime_test_env(
        project_root
    )
    observed_tool_calls: list[ToolCall] = []
    call_count = 0

    tool_registry = ToolRegistry()
    tool_registry.register(
        LIST_DIRECTORY_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                "Directory: config/prompts\n"
                "- system.txt\n"
                "- assistant.txt"
            ),
        ),
    )
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Phase 1A prompt body.",
        ),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            system_messages = [
                message.content for message in messages if message.role is LLMRole.SYSTEM
            ]
            tool_messages = [
                message.content for message in messages if message.role is LLMRole.TOOL
            ]
            if call_count == 1:
                assert all(
                    "Current request routing hint:" not in message
                    for message in system_messages
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T11:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="list_directory",
                            arguments={"path": "config/prompts", "max_depth": 1},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "list_directory",
                                        "arguments": {
                                            "path": "config/prompts",
                                            "max_depth": 1,
                                        },
                                    }
                                }
                            ],
                        }
                    },
                )

            if call_count == 2:
                assert len(tool_messages) == 1
                assert "system.txt" in tool_messages[0]
                assert any(
                    "call read_text_file next instead of stopping at the listing."
                    in message
                    for message in system_messages
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T11:00:01Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="read_text_file",
                            arguments={"path": "config/prompts/system.txt"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "read_text_file",
                                        "arguments": {
                                            "path": "config/prompts/system.txt",
                                        },
                                    }
                                }
                            ],
                        }
                    },
                )

            assert len(tool_messages) == 2
            assert "Phase 1A prompt body." in tool_messages[1]
            assert any(
                message.startswith("Post-tool grounding note:")
                for message in system_messages
            )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="`system.txt` says: Phase 1A prompt body.",
                created_at="2026-03-24T11:00:02Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Inspect the prompts directory and tell me what `system.txt` says.",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Inspect the prompts directory and tell me what `system.txt` says.",
            tracer=tracer,
            tool_registry=tool_registry,
            tool_call_callback=observed_tool_calls.append,
        )

        assert reply == "`system.txt` says: Phase 1A prompt body."
        assert call_count == 3
        assert observed_tool_calls == [
            ToolCall(
                tool_name="list_directory",
                arguments={"path": "config/prompts", "max_depth": 1},
            ),
            ToolCall(
                tool_name="read_text_file",
                arguments={"path": "config/prompts/system.txt"},
            ),
        ]
        stored_messages = session_manager.list_messages(session.id)
        assert [message.role for message in stored_messages] == [
            MessageRole.USER,
            MessageRole.TOOL,
            MessageRole.TOOL,
            MessageRole.ASSISTANT,
        ]
    finally:
        session_manager.close()


def test_runtime_explicit_web_lookup_uses_search_and_post_tool_grounding(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_runtime_test_env(
        project_root
    )
    observed_tool_calls: list[ToolCall] = []
    call_count = 0

    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                f"Search query: {call.arguments['query']}\n"
                "Supported fact: Example Product released on 2026-02-10."
            ),
        ),
    )

    user_input = "Search the web for Example Product release date."

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            system_messages = [
                message.content for message in messages if message.role is LLMRole.SYSTEM
            ]
            if call_count == 1:
                assert all(
                    "Current request routing hint:" not in message
                    for message in system_messages
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T12:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="search_web",
                            arguments={"query": user_input},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "search_web",
                                        "arguments": {"query": user_input},
                                    }
                                }
                            ],
                        }
                    },
                )

            assert any(
                "Do not contradict successful tool output" in message
                for message in system_messages
            )
            assert any(
                message.role is LLMRole.TOOL
                and "Example Product released on 2026-02-10." in message.content
                for message in messages
            )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Example Product released on 2026-02-10.",
                created_at="2026-03-24T12:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            user_input,
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=user_input,
            tracer=tracer,
            tool_registry=tool_registry,
            tool_call_callback=observed_tool_calls.append,
        )

        assert reply == "Example Product released on 2026-02-10."
        assert call_count == 3
        assert observed_tool_calls == [
            ToolCall(
                tool_name="search_web",
                arguments={"query": user_input},
            )
        ]
    finally:
        session_manager.close()
