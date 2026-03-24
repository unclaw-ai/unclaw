from __future__ import annotations

from types import SimpleNamespace

import pytest

import unclaw.core.routing as _routing
from unclaw.core.command_handler import CommandHandler
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import FETCH_URL_TEXT_DEFINITION, SEARCH_WEB_DEFINITION

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
    return session_manager, tracer, command_handler


def _track_legacy_routing_note_builder(monkeypatch) -> list[str]:
    original_builder = _routing._build_request_routing_note
    build_calls: list[str] = []

    def _wrapped_builder(*, user_input, capability_summary):
        build_calls.append(user_input)
        return original_builder(
            user_input=user_input,
            capability_summary=capability_summary,
        )

    monkeypatch.setattr(_routing, "_build_request_routing_note", _wrapped_builder)
    return build_calls


def test_native_first_tool_call_skips_legacy_routing_retry(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    build_calls = _track_legacy_routing_note_builder(monkeypatch)
    call_count = 0

    tool_registry = ToolRegistry()
    tool_registry.register(
        FETCH_URL_TEXT_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Fetched example page.",
        ),
    )

    class FakeOllamaProvider:
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
            call_count += 1
            system_messages = [
                message.content for message in messages if message.role is LLMRole.SYSTEM
            ]
            assert all(
                "Current request routing hint:" not in message
                for message in system_messages
            )
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T10:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fetch_url_text",
                            arguments={"url": "https://example.com"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fetch_url_text",
                                        "arguments": {"url": "https://example.com"},
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
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Fetched summary.",
                created_at="2026-03-24T10:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        user_input = "Fetch https://example.com and summarize it."
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
        )

        assert reply == "Fetched summary."
        assert call_count == 2
        assert build_calls == []
    finally:
        session_manager.close()


def test_native_legacy_routing_retry_recovers_tool_call(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    build_calls = _track_legacy_routing_note_builder(monkeypatch)
    call_count = 0

    tool_registry = ToolRegistry()
    tool_registry.register(
        FETCH_URL_TEXT_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Recovered fetched page.",
        ),
    )

    class FakeOllamaProvider:
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
                    content="First pass answered without tools.",
                    created_at="2026-03-24T11:00:00Z",
                    finish_reason="stop",
                )

            if call_count == 2:
                assert any(
                    "Current request routing hint: the user gave a specific public URL."
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
                            tool_name="fetch_url_text",
                            arguments={"url": "https://example.com"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fetch_url_text",
                                        "arguments": {"url": "https://example.com"},
                                    }
                                }
                            ],
                        }
                    },
                )

            assert any(
                "Current request routing hint: the user gave a specific public URL."
                in message
                for message in system_messages
            )
            assert any(
                message.startswith("Post-tool grounding note:")
                for message in system_messages
            )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Recovered summary after one legacy retry.",
                created_at="2026-03-24T11:00:02Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        user_input = "Fetch https://example.com and summarize it."
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
        )

        assert reply == "Recovered summary after one legacy retry."
        assert call_count == 3
        assert build_calls == [user_input]
    finally:
        session_manager.close()


def test_native_legacy_routing_retry_stops_after_one_no_tool_retry(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    build_calls = _track_legacy_routing_note_builder(monkeypatch)
    call_count = 0

    tool_registry = ToolRegistry()
    tool_registry.register(
        FETCH_URL_TEXT_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="unused",
        ),
    )

    class FakeOllamaProvider:
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
                    content="First pass stayed text-only.",
                    created_at="2026-03-24T12:00:00Z",
                    finish_reason="stop",
                )

            assert any(
                "Current request routing hint: the user gave a specific public URL."
                in message
                for message in system_messages
            )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Second pass still stayed text-only.",
                created_at="2026-03-24T12:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        user_input = "Fetch https://example.com and summarize it."
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
        )

        assert reply == "Second pass still stayed text-only."
        assert call_count == 2
        assert build_calls == [user_input]
    finally:
        session_manager.close()


def test_native_no_legacy_routing_note_means_no_retry(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    build_calls = _track_legacy_routing_note_builder(monkeypatch)
    call_count = 0

    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="unused",
        ),
    )

    class FakeOllamaProvider:
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
            call_count += 1
            system_messages = [
                message.content for message in messages if message.role is LLMRole.SYSTEM
            ]
            assert all(
                "Current request routing hint:" not in message
                for message in system_messages
            )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Direct answer without any retry.",
                created_at="2026-03-24T13:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        user_input = "Explain recursion in one sentence."
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
        )

        assert reply == "Direct answer without any retry."
        assert call_count == 1
        assert build_calls == [user_input]
    finally:
        session_manager.close()
