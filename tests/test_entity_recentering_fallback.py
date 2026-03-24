from __future__ import annotations

from types import SimpleNamespace

import pytest

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
from unclaw.tools.web_tools import FAST_WEB_SEARCH_DEFINITION, FETCH_URL_TEXT_DEFINITION

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


def test_native_first_pass_skips_entity_recentering_note_even_when_anchor_exists(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    executed_queries: list[str] = []
    system_messages_by_call: list[list[str]] = []
    call_count = 0

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
            system_messages_by_call.append(system_messages)
            assert all(
                "Entity recentering hint:" not in message for message in system_messages
            )
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T11:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fast_web_search",
                            arguments={"query": "Marty McFly and Carlito"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fast_web_search",
                                        "arguments": {
                                            "query": "Marty McFly and Carlito"
                                        },
                                    }
                                }
                            ],
                        }
                    },
                )

            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="McFly et Carlito are a French YouTube comedy duo.",
                created_at="2026-03-24T11:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    tool_registry = ToolRegistry()
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: (
            executed_queries.append(call.arguments["query"]) or ToolResult.ok(
                tool_name=call.tool_name,
                output_text=(
                    "McFly et Carlito\n- McFly et Carlito are a French YouTube comedy duo."
                ),
                payload={
                    "query": call.arguments["query"],
                    "result_count": 1,
                    "grounding_note": (
                        "McFly et Carlito\n"
                        "- McFly et Carlito are a French YouTube comedy duo."
                    ),
                },
            )
        ),
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Qui sont McFly et Carlito ?",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Je pensais a d'autres personnalites.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.USER,
            "Non, je parle bien du duo YouTube francais McFly et Carlito.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Compris.",
            session_id=session.id,
        )
        user_input = "Fais leur bio courte."
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

        assert reply == "McFly et Carlito are a French YouTube comedy duo."
        assert call_count == 2
        assert executed_queries == ["McFly et Carlito"]
        assert len(system_messages_by_call) == 2
    finally:
        session_manager.close()


def test_native_legacy_retry_injects_entity_recentering_note_only_on_fallback(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    system_messages_by_call: list[list[str]] = []
    call_count = 0

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
            system_messages_by_call.append(system_messages)
            if call_count == 1:
                assert all(
                    "Entity recentering hint:" not in message
                    for message in system_messages
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="First pass stayed text-only.",
                    created_at="2026-03-24T12:00:00Z",
                    finish_reason="stop",
                )

            if call_count == 2:
                assert any(
                    "Current request routing hint:" in message
                    for message in system_messages
                )
                assert any(
                    "Entity recentering hint:" in message
                    and "McFly et Carlito" in message
                    for message in system_messages
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="Fallback answer after one legacy retry.",
                    created_at="2026-03-24T12:00:01Z",
                    finish_reason="stop",
                )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Qui sont McFly et Carlito ?",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Je pensais a d'autres personnalites.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.USER,
            "Non, je parle bien du duo YouTube francais McFly et Carlito.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Compris.",
            session_id=session.id,
        )
        user_input = "Fais leur bio courte."
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
        )

        assert reply == "Fallback answer after one legacy retry."
        assert call_count == 2
        assert len(system_messages_by_call) == 2
    finally:
        session_manager.close()


def test_native_direct_success_never_injects_entity_recentering_note_without_retry(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    call_count = 0

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
                "Entity recentering hint:" not in message for message in system_messages
            )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Follow-up answer without legacy retry.",
                created_at="2026-03-24T13:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Qui sont McFly et Carlito ?",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "McFly et Carlito are a French YouTube comedy duo.",
            session_id=session.id,
        )
        user_input = "Tell me more about them."
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
        )

        assert reply == "Follow-up answer without legacy retry."
        assert call_count == 1
    finally:
        session_manager.close()


def test_native_legacy_retry_without_anchor_skips_entity_recentering_note(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    call_count = 0

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
                "Entity recentering hint:" not in message for message in system_messages
            )
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="First pass stayed text-only.",
                    created_at="2026-03-24T14:00:00Z",
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
                    created_at="2026-03-24T14:00:01Z",
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

            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Recovered summary after one legacy retry.",
                created_at="2026-03-24T14:00:02Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    tool_registry = ToolRegistry()
    tool_registry.register(
        FETCH_URL_TEXT_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Recovered fetched page.",
        ),
    )

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
    finally:
        session_manager.close()
