from __future__ import annotations

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
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_entity_guard import (
    apply_entity_guard_to_tool_calls,
    extract_user_entity_surface,
)
from unclaw.tools.web_tools import FAST_WEB_SEARCH_DEFINITION, SEARCH_WEB_DEFINITION


def _make_search_call(tool_name: str, query: str) -> ToolCall:
    return ToolCall(tool_name=tool_name, arguments={"query": query})


def test_fast_web_search_preserves_explicit_literal_entity() -> None:
    guarded = apply_entity_guard_to_tool_calls(
        [_make_search_call("fast_web_search", "Marine Le Pen biographie")],
        "Marine Leleu",
    )

    assert guarded[0].arguments["query"] == "Marine Leleu biographie"


def test_search_web_preserves_explicit_literal_entity() -> None:
    guarded = apply_entity_guard_to_tool_calls(
        [_make_search_call("search_web", "Marine Le Pen recent profile")],
        "Marine Leleu",
    )

    assert guarded[0].arguments["query"] == "Marine Leleu recent profile"


def test_guard_is_noop_when_query_already_uses_literal_entity() -> None:
    guarded = apply_entity_guard_to_tool_calls(
        [_make_search_call("fast_web_search", "Marine Leleu biographie")],
        "Marine Leleu",
    )

    assert guarded[0].arguments["query"] == "Marine Leleu biographie"


def test_guard_does_not_mutate_without_explicit_literal_entity_in_current_input() -> None:
    user_input = "can you search the web for the latest updates on climate change and renewable energy"
    user_entity_surface = extract_user_entity_surface(user_input)

    assert user_entity_surface == ""

    guarded = apply_entity_guard_to_tool_calls(
        [_make_search_call("search_web", "Marine Le Pen")],
        user_entity_surface,
    )

    assert guarded[0].arguments["query"] == "Marine Le Pen"


def test_explicit_literal_surface_wins_over_trailing_noise_tolerance() -> None:
    guarded = apply_entity_guard_to_tool_calls(
        [_make_search_call("fast_web_search", "Inoxtag")],
        "Inoxtag N",
    )

    assert guarded[0].arguments["query"] == "Inoxtag N"


@pytest.mark.integration
def test_runtime_no_longer_applies_literal_entity_guard_to_tool_calls(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
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
    observed_tool_calls: list[ToolCall] = []
    call_count = 0

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del messages, timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-25T10:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fast_web_search",
                            arguments={"query": "Marine Le Pen"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fast_web_search",
                                        "arguments": {"query": "Marine Le Pen"},
                                    }
                                }
                            ],
                        }
                    },
                )

            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Marine Leleu is a French endurance athlete.",
                created_at="2026-03-25T10:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    tool_registry = ToolRegistry()
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=f"Grounding note: {call.arguments['query']}",
            payload={
                "query": call.arguments["query"],
                "result_count": 1,
                "grounding_note": f"Grounding note: {call.arguments['query']}",
            },
        ),
    )
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=f"Search query: {call.arguments['query']}",
        ),
    )

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

        assert reply == "Marine Leleu is a French endurance athlete."
        assert call_count == 2
        assert observed_tool_calls == [
            ToolCall(
                tool_name="fast_web_search",
                arguments={"query": "Marine Le Pen"},
            )
        ]
    finally:
        session_manager.close()
