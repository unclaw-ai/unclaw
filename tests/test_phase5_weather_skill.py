from __future__ import annotations

from types import SimpleNamespace

import pytest

from unclaw.core.command_handler import CommandHandler
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMMessage, LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import TraceEvent, Tracer
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("profile_name", "expect_weather_skill", "expect_tools"),
    (
        ("main", True, True),
        ("deep", True, True),
        ("fast", False, False),
    ),
)
def test_phase5_weather_skill_reaches_live_runtime_without_giving_fast_tools(
    monkeypatch,
    make_temp_project,
    profile_name: str,
    expect_weather_skill: bool,
    expect_tools: bool,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    event_bus = EventBus()
    published_events: list[TraceEvent] = []
    event_bus.subscribe(published_events.append)
    tracer = Tracer(
        event_bus=event_bus,
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )
    command_handler.current_model_profile_name = profile_name

    tool_registry = ToolRegistry()

    def _search_tool(call: ToolCall) -> ToolResult:
        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text="phase5 weather search grounding",
        )

    tool_registry.register(SEARCH_WEB_DEFINITION, _search_tool)

    captured_messages: list[list[LLMMessage]] = []
    captured_tools: list[object | None] = []

    class RouterShouldNotRun:
        provider_name = "ollama"

        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            del kwargs
            raise AssertionError("default direct runtime turns must not instantiate the router")

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
            del timeout_seconds, thinking_enabled, content_callback
            assert profile.name == profile_name
            captured_messages.append(list(messages))
            captured_tools.append(tools)
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content=f"{profile_name} weather reply",
                created_at="2026-03-21T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.router.OllamaProvider", RouterShouldNotRun)
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="What is the weather in Paris tomorrow?",
            tracer=tracer,
            event_bus=event_bus,
            tool_registry=tool_registry,
        )

        assert reply == f"{profile_name} weather reply"
        assert len(captured_messages) == 1
        assert len(captured_tools) == 1

        system_messages = [
            message.content
            for message in captured_messages[0]
            if message.role is LLMRole.SYSTEM
        ]
        weather_notes = [
            message for message in system_messages if "Active optional skill: Weather" in message
        ]

        assert bool(weather_notes) is expect_weather_skill
        if expect_weather_skill:
            assert "use search_web before stating weather details" in weather_notes[0]
        else:
            assert not weather_notes

        if expect_tools:
            assert captured_tools[0] is not None
            assert any(tool.name == "search_web" for tool in captured_tools[0])
        else:
            assert captured_tools[0] is None

        capability_message = next(
            message
            for message in system_messages
            if "Runtime capability status:" in message
        )
        if profile_name == "fast":
            assert "user-initiated slash commands only" in capability_message
        else:
            assert "you may call tools directly this turn" in capability_message
    finally:
        session_manager.close()
