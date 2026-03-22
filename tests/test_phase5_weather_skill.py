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
from skills.weather.tool import GET_WEATHER_DEFINITION

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("profile_name", "expect_weather_skill", "expect_tools"),
    (
        # Weather catalog is always injected when skills are enabled (file-first
        # catalog has no profile gating). Tools remain profile-gated.
        ("main", True, True),
        ("deep", True, True),
        ("fast", True, False),
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

    def _weather_tool(call: ToolCall) -> ToolResult:
        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text="phase7A weather grounding",
            payload={
                "provider": "open-meteo",
                "location_query": call.arguments["location"],
                "forecast_days": 7,
                "resolved_location": {
                    "name": "Paris",
                    "latitude": 48.8566,
                    "longitude": 2.3522,
                    "timezone": "Europe/Paris",
                    "admin1": "Ile-de-France",
                    "country": "France",
                },
                "current": None,
                "daily_forecast": [],
            },
        )

    tool_registry.register(GET_WEATHER_DEFINITION, _weather_tool)

    captured_messages: list[list[LLMMessage]] = []
    captured_tools: list[object | None] = []

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
        # Weather prompt ownership is now in the file-first skill bundle.
        # The compact catalog is injected as a system message when skills are enabled.
        catalog_msgs = [
            message for message in system_messages if "Active optional skills:" in message
        ]
        if expect_weather_skill:
            assert len(catalog_msgs) == 1
            assert "weather" in catalog_msgs[0]
            assert "get_weather" in catalog_msgs[0]
        else:
            assert not catalog_msgs

        if expect_tools:
            assert captured_tools[0] is not None
            assert any(tool.name == "get_weather" for tool in captured_tools[0])
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
