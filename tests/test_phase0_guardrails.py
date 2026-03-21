from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from unclaw.core.command_handler import CommandHandler
from unclaw.core.orchestrator import Orchestrator
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMMessage, LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import TraceEvent, Tracer
from unclaw.model_packs import get_model_pack_profiles
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolResult
from unclaw.tools.file_tools import READ_TEXT_FILE_DEFINITION
from unclaw.tools.registry import ToolRegistry


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_phase0_repo_settings_lock_shipped_pack_resolution_contract() -> None:
    repo_root = _repo_root()
    settings = load_settings(project_root=repo_root)
    shipped_models_payload = yaml.safe_load(
        (repo_root / "config" / "models.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(shipped_models_payload, dict)
    raw_dev_profiles = shipped_models_payload["dev_profiles"]
    assert isinstance(raw_dev_profiles, dict)

    power_profiles = get_model_pack_profiles("power")

    assert settings.model_pack == "power"
    assert settings.app.default_model_profile == "main"
    assert set(settings.models) == set(power_profiles)
    assert set(settings.dev_profiles) == set(raw_dev_profiles)
    assert settings.default_model.model_name == power_profiles["main"].model_name

    for profile_name, expected_profile in power_profiles.items():
        active_profile = settings.models[profile_name]
        assert active_profile.provider == expected_profile.provider
        assert active_profile.model_name == expected_profile.model_name
        assert active_profile.temperature == expected_profile.temperature
        assert (
            active_profile.thinking_supported
            == expected_profile.thinking_supported
        )
        assert active_profile.tool_mode == expected_profile.tool_mode
        assert active_profile.num_ctx == expected_profile.num_ctx
        assert active_profile.keep_alive == expected_profile.keep_alive
        assert active_profile.planner_profile == expected_profile.planner_profile

    for profile_name, raw_profile in raw_dev_profiles.items():
        assert isinstance(raw_profile, dict)
        dev_profile = settings.dev_profiles[profile_name]
        assert dev_profile.provider == raw_profile["provider"]
        assert dev_profile.model_name == raw_profile["model_name"]
        assert dev_profile.temperature == raw_profile["temperature"]
        assert dev_profile.thinking_supported == raw_profile["thinking_supported"]
        assert dev_profile.tool_mode == raw_profile["tool_mode"]
        assert dev_profile.num_ctx == raw_profile["num_ctx"]
        assert dev_profile.keep_alive == raw_profile["keep_alive"]


@pytest.mark.unit
def test_phase0_orchestrator_run_turn_uses_build_context_messages_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    orchestrator = Orchestrator(
        settings=settings,
        session_manager=session_manager,
        tracer=tracer,
    )

    captured: dict[str, object] = {}
    sentinel_context = [
        LLMMessage(role=LLMRole.SYSTEM, content="phase0 system prompt"),
        LLMMessage(role=LLMRole.SYSTEM, content="phase0 capability context"),
        LLMMessage(role=LLMRole.USER, content="phase0 user turn"),
    ]
    capability_summary = SimpleNamespace(name="phase0-summary")
    system_context_notes = ("phase0 note",)

    def fake_build_context_messages(**kwargs):  # type: ignore[no-untyped-def]
        captured["build_context_messages_kwargs"] = kwargs
        return list(sentinel_context)

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(
            self,
            profile,
            messages,
            *,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del thinking_enabled, content_callback, tools
            captured["profile_name"] = profile.name
            captured["provider_messages"] = list(messages)
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="phase0 reply",
                created_at="2026-03-21T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr(
        "unclaw.core.orchestrator.build_context_messages",
        fake_build_context_messages,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()

        result = orchestrator.run_turn(
            session_id=session.id,
            user_message="Protect the assembly boundary.",
            model_profile_name="main",
            capability_summary=capability_summary,
            system_context_notes=system_context_notes,
        )

        build_kwargs = captured["build_context_messages_kwargs"]
        assert isinstance(build_kwargs, dict)
        assert build_kwargs["session_manager"] is session_manager
        assert build_kwargs["session_id"] == session.id
        assert build_kwargs["user_message"] == "Protect the assembly boundary."
        assert build_kwargs["capability_summary"] is capability_summary
        assert build_kwargs["system_context_notes"] == system_context_notes
        assert build_kwargs["model_profile_name"] == "main"
        assert captured["profile_name"] == "main"
        assert captured["provider_messages"] == sentinel_context
        assert result.context_messages == tuple(sentinel_context)
        assert result.response.content == "phase0 reply"
    finally:
        session_manager.close()


@pytest.mark.integration
@pytest.mark.parametrize("profile_name", ["fast", "main", "deep", "codex"])
def test_phase0_direct_runtime_profiles_keep_expected_tool_guidance(
    monkeypatch: pytest.MonkeyPatch,
    make_temp_project,
    profile_name: str,
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
    tool_registry.register(
        READ_TEXT_FILE_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="phase0 file content",
        ),
    )

    captured_capability_messages: list[str] = []
    captured_tools: list[object | None] = []

    class RouterShouldNotRun:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds
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

        def chat(
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
            captured_tools.append(tools)
            captured_capability_messages.append(
                next(
                    message.content
                    for message in messages
                    if message.role is LLMRole.SYSTEM
                    and "Runtime capability status:" in message.content
                )
            )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content=f"{profile_name} direct reply",
                created_at="2026-03-21T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.router.OllamaProvider", RouterShouldNotRun)
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    expected_native_runtime = settings.models[profile_name].tool_mode == "native"

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Explain the current runtime contract in one sentence.",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Explain the current runtime contract in one sentence.",
            tracer=tracer,
            event_bus=event_bus,
            tool_registry=tool_registry,
        )

        assert reply == f"{profile_name} direct reply"
        assert len(captured_capability_messages) == 1
        assert len(captured_tools) == 1
        capability_message = captured_capability_messages[0]

        if expected_native_runtime:
            assert captured_tools[0] is not None
            assert "you may call tools directly this turn" in capability_message
            assert "user-initiated slash commands only" not in capability_message
        else:
            assert captured_tools[0] is None
            assert "user-initiated slash commands only" in capability_message
            assert "you cannot call tools directly this turn" in capability_message
            assert "you may call tools directly this turn" not in capability_message

        route_event = next(
            event for event in published_events if event.event_type == "route.selected"
        )
        assert route_event.payload["route_kind"] == "chat"
    finally:
        session_manager.close()
