from __future__ import annotations

from types import SimpleNamespace

import pytest

from unclaw.core.command_handler import CommandHandler
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolDefinition, ToolPermissionLevel, ToolResult
from unclaw.tools.registry import ToolRegistry

pytestmark = [pytest.mark.integration, pytest.mark.real_ollama]


def _override_main_profile(settings, *, model_name: str, tool_mode: str) -> None:
    profile = settings.models["main"]
    settings.models["main"] = profile.__class__(
        name=profile.name,
        provider=profile.provider,
        model_name=model_name,
        temperature=0.0,
        thinking_supported=profile.thinking_supported,
        tool_mode=tool_mode,
        keep_alive=profile.keep_alive,
    )


def test_real_ollama_run_user_turn_streams_end_to_end(
    make_temp_project,
    real_ollama_model_name: str,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    _override_main_profile(
        settings,
        model_name=real_ollama_model_name,
        tool_mode="native",
    )
    session_manager = SessionManager.from_settings(settings)
    event_bus = EventBus()
    published_events: list[object] = []
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
    streamed_chunks: list[str] = []

    try:
        session = session_manager.ensure_current_session()
        user_input = "Reply with one short sentence about local-only AI assistants."
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
            event_bus=event_bus,
            stream_output_func=streamed_chunks.append,
            tool_registry=ToolRegistry(),
        )

        assert reply.strip()
        assert streamed_chunks
        assert "".join(streamed_chunks) == reply

        stored_messages = session_manager.list_messages(session.id)
        assert stored_messages[-1].role is MessageRole.ASSISTANT
        assert stored_messages[-1].content == reply

        event_types = [getattr(event, "event_type", None) for event in published_events]
        assert event_types == [
            "runtime.started",
            "route.selected",
            "model.called",
            "model.succeeded",
            "assistant.reply.persisted",
        ]
    finally:
        session_manager.close()


def test_real_ollama_run_user_turn_streaming_native_tool_path_when_model_calls_tool(
    make_temp_project,
    real_ollama_model_name: str,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    _override_main_profile(
        settings,
        model_name=real_ollama_model_name,
        tool_mode="native",
    )
    session_manager = SessionManager.from_settings(settings)
    event_bus = EventBus()
    published_events: list[object] = []
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
    streamed_chunks: list[str] = []
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolDefinition(
            name="get_runtime_token",
            description="Return the fixed runtime token for integration testing.",
            permission_level=ToolPermissionLevel.LOCAL_READ,
            arguments={},
        ),
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="real-tool-ok",
        ),
    )

    try:
        session = session_manager.ensure_current_session()
        user_input = (
            "Call the get_runtime_token tool before answering. "
            "After the tool returns, reply with exactly the tool result."
        )
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
            event_bus=event_bus,
            stream_output_func=streamed_chunks.append,
            tool_registry=tool_registry,
        )

        tool_finished_events = [
            event
            for event in published_events
            if getattr(event, "event_type", None) == "tool.finished"
        ]
        if not tool_finished_events:
            pytest.skip(
                "The live model did not emit a native tool call for the constrained "
                "runtime prompt. Real provider, streaming, and runtime boundaries are "
                "still covered by the other real_ollama tests."
            )

        assert len(tool_finished_events) == 1
        assert getattr(tool_finished_events[0], "payload", {}).get("success") is True
        assert "real-tool-ok" in reply
        assert streamed_chunks
        assert "".join(streamed_chunks) == reply

        stored_tool_messages = [
            message.content
            for message in session_manager.list_messages(session.id)
            if message.role is MessageRole.TOOL
        ]
        assert len(stored_tool_messages) == 1
        assert "Tool: get_runtime_token" in stored_tool_messages[0]
        assert "Outcome: success" in stored_tool_messages[0]
        assert "real-tool-ok" in stored_tool_messages[0]
    finally:
        session_manager.close()
