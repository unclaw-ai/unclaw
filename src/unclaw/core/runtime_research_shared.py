"""Shared runtime/research bridge helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from unclaw.constants import DEFAULT_RUNTIME_AGENT_STEP_LIMIT

if TYPE_CHECKING:
    from unclaw.core.agent_loop import RuntimeTurnCancellation
    from unclaw.core.command_handler import CommandHandler
    from unclaw.core.session_manager import SessionManager
    from unclaw.llm.base import LLMContentCallback
    from unclaw.logs.event_bus import EventBus
    from unclaw.logs.tracer import Tracer
    from unclaw.tools.contracts import ToolCall
    from unclaw.tools.registry import ToolRegistry


def run_user_turn(
    *,
    session_manager: SessionManager,
    command_handler: CommandHandler,
    user_input: str,
    tracer: Tracer | None = None,
    event_bus: EventBus | None = None,
    stream_output_func: LLMContentCallback | None = None,
    tool_registry: ToolRegistry | None = None,
    explicit_tool_call: ToolCall | None = None,
    assistant_reply_transform: Callable[[str], str] | None = None,
    tool_call_callback: Callable[[ToolCall], None] | None = None,
    max_agent_steps: int = DEFAULT_RUNTIME_AGENT_STEP_LIMIT,
    turn_cancellation: RuntimeTurnCancellation | None = None,
) -> str:
    """Run one user turn through the shared runtime entrypoint."""
    from unclaw.core.runtime import run_user_turn as _run_user_turn

    return _run_user_turn(
        session_manager=session_manager,
        command_handler=command_handler,
        user_input=user_input,
        tracer=tracer,
        event_bus=event_bus,
        stream_output_func=stream_output_func,
        tool_registry=tool_registry,
        explicit_tool_call=explicit_tool_call,
        assistant_reply_transform=assistant_reply_transform,
        tool_call_callback=tool_call_callback,
        max_agent_steps=max_agent_steps,
        turn_cancellation=turn_cancellation,
    )


__all__ = ["run_user_turn"]
