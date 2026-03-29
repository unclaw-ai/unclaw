"""Runtime entrypoint for the single-agent mission loop."""

from __future__ import annotations

from collections.abc import Callable
import json
from time import perf_counter

import unclaw.core.agent_kernel as _agent_kernel
import unclaw.core.agent_loop as _agent_loop
import unclaw.core.runtime_support as _runtime_support
from unclaw.core.command_handler import CommandHandler
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.orchestrator import (
    ModelCallFailedError,
    Orchestrator,
    OrchestratorError,
)
from unclaw.core.session_manager import SessionManager, SessionManagerError
from unclaw.core.timing import elapsed_ms
from unclaw.constants import (
    DEFAULT_RUNTIME_AGENT_STEP_LIMIT,
    EMPTY_RESPONSE_REPLY,
    RUNTIME_ERROR_REPLY,
)
from unclaw.errors import ConfigurationError, UnclawError
from unclaw.llm.base import LLMContentCallback, LLMError
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
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
    mission_event_callback: Callable[[str], None] | None = None,
    max_agent_steps: int = DEFAULT_RUNTIME_AGENT_STEP_LIMIT,
    turn_cancellation: _agent_loop.RuntimeTurnCancellation | None = None,
) -> str:
    """Run one user turn through the single-agent mission loop."""

    session = session_manager.ensure_current_session()
    active_tracer = tracer or Tracer(
        event_bus=event_bus or EventBus(),
        event_repository=session_manager.event_repository,
        include_reasoning_text=(
            session_manager.settings.app.logging.include_reasoning_text
        ),
    )
    active_tracer.runtime_log_path = (
        session_manager.settings.paths.log_file_path
        if session_manager.settings.app.logging.file_enabled
        else None
    )

    selected_model = command_handler.current_model_profile
    selected_model_profile_name = selected_model.name
    thinking_enabled = command_handler.thinking_enabled is True
    active_tool_registry = tool_registry or create_default_tool_registry(
        session_manager.settings,
        session_manager=session_manager,
    )
    tool_definitions = (
        ()
        if _runtime_support._is_tool_mode_none_profile(selected_model)
        else tuple(active_tool_registry.list_tools())
    )
    tool_guard_state = _agent_loop._RuntimeToolGuardState(
        tool_timeout_seconds=(
            session_manager.settings.app.runtime.tool_timeout_seconds
        ),
        max_tool_calls_per_turn=(
            session_manager.settings.app.runtime.max_tool_calls_per_turn
        ),
        cancellation=turn_cancellation,
    )
    turn_started_at = perf_counter()

    active_tracer.trace_runtime_started(
        session_id=session.id,
        model_profile_name=selected_model_profile_name,
        provider=selected_model.provider,
        model_name=selected_model.model_name,
        thinking_enabled=thinking_enabled,
        input_length=len(user_input),
    )

    assistant_reply: str | None = None

    try:
        if explicit_tool_call is not None:
            assistant_reply = _run_explicit_tool_call(
                session_manager=session_manager,
                session_id=session.id,
                tracer=active_tracer,
                tool_registry=active_tool_registry,
                tool_call=explicit_tool_call,
                tool_guard_state=tool_guard_state,
                tool_call_callback=tool_call_callback,
            )
        else:
            orchestrator = Orchestrator(
                settings=session_manager.settings,
                session_manager=session_manager,
                tracer=active_tracer,
            )
            mission_result = _agent_kernel.run_agent_kernel_turn(
                session_manager=session_manager,
                session_id=session.id,
                user_input=user_input,
                orchestrator=orchestrator,
                tracer=active_tracer,
                tool_registry=active_tool_registry,
                tool_definitions=tool_definitions,
                model_profile_name=selected_model_profile_name,
                thinking_enabled=thinking_enabled,
                capability_summary=None,
                system_context_notes=(),
                max_steps=max_agent_steps,
                tool_guard_state=tool_guard_state,
                tool_call_callback=tool_call_callback,
                mission_event_callback=mission_event_callback,
                content_callback=None,
                existing_mission_state=session_manager.get_current_mission_state(
                    session.id
                ),
            )
            assistant_reply = mission_result.assistant_reply or EMPTY_RESPONSE_REPLY
            if mission_result.persisted and mission_result.mission_state is not None:
                session_manager.persist_legacy_mission_projection(
                    mission_state=mission_result.mission_state,
                    session_id=session.id,
                )

    except ModelCallFailedError as exc:
        active_tracer.trace_model_failed(
            session_id=session.id,
            provider=exc.provider,
            model_profile_name=exc.model_profile_name,
            model_name=exc.model_name,
            model_duration_ms=exc.duration_ms,
            error=str(exc),
        )
        assistant_reply = _runtime_support._build_model_failure_reply(exc)
    except (
        ConfigurationError,
        LLMError,
        OrchestratorError,
        SessionManagerError,
    ) as exc:
        active_tracer.trace_model_failed(
            session_id=session.id,
            provider=selected_model.provider,
            model_profile_name=selected_model_profile_name,
            model_name=selected_model.model_name,
            error=str(exc),
        )
        assistant_reply = RUNTIME_ERROR_REPLY

    if assistant_reply is None:
        raise UnclawError(
            "Runtime turn completed without producing an assistant reply."
        )

    assistant_reply = _sanitize_assistant_reply(
        assistant_reply=assistant_reply,
        session_manager=session_manager,
        session_id=session.id,
    )
    if assistant_reply_transform is not None:
        assistant_reply = assistant_reply_transform(assistant_reply)
    if stream_output_func is not None:
        stream_output_func(assistant_reply)

    session_manager.add_message(
        MessageRole.ASSISTANT,
        assistant_reply,
        session_id=session.id,
    )
    active_tracer.trace_assistant_reply_persisted(
        session_id=session.id,
        output_length=len(assistant_reply),
        turn_duration_ms=elapsed_ms(turn_started_at),
    )
    return assistant_reply


def _run_explicit_tool_call(
    *,
    session_manager: SessionManager,
    session_id: str,
    tracer: Tracer,
    tool_registry: ToolRegistry,
    tool_call: ToolCall,
    tool_guard_state: _agent_loop._RuntimeToolGuardState,
    tool_call_callback: Callable[[ToolCall], None] | None,
) -> str:
    stop_reply = _agent_loop._preflight_runtime_tool_batch(
        tool_calls=(tool_call,),
        tool_guard_state=tool_guard_state,
    )
    if stop_reply is not None:
        return stop_reply
    tool_results = _agent_loop._execute_runtime_tool_calls(
        session_manager=session_manager,
        session_id=session_id,
        tracer=tracer,
        tool_registry=tool_registry,
        tool_calls=(tool_call,),
        tool_guard_state=tool_guard_state,
        tool_call_callback=tool_call_callback,
    )
    if not tool_results:
        return EMPTY_RESPONSE_REPLY
    result = tool_results[-1]
    if result.success:
        return result.output_text or EMPTY_RESPONSE_REPLY
    return result.error or result.output_text or RUNTIME_ERROR_REPLY


def _sanitize_assistant_reply(
    *,
    assistant_reply: str,
    session_manager: SessionManager,
    session_id: str,
) -> str:
    stripped = assistant_reply.strip()
    if not stripped:
        return assistant_reply
    if not (
        _looks_like_internal_json_reply(stripped)
        or _looks_like_internal_scaffolding_reply(stripped)
    ):
        return assistant_reply
    mission_state = session_manager.get_current_mission_state(session_id)
    if mission_state is None:
        return EMPTY_RESPONSE_REPLY
    return _agent_kernel._build_mission_status_reply(mission_state)


def _looks_like_internal_json_reply(reply_text: str) -> bool:
    if not reply_text.startswith(("{", "[")):
        return False
    try:
        payload = json.loads(reply_text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    keys = set(payload)
    if {"mission_action", "task_board"}.issubset(keys):
        return True
    return {"mission_id", "tasks", "status"}.issubset(keys)


def _looks_like_internal_scaffolding_reply(reply_text: str) -> bool:
    lines = [line.strip() for line in reply_text.splitlines() if line.strip()]
    if not lines:
        return False
    prefixes = (
        "Mission goal:",
        "Current active task:",
        "Completed tasks:",
    )
    return sum(
        1 for line in lines if any(line.startswith(prefix) for prefix in prefixes)
    ) >= 2


__all__ = ["run_user_turn"]
