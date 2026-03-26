"""Minimal runtime entrypoint for one non-command chat turn."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from time import perf_counter

import unclaw.core.agent_loop as _agent_loop
from unclaw.core.capabilities import build_runtime_capability_summary
from unclaw.core.command_handler import CommandHandler
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.orchestrator import (
    ModelCallFailedError,
    Orchestrator,
    OrchestratorError,
    OrchestratorTurnResult,
)
import unclaw.core.runtime_support as _runtime_support
from unclaw.core.session_manager import SessionManager, SessionManagerError
from unclaw.core.timing import elapsed_ms
from unclaw.constants import (
    DEFAULT_RUNTIME_AGENT_STEP_LIMIT,
    EMPTY_RESPONSE_REPLY,
    RUNTIME_ERROR_REPLY,
)
from unclaw.errors import ConfigurationError, UnclawError
from unclaw.llm.base import LLMContentCallback, LLMError, LLMMessage, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.tools.contracts import ToolCall, ToolResult
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
    turn_cancellation: _agent_loop.RuntimeTurnCancellation | None = None,
) -> str:
    """Run the minimal runtime path and persist the assistant reply."""
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

    selected_model_profile_name = command_handler.current_model_profile.name
    selected_model = command_handler.current_model_profile
    thinking_enabled = command_handler.thinking_enabled is True
    active_tool_registry = tool_registry or create_default_tool_registry(
        session_manager.settings,
        session_manager=session_manager,
    )
    capability_tool_registry = (
        ToolRegistry()
        if _runtime_support._is_tool_mode_none_profile(selected_model)
        else active_tool_registry
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

    legacy_tool_definitions = _runtime_support._resolve_tool_definitions(
        tool_registry=active_tool_registry,
        model_profile=selected_model,
    )
    legacy_can_call_tools = legacy_tool_definitions is not None
    capability_summary = build_runtime_capability_summary(
        tool_registry=capability_tool_registry,
        memory_summary_available=command_handler.memory_manager is not None,
        model_can_call_tools=legacy_can_call_tools,
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

    active_assistant_reply_transform = assistant_reply_transform
    system_context_notes: tuple[str, ...] = ()
    runtime_explicit_tool_call = explicit_tool_call
    assistant_reply: str | None = None
    turn_tool_results: list[ToolResult] = []
    turn_start_message_count = len(session_manager.list_messages(session.id))
    pre_turn_goal_state = session_manager.get_session_goal_state(session.id)
    pre_turn_progress_ledger = session_manager.get_session_progress_ledger(session.id)
    orchestrator: Orchestrator | None = None
    grounded_finalization_used = False

    buffered_stream_chunks: list[str] | None
    _effective_stream_func: LLMContentCallback | None
    # Stream directly to the caller during the agent loop so that output
    # appears progressively instead of being held until the whole turn ends.
    # Only the initial model call is separately buffered (below) so we can
    # discard its text if the model decides to call tools instead.
    live_streamed = False
    if stream_output_func is not None:
        buffered_stream_chunks = []

        def _live_stream_func(text: str) -> None:
            nonlocal live_streamed
            live_streamed = True
            stream_output_func(text)

        _effective_stream_func = _live_stream_func
    else:
        buffered_stream_chunks = None
        _effective_stream_func = None

    if runtime_explicit_tool_call is None and legacy_tool_definitions is not None:
        active_assistant_reply_transform = _runtime_support._compose_reply_transforms(
            _runtime_support._build_default_search_grounding_transform(
                session_manager=session_manager,
                session_id=session.id,
                query=user_input,
                turn_start_message_count=turn_start_message_count,
                model_profile_name=selected_model_profile_name,
            ),
            active_assistant_reply_transform,
        )

    try:
        memory_context_note = _runtime_support._build_session_memory_context_note(
            command_handler=command_handler,
            session_id=session.id,
        )
        task_continuity_note = _runtime_support._build_session_task_continuity_note(
            session_manager=session_manager,
            session_id=session.id,
        )
        local_access_note = _runtime_support._build_local_access_control_note(
            command_handler=command_handler,
        )

        system_context_notes = tuple(
            note
            for note in (
                memory_context_note,
                task_continuity_note,
                local_access_note,
            )
            if note is not None
        )

        responder_tool_definitions = legacy_tool_definitions
        responder_capability_summary = capability_summary

        if runtime_explicit_tool_call is not None and responder_tool_definitions is None:
            assistant_reply = _agent_loop._preflight_runtime_tool_batch(
                tool_calls=(runtime_explicit_tool_call,),
                tool_guard_state=tool_guard_state,
            )
            if assistant_reply is None:
                turn_tool_results.extend(
                    _agent_loop._execute_runtime_tool_calls(
                        session_manager=session_manager,
                        session_id=session.id,
                        tracer=active_tracer,
                        tool_registry=active_tool_registry,
                        tool_calls=(runtime_explicit_tool_call,),
                        tool_guard_state=tool_guard_state,
                    )
                )
                if tool_guard_state.is_cancelled():
                    assistant_reply = _agent_loop._TURN_CANCELLED_REPLY

        if assistant_reply is None and tool_guard_state.is_cancelled():
            assistant_reply = _agent_loop._TURN_CANCELLED_REPLY

        if assistant_reply is None:
            if orchestrator is None:
                orchestrator = Orchestrator(
                    settings=session_manager.settings,
                    session_manager=session_manager,
                    tracer=active_tracer,
                )

            def _finalize_responder_turn(
                turn_result: OrchestratorTurnResult,
            ) -> tuple[OrchestratorTurnResult, str | None]:
                active_tracer.trace_model_succeeded(
                    session_id=session.id,
                    provider=turn_result.response.provider,
                    model_name=turn_result.response.model_name,
                    finish_reason=turn_result.response.finish_reason,
                    output_length=len(turn_result.response.content),
                    model_duration_ms=turn_result.model_duration_ms,
                    reasoning=turn_result.response.reasoning,
                )

                inline_tool_reply: str | None = None
                if not turn_result.response.tool_calls:
                    inline_tool_response, inline_tool_reply = (
                        _agent_loop._recover_inline_native_tool_response(
                            turn_result.response,
                            tool_definitions=responder_tool_definitions or (),
                            max_agent_steps=max_agent_steps,
                        )
                    )
                    if inline_tool_response is not None:
                        turn_result = replace(
                            turn_result,
                            response=inline_tool_response,
                        )

                return turn_result, inline_tool_reply

            def _run_responder_turn(
                *,
                system_notes: tuple[str, ...],
                content_callback: LLMContentCallback | None,
            ) -> tuple[OrchestratorTurnResult, str | None]:
                turn_result = orchestrator.run_turn(
                    session_id=session.id,
                    user_message=user_input,
                    model_profile_name=selected_model_profile_name,
                    capability_summary=responder_capability_summary,
                    system_context_notes=system_notes,
                    thinking_enabled=thinking_enabled,
                    content_callback=content_callback,
                    tools=responder_tool_definitions,
                )
                return _finalize_responder_turn(turn_result)

            def _run_responder_recovery_turn(
                *,
                prior_context_messages: tuple[LLMMessage, ...],
                recovery_note: str,
                content_callback: LLMContentCallback | None,
            ) -> tuple[OrchestratorTurnResult, str | None]:
                recovery_messages = list(prior_context_messages)
                insert_index = len(recovery_messages)
                for index in range(len(recovery_messages) - 1, -1, -1):
                    if recovery_messages[index].role is LLMRole.USER:
                        insert_index = index
                        break
                recovery_messages.insert(
                    insert_index,
                    LLMMessage(role=LLMRole.SYSTEM, content=recovery_note),
                )
                turn_result = orchestrator.call_model(
                    session_id=session.id,
                    messages=tuple(recovery_messages),
                    model_profile_name=selected_model_profile_name,
                    thinking_enabled=thinking_enabled,
                    content_callback=content_callback,
                    tools=responder_tool_definitions,
                )
                return _finalize_responder_turn(turn_result)

            initial_stream_chunks: list[str] | None = None
            initial_content_callback = _effective_stream_func
            if responder_tool_definitions is not None and _effective_stream_func is not None:
                initial_stream_chunks = []

                def _buffer_initial_stream(text: str) -> None:
                    initial_stream_chunks.append(text)

                initial_content_callback = _buffer_initial_stream

            def _flush_initial_stream_chunks() -> None:
                if (
                    initial_stream_chunks is None
                    or _effective_stream_func is None
                    or not initial_stream_chunks
                ):
                    return
                for chunk in initial_stream_chunks:
                    _effective_stream_func(chunk)
                initial_stream_chunks.clear()

            if assistant_reply is None:
                response, assistant_reply = _run_responder_turn(
                    system_notes=system_context_notes,
                    content_callback=initial_content_callback,
                )

                if (
                    assistant_reply is None
                    and not response.response.tool_calls
                    and responder_tool_definitions is not None
                    and not turn_tool_results
                    and not tool_guard_state.is_cancelled()
                ):
                    response, assistant_reply = _run_responder_recovery_turn(
                        prior_context_messages=response.context_messages,
                        recovery_note=(
                            _runtime_support._build_model_native_tool_recovery_note()
                        ),
                        content_callback=_effective_stream_func,
                    )
                elif assistant_reply is None and not response.response.tool_calls:
                    _flush_initial_stream_chunks()

                # --- Legacy native tool loop fallback ---
                if (
                    assistant_reply is None
                    and response.response.tool_calls
                    and responder_tool_definitions
                    and max_agent_steps > 0
                ):
                    assistant_reply = _agent_loop._run_agent_loop(
                        first_response=response,
                        orchestrator=orchestrator,
                        session_id=session.id,
                        session_manager=session_manager,
                        tracer=active_tracer,
                        tool_registry=active_tool_registry,
                        tool_definitions=responder_tool_definitions,
                        model_profile_name=selected_model_profile_name,
                        thinking_enabled=thinking_enabled,
                        content_callback=_effective_stream_func,
                        max_steps=max_agent_steps,
                        tool_guard_state=tool_guard_state,
                        tool_call_callback=tool_call_callback,
                        build_post_tool_grounding_note=(
                            _runtime_support._build_post_tool_grounding_note
                        ),
                        collected_tool_results=turn_tool_results,
                        user_input=user_input,
                    )
                elif assistant_reply is None:
                    assistant_reply = (
                        response.response.content.strip() or EMPTY_RESPONSE_REPLY
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

    if active_assistant_reply_transform is not None:
        assistant_reply = active_assistant_reply_transform(assistant_reply)
    if orchestrator is None:
        orchestrator = Orchestrator(
            settings=session_manager.settings,
            session_manager=session_manager,
            tracer=active_tracer,
        )
    assistant_reply, grounded_finalization_used = (
        _runtime_support._finalize_grounded_reply(
            orchestrator=orchestrator,
            session_id=session.id,
            model_profile_name=selected_model_profile_name,
            thinking_enabled=thinking_enabled,
            tracer=active_tracer,
            user_input=user_input,
            assistant_draft_reply=assistant_reply,
            tool_results=turn_tool_results,
            session_goal_state=pre_turn_goal_state,
            session_progress_ledger=pre_turn_progress_ledger,
            turn_cancelled_reply=_agent_loop._TURN_CANCELLED_REPLY,
            tool_registry=tool_registry,
        )
    )

    # Streaming emission: avoid duplicating already-shown live content.
    # The caller's finish() method handles showing the final text and
    # detecting refinements vs streamed drafts.
    if stream_output_func is not None and not live_streamed:
        # No live streaming happened — emit the final reply.
        stream_output_func(assistant_reply)

    session_manager.add_message(
        MessageRole.ASSISTANT,
        assistant_reply,
        session_id=session.id,
    )
    _runtime_support._persist_session_goal_state_from_runtime_facts(
        session_manager=session_manager,
        session_id=session.id,
        user_input=user_input,
        tool_results=turn_tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=_agent_loop._TURN_CANCELLED_REPLY,
    )
    _runtime_support._persist_session_progress_ledger_from_runtime_facts(
        session_manager=session_manager,
        session_id=session.id,
        tool_results=turn_tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=_agent_loop._TURN_CANCELLED_REPLY,
    )
    active_tracer.trace_assistant_reply_persisted(
        session_id=session.id,
        output_length=len(assistant_reply),
        turn_duration_ms=elapsed_ms(turn_started_at),
    )
    return assistant_reply
