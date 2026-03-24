"""Minimal runtime entrypoint for one non-command chat turn."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from time import perf_counter

from unclaw.core.agent_loop import (
    RuntimeTurnCancellation,
    _INLINE_TOOL_PAYLOAD_FALLBACK_REPLY,
    _InlineToolPayloadAnalysis,
    _MAX_STEPS_FALLBACK_REPLY,
    _PendingToolExecution,
    _RuntimeToolGuardState,
    _TOOL_BUDGET_FALLBACK_REPLY,
    _TURN_CANCELLED_REPLY,
    _execute_runtime_tool_calls,
    _preflight_runtime_tool_batch,
    _recover_inline_native_tool_response,
    _run_agent_loop,
)
from unclaw.core.capabilities import build_runtime_capability_summary
from unclaw.core.command_handler import CommandHandler
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.orchestrator import (
    ModelCallFailedError,
    Orchestrator,
    OrchestratorError,
)
from unclaw.core.reply_discipline import (
    _apply_post_tool_reply_discipline,
)
from unclaw.core.routing import (
    _build_request_routing_note,
    _deduplicate_search_tool_calls,
    _looks_like_deep_search_request,
    _looks_like_joint_entity_request,
    _normalize_runtime_routing_text,
    _resolve_entity_anchor_for_turn,
)
from unclaw.core.runtime_support import (
    _ENTITY_RECENTERING_NOTE_PREFIX,
    _POST_TOOL_GROUNDING_NOTE_PREFIX,
    _build_default_search_grounding_transform,
    _build_entity_recentering_note,
    _build_local_access_control_note,
    _build_model_failure_reply,
    _build_post_tool_grounding_note,
    _build_session_memory_context_note,
    _compose_reply_transforms,
    _is_tool_mode_none_profile,
    _resolve_tool_definitions,
    _supports_native_tool_calling,
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
    turn_cancellation: RuntimeTurnCancellation | None = None,
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
        ToolRegistry() if _is_tool_mode_none_profile(selected_model) else active_tool_registry
    )
    tool_guard_state = _RuntimeToolGuardState(
        tool_timeout_seconds=(
            session_manager.settings.app.runtime.tool_timeout_seconds
        ),
        max_tool_calls_per_turn=(
            session_manager.settings.app.runtime.max_tool_calls_per_turn
        ),
        cancellation=turn_cancellation,
    )

    legacy_tool_definitions = _resolve_tool_definitions(
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

    # Track whether the streaming callback was ever invoked by the provider.
    # If a final reply exists but nothing was streamed, we emit it once as a
    # fallback so callers always receive output when a reply was produced.
    _streamed_something = False
    _effective_stream_func: LLMContentCallback | None
    if stream_output_func is not None:
        _orig_stream_func = stream_output_func

        def _tracking_stream_func(text: str) -> None:
            nonlocal _streamed_something
            _streamed_something = True
            _orig_stream_func(text)

        _effective_stream_func = _tracking_stream_func
    else:
        _effective_stream_func = None

    if runtime_explicit_tool_call is None and legacy_tool_definitions is not None:
        active_assistant_reply_transform = _compose_reply_transforms(
            _build_default_search_grounding_transform(
                session_manager=session_manager,
                session_id=session.id,
                query=user_input,
                turn_start_message_count=turn_start_message_count,
                model_profile_name=selected_model_profile_name,
            ),
            active_assistant_reply_transform,
        )

    try:
        memory_context_note = _build_session_memory_context_note(
            command_handler=command_handler,
            session_id=session.id,
        )
        local_access_note = _build_local_access_control_note(
            command_handler=command_handler,
        )
        request_routing_note = _build_request_routing_note(
            user_input=user_input,
            capability_summary=capability_summary,
        )
        entity_anchor = _resolve_entity_anchor_for_turn(
            session_manager=session_manager,
            session_id=session.id,
            user_input=user_input,
        )
        entity_recentering_note = _build_entity_recentering_note(
            entity_anchor=entity_anchor,
            user_input=user_input,
        )
        system_context_notes = tuple(
            note
            for note in (
                memory_context_note,
                local_access_note,
                request_routing_note,
                entity_recentering_note,
            )
            if note is not None
        )

        responder_tool_definitions = legacy_tool_definitions
        responder_capability_summary = capability_summary

        if runtime_explicit_tool_call is not None and responder_tool_definitions is None:
            assistant_reply = _preflight_runtime_tool_batch(
                tool_calls=(runtime_explicit_tool_call,),
                tool_guard_state=tool_guard_state,
            )
            if assistant_reply is None:
                turn_tool_results.extend(
                    _execute_runtime_tool_calls(
                        session_manager=session_manager,
                        session_id=session.id,
                        tracer=active_tracer,
                        tool_registry=active_tool_registry,
                        tool_calls=(runtime_explicit_tool_call,),
                        tool_guard_state=tool_guard_state,
                    )
                )
                if tool_guard_state.is_cancelled():
                    assistant_reply = _TURN_CANCELLED_REPLY

        if assistant_reply is None and tool_guard_state.is_cancelled():
            assistant_reply = _TURN_CANCELLED_REPLY

        if assistant_reply is None:
            orchestrator = Orchestrator(
                settings=session_manager.settings,
                session_manager=session_manager,
                tracer=active_tracer,
            )
            if assistant_reply is None:
                response = orchestrator.run_turn(
                    session_id=session.id,
                    user_message=user_input,
                    model_profile_name=selected_model_profile_name,
                    capability_summary=responder_capability_summary,
                    system_context_notes=system_context_notes,
                    thinking_enabled=thinking_enabled,
                    content_callback=_effective_stream_func,
                    tools=responder_tool_definitions,
                )
                active_tracer.trace_model_succeeded(
                    session_id=session.id,
                    provider=response.response.provider,
                    model_name=response.response.model_name,
                    finish_reason=response.response.finish_reason,
                    output_length=len(response.response.content),
                    model_duration_ms=response.model_duration_ms,
                    reasoning=response.response.reasoning,
                )

                # --- Inline tool-call recovery (codex regression fix) ---
                # Some native-capable models (e.g. qwen2.5-coder:7b) emit tool
                # call requests as JSON text in `content` instead of the
                # structured `tool_calls` field that the Ollama API expects.
                # When that happens `response.response.tool_calls` is None and
                # the raw JSON reaches the user verbatim.  We recover here by
                # trying to parse the content as a tool call.
                # Explicitly required by BIG-FIX-ROUTER-1R3.  Complies with
                # mandatory_rules rule 5 allowed exception: defensive, scoped to
                # the native legacy path only, easy to audit and remove.
                if not response.response.tool_calls:
                    inline_tool_response, inline_tool_reply = (
                        _recover_inline_native_tool_response(
                            response.response,
                            tool_definitions=responder_tool_definitions or (),
                            max_agent_steps=max_agent_steps,
                        )
                    )
                    if inline_tool_response is not None:
                        response = replace(response, response=inline_tool_response)
                    elif inline_tool_reply is not None:
                        assistant_reply = inline_tool_reply

                # --- Legacy native tool loop fallback ---
                if (
                    assistant_reply is None
                    and response.response.tool_calls
                    and responder_tool_definitions
                    and max_agent_steps > 0
                ):
                    assistant_reply = _run_agent_loop(
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
                        build_post_tool_grounding_note=_build_post_tool_grounding_note,
                        user_entity_surface=(
                            # Only activate the pre-tool entity guard when the anchor
                            # came from the current turn (explicit mention) or from an
                            # explicit correction by the user.  History-fallback anchors
                            # must not override the model's entity choice — they only
                            # inform the recentering note injected into the system prompt.
                            entity_anchor.surface
                            if entity_anchor is not None
                            and (entity_anchor.from_current_turn or entity_anchor.corrected)
                            else ""
                        ),
                        collected_tool_results=turn_tool_results,
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
        assistant_reply = _build_model_failure_reply(exc)
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
    assistant_reply = _apply_post_tool_reply_discipline(
        reply=assistant_reply,
        user_input=user_input,
        tool_results=turn_tool_results,
        turn_cancelled_reply=_TURN_CANCELLED_REPLY,
    )

    # Fallback: if a callback was registered but the provider never invoked it
    # (e.g. the model returned all content in a final aggregated chunk with no
    # streaming deltas), emit the final reply once so callers always receive
    # output through the callback when a reply was produced.
    if stream_output_func is not None and not _streamed_something and assistant_reply:
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
