"""Minimal runtime entrypoint for one non-command chat turn."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
import threading
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any

from unclaw.core.capabilities import build_runtime_capability_summary
from unclaw.core.context_builder import build_untrusted_tool_message_content
from unclaw.core.command_handler import CommandHandler
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.orchestrator import (
    ModelCallFailedError,
    Orchestrator,
    OrchestratorError,
    OrchestratorTurnResult,
)
from unclaw.core.session_manager import SessionManager, SessionManagerError
from unclaw.core.timing import elapsed_ms
from unclaw.constants import (
    DEFAULT_RUNTIME_AGENT_STEP_LIMIT,
    EMPTY_RESPONSE_REPLY,
    RUNTIME_ERROR_REPLY,
    RUNTIME_TOOL_RESULT_POLL_INTERVAL_SECONDS,
)
from unclaw.errors import ConfigurationError, UnclawError
from unclaw.llm.base import LLMContentCallback, LLMError, LLMMessage, LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.memory.protocols import SessionMemoryContextProvider
from unclaw.schemas.chat import MessageRole
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolResult
from unclaw.tools.dispatcher import ToolDispatcher
from unclaw.tools.registry import ToolRegistry
_MAX_STEPS_FALLBACK_REPLY = (
    "I reached the maximum number of steps for this request. "
    "Here is what I found so far."
)
_TOOL_BUDGET_FALLBACK_REPLY = (
    "I stopped after reaching the tool-call limit for this request."
)
_TURN_CANCELLED_REPLY = "This request was cancelled before tool work completed."
_INLINE_TOOL_PAYLOAD_FALLBACK_REPLY = (
    "I couldn't safely execute a tool request because the model returned an "
    "invalid tool payload. Please try again or rephrase the request."
)


@dataclass(slots=True)
class RuntimeTurnCancellation:
    """Minimal turn-local cancellation handle for runtime tool execution."""

    _cancel_event: threading.Event = field(
        default_factory=threading.Event,
        init=False,
        repr=False,
    )

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()


@dataclass(slots=True)
class _RuntimeToolGuardState:
    tool_timeout_seconds: float
    max_tool_calls_per_turn: int
    cancellation: RuntimeTurnCancellation | None = None
    tool_calls_started: int = 0

    def is_cancelled(self) -> bool:
        return self.cancellation is not None and self.cancellation.is_cancelled()


@dataclass(slots=True)
class _PendingToolExecution:
    tool_call: ToolCall
    started_at: float
    done_event: threading.Event
    thread: threading.Thread | None = None
    result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class _InlineToolPayloadAnalysis:
    tool_calls: tuple[ToolCall, ...] | None = None
    raw_tool_calls_payload: tuple[dict[str, Any], ...] | None = None
    invalid_tool_payload: bool = False


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
        system_context_notes = tuple(
            note
            for note in (memory_context_note, local_access_note)
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
                _execute_runtime_tool_calls(
                    session_manager=session_manager,
                    session_id=session.id,
                    tracer=active_tracer,
                    tool_registry=active_tool_registry,
                    tool_calls=(runtime_explicit_tool_call,),
                    tool_guard_state=tool_guard_state,
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


def _build_session_memory_context_note(
    *,
    command_handler: CommandHandler,
    session_id: str,
) -> str | None:
    memory_manager = command_handler.memory_manager
    if (
        memory_manager is None
        or not isinstance(memory_manager, SessionMemoryContextProvider)
    ):
        return None

    note = memory_manager.build_context_note(session_id)
    if not isinstance(note, str):
        return None

    normalized_note = note.strip()
    return normalized_note or None


def _build_local_access_control_note(
    *,
    command_handler: CommandHandler,
) -> str:
    preset_name = command_handler.settings.app.security.tools.files.control_preset
    return (
        "Local access control: the current control preset "
        f"('{preset_name}') only changes elevated file and terminal boundaries. "
        "It never disables system_info, web tools, session history, long-term "
        "memory, or active skill tools when the active model profile can call tools."
    )


def _is_tool_mode_none_profile(model_profile: Any) -> bool:
    """Return True when the profile has tool_mode=none (e.g. fast)."""
    tool_mode = getattr(model_profile, "tool_mode", None)
    if not isinstance(tool_mode, str):
        return False
    return tool_mode.strip().lower() == "none"


def _compose_reply_transforms(
    first: Callable[[str], str] | None,
    second: Callable[[str], str] | None,
) -> Callable[[str], str] | None:
    if first is None:
        return second
    if second is None:
        return first

    def _composed(reply: str) -> str:
        return second(first(reply))

    return _composed


def _build_default_search_grounding_transform(
    *,
    session_manager: SessionManager,
    session_id: str,
    query: str,
    turn_start_message_count: int,
    model_profile_name: str,
) -> Callable[[str], str]:
    from unclaw.core.research_flow import apply_search_grounding_from_history

    def _grounding_transform(reply: str) -> str:
        return apply_search_grounding_from_history(
            reply,
            query=query,
            session_manager=session_manager,
            session_id=session_id,
            turn_start_message_count=turn_start_message_count,
            model_profile_name=model_profile_name,
        )

    return _grounding_transform


def _run_agent_loop(
    *,
    first_response: OrchestratorTurnResult,
    orchestrator: Orchestrator,
    session_id: str,
    session_manager: SessionManager,
    tracer: Tracer,
    tool_registry: ToolRegistry,
    tool_definitions: Sequence[ToolDefinition],
    model_profile_name: str,
    thinking_enabled: bool,
    content_callback: LLMContentCallback | None,
    max_steps: int,
    tool_guard_state: _RuntimeToolGuardState,
    tool_call_callback: Callable[[ToolCall], None] | None,
    user_input: str = "",
) -> str:
    """Execute the observation-action loop until text reply or step limit."""
    from unclaw.tools.web_entity_guard import (
        apply_entity_guard_to_tool_calls,
        extract_user_entity_surface,
    )

    user_entity_surface = extract_user_entity_surface(user_input)
    context_messages: list[LLMMessage] = list(first_response.context_messages)
    current_response = first_response

    for _step in range(max_steps):
        tool_calls = current_response.response.tool_calls
        if not tool_calls:
            return current_response.response.content.strip() or EMPTY_RESPONSE_REPLY

        stop_reply = _preflight_runtime_tool_batch(
            tool_calls=tool_calls,
            tool_guard_state=tool_guard_state,
        )
        if stop_reply is not None:
            return stop_reply

        # Append the assistant message (with tool_calls) to context.
        context_messages.append(
            LLMMessage(
                role=LLMRole.ASSISTANT,
                content=current_response.response.content,
                tool_calls_payload=_extract_raw_tool_calls(current_response.response),
            )
        )

        # Apply pre-tool entity guard: restore literal entity if model drifted.
        guarded_tool_calls = apply_entity_guard_to_tool_calls(
            tool_calls,
            user_entity_surface,
        )

        # Execute tool calls concurrently, then replay results in model order.
        tool_results = _execute_runtime_tool_calls(
            session_manager=session_manager,
            session_id=session_id,
            tracer=tracer,
            tool_registry=tool_registry,
            tool_calls=guarded_tool_calls,
            tool_guard_state=tool_guard_state,
            tool_call_callback=tool_call_callback,
        )
        for tool_result in tool_results:
            context_messages.append(
                LLMMessage(
                    role=LLMRole.TOOL,
                    content=build_untrusted_tool_message_content(
                        tool_result.output_text
                    ),
                )
            )

        if tool_guard_state.is_cancelled():
            return _TURN_CANCELLED_REPLY

        # Call the model again with extended context.
        current_response = orchestrator.call_model(
            session_id=session_id,
            messages=context_messages,
            model_profile_name=model_profile_name,
            thinking_enabled=thinking_enabled,
            content_callback=content_callback,
            tools=tool_definitions,
        )
        tracer.trace_model_succeeded(
            session_id=session_id,
            provider=current_response.response.provider,
            model_name=current_response.response.model_name,
            finish_reason=current_response.response.finish_reason,
            output_length=len(current_response.response.content),
            model_duration_ms=current_response.model_duration_ms,
            reasoning=current_response.response.reasoning,
        )

    # Final check: last model call may have produced a text reply.
    if not current_response.response.tool_calls:
        return current_response.response.content.strip() or EMPTY_RESPONSE_REPLY

    # max_steps reached without a final text reply.
    return _MAX_STEPS_FALLBACK_REPLY


def _preflight_runtime_tool_batch(
    *,
    tool_calls: Sequence[ToolCall],
    tool_guard_state: _RuntimeToolGuardState,
) -> str | None:
    if tool_guard_state.is_cancelled():
        return _TURN_CANCELLED_REPLY
    if (
        tool_guard_state.tool_calls_started + len(tool_calls)
        > tool_guard_state.max_tool_calls_per_turn
    ):
        return _TOOL_BUDGET_FALLBACK_REPLY
    return None


def _execute_runtime_tool_calls(
    *,
    session_manager: SessionManager,
    session_id: str,
    tracer: Tracer,
    tool_registry: ToolRegistry,
    tool_calls: Sequence[ToolCall],
    tool_guard_state: _RuntimeToolGuardState,
    tool_call_callback: Callable[[ToolCall], None] | None = None,
) -> tuple[ToolResult, ...]:
    """Execute one tool-call batch while keeping context and persistence ordered."""
    pending_calls = _start_pending_tool_executions(
        session_id=session_id,
        tracer=tracer,
        tool_registry=tool_registry,
        tool_calls=tool_calls,
        tool_guard_state=tool_guard_state,
        tool_call_callback=tool_call_callback,
    )
    if not pending_calls:
        return ()

    resolved_results = _collect_pending_tool_results(
        pending_calls=pending_calls,
        tool_guard_state=tool_guard_state,
    )
    tool_results: list[ToolResult] = []
    for pending_call, tool_result, finished_at in resolved_results:
        _finalize_runtime_tool_call(
            session_manager=session_manager,
            session_id=session_id,
            tracer=tracer,
            tool_call=pending_call.tool_call,
            tool_result=tool_result,
            tool_duration_ms=max(
                0,
                round((finished_at - pending_call.started_at) * 1000),
            ),
            skill_id=tool_registry.get_owner_skill_id(pending_call.tool_call.tool_name),
        )
        tool_results.append(tool_result)

    return tuple(tool_results)


def _start_pending_tool_executions(
    *,
    session_id: str,
    tracer: Tracer,
    tool_registry: ToolRegistry,
    tool_calls: Sequence[ToolCall],
    tool_guard_state: _RuntimeToolGuardState,
    tool_call_callback: Callable[[ToolCall], None] | None = None,
) -> list[_PendingToolExecution]:
    pending_calls: list[_PendingToolExecution] = []
    for tool_call in tool_calls:
        if tool_guard_state.is_cancelled():
            break

        started_at = perf_counter()
        if tool_call_callback is not None:
            tool_call_callback(tool_call)
        tracer.trace_tool_started(
            session_id=session_id,
            tool_name=tool_call.tool_name,
            arguments=tool_call.arguments,
            skill_id=tool_registry.get_owner_skill_id(tool_call.tool_name),
        )
        done_event = threading.Event()
        pending_call = _PendingToolExecution(
            tool_call=tool_call,
            started_at=started_at,
            done_event=done_event,
        )
        pending_call.thread = threading.Thread(
            target=_run_pending_tool_execution,
            kwargs={
                "pending_call": pending_call,
                "tool_registry": tool_registry,
            },
            daemon=True,
        )
        pending_call.thread.start()
        pending_calls.append(pending_call)

    tool_guard_state.tool_calls_started += len(pending_calls)
    return pending_calls


def _run_pending_tool_execution(
    *,
    pending_call: _PendingToolExecution,
    tool_registry: ToolRegistry,
) -> None:
    try:
        pending_call.result = _dispatch_runtime_tool_call(
            tool_registry=tool_registry,
            tool_call=pending_call.tool_call,
        )
    except Exception as exc:
        pending_call.result = ToolResult.failure(
            tool_name=pending_call.tool_call.tool_name,
            error=(
                f"Tool '{pending_call.tool_call.tool_name}' failed unexpectedly: {exc}"
            ),
        )
    finally:
        pending_call.done_event.set()


def _collect_pending_tool_results(
    *,
    pending_calls: Sequence[_PendingToolExecution],
    tool_guard_state: _RuntimeToolGuardState,
) -> tuple[tuple[_PendingToolExecution, ToolResult, float], ...]:
    resolved_by_index: list[tuple[ToolResult, float] | None] = [None] * len(pending_calls)
    timeout_seconds = tool_guard_state.tool_timeout_seconds

    while any(result is None for result in resolved_by_index):
        cancellation_requested = tool_guard_state.is_cancelled()
        now = perf_counter()
        for index, pending_call in enumerate(pending_calls):
            if resolved_by_index[index] is not None:
                continue

            if pending_call.done_event.is_set():
                resolved_result = pending_call.result or ToolResult.failure(
                    tool_name=pending_call.tool_call.tool_name,
                    error=(
                        f"Tool '{pending_call.tool_call.tool_name}' returned no result."
                    ),
                )
                resolved_by_index[index] = (resolved_result, perf_counter())
                continue

            if cancellation_requested:
                resolved_by_index[index] = (
                    _build_cancelled_tool_result(pending_call.tool_call),
                    now,
                )
                continue

            if now - pending_call.started_at >= timeout_seconds:
                resolved_by_index[index] = (
                    _build_timed_out_tool_result(
                        pending_call.tool_call,
                        timeout_seconds=timeout_seconds,
                    ),
                    now,
                )

        if any(result is None for result in resolved_by_index):
            sleep(RUNTIME_TOOL_RESULT_POLL_INTERVAL_SECONDS)

    return tuple(
        (pending_call, resolved_result, finished_at)
        for pending_call, resolved in zip(pending_calls, resolved_by_index, strict=True)
        if resolved is not None
        for resolved_result, finished_at in (resolved,)
    )


def _build_timed_out_tool_result(
    tool_call: ToolCall,
    *,
    timeout_seconds: float,
) -> ToolResult:
    return ToolResult.failure(
        tool_name=tool_call.tool_name,
        error=(
            f"Tool '{tool_call.tool_name}' timed out after "
            f"{timeout_seconds:g} seconds."
        ),
    )


def _build_cancelled_tool_result(tool_call: ToolCall) -> ToolResult:
    return ToolResult.failure(
        tool_name=tool_call.tool_name,
        error=f"Tool '{tool_call.tool_name}' was cancelled.",
    )


def _resolve_tool_definitions(
    *,
    tool_registry: ToolRegistry,
    model_profile: Any,
) -> list[ToolDefinition] | None:
    """Return tool definitions only for models with native tool-calling support."""
    if not _supports_native_tool_calling(model_profile):
        return None
    tools = tool_registry.list_tools()
    return tools if tools else None


def _supports_native_tool_calling(model_profile: Any) -> bool:
    """Check runtime/profile metadata for explicit native tool-calling support."""
    capabilities = getattr(model_profile, "capabilities", None)
    supports_native = getattr(capabilities, "supports_native_tool_calling", None)
    if isinstance(supports_native, bool):
        return supports_native

    tool_mode = getattr(model_profile, "tool_mode", None)
    if not isinstance(tool_mode, str):
        return False

    return tool_mode.strip().lower() == "native"


def _recover_inline_native_tool_response(
    response: LLMResponse,
    *,
    tool_definitions: Sequence[ToolDefinition],
    max_agent_steps: int,
) -> tuple[LLMResponse | None, str | None]:
    """Normalize inline JSON tool payloads into the shared native tool path.

    Some native-capable local models emit tool requests as assistant-visible
    JSON instead of filling the structured ``tool_calls`` field. When recovery
    is safe, rebuild a synthetic native payload so the existing agent loop can
    run unchanged. When the content clearly looks like a tool payload but is
    invalid or tool execution is unavailable, return a normal assistant reply
    instead of exposing raw JSON to the user.
    """
    analysis = _analyze_inline_tool_payload(
        response.content,
        tool_definitions=tool_definitions,
    )
    if analysis.tool_calls is None:
        if analysis.invalid_tool_payload:
            return None, _INLINE_TOOL_PAYLOAD_FALLBACK_REPLY
        return None, None

    if max_agent_steps <= 0:
        return None, _INLINE_TOOL_PAYLOAD_FALLBACK_REPLY

    raw_message = response.raw_payload.get("message")
    normalized_message = dict(raw_message) if isinstance(raw_message, dict) else {}
    normalized_message["content"] = ""
    normalized_message["tool_calls"] = list(analysis.raw_tool_calls_payload or ())

    normalized_payload = dict(response.raw_payload)
    normalized_payload["message"] = normalized_message
    return (
        replace(
            response,
            content="",
            tool_calls=analysis.tool_calls,
            raw_payload=normalized_payload,
        ),
        None,
    )


def _analyze_inline_tool_payload(
    content: str,
    *,
    tool_definitions: Sequence[ToolDefinition],
) -> _InlineToolPayloadAnalysis:
    candidate = _extract_inline_tool_payload_candidate(content)
    if candidate is None:
        return _InlineToolPayloadAnalysis()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return _InlineToolPayloadAnalysis(
            invalid_tool_payload=_looks_like_inline_tool_payload_text(candidate)
        )

    if not _looks_like_inline_tool_payload(payload):
        return _InlineToolPayloadAnalysis()

    parsed = _parse_inline_tool_payload(
        payload,
        allowed_tool_names={tool.name for tool in tool_definitions},
    )
    if parsed is None:
        return _InlineToolPayloadAnalysis(invalid_tool_payload=True)

    tool_calls, raw_tool_calls_payload = parsed
    return _InlineToolPayloadAnalysis(
        tool_calls=tool_calls,
        raw_tool_calls_payload=raw_tool_calls_payload,
    )


def _extract_inline_tool_payload_candidate(content: str) -> str | None:
    stripped = content.strip()
    if not stripped:
        return None

    fenced_candidate = _extract_fenced_json_candidate(stripped)
    if fenced_candidate is not None:
        return fenced_candidate

    if stripped[0] in ("{", "["):
        return stripped

    return None


def _extract_fenced_json_candidate(content: str) -> str | None:
    fence_start = content.find("```")
    if fence_start == -1:
        return None

    fence_end = content.find("```", fence_start + 3)
    if fence_end == -1:
        return None

    candidate = content[fence_start + 3 : fence_end].strip()
    if not candidate:
        return None

    if "\n" in candidate:
        first_line, remainder = candidate.split("\n", 1)
        if first_line.strip().lower() in {"json", "jsonc", "javascript", "js"}:
            candidate = remainder.strip()

    if not candidate or candidate[0] not in ("{", "["):
        return None
    return candidate


def _looks_like_inline_tool_payload_text(candidate: str) -> bool:
    lowered = candidate.lower()
    if "\"tool_calls\"" in lowered or "\"function\"" in lowered:
        return True
    return "\"arguments\"" in lowered and (
        "\"name\"" in lowered or "\"tool_name\"" in lowered
    )


def _looks_like_inline_tool_payload(payload: Any) -> bool:
    if isinstance(payload, list):
        return bool(payload) and all(
            _looks_like_inline_tool_payload(item) for item in payload
        )

    if not isinstance(payload, dict):
        return False

    if "tool_calls" in payload:
        return set(payload).issubset({"tool_calls", "id", "type"})

    if "function" in payload:
        return set(payload).issubset({"function", "id", "type", "index"})

    name = payload.get("name")
    tool_name = payload.get("tool_name")
    return (
        (isinstance(name, str) or isinstance(tool_name, str))
        and set(payload).issubset({"name", "tool_name", "arguments", "id", "type"})
    )


def _parse_inline_tool_payload(
    payload: Any,
    *,
    allowed_tool_names: set[str],
) -> tuple[tuple[ToolCall, ...], tuple[dict[str, Any], ...]] | None:
    if isinstance(payload, dict) and "tool_calls" in payload:
        raw_calls = payload.get("tool_calls")
    elif isinstance(payload, list):
        raw_calls = payload
    else:
        raw_calls = [payload]

    if not isinstance(raw_calls, list) or not raw_calls:
        return None

    parsed_calls: list[ToolCall] = []
    raw_payloads: list[dict[str, Any]] = []
    for raw_call in raw_calls:
        parsed = _parse_one_inline_tool_call(
            raw_call,
            allowed_tool_names=allowed_tool_names,
        )
        if parsed is None:
            return None
        tool_call, raw_payload = parsed
        parsed_calls.append(tool_call)
        raw_payloads.append(raw_payload)

    return tuple(parsed_calls), tuple(raw_payloads)


def _parse_one_inline_tool_call(
    payload: Any,
    *,
    allowed_tool_names: set[str],
) -> tuple[ToolCall, dict[str, Any]] | None:
    if not isinstance(payload, dict):
        return None

    if "function" in payload:
        function = payload.get("function")
        if not isinstance(function, dict):
            return None
        raw_name = function.get("name")
        raw_arguments = function.get("arguments")
    else:
        raw_name = payload.get("name")
        if not isinstance(raw_name, str):
            raw_name = payload.get("tool_name")
        raw_arguments = payload.get("arguments")

    if not isinstance(raw_name, str) or raw_name not in allowed_tool_names:
        return None

    arguments = _normalize_inline_tool_arguments(raw_arguments)
    if arguments is None:
        return None

    return (
        ToolCall(tool_name=raw_name, arguments=arguments),
        {
            "function": {
                "name": raw_name,
                "arguments": arguments,
            }
        },
    )


def _normalize_inline_tool_arguments(arguments: Any) -> dict[str, Any] | None:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return None

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def _extract_raw_tool_calls(response: LLMResponse) -> tuple[dict[str, Any], ...] | None:
    """Extract raw tool_calls from the provider response for re-sending."""
    message = response.raw_payload.get("message")
    if not isinstance(message, dict):
        return None
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
        return None
    return tuple(raw_tool_calls)


def _dispatch_runtime_tool_call(
    *,
    tool_registry: ToolRegistry,
    tool_call: ToolCall,
) -> ToolResult:
    dispatcher = ToolDispatcher(tool_registry)
    return dispatcher.dispatch(tool_call)


def _finalize_runtime_tool_call(
    *,
    session_manager: SessionManager,
    session_id: str,
    tracer: Tracer,
    tool_call: ToolCall,
    tool_result: ToolResult,
    tool_duration_ms: int,
    skill_id: str | None = None,
) -> None:
    from unclaw.core.research_flow import persist_tool_result  # lazy to avoid circular import

    tracer.trace_tool_finished(
        session_id=session_id,
        tool_name=tool_result.tool_name,
        success=tool_result.success,
        output_length=len(tool_result.output_text),
        error=tool_result.error,
        tool_duration_ms=tool_duration_ms,
        skill_id=skill_id,
    )
    persist_tool_result(
        session_manager=session_manager,
        session_id=session_id,
        result=tool_result,
        tool_call=tool_call,
    )


def _build_model_failure_reply(error: ModelCallFailedError) -> str:
    message = error.error.strip()
    if message:
        return message
    return RUNTIME_ERROR_REPLY
