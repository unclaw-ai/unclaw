"""Minimal runtime entrypoint for one non-command chat turn."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
from unclaw.core.router import RouteKind, route_request
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

if TYPE_CHECKING:
    from unclaw.tools.registry import ToolRegistry

_MAX_STEPS_FALLBACK_REPLY = (
    "I reached the maximum number of steps for this request. "
    "Here is what I found so far."
)
_TOOL_BUDGET_FALLBACK_REPLY = (
    "I stopped after reaching the tool-call limit for this request."
)
_TURN_CANCELLED_REPLY = "This request was cancelled before tool work completed."

# ---------------------------------------------------------------------------
# Overwrite intent guard — P3-4 corrective mission.
# Explicitly required by the P3-4 mission spec; satisfies mandatory_rules.md
# rule 5 allowed exception (tiny local safeguard, scoped, not core routing)
# and rule 10 (explicitly justified, easy to audit, easy to remove).
# ---------------------------------------------------------------------------

_OVERWRITE_INTENT_KEYWORDS: frozenset[str] = frozenset(
    {
        "écrase",
        "écraser",
        "remplace",
        "remplacer",
        "réécris par-dessus",
        "overwrite",
        "replace existing",
        "overwrite the file",
        "replace the file",
    }
)


def _has_explicit_overwrite_intent(user_input: str) -> bool:
    """Return True if user_input contains an explicit overwrite intent keyword.

    Keyword list is defined in the P3-4 corrective mission spec.
    Checks the latest user turn only, never the full conversation history.
    """
    lowered = user_input.lower()
    return any(keyword in lowered for keyword in _OVERWRITE_INTENT_KEYWORDS)


def _guard_write_overwrite_intent(
    tool_calls: Sequence[ToolCall],
    user_input: str,
) -> Sequence[ToolCall]:
    """Strip overwrite=True from write_text_file calls when user intent is absent.

    When the user's message contains an explicit overwrite keyword, calls pass
    through unchanged. Otherwise overwrite=True is removed so the tool naturally
    returns a 'file already exists' error, prompting the assistant to surface
    the conflict and ask the user for explicit confirmation before retrying.
    """
    if _has_explicit_overwrite_intent(user_input):
        return tool_calls

    guarded: list[ToolCall] = []
    for call in tool_calls:
        if call.tool_name == "write_text_file":
            overwrite_val = call.arguments.get("overwrite")
            overwrite_is_true = overwrite_val is True or overwrite_val == "true"
            if overwrite_is_true:
                new_args = {k: v for k, v in call.arguments.items() if k != "overwrite"}
                guarded.append(ToolCall(tool_name=call.tool_name, arguments=new_args))
                continue
        guarded.append(call)
    return guarded


# ---------------------------------------------------------------------------
# Overwrite-refusal deterministic reply — P3-4 corrective follow-up.
# Prevents the model from falsely claiming success after write_text_file
# returns the specific "already exists" overwrite-protection failure.
# Short-circuits the agent loop so the model is never called with the
# tool-failure context, eliminating the false-success response path.
# Explicitly required by the P3-4 corrective mission; satisfies
# mandatory_rules.md rule 7 (smallest correct patch) and rule 10
# (explicitly justified, visibly scoped, easy to audit and remove).
# ---------------------------------------------------------------------------

_WRITE_FILE_ALREADY_EXISTS_MARKER = "already exists"
_OVERWRITE_REFUSED_REPLY_TEMPLATE = (
    "The file already exists and was not overwritten.\n"
    "{error}\n"
    'Please confirm explicitly if you want to replace it '
    '(for example: "overwrite the file" or "replace the file").'
)


def _build_overwrite_refusal_reply(tool_results: tuple[ToolResult, ...]) -> str | None:
    """Return a deterministic reply if write_text_file was blocked by overwrite protection.

    Checks tool_results for the specific write_text_file "already exists" error.
    When found, returns a pre-built refusal reply so the model is never called
    again on that path — eliminating the false-success assistant reply.
    Returns None if no such failure is present.
    """
    for result in tool_results:
        if (
            result.tool_name == "write_text_file"
            and not result.success
            and result.error is not None
            and _WRITE_FILE_ALREADY_EXISTS_MARKER in result.error
        ):
            return _OVERWRITE_REFUSED_REPLY_TEMPLATE.format(error=result.error)
    return None


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
    tool_guard_state = _RuntimeToolGuardState(
        tool_timeout_seconds=(
            session_manager.settings.app.runtime.tool_timeout_seconds
        ),
        max_tool_calls_per_turn=(
            session_manager.settings.app.runtime.max_tool_calls_per_turn
        ),
        cancellation=turn_cancellation,
    )

    # Determine tool definitions for agent loop first, so the capability
    # summary can honestly report whether model-driven tool use is active.
    tool_definitions = _resolve_tool_definitions(
        tool_registry=active_tool_registry,
        model_profile=selected_model,
    )
    capability_summary = build_runtime_capability_summary(
        tool_registry=active_tool_registry,
        memory_summary_available=command_handler.memory_manager is not None,
        model_can_call_tools=tool_definitions is not None,
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

    try:
        memory_context_note = _build_session_memory_context_note(
            command_handler=command_handler,
            session_id=session.id,
        )
        if memory_context_note is not None:
            system_context_notes = (memory_context_note,)

        route = route_request(
            settings=session_manager.settings,
            model_profile_name=selected_model_profile_name,
            user_input=user_input,
            thinking_enabled=thinking_enabled,
            capability_summary=capability_summary,
            allow_web_search_routing=explicit_tool_call is None,
        )
        active_tracer.trace_route_selected(
            session_id=session.id,
            route_kind=route.kind.value,
            model_profile_name=route.model_profile_name,
        )

        if route.kind is RouteKind.WEB_SEARCH:
            (
                route_context_notes,
                active_assistant_reply_transform,
                runtime_explicit_tool_call,
            ) = _prepare_web_search_route(
                session_manager=session_manager,
                session_id=session.id,
                user_input=user_input,
                route=route,
                assistant_reply_transform=assistant_reply_transform,
            )
            system_context_notes = (*system_context_notes, *route_context_notes)

        if runtime_explicit_tool_call is not None and (
            route.kind is RouteKind.WEB_SEARCH or tool_definitions is None
        ):
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
            response = orchestrator.run_turn(
                session_id=session.id,
                user_message=user_input,
                model_profile_name=route.model_profile_name,
                capability_summary=capability_summary,
                system_context_notes=system_context_notes,
                thinking_enabled=thinking_enabled,
                content_callback=stream_output_func,
                tools=tool_definitions,
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

            # --- Agent observation-action loop ---
            if response.response.tool_calls and tool_definitions and max_agent_steps > 0:
                assistant_reply = _run_agent_loop(
                    first_response=response,
                    orchestrator=orchestrator,
                    session_id=session.id,
                    session_manager=session_manager,
                    tracer=active_tracer,
                    tool_registry=active_tool_registry,
                    tool_definitions=tool_definitions,
                    model_profile_name=route.model_profile_name,
                    thinking_enabled=thinking_enabled,
                    content_callback=stream_output_func,
                    max_steps=max_agent_steps,
                    tool_guard_state=tool_guard_state,
                    user_input=user_input,
                )
            else:
                assistant_reply = response.response.content.strip() or EMPTY_RESPONSE_REPLY

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


def _prepare_web_search_route(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
    route: Any,
    assistant_reply_transform: Callable[[str], str] | None,
) -> tuple[tuple[str, ...], Callable[[str], str] | None, ToolCall | None]:
    # ---------------------------------------------------------------------------
    # P5-2: guarantee one initial search_web execution on every WEB_SEARCH route,
    # regardless of whether the model profile uses native tool calling.
    # Previously, explicit_search_call was only created when search_results_ready=True
    # (i.e. json_plan / non-native profiles).  Native profiles received no initial
    # forced search — the model was expected to call search_web itself, which caused
    # entity drift and weather-query refusals.
    # This function now always creates the explicit ToolCall.  The caller's condition
    #   `runtime_explicit_tool_call is not None and route.kind is WEB_SEARCH`
    # already handles both native and non-native execution paths identically.
    # Complies with mandatory_rules.md rule 5 allowed exception (explicitly required
    # by mission P5-2, scoped to the WEB_SEARCH execution path only, not routing
    # architecture) and rule 10 (explicitly justified, visibly scoped, easy to remove).
    # ---------------------------------------------------------------------------
    from unclaw.core.research_flow import (
        apply_search_grounding_from_history,
        build_web_search_route_note,
    )

    search_query = (
        route.search_query.strip()
        if isinstance(route.search_query, str) and route.search_query.strip()
        else user_input.strip()
    )
    system_context_notes = (
        build_web_search_route_note(
            query=search_query,
            search_results_ready=True,  # P5-2: search is always forced before model call
        ),
    )

    # Capture the session message count before any search tool executes for
    # this turn.  The grounding transform uses this floor so it only scans
    # messages added during the current turn, preventing stale display_sources
    # from an earlier turn from contaminating a new grounded reply.
    _list_messages = getattr(session_manager, "list_messages", None)
    _turn_start_count = (
        len(_list_messages(session_id)) if callable(_list_messages) else 0
    )

    def _grounding_transform(reply: str) -> str:
        return apply_search_grounding_from_history(
            reply,
            query=search_query,
            session_manager=session_manager,
            session_id=session_id,
            turn_start_message_count=_turn_start_count,
        )

    # P5-2: always create the explicit search call — native and non-native paths both
    # need a guaranteed initial search_web execution with the routed query.
    explicit_search_call = ToolCall(
        tool_name="search_web",
        arguments={"query": search_query},
    )

    return (
        system_context_notes,
        _compose_reply_transforms(_grounding_transform, assistant_reply_transform),
        explicit_search_call,
    )


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
    user_input: str = "",
) -> str:
    """Execute the observation-action loop until text reply or step limit."""
    context_messages: list[LLMMessage] = list(first_response.context_messages)
    current_response = first_response

    for _step in range(max_steps):
        tool_calls = current_response.response.tool_calls
        if not tool_calls:
            return current_response.response.content.strip() or EMPTY_RESPONSE_REPLY

        # Apply overwrite intent guard before executing write_text_file calls.
        tool_calls = _guard_write_overwrite_intent(tool_calls, user_input)

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

        # Execute tool calls concurrently, then replay results in model order.
        tool_results = _execute_runtime_tool_calls(
            session_manager=session_manager,
            session_id=session_id,
            tracer=tracer,
            tool_registry=tool_registry,
            tool_calls=tool_calls,
            tool_guard_state=tool_guard_state,
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

        # Overwrite-refusal short-circuit: if write_text_file was blocked by the
        # overwrite protection guard, return a deterministic reply immediately.
        # The model is never called again after this path — it cannot claim success.
        overwrite_refusal = _build_overwrite_refusal_reply(tool_results)
        if overwrite_refusal is not None:
            return overwrite_refusal

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
) -> tuple[ToolResult, ...]:
    """Execute one tool-call batch while keeping context and persistence ordered."""
    pending_calls = _start_pending_tool_executions(
        session_id=session_id,
        tracer=tracer,
        tool_registry=tool_registry,
        tool_calls=tool_calls,
        tool_guard_state=tool_guard_state,
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
) -> list[_PendingToolExecution]:
    pending_calls: list[_PendingToolExecution] = []
    for tool_call in tool_calls:
        if tool_guard_state.is_cancelled():
            break

        started_at = perf_counter()
        tracer.trace_tool_started(
            session_id=session_id,
            tool_name=tool_call.tool_name,
            arguments=tool_call.arguments,
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
) -> None:
    from unclaw.core.research_flow import persist_tool_result  # lazy to avoid circular import

    tracer.trace_tool_finished(
        session_id=session_id,
        tool_name=tool_result.tool_name,
        success=tool_result.success,
        output_length=len(tool_result.output_text),
        error=tool_result.error,
        tool_duration_ms=tool_duration_ms,
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
