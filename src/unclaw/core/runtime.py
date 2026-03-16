"""Minimal runtime entrypoint for one non-command chat turn."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter
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
from unclaw.constants import EMPTY_RESPONSE_REPLY, RUNTIME_ERROR_REPLY
from unclaw.errors import ConfigurationError
from unclaw.llm.base import LLMContentCallback, LLMError, LLMMessage, LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolResult
from unclaw.tools.dispatcher import ToolDispatcher

if TYPE_CHECKING:
    from unclaw.tools.registry import ToolRegistry

_MAX_AGENT_STEPS_DEFAULT = 6
_MAX_STEPS_FALLBACK_REPLY = (
    "I reached the maximum number of steps for this request. "
    "Here is what I found so far."
)


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
    max_agent_steps: int = _MAX_AGENT_STEPS_DEFAULT,
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
        session_manager.settings
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
            _execute_runtime_tool_call(
                session_manager=session_manager,
                session_id=session.id,
                tracer=active_tracer,
                tool_registry=active_tool_registry,
                tool_call=runtime_explicit_tool_call,
            )

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
        assistant_reply = RUNTIME_ERROR_REPLY
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
    memory_manager = getattr(command_handler, "memory_manager", None)
    if memory_manager is None:
        return None

    build_context_note = getattr(memory_manager, "build_context_note", None)
    if not callable(build_context_note):
        return None

    note = build_context_note(session_id)
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
            search_results_ready=True,
        ),
    )

    def _grounding_transform(reply: str) -> str:
        return apply_search_grounding_from_history(
            reply,
            query=search_query,
            session_manager=session_manager,
            session_id=session_id,
        )

    return (
        system_context_notes,
        _compose_reply_transforms(_grounding_transform, assistant_reply_transform),
        ToolCall(
            tool_name="search_web",
            arguments={"query": search_query},
        ),
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
) -> str:
    """Execute the observation-action loop until text reply or step limit."""
    context_messages: list[LLMMessage] = list(first_response.context_messages)
    current_response = first_response

    for _step in range(max_steps):
        tool_calls = current_response.response.tool_calls
        if not tool_calls:
            return current_response.response.content.strip() or EMPTY_RESPONSE_REPLY

        # Append the assistant message (with tool_calls) to context.
        context_messages.append(
            LLMMessage(
                role=LLMRole.ASSISTANT,
                content=current_response.response.content,
                tool_calls_payload=_extract_raw_tool_calls(current_response.response),
            )
        )

        # Execute each tool call through the existing dispatcher.
        for tool_call in tool_calls:
            tool_result = _execute_runtime_tool_call(
                session_manager=session_manager,
                session_id=session_id,
                tracer=tracer,
                tool_registry=tool_registry,
                tool_call=tool_call,
            )
            context_messages.append(
                LLMMessage(
                    role=LLMRole.TOOL,
                    content=build_untrusted_tool_message_content(
                        tool_result.output_text
                    ),
                )
            )

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


def _execute_runtime_tool_call(
    *,
    session_manager: SessionManager,
    session_id: str,
    tracer: Tracer,
    tool_registry: ToolRegistry,
    tool_call: ToolCall,
) -> ToolResult:
    """Execute one runtime-level tool call with shared tracing and persistence."""
    from unclaw.core.research_flow import persist_tool_result  # lazy to avoid circular import

    dispatcher = ToolDispatcher(tool_registry)
    tool_started_at = perf_counter()
    tracer.trace_tool_started(
        session_id=session_id,
        tool_name=tool_call.tool_name,
        arguments=tool_call.arguments,
    )
    tool_result = dispatcher.dispatch(tool_call)
    tracer.trace_tool_finished(
        session_id=session_id,
        tool_name=tool_result.tool_name,
        success=tool_result.success,
        output_length=len(tool_result.output_text),
        error=tool_result.error,
        tool_duration_ms=elapsed_ms(tool_started_at),
    )
    persist_tool_result(
        session_manager=session_manager,
        session_id=session_id,
        result=tool_result,
        tool_call=tool_call,
    )
    return tool_result
