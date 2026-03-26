"""Agent-loop runtime helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import json
import threading
from time import perf_counter, sleep
from typing import Any

from unclaw.core.context_builder import build_untrusted_tool_message_content
from unclaw.core.orchestrator import Orchestrator, OrchestratorTurnResult
from unclaw.core.reply_discipline import (
    _build_grounded_reply_facts,
    _tool_result_has_thin_search_evidence,
)
from unclaw.core.session_manager import SessionManager
from unclaw.constants import EMPTY_RESPONSE_REPLY, RUNTIME_TOOL_RESULT_POLL_INTERVAL_SECONDS
from unclaw.llm.base import LLMContentCallback, LLMMessage, LLMResponse, LLMRole
from unclaw.logs.tracer import Tracer
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolResult, resolve_tool_argument_spec
from unclaw.tools.dispatcher import ToolDispatcher, normalize_tool_call_for_execution
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
_SEARCH_WEB_TIMEOUT_RETRY_MAX_RESULTS = 3
# Test hook: set to False to suppress the continuation check in tests
# that do not script for the extra model call.
_continuation_check_enabled = True
_PRE_WRITE_GROUNDING_NOTE_PREFIX = "Pre-write grounding check:"
_SEARCH_TOOL_NAMES = frozenset({"fast_web_search", "search_web"})
_STATE_CONFLICT_FAILURE_KINDS = frozenset(
    {
        "access_denied",
        "collision_conflict",
        "confirmation_required",
        "permission_denied",
        "unsupported_input",
    }
)
_FINALIZATION_TRIGGERING_TOOL_NAMES = frozenset({
    "write_text_file",
    "fast_web_search",
    "search_web",
})


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
    search_web_timeout_retry_used: bool = False

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
    build_post_tool_grounding_note: Callable[..., str],
    collected_tool_results: list[ToolResult] | None = None,
    user_input: str = "",
    session_goal_state: Any = None,
    session_progress_ledger: Sequence[Any] = (),
) -> str:
    """Execute the observation-action loop until text reply or step limit."""
    context_messages: list[LLMMessage] = list(first_response.context_messages)
    current_response = first_response
    tools_ran_in_turn = False
    last_continuation_checked_tool_count = 0
    state_conflict_recovery_used = False
    pre_write_grounding_used = False

    def _effective_callback() -> LLMContentCallback | None:
        """Suppress live streaming when grounded finalization will run.

        When collected tool results include tools that trigger grounded
        finalization (search, write), the model draft will be rewritten.
        Suppressing the stream here prevents the user from seeing a draft
        that will be replaced, eliminating duplicate/refined output in
        the terminal.
        """
        if collected_tool_results and any(
            tr.tool_name in _FINALIZATION_TRIGGERING_TOOL_NAMES
            for tr in collected_tool_results
        ):
            return None
        return content_callback

    for _step in range(max_steps):
        tool_calls = current_response.response.tool_calls
        if not tool_calls:
            draft_reply = current_response.response.content.strip() or EMPTY_RESPONSE_REPLY
            turn_tool_results = tuple(collected_tool_results or ())

            # Model-native continuation checkpoint: after a tool-backed phase
            # returns text, ask the model to validate whether the original
            # request is fully satisfied or whether another tool step is still
            # required. The checkpoint runs at most once for each distinct
            # amount of accumulated tool progress, so the loop stays bounded.
            if (
                _continuation_check_enabled
                and tools_ran_in_turn
                and not tool_guard_state.is_cancelled()
                and _step + 1 < max_steps
                and user_input
                and len(turn_tool_results) > last_continuation_checked_tool_count
                and _should_run_continuation_check(
                    user_input=user_input,
                    draft_reply=draft_reply,
                    tool_results=turn_tool_results,
                    tool_definitions=tool_definitions,
                    session_goal_state=session_goal_state,
                    session_progress_ledger=session_progress_ledger,
                )
            ):
                last_continuation_checked_tool_count = len(turn_tool_results)
                continuation_note = _build_continuation_check_note(
                    user_input=user_input,
                    draft_reply=draft_reply,
                    tool_results=turn_tool_results,
                    tool_definitions=tool_definitions,
                    session_goal_state=session_goal_state,
                    session_progress_ledger=session_progress_ledger,
                )
                context_messages.append(
                    LLMMessage(role=LLMRole.ASSISTANT, content=draft_reply)
                )
                context_messages.append(
                    LLMMessage(role=LLMRole.SYSTEM, content=continuation_note)
                )
                current_response = orchestrator.call_model(
                    session_id=session_id,
                    messages=context_messages,
                    model_profile_name=model_profile_name,
                    thinking_enabled=thinking_enabled,
                    content_callback=_effective_callback(),
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
                # Re-enter loop: if the model now emits tool calls, they'll
                # be handled by the next iteration; if text, we return it.
                continue

            return draft_reply

        stop_reply = _preflight_runtime_tool_batch(
            tool_calls=tool_calls,
            tool_guard_state=tool_guard_state,
        )
        if stop_reply is not None:
            return stop_reply

        context_messages.append(
            LLMMessage(
                role=LLMRole.ASSISTANT,
                content=current_response.response.content,
                tool_calls_payload=_extract_raw_tool_calls(current_response.response),
            )
        )

        turn_tool_results = tuple(collected_tool_results or ())
        if (
            not pre_write_grounding_used
            and not tool_guard_state.is_cancelled()
            and _step + 1 < max_steps
            and _should_run_pre_write_grounding_check(
                prior_tool_results=turn_tool_results,
                pending_tool_calls=tool_calls,
            )
        ):
            pre_write_grounding_used = True
            context_messages.append(
                LLMMessage(
                    role=LLMRole.SYSTEM,
                    content=_build_pre_write_grounding_note(
                        user_input=user_input,
                        prior_tool_results=turn_tool_results,
                        pending_tool_calls=tool_calls,
                    ),
                )
            )
            current_response = orchestrator.call_model(
                session_id=session_id,
                messages=context_messages,
                model_profile_name=model_profile_name,
                thinking_enabled=thinking_enabled,
                content_callback=None,
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
            continue

        tool_results = _execute_runtime_tool_calls(
            session_manager=session_manager,
            session_id=session_id,
            tracer=tracer,
            tool_registry=tool_registry,
            tool_calls=tool_calls,
            tool_guard_state=tool_guard_state,
            tool_call_callback=tool_call_callback,
        )
        tools_ran_in_turn = True
        if collected_tool_results is not None:
            collected_tool_results.extend(tool_results)

        # Model-native tool-argument repair: if every tool in the batch
        # failed with a structural/contract error, give the model one
        # bounded chance to emit corrected tool calls.
        if (
            tool_results
            and all(tr.success is False for tr in tool_results)
            and not tool_guard_state.is_cancelled()
            and _step + 1 < max_steps
            and _all_failures_look_structural(tool_results)
        ):
            repair_note = _build_tool_argument_repair_note(
                failed_tool_calls=tool_calls,
                failed_tool_results=tool_results,
                tool_definitions=tool_definitions,
                user_input=user_input,
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
            context_messages.append(
                LLMMessage(role=LLMRole.SYSTEM, content=repair_note)
            )
            current_response = orchestrator.call_model(
                session_id=session_id,
                messages=context_messages,
                model_profile_name=model_profile_name,
                thinking_enabled=thinking_enabled,
                content_callback=_effective_callback(),
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
            # The repair response re-enters the loop: if the model now
            # emits valid tool calls, they'll execute next iteration.
            continue

        if (
            tool_results
            and not state_conflict_recovery_used
            and not tool_guard_state.is_cancelled()
            and _step + 1 < max_steps
            and _tool_results_include_state_conflict(tool_results)
        ):
            state_conflict_recovery_used = True
            _append_tool_result_messages(
                context_messages=context_messages,
                tool_results=tool_results,
            )
            context_messages.append(
                LLMMessage(
                    role=LLMRole.SYSTEM,
                    content=_build_state_conflict_recovery_note(
                        user_input=user_input,
                        failed_tool_calls=tool_calls,
                        tool_results=tool_results,
                    ),
                )
            )
            current_response = orchestrator.call_model(
                session_id=session_id,
                messages=context_messages,
                model_profile_name=model_profile_name,
                thinking_enabled=thinking_enabled,
                content_callback=None,
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
            continue

        _append_tool_result_messages(
            context_messages=context_messages,
            tool_results=tool_results,
        )

        if tool_guard_state.is_cancelled():
            return _TURN_CANCELLED_REPLY

        if tool_results:
            context_messages.append(
                LLMMessage(
                    role=LLMRole.SYSTEM,
                    content=build_post_tool_grounding_note(
                        tool_results=tool_results,
                        tool_definitions=tool_definitions,
                    ),
                )
            )

        current_response = orchestrator.call_model(
            session_id=session_id,
            messages=context_messages,
            model_profile_name=model_profile_name,
            thinking_enabled=thinking_enabled,
            content_callback=_effective_callback(),
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

    if not current_response.response.tool_calls:
        return current_response.response.content.strip() or EMPTY_RESPONSE_REPLY

    return _MAX_STEPS_FALLBACK_REPLY


_CONTINUATION_CHECK_NOTE_PREFIX = "Task completion check:"


def _build_continuation_check_note(
    *,
    user_input: str,
    draft_reply: str,
    tool_results: Sequence[ToolResult] = (),
    tool_definitions: Sequence[ToolDefinition] = (),
    session_goal_state: Any = None,
    session_progress_ledger: Sequence[Any] = (),
) -> str:
    lines = [
        (
            f"{_CONTINUATION_CHECK_NOTE_PREFIX} "
            "the model produced a text reply after running tools this turn."
        ),
        f"Original user request: {user_input}",
        f"Current draft answer: {draft_reply[:500]}",
    ]
    if tool_results:
        lines.extend(
            [
                "Structured runtime facts (JSON):",
                _serialize_note_payload(
                    _build_continuation_check_runtime_facts(
                        user_input=user_input,
                        draft_reply=draft_reply,
                        tool_results=tool_results,
                        tool_definitions=tool_definitions,
                        session_goal_state=session_goal_state,
                        session_progress_ledger=session_progress_ledger,
                    )
                ),
            ]
        )
    lines.extend(
        (
            "Decide: is the user's full request satisfied by the work done so far?",
            "If YES: produce the full final answer text now, including any short non-tool deliverable still requested by the user.",
            "If NO and an available tool call would make meaningful progress: "
            "emit that tool call instead of answering.",
            "If completion_risks.phase_checkpoint_required is true, treat this as a real phase checkpoint instead of trusting the current draft.",
            "If a blocking failure remains or any action_performed flag is false, do not claim success.",
            "Do not repeat tool calls that already ran this turn unless the structured facts above clearly support a safe retry.",
        )
    )
    return "\n".join(lines)


def _serialize_note_payload(payload: dict[str, Any]) -> str:
    normalized_payload = json.loads(
        json.dumps(payload, ensure_ascii=False, default=str)
    )
    return json.dumps(normalized_payload, ensure_ascii=False, indent=2, sort_keys=True)


def _build_continuation_check_runtime_facts(
    *,
    user_input: str,
    draft_reply: str,
    tool_results: Sequence[ToolResult],
    tool_definitions: Sequence[ToolDefinition],
    session_goal_state: Any,
    session_progress_ledger: Sequence[Any],
) -> dict[str, Any]:
    return _build_grounded_reply_facts(
        user_input=user_input,
        assistant_draft_reply=draft_reply,
        tool_results=tool_results,
        session_goal_state=session_goal_state,
        session_progress_ledger=session_progress_ledger,
        available_tool_definitions=tool_definitions,
    )


def _tool_result_payload_dict(tool_result: ToolResult) -> dict[str, Any]:
    return tool_result.payload if isinstance(tool_result.payload, dict) else {}


def _tool_result_payload_flag(tool_result: ToolResult, key: str) -> bool | None:
    value = _tool_result_payload_dict(tool_result).get(key)
    return value if isinstance(value, bool) else None


def _tool_result_has_thin_or_ambiguous_search_evidence(tool_result: ToolResult) -> bool:
    if tool_result.tool_name not in _SEARCH_TOOL_NAMES:
        return False
    if tool_result.success is not True:
        return True
    payload = _tool_result_payload_dict(tool_result)
    if tool_result.tool_name == "fast_web_search":
        match_quality = payload.get("match_quality")
        if isinstance(match_quality, str) and match_quality in {
            "mismatch",
            "no_results",
            "partial",
        }:
            return True
    return _tool_result_has_thin_search_evidence(tool_result)


def _should_run_continuation_check(
    *,
    user_input: str,
    draft_reply: str,
    tool_results: Sequence[ToolResult],
    tool_definitions: Sequence[ToolDefinition],
    session_goal_state: Any,
    session_progress_ledger: Sequence[Any],
) -> bool:
    if not tool_results:
        return False
    continuation_facts = _build_continuation_check_runtime_facts(
        user_input=user_input,
        draft_reply=draft_reply,
        tool_results=tool_results,
        tool_definitions=tool_definitions,
        session_goal_state=session_goal_state,
        session_progress_ledger=session_progress_ledger,
    )
    completion_risks = continuation_facts.get("completion_risks")
    if not isinstance(completion_risks, dict):
        return False
    return completion_risks.get("phase_checkpoint_required") is True


def _should_run_pre_write_grounding_check(
    *,
    prior_tool_results: Sequence[ToolResult],
    pending_tool_calls: Sequence[ToolCall],
) -> bool:
    if not any(tool_call.tool_name == "write_text_file" for tool_call in pending_tool_calls):
        return False
    return any(
        _tool_result_has_thin_or_ambiguous_search_evidence(tool_result)
        for tool_result in prior_tool_results
    )


def _build_pre_write_grounding_note(
    *,
    user_input: str,
    prior_tool_results: Sequence[ToolResult],
    pending_tool_calls: Sequence[ToolCall],
) -> str:
    search_facts = [
        _build_search_fact(tool_result)
        for tool_result in prior_tool_results
        if tool_result.tool_name in _SEARCH_TOOL_NAMES
    ]
    pending_write_calls = [
        _build_pending_write_call_fact(tool_call)
        for tool_call in pending_tool_calls
        if tool_call.tool_name == "write_text_file"
    ]
    return "\n".join(
        (
            (
                f"{_PRE_WRITE_GROUNDING_NOTE_PREFIX} "
                "a write_text_file call is pending after weak or incomplete search evidence in this turn."
            ),
            "Use only the structured search facts and pending write facts below for this review.",
            "If evidence is thin, ambiguous, mismatch-heavy, or failed, revise the file content into a conservative version limited to directly supported facts and explicit uncertainty.",
            "If more evidence is needed and an available search tool would help, emit that tool call instead of writing.",
            "If the write content is already conservative and supported, you may repeat the write tool call with corrected arguments if needed.",
            "Original user request:",
            user_input,
            "Structured review facts (JSON):",
            _serialize_note_payload(
                {
                    "search_facts": search_facts,
                    "pending_write_calls": pending_write_calls,
                }
            ),
        )
    )


def _build_search_fact(tool_result: ToolResult) -> dict[str, Any]:
    payload = _tool_result_payload_dict(tool_result)
    search_fact: dict[str, Any] = {
        "tool_name": tool_result.tool_name,
        "success": tool_result.success,
        "failure_kind": tool_result.failure_kind,
        "query": payload.get("query"),
        "match_quality": payload.get("match_quality"),
        "result_count": payload.get("result_count"),
        "supported_point_count": payload.get("supported_point_count"),
        "evidence_count": payload.get("evidence_count"),
        "finding_count": payload.get("finding_count"),
        "summary_points": payload.get("summary_points"),
        "display_sources": payload.get("display_sources"),
        "action_performed": _tool_result_payload_flag(tool_result, "action_performed"),
        "thin_or_ambiguous": _tool_result_has_thin_or_ambiguous_search_evidence(tool_result),
    }
    return search_fact


def _build_pending_write_call_fact(tool_call: ToolCall) -> dict[str, Any]:
    content = tool_call.arguments.get("content")
    content_text = content if isinstance(content, str) else ""
    max_chars = 4000
    content_truncated = len(content_text) > max_chars
    return {
        "path": tool_call.arguments.get("path"),
        "collision_policy": tool_call.arguments.get("collision_policy", "fail"),
        "content": content_text[:max_chars] if content_truncated else content_text,
        "content_size_chars": len(content_text),
        "content_truncated": content_truncated,
    }


def _tool_results_include_state_conflict(tool_results: Sequence[ToolResult]) -> bool:
    return any(
        tool_result.success is False
        and tool_result.failure_kind in _STATE_CONFLICT_FAILURE_KINDS
        for tool_result in tool_results
    )


def _build_state_conflict_recovery_note(
    *,
    user_input: str,
    failed_tool_calls: Sequence[ToolCall],
    tool_results: Sequence[ToolResult],
) -> str:
    del failed_tool_calls
    state_conflict_facts = [
        _build_state_conflict_fact(tool_result)
        for tool_result in tool_results
        if tool_result.success is False
        and tool_result.failure_kind in _STATE_CONFLICT_FAILURE_KINDS
    ]
    lines = [
        "State conflict recovery: the latest tool step was blocked by structured runtime state constraints.",
        f"Original user request: {user_input}",
        "Structured blocker facts (JSON):",
        _serialize_note_payload({"blocked_tools": state_conflict_facts}),
        "Use the structured blocker metadata above instead of guessing from prose.",
        "If a safe retry is available, emit the corrected tool call now. Otherwise answer honestly about what was not performed.",
        "Do not claim success for any blocker with action_performed=false.",
    ]
    for fact in state_conflict_facts:
        if fact["failure_kind"] == "collision_conflict" and fact.get("suggested_version_path"):
            lines.append(
                "For the write collision above, the safe retry is write_text_file "
                "with collision_policy='version'. Suggested version path: "
                f"{fact['suggested_version_path']}."
            )
        if fact["failure_kind"] == "confirmation_required":
            lines.append(
                "For the confirmation-required blocker above, the action was not performed."
            )
    return "\n".join(lines)


def _build_state_conflict_fact(tool_result: ToolResult) -> dict[str, Any]:
    payload = _tool_result_payload_dict(tool_result)
    return {
        "tool_name": tool_result.tool_name,
        "failure_kind": tool_result.failure_kind,
        "action_performed": _tool_result_payload_flag(tool_result, "action_performed"),
        "requested_path": payload.get("requested_path"),
        "resolved_path": payload.get("resolved_path"),
        "path": payload.get("path"),
        "confirm_required": payload.get("confirm_required"),
        "suggested_collision_policy": payload.get("suggested_collision_policy"),
        "suggested_version_path": payload.get("suggested_version_path"),
    }


def _append_tool_result_messages(
    *,
    context_messages: list[LLMMessage],
    tool_results: Sequence[ToolResult],
) -> None:
    for tool_result in tool_results:
        context_messages.append(
            LLMMessage(
                role=LLMRole.TOOL,
                content=build_untrusted_tool_message_content(tool_result.output_text),
            )
        )


_STRUCTURAL_FAILURE_KINDS = frozenset({
    "schema_error",
    "unknown_tool",
    "contract_error",
})


def _all_failures_look_structural(tool_results: Sequence[ToolResult]) -> bool:
    """Return True when every failed result has structured failure metadata
    indicating a schema, contract, or unknown-tool error.

    Only failures with an explicit ``failure_kind`` in the structural set
    qualify.  Failures without metadata (e.g. tool-handler runtime errors)
    are treated as non-structural so that the model does not waste a repair
    attempt on them.
    """
    for tool_result in tool_results:
        if tool_result.success is not False:
            continue
        if tool_result.failure_kind not in _STRUCTURAL_FAILURE_KINDS:
            return False
    return True


_TOOL_ARGUMENT_REPAIR_NOTE_PREFIX = "Tool argument repair:"


def _build_tool_argument_repair_note(
    *,
    failed_tool_calls: Sequence[ToolCall],
    failed_tool_results: Sequence[ToolResult],
    tool_definitions: Sequence[ToolDefinition],
    user_input: str,
) -> str:
    tool_schema_lines: list[str] = []
    failed_tool_names = {call.tool_name for call in failed_tool_calls}
    for tool_def in tool_definitions:
        if tool_def.name in failed_tool_names:
            arg_parts: list[str] = []
            for arg_name, arg_raw in tool_def.arguments.items():
                spec = resolve_tool_argument_spec(arg_raw)
                arg_parts.append(f"{arg_name}: {spec.value_type}")
            tool_schema_lines.append(f"  {tool_def.name}({', '.join(arg_parts)})")

    error_lines: list[str] = []
    for i, (call, result) in enumerate(zip(failed_tool_calls, failed_tool_results)):
        error_lines.append(
            f"  [{i}] {call.tool_name}({call.arguments}) -> error: {result.error}"
        )

    parts = [
        f"{_TOOL_ARGUMENT_REPAIR_NOTE_PREFIX} "
        "the previous tool call(s) failed due to argument or schema errors.",
        f"Original user request: {user_input}",
        "Failed calls with errors:",
        *error_lines,
        "Available tool schemas (* = required):",
        *tool_schema_lines,
        "Either emit a corrected tool call with valid arguments, "
        "or answer honestly that you cannot proceed.",
    ]
    return "\n".join(parts)


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
        tool_result, finished_at = _maybe_retry_timed_out_search_web_result(
            pending_call=pending_call,
            tool_result=tool_result,
            finished_at=finished_at,
            tool_registry=tool_registry,
            tool_guard_state=tool_guard_state,
        )
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

        normalized_tool_call = normalize_tool_call_for_execution(
            tool_registry,
            tool_call,
        )
        started_at = perf_counter()
        if tool_call_callback is not None:
            tool_call_callback(normalized_tool_call)
        tracer.trace_tool_started(
            session_id=session_id,
            tool_name=normalized_tool_call.tool_name,
            arguments=normalized_tool_call.arguments,
            skill_id=tool_registry.get_owner_skill_id(normalized_tool_call.tool_name),
        )
        pending_call = _launch_pending_tool_execution(
            tool_call=normalized_tool_call,
            tool_registry=tool_registry,
            started_at=started_at,
        )
        pending_calls.append(pending_call)

    tool_guard_state.tool_calls_started += len(pending_calls)
    return pending_calls


def _launch_pending_tool_execution(
    *,
    tool_call: ToolCall,
    tool_registry: ToolRegistry,
    started_at: float | None = None,
) -> _PendingToolExecution:
    done_event = threading.Event()
    pending_call = _PendingToolExecution(
        tool_call=tool_call,
        started_at=perf_counter() if started_at is None else started_at,
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
    return pending_call


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
            failure_kind="execution_error",
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


def _maybe_retry_timed_out_search_web_result(
    *,
    pending_call: _PendingToolExecution,
    tool_result: ToolResult,
    finished_at: float,
    tool_registry: ToolRegistry,
    tool_guard_state: _RuntimeToolGuardState,
) -> tuple[ToolResult, float]:
    if not _should_retry_timed_out_search_web_result(
        pending_call=pending_call,
        tool_result=tool_result,
        tool_guard_state=tool_guard_state,
    ):
        return tool_result, finished_at

    tool_guard_state.search_web_timeout_retry_used = True
    tool_guard_state.tool_calls_started += 1
    retry_pending_call = _launch_pending_tool_execution(
        tool_call=_build_search_web_timeout_retry_tool_call(pending_call.tool_call),
        tool_registry=tool_registry,
    )
    retry_result, retry_finished_at = _collect_pending_tool_results(
        pending_calls=(retry_pending_call,),
        tool_guard_state=tool_guard_state,
    )[0][1:]
    return retry_result, retry_finished_at


def _should_retry_timed_out_search_web_result(
    *,
    pending_call: _PendingToolExecution,
    tool_result: ToolResult,
    tool_guard_state: _RuntimeToolGuardState,
) -> bool:
    if pending_call.tool_call.tool_name != "search_web":
        return False
    if tool_result.success is not False or not _tool_result_timed_out(tool_result):
        return False
    if tool_guard_state.is_cancelled():
        return False
    if tool_guard_state.search_web_timeout_retry_used:
        return False
    return (
        tool_guard_state.tool_calls_started
        < tool_guard_state.max_tool_calls_per_turn
    )


def _build_search_web_timeout_retry_tool_call(tool_call: ToolCall) -> ToolCall:
    retry_arguments = dict(tool_call.arguments)
    retry_arguments["max_results"] = _resolve_search_web_timeout_retry_max_results(
        retry_arguments.get("max_results")
    )
    return ToolCall(
        tool_name=tool_call.tool_name,
        arguments=retry_arguments,
    )


def _resolve_search_web_timeout_retry_max_results(raw_value: Any) -> int:
    if isinstance(raw_value, int) and not isinstance(raw_value, bool):
        if raw_value <= 2:
            return 1
        return min(raw_value - 1, _SEARCH_WEB_TIMEOUT_RETRY_MAX_RESULTS)
    return _SEARCH_WEB_TIMEOUT_RETRY_MAX_RESULTS


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
        payload={"execution_state": "timed_out"},
        failure_kind="timeout",
    )


def _tool_result_timed_out(tool_result: ToolResult) -> bool:
    if tool_result.failure_kind == "timeout":
        return True
    payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
    execution_state = payload.get("execution_state")
    return isinstance(execution_state, str) and execution_state == "timed_out"


def _build_cancelled_tool_result(tool_call: ToolCall) -> ToolResult:
    return ToolResult.failure(
        tool_name=tool_call.tool_name,
        error=f"Tool '{tool_call.tool_name}' was cancelled.",
        payload={"execution_state": "cancelled"},
        failure_kind="cancelled",
    )


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
    from unclaw.core.research_flow import persist_tool_result

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
