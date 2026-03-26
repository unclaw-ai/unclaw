"""Runtime support helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

from unclaw.core.command_handler import CommandHandler
from unclaw.core.orchestrator import (
    ModelCallFailedError,
    Orchestrator,
    OrchestratorError,
)
from unclaw.core.reply_discipline import (
    _build_grounded_reply_facts,
    _build_structural_finalization_fallback,
    _tool_result_timed_out,
)
from unclaw.core.session_manager import SessionManager
from unclaw.constants import EMPTY_RESPONSE_REPLY, RUNTIME_ERROR_REPLY
from unclaw.errors import ConfigurationError
from unclaw.llm.base import LLMMessage, LLMRole
from unclaw.memory.protocols import SessionMemoryContextProvider
from unclaw.schemas.chat import MessageRole
from unclaw.tools.contracts import ToolDefinition, ToolResult
from unclaw.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from unclaw.core.routing import _EntityAnchor

_ENTITY_RECENTERING_NOTE_PREFIX = "Entity recentering hint:"
_POST_TOOL_GROUNDING_NOTE_PREFIX = "Post-tool grounding note:"
_MODEL_NATIVE_TOOL_RECOVERY_NOTE_PREFIX = "Tool reconsideration note:"
_SEARCH_REPAIR_FACTS_NOTE_PREFIX = "Search repair facts:"
_SESSION_GOAL_STATE_NOTE_PREFIX = "Session goal state:"
_SESSION_PROGRESS_LEDGER_NOTE_PREFIX = "Session progress ledger:"
_SESSION_TASK_CONTINUITY_NOTE_PREFIX = "Session task continuity:"
_WRITE_SUCCESS_TOOL_NAME = "write_text_file"
_STATE_CONFLICT_FAILURE_KINDS = frozenset(
    {
        "access_denied",
        "collision_conflict",
        "confirmation_required",
        "permission_denied",
        "unsupported_input",
    }
)
_BLOCKED_GOAL_CONTINUATION_MAX_TOKENS = 2
_BLOCKED_GOAL_CONTINUATION_MAX_TOKEN_ALNUM_CHARS = 3
_BLOCKED_GOAL_CONTINUATION_ALLOWED_PUNCTUATION = ".,!?;:'\"-()[]{}"
_GROUNDED_REPLY_FINALIZER_SYSTEM_PROMPT = "\n".join(
    (
        "Grounded reply finalizer for one runtime turn.",
        "Rewrite the draft into the final user reply using only the provided runtime facts and evidence.",
        "Hard rules:",
        "- Treat the structured tool facts and raw tool outputs as the only evidence for current-turn execution claims.",
        "- Do not claim that a file was created, saved, updated, or modified unless the current turn includes a successful `write_text_file` tool result.",
        "- Do not claim that a file was deleted unless the current turn includes a successful `delete_file` tool result.",
        "- If any tool payload says `action_performed` is false, explicitly say that the action did not happen.",
        "- Do not claim research, search completion, biography completion, or overall task completion unless the evidence and persisted task state support it.",
        "- If the user is asking about task status or progress, answer from the persisted task state before the turn plus the current-turn execution facts.",
        "- If requested deliverables are still unsupported or blocked, say that plainly instead of pretending they are complete.",
        "- Use the original user request plus `completion_risks` to make sure every requested deliverable is addressed once. If tool-backed work succeeded but the draft only covers part of the request, add the remaining short textual deliverable in the final reply.",
        "- If blocking failures remain, say exactly what completed and what remains blocked. Do not flatten blocked turns into full success.",
        "- If `execution_claim_risks.no_tools_ran_this_turn` is true, do not preserve unsupported claims that side effects, research, or task completion happened in this turn.",
        "- If `execution_claim_risks.unsupported_execution_claim_risk` or `execution_claim_risks.completion_without_execution_risk` is true, explicitly say what did not happen yet and whether another tool step is still needed.",
        "- Keep the reply in the user's language when it is inferable from the request.",
        "- Do not end with promises about actions that did not happen in this turn.",
        "- Preserve supported details from the draft when they remain grounded.",
        "- If the evidence_quality section indicates thin or partial search evidence, do NOT expand limited search results into detailed claims, biographies, or timelines. Only state what the evidence directly supports.",
        "- When a file was written based on thin search evidence (write_after_thin_search is true), the reply must acknowledge that the file content is based on limited evidence. Do not present thin-evidence content as comprehensive or complete.",
        "Return JSON only with this shape: {\"final_reply\": \"...\"}",
    )
)


@dataclass(frozen=True, slots=True)
class _NoToolExecutionClaimRiskAssessment:
    assistant_reply: str
    unsupported_execution_claim_risk: bool = False
    completion_without_execution_risk: bool = False
    multi_deliverable_request_risk: bool = False

    def as_payload(self) -> dict[str, Any]:
        return {
            "unsupported_execution_claim_risk": (
                self.unsupported_execution_claim_risk
            ),
            "completion_without_execution_risk": (
                self.completion_without_execution_risk
            ),
            "multi_deliverable_request_risk": (
                self.multi_deliverable_request_risk
            ),
            "no_tool_honesty_rescue_used": True,
        }


def _tool_result_payload_dict(tool_result: ToolResult) -> dict[str, Any]:
    return tool_result.payload if isinstance(tool_result.payload, dict) else {}


def _fast_web_search_match_quality(tool_result: ToolResult) -> str | None:
    payload = _tool_result_payload_dict(tool_result)
    match_quality = payload.get("match_quality")
    if isinstance(match_quality, str):
        return match_quality
    result_count = payload.get("result_count")
    if isinstance(result_count, int) and result_count <= 0:
        return "no_results"
    return None


def _fast_web_search_result_is_mismatch(tool_result: ToolResult) -> bool:
    if tool_result.success is not True:
        return False
    return _fast_web_search_match_quality(tool_result) in {"mismatch", "no_results"}


def _fast_web_search_result_is_thin(tool_result: ToolResult) -> bool:
    if tool_result.success is not True:
        return True

    payload = _tool_result_payload_dict(tool_result)
    result_count = payload.get("result_count")
    supported_point_count = payload.get("supported_point_count")

    if _fast_web_search_result_is_mismatch(tool_result):
        return True
    if isinstance(result_count, int) and result_count <= 1:
        return True
    if isinstance(supported_point_count, int):
        return supported_point_count <= 1
    return False


def _search_web_result_is_thin(tool_result: ToolResult) -> bool:
    if tool_result.success is not True:
        return True

    payload = _tool_result_payload_dict(tool_result)
    evidence_count = payload.get("evidence_count")
    finding_count = payload.get("finding_count")
    display_sources = payload.get("display_sources")

    if isinstance(finding_count, int) and finding_count <= 1:
        return True
    if isinstance(evidence_count, int) and evidence_count <= 1:
        return True
    if isinstance(display_sources, list) and len(display_sources) <= 1:
        return True
    return False


def _tool_result_needs_model_native_search_repair(tool_result: ToolResult) -> bool:
    if tool_result.tool_name == "fast_web_search":
        return _fast_web_search_result_is_thin(tool_result)
    if tool_result.tool_name == "search_web":
        return _search_web_result_is_thin(tool_result)
    return False


def _build_model_native_search_repair_fact_line(tool_result: ToolResult) -> str:
    payload = _tool_result_payload_dict(tool_result)
    parts = [
        f"tool_name={tool_result.tool_name}",
        f"success={tool_result.success}",
    ]

    for field_name in (
        "result_count",
        "supported_point_count",
        "evidence_count",
        "finding_count",
        "match_quality",
    ):
        field_value = payload.get(field_name)
        if isinstance(field_value, bool):
            continue
        if isinstance(field_value, (int, str)):
            parts.append(f"{field_name}={field_value}")

    display_sources = payload.get("display_sources")
    if isinstance(display_sources, list):
        parts.append(f"display_sources_count={len(display_sources)}")

    return "; ".join(parts) + "."


def _serialize_grounded_reply_facts(facts: dict[str, Any]) -> str:
    normalized_facts = json.loads(
        json.dumps(facts, ensure_ascii=False, default=str)
    )
    return json.dumps(normalized_facts, ensure_ascii=False, indent=2, sort_keys=True)


def _build_grounded_reply_finalization_messages(
    *,
    user_input: str,
    assistant_draft_reply: str,
    tool_results: Sequence[ToolResult],
    session_goal_state: Any,
    session_progress_ledger: Sequence[Any],
    available_tool_definitions: Sequence[ToolDefinition],
    no_tool_execution_claim_risks: dict[str, Any] | None,
    turn_cancelled_reply: str,
) -> tuple[LLMMessage, ...]:
    facts = _build_grounded_reply_facts(
        user_input=user_input,
        assistant_draft_reply=assistant_draft_reply,
        tool_results=tool_results,
        turn_cancelled_reply=turn_cancelled_reply,
        session_goal_state=session_goal_state,
        session_progress_ledger=session_progress_ledger,
        available_tool_definitions=available_tool_definitions,
        no_tool_execution_claim_risks=no_tool_execution_claim_risks,
    )
    return (
        LLMMessage(
            role=LLMRole.SYSTEM,
            content=_GROUNDED_REPLY_FINALIZER_SYSTEM_PROMPT,
        ),
        LLMMessage(
            role=LLMRole.USER,
            content=_serialize_grounded_reply_facts(facts),
        ),
    )


def _parse_grounded_reply_finalization_response(response_text: str) -> str | None:
    stripped_response = response_text.strip()
    if not stripped_response:
        return None

    try:
        payload = json.loads(stripped_response)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    final_reply = payload.get("final_reply")
    if not isinstance(final_reply, str):
        return None

    normalized_reply = final_reply.strip()
    return normalized_reply or None


def _turn_requires_grounded_reply_finalization(
    *,
    user_input: str,
    assistant_draft_reply: str,
    tool_results: Sequence[ToolResult],
    session_goal_state: Any,
    session_progress_ledger: Sequence[Any],
    available_tool_definitions: Sequence[ToolDefinition],
    no_tool_execution_claim_risk_assessment: _NoToolExecutionClaimRiskAssessment | None,
) -> bool:
    grounded_facts = _build_grounded_reply_facts(
        user_input=user_input,
        assistant_draft_reply=assistant_draft_reply,
        tool_results=tool_results,
        session_goal_state=session_goal_state,
        session_progress_ledger=session_progress_ledger,
        available_tool_definitions=available_tool_definitions,
        no_tool_execution_claim_risks=(
            no_tool_execution_claim_risk_assessment.as_payload()
            if no_tool_execution_claim_risk_assessment is not None
            else None
        ),
    )
    if tool_results:
        current_turn_tool_summary = grounded_facts["current_turn_tool_summary"]
        completion_risks = grounded_facts["completion_risks"]
        if (
            completion_risks["deliverable_check_required"]
            or current_turn_tool_summary["grounded_search_succeeded"]
        ):
            return True
        return any(_tool_result_requires_honesty_finalization(tool_result) for tool_result in tool_results)
    # No tools ran. Skip finalization for most no-tool turns so simple
    # arithmetic/chat stays cheap. We still finalize when persisted task
    # state is terminal or when the bounded no-tool honesty rescue returned
    # structured risk flags that the draft implied unsupported completion.
    execution_claim_risks = grounded_facts["execution_claim_risks"]
    completion_risks = grounded_facts["completion_risks"]
    return any(
        (
            execution_claim_risks["unsupported_execution_claim_risk"],
            execution_claim_risks["completion_without_execution_risk"],
            execution_claim_risks["multi_deliverable_request_risk"],
            completion_risks["deliverable_check_required"],
        )
    )


def _tool_result_requires_honesty_finalization(tool_result: ToolResult) -> bool:
    if tool_result.success is True:
        return False
    payload = _tool_result_payload_dict(tool_result)
    if payload.get("action_performed") is False:
        return True
    return tool_result.failure_kind in _STATE_CONFLICT_FAILURE_KINDS


def _finalize_grounded_reply(
    *,
    orchestrator: Orchestrator,
    session_id: str,
    model_profile_name: str,
    thinking_enabled: bool,
    tracer: Any,
    user_input: str,
    assistant_draft_reply: str,
    tool_results: Sequence[ToolResult],
    session_goal_state: Any,
    session_progress_ledger: Sequence[Any],
    available_tool_definitions: Sequence[ToolDefinition],
    no_tool_execution_claim_risk_assessment: _NoToolExecutionClaimRiskAssessment | None,
    turn_cancelled_reply: str,
) -> tuple[str, bool]:
    stripped_draft_reply = assistant_draft_reply.strip() or EMPTY_RESPONSE_REPLY
    no_tool_execution_claim_risks = (
        no_tool_execution_claim_risk_assessment.as_payload()
        if no_tool_execution_claim_risk_assessment is not None
        else None
    )

    def _build_no_tool_risk_fallback() -> str:
        if (
            not tool_results
            and no_tool_execution_claim_risk_assessment is not None
            and no_tool_execution_claim_risk_assessment.assistant_reply.strip()
        ):
            return no_tool_execution_claim_risk_assessment.assistant_reply.strip()
        return _build_structural_finalization_fallback(
            reply=stripped_draft_reply,
            tool_results=tool_results,
            turn_cancelled_reply=turn_cancelled_reply,
            session_goal_state=session_goal_state,
            session_progress_ledger=session_progress_ledger,
            finalization_required=True,
        )

    finalization_required = _turn_requires_grounded_reply_finalization(
        user_input=user_input,
        assistant_draft_reply=stripped_draft_reply,
        tool_results=tool_results,
        session_goal_state=session_goal_state,
        session_progress_ledger=session_progress_ledger,
        available_tool_definitions=available_tool_definitions,
        no_tool_execution_claim_risk_assessment=(
            no_tool_execution_claim_risk_assessment
        ),
    )
    if not finalization_required:
        if tool_results and all(tool_result.success is False for tool_result in tool_results):
            return (
                _build_structural_finalization_fallback(
                    reply=stripped_draft_reply,
                    tool_results=tool_results,
                    turn_cancelled_reply=turn_cancelled_reply,
                    session_goal_state=session_goal_state,
                    session_progress_ledger=session_progress_ledger,
                ),
                False,
            )
        return stripped_draft_reply, False

    messages = _build_grounded_reply_finalization_messages(
        user_input=user_input,
        assistant_draft_reply=stripped_draft_reply,
        tool_results=tool_results,
        session_goal_state=session_goal_state,
        session_progress_ledger=session_progress_ledger,
        available_tool_definitions=available_tool_definitions,
        no_tool_execution_claim_risks=no_tool_execution_claim_risks,
        turn_cancelled_reply=turn_cancelled_reply,
    )

    try:
        turn_result = orchestrator.call_model(
            session_id=session_id,
            messages=messages,
            model_profile_name=model_profile_name,
            thinking_enabled=thinking_enabled,
        )
        tracer.trace_model_succeeded(
            session_id=session_id,
            provider=turn_result.response.provider,
            model_name=turn_result.response.model_name,
            finish_reason=turn_result.response.finish_reason,
            output_length=len(turn_result.response.content),
            model_duration_ms=turn_result.model_duration_ms,
            reasoning=turn_result.response.reasoning,
        )
    except (AssertionError, ConfigurationError, ModelCallFailedError, OrchestratorError) as exc:
        if isinstance(exc, ModelCallFailedError):
            tracer.trace_model_failed(
                session_id=session_id,
                provider=exc.provider,
                model_profile_name=exc.model_profile_name,
                model_name=exc.model_name,
                model_duration_ms=exc.duration_ms,
                error=str(exc),
            )
        return (_build_no_tool_risk_fallback(), True)

    finalized_reply = _parse_grounded_reply_finalization_response(
        turn_result.response.content
    )
    if finalized_reply is None:
        return (_build_no_tool_risk_fallback(), True)

    return finalized_reply, True


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


def _build_session_goal_state_context_note(
    *,
    session_manager: SessionManager,
    session_id: str,
) -> str | None:
    goal_state = session_manager.get_session_goal_state(session_id)
    if goal_state is None:
        return None

    return (
        f"{_SESSION_GOAL_STATE_NOTE_PREFIX} "
        f"goal={_format_goal_state_note_value(goal_state.goal)}; "
        f"status={_format_goal_state_note_value(goal_state.status)}; "
        f"current_step={_format_goal_state_note_value(goal_state.current_step)}; "
        f"last_blocker={_format_goal_state_note_value(goal_state.last_blocker)}; "
        f"updated_at={_format_goal_state_note_value(goal_state.updated_at)}."
    )


def _build_session_progress_ledger_context_note(
    *,
    session_manager: SessionManager,
    session_id: str,
) -> str | None:
    if session_manager.get_session_goal_state(session_id) is None:
        return None

    ledger = session_manager.get_session_progress_ledger(session_id)
    if not ledger:
        return None

    entries = " | ".join(
        (
            "["
            f"status={_format_goal_state_note_value(entry.status)}; "
            f"step={_format_goal_state_note_value(entry.step)}; "
            f"detail={_format_goal_state_note_value(entry.detail)}; "
            f"updated_at={_format_goal_state_note_value(entry.updated_at)}"
            "]"
        )
        for entry in ledger
    )
    return f"{_SESSION_PROGRESS_LEDGER_NOTE_PREFIX} {entries}."


def _build_session_task_continuity_note(
    *,
    session_manager: SessionManager,
    session_id: str,
) -> str | None:
    goal_state = session_manager.get_session_goal_state(session_id)
    if goal_state is None:
        return None

    parts = [
        f"{_SESSION_TASK_CONTINUITY_NOTE_PREFIX} "
        f"goal={_format_goal_state_note_value(goal_state.goal)}",
        f"status={_format_goal_state_note_value(goal_state.status)}",
    ]
    if goal_state.current_step is not None:
        parts.append(
            f"current_step={_format_goal_state_note_value(goal_state.current_step)}"
        )

    if goal_state.status == "blocked" and goal_state.last_blocker is not None:
        parts.append(
            f"last_blocker={_format_goal_state_note_value(goal_state.last_blocker)}"
        )
    elif goal_state.status == "active":
        progress_ledger = session_manager.get_session_progress_ledger(session_id)
        if progress_ledger:
            latest_entry = progress_ledger[-1]
            parts.append(
                "latest_progress=["
                f"status={_format_goal_state_note_value(latest_entry.status)}; "
                f"step={_format_goal_state_note_value(latest_entry.step)}; "
                f"detail={_format_goal_state_note_value(latest_entry.detail)}"
                "]"
            )

    return "; ".join(parts) + "."


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


def _turn_qualifies_for_session_goal_state_persistence(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> bool:
    return _turn_may_create_session_goal_state(
        session_manager=session_manager,
        session_id=session_id,
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    ) or _turn_may_update_existing_session_goal_state(
        session_manager=session_manager,
        session_id=session_id,
        user_input=user_input,
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    )


def _turn_may_create_session_goal_state(
    *,
    session_manager: SessionManager,
    session_id: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> bool:
    if session_manager.get_session_goal_state(session_id) is not None:
        return False
    return _turn_has_task_like_runtime_facts(
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    )


def _turn_may_update_existing_session_goal_state(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> bool:
    existing_goal_state = session_manager.get_session_goal_state(session_id)
    if existing_goal_state is None:
        return False
    return _turn_has_task_like_runtime_facts(
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    ) or (
        existing_goal_state.status in {"blocked", "completed"}
        and _turn_can_handoff_terminal_session_goal_state(
            user_input=user_input,
            tool_results=tool_results,
            assistant_reply=assistant_reply,
            turn_cancelled_reply=turn_cancelled_reply,
        )
    )


def _turn_has_task_like_runtime_facts(
    *,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> bool:
    if not tool_results:
        return False

    latest_tool_result = tool_results[-1]
    if len(tool_results) >= 2:
        return True
    if assistant_reply.strip() == turn_cancelled_reply.strip():
        return True
    if latest_tool_result.success is False:
        return True
    return (
        latest_tool_result.success is True
        and latest_tool_result.tool_name == _WRITE_SUCCESS_TOOL_NAME
    )


def _turn_can_handoff_terminal_session_goal_state(
    *,
    user_input: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> bool:
    if assistant_reply.strip() == turn_cancelled_reply.strip():
        return False
    if _user_input_has_compact_blocked_goal_continuation_shape(user_input):
        return False
    if len(tool_results) != 1:
        return False
    return _tool_result_shows_meaningful_forward_progress(tool_results[0])


def _tool_result_shows_meaningful_forward_progress(tool_result: ToolResult) -> bool:
    # Only treat richer, structured research-like payloads as early handoff progress.
    if (
        tool_result.success is not True
        or tool_result.tool_name == _WRITE_SUCCESS_TOOL_NAME
    ):
        return False

    payload = tool_result.payload
    if not isinstance(payload, dict):
        return False

    summary_points = payload.get("summary_points")
    display_sources = payload.get("display_sources")
    evidence_count = payload.get("evidence_count")
    finding_count = payload.get("finding_count")
    return (
        isinstance(summary_points, list)
        and len(summary_points) >= 1
        and isinstance(display_sources, list)
        and len(display_sources) >= 1
        and (
            isinstance(evidence_count, int)
            or isinstance(finding_count, int)
        )
    )


def _get_latest_session_user_input(
    *,
    session_manager: SessionManager,
    session_id: str,
) -> str:
    for message in reversed(session_manager.list_messages(session_id)):
        if message.role is MessageRole.USER:
            return message.content
    return ""


def _turn_requires_progress_entry_after_terminal_goal_handoff(
    *,
    existing_goal_state: Any,
    user_input: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> bool:
    return (
        existing_goal_state is not None
        and existing_goal_state.status in {"blocked", "completed"}
        and not _turn_has_task_like_runtime_facts(
            tool_results=tool_results,
            assistant_reply=assistant_reply,
            turn_cancelled_reply=turn_cancelled_reply,
        )
        and _turn_can_handoff_terminal_session_goal_state(
            user_input=user_input,
            tool_results=tool_results,
            assistant_reply=assistant_reply,
            turn_cancelled_reply=turn_cancelled_reply,
        )
    )


def _find_latest_failed_non_write_tool_result(
    tool_results: Sequence[ToolResult],
) -> ToolResult | None:
    for tool_result in reversed(tool_results[:-1]):
        if (
            tool_result.success is False
            and tool_result.tool_name != _WRITE_SUCCESS_TOOL_NAME
        ):
            return tool_result
    return None


def _find_latest_completion_blocking_web_tool_result(
    tool_results: Sequence[ToolResult],
) -> ToolResult | None:
    for tool_result in reversed(tool_results[:-1]):
        if tool_result.success is not True:
            continue
        if tool_result.tool_name == "fast_web_search":
            return (
                tool_result if _fast_web_search_result_is_thin(tool_result) else None
            )
        if tool_result.tool_name == "search_web":
            return tool_result if _search_web_result_is_thin(tool_result) else None
    return None


def _build_completion_blocking_web_detail(tool_result: ToolResult) -> str:
    if tool_result.tool_name == "fast_web_search":
        if _fast_web_search_result_is_mismatch(tool_result):
            return (
                "Quick web grounding matched a different entity or found no exact "
                "usable match."
            )
        return "Quick web grounding was too thin to confirm requested details."
    return "Web evidence was too thin to confirm requested details."


def _turn_should_mark_goal_state_completed(
    *,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> bool:
    if not tool_results:
        return False
    if assistant_reply.strip() == turn_cancelled_reply.strip():
        return False

    latest_tool_result = tool_results[-1]
    return (
        latest_tool_result.success is True
        and latest_tool_result.tool_name == _WRITE_SUCCESS_TOOL_NAME
        and _find_latest_failed_non_write_tool_result(tool_results) is None
        and _find_latest_completion_blocking_web_tool_result(tool_results) is None
    )


def _resolve_session_goal_text_for_runtime_persistence(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> str:
    existing_goal_state = session_manager.get_session_goal_state(session_id)
    if existing_goal_state is None:
        return user_input

    if existing_goal_state.status == "active":
        return existing_goal_state.goal

    if existing_goal_state.status != "blocked":
        return user_input

    if _turn_can_handoff_terminal_session_goal_state(
        user_input=user_input,
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    ):
        return user_input

    if _turn_clearly_replaces_blocked_session_goal_state(
        user_input=user_input,
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    ):
        return user_input

    return existing_goal_state.goal


def _turn_clearly_replaces_blocked_session_goal_state(
    *,
    user_input: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> bool:
    if not _turn_should_mark_goal_state_completed(
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    ):
        return False

    if _user_input_has_compact_blocked_goal_continuation_shape(user_input):
        return False

    return True


def _user_input_has_compact_blocked_goal_continuation_shape(user_input: str) -> bool:
    normalized_input = " ".join(user_input.split()).strip()
    if not normalized_input:
        return False

    tokens = normalized_input.split(" ")
    if len(tokens) > _BLOCKED_GOAL_CONTINUATION_MAX_TOKENS:
        return False

    saw_alnum = False
    for token in tokens:
        stripped_token = token.strip(_BLOCKED_GOAL_CONTINUATION_ALLOWED_PUNCTUATION)
        if not stripped_token:
            return False
        if not stripped_token.isalnum():
            return False
        if len(stripped_token) > _BLOCKED_GOAL_CONTINUATION_MAX_TOKEN_ALNUM_CHARS:
            return False
        saw_alnum = True

    if not saw_alnum:
        return False

    return all(
        character.isalnum()
        or character.isspace()
        or character in _BLOCKED_GOAL_CONTINUATION_ALLOWED_PUNCTUATION
        for character in normalized_input
    )


def _persist_session_goal_state_from_runtime_facts(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> None:
    if not _turn_qualifies_for_session_goal_state_persistence(
        session_manager=session_manager,
        session_id=session_id,
        user_input=user_input,
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    ):
        return

    existing_goal_state = session_manager.get_session_goal_state(session_id)
    latest_tool_result = tool_results[-1]
    latest_failed_non_write_tool_result = _find_latest_failed_non_write_tool_result(
        tool_results
    )
    latest_completion_blocking_web_tool_result = (
        _find_latest_completion_blocking_web_tool_result(tool_results)
    )
    last_blocker: str | None = None
    status = "active"
    current_step = latest_tool_result.tool_name
    if _turn_should_mark_goal_state_completed(
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    ):
        status = "completed"
    elif assistant_reply.strip() == turn_cancelled_reply.strip():
        status = "blocked"
        last_blocker = turn_cancelled_reply
    elif (
        latest_tool_result.success is True
        and latest_tool_result.tool_name == _WRITE_SUCCESS_TOOL_NAME
        and latest_failed_non_write_tool_result is not None
    ):
        status = "blocked"
        current_step = latest_failed_non_write_tool_result.tool_name
        last_blocker = (
            latest_failed_non_write_tool_result.error
            or latest_failed_non_write_tool_result.output_text
        )
    elif (
        latest_tool_result.success is True
        and latest_tool_result.tool_name == _WRITE_SUCCESS_TOOL_NAME
        and latest_completion_blocking_web_tool_result is not None
    ):
        status = "blocked"
        current_step = latest_completion_blocking_web_tool_result.tool_name
        last_blocker = _build_completion_blocking_web_detail(
            latest_completion_blocking_web_tool_result
        )
    elif latest_tool_result.success is False:
        status = "blocked"
        last_blocker = latest_tool_result.error or latest_tool_result.output_text

    goal_text = _resolve_session_goal_text_for_runtime_persistence(
        session_manager=session_manager,
        session_id=session_id,
        user_input=user_input,
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    )

    session_manager.persist_session_goal_state(
        session_id=session_id,
        goal=goal_text,
        status=status,
        current_step=current_step,
        last_blocker=last_blocker,
    )
    if _turn_requires_progress_entry_after_terminal_goal_handoff(
        existing_goal_state=existing_goal_state,
        user_input=user_input,
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    ):
        session_manager.persist_session_progress_entry(
            session_id=session_id,
            status="active",
            step=latest_tool_result.tool_name,
            detail="tool succeeded",
        )


def _persist_session_progress_ledger_from_runtime_facts(
    *,
    session_manager: SessionManager,
    session_id: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> None:
    user_input = _get_latest_session_user_input(
        session_manager=session_manager,
        session_id=session_id,
    )
    if not _turn_qualifies_for_session_goal_state_persistence(
        session_manager=session_manager,
        session_id=session_id,
        user_input=user_input,
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
    ):
        return

    latest_tool_result = tool_results[-1]
    latest_failed_non_write_tool_result = _find_latest_failed_non_write_tool_result(
        tool_results
    )
    latest_completion_blocking_web_tool_result = (
        _find_latest_completion_blocking_web_tool_result(tool_results)
    )
    status = "active"
    detail = "tool succeeded"
    step = latest_tool_result.tool_name
    if assistant_reply.strip() == turn_cancelled_reply.strip():
        status = "blocked"
        detail = "request cancelled before tool work completed"
    elif (
        latest_tool_result.success is True
        and latest_tool_result.tool_name == _WRITE_SUCCESS_TOOL_NAME
        and latest_failed_non_write_tool_result is not None
    ):
        status = "blocked"
        step = latest_failed_non_write_tool_result.tool_name
        detail = _build_progress_detail_from_failed_tool_result(
            latest_failed_non_write_tool_result
        )
    elif (
        latest_tool_result.success is True
        and latest_tool_result.tool_name == _WRITE_SUCCESS_TOOL_NAME
        and latest_completion_blocking_web_tool_result is not None
    ):
        status = "blocked"
        step = latest_completion_blocking_web_tool_result.tool_name
        detail = _build_completion_blocking_web_detail(
            latest_completion_blocking_web_tool_result
        )
    elif latest_tool_result.success is True:
        if latest_tool_result.tool_name == _WRITE_SUCCESS_TOOL_NAME:
            detail = "file write succeeded"
    else:
        status = "blocked"
        detail = _build_progress_detail_from_failed_tool_result(latest_tool_result)

    session_manager.persist_session_progress_entry(
        session_id=session_id,
        status=status,
        step=step,
        detail=detail,
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


def _parse_no_tool_execution_claim_risk_response(
    response_text: str,
) -> _NoToolExecutionClaimRiskAssessment | None:
    stripped_response = response_text.strip()
    if not stripped_response:
        return None

    try:
        payload = json.loads(stripped_response)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    assistant_reply = payload.get("assistant_reply")
    if not isinstance(assistant_reply, str):
        return None

    normalized_reply = assistant_reply.strip()
    if not normalized_reply:
        return None

    return _NoToolExecutionClaimRiskAssessment(
        assistant_reply=normalized_reply,
        unsupported_execution_claim_risk=(
            payload.get("unsupported_execution_claim_risk") is True
        ),
        completion_without_execution_risk=(
            payload.get("completion_without_execution_risk") is True
        ),
        multi_deliverable_request_risk=(
            payload.get("multi_deliverable_request_risk") is True
        ),
    )


def _build_model_native_tool_recovery_note(
    *,
    user_input: str,
    assistant_draft_reply: str,
    tool_definitions: Sequence[ToolDefinition],
    session_goal_state: Any,
    session_progress_ledger: Sequence[Any],
) -> str:
    facts = _build_grounded_reply_facts(
        user_input=user_input,
        assistant_draft_reply=assistant_draft_reply,
        tool_results=(),
        session_goal_state=session_goal_state,
        session_progress_ledger=session_progress_ledger,
        available_tool_definitions=tool_definitions,
    )
    json_shape = (
        '{"assistant_reply": "...", '
        '"unsupported_execution_claim_risk": false, '
        '"completion_without_execution_risk": false, '
        '"multi_deliverable_request_risk": false}'
    )
    return "\n".join(
        (
            (
                f"{_MODEL_NATIVE_TOOL_RECOVERY_NOTE_PREFIX} "
                "the first pass returned text without tool calls."
            ),
            "Reconsider the same user request using the prior draft plus the structured runtime facts below.",
            "If an available tool call is needed for a grounded answer or meaningful progress, emit that tool call now.",
            f"If no tool is needed, return JSON only with this shape: {json_shape}",
            "Set unsupported_execution_claim_risk=true if the draft or revised reply would claim completed side effects, completed evidence gathering, or task completion without support from current-turn tool results or persisted task state.",
            "Set completion_without_execution_risk=true if additional tool work is still needed before the request can honestly be called complete.",
            "Set multi_deliverable_request_risk=true only if multiple requested deliverables remain unevenly satisfied.",
            "If no tool ran, do not claim that any file was created, modified, deleted, searched, researched, or completed unless the persisted task state already confirms it.",
            "Structured runtime facts (JSON):",
            _serialize_grounded_reply_facts(facts),
        )
    )


def _build_entity_recentering_note(
    *,
    entity_anchor: _EntityAnchor | None,
    user_input: str,
) -> str | None:
    if entity_anchor is None:
        return None

    explicit_surface_in_turn = bool(entity_anchor.surface) and (
        entity_anchor.surface.casefold() in user_input.casefold()
    )

    if entity_anchor.corrected and explicit_surface_in_turn:
        return (
            f"{_ENTITY_RECENTERING_NOTE_PREFIX} "
            f"the user just corrected the target entity or context to "
            f"'{entity_anchor.surface}'. Reuse that exact entity on the next "
            "search or biography step and do not drift back to the earlier mistake."
        )

    if entity_anchor.corrected:
        return (
            f"{_ENTITY_RECENTERING_NOTE_PREFIX} "
            f"this turn appears to follow the user's corrected entity or context "
            f"'{entity_anchor.surface}'. Reuse it for the next search or biography "
            "step and do not ask generic clarification again unless a new ambiguity remains."
        )

    if explicit_surface_in_turn:
        return None

    return (
        f"{_ENTITY_RECENTERING_NOTE_PREFIX} "
        f"this turn looks like a follow-up about '{entity_anchor.surface}'. "
        "Keep that entity centered on the next tool step unless the user changes it."
    )


def _build_post_tool_grounding_note(
    *,
    tool_results: Sequence[ToolResult],
    tool_definitions: Sequence[ToolDefinition],
) -> str:
    success_count = sum(1 for tool_result in tool_results if tool_result.success)
    error_count = len(tool_results) - success_count
    available_tool_names = {
        tool_definition.name for tool_definition in tool_definitions
    }
    latest_tool_names = {tool_result.tool_name for tool_result in tool_results}

    lines = [
        (
            f"{_POST_TOOL_GROUNDING_NOTE_PREFIX} the latest tool step returned "
            f"{success_count} success(es) and {error_count} error(s)."
        ),
        "Base the next reply on the latest tool results above.",
        "Only state facts that the latest tool outputs directly support.",
        "If the user's request is now answered, reply directly from that evidence.",
        "If one obvious missing fact remains and a listed tool can get it, call that tool now instead of stopping early.",
        "Do not ask for clarification when the current request already gives enough information for the next obvious tool step.",
        "Do not contradict successful tool output, and do not say a tool failed unless the tool result above says it failed.",
        "Do not infer missing names, professions, achievements, biographies, timelines, or background details from weak clues.",
    ]

    if any(
        _fast_web_search_result_is_thin(tool_result)
        or _search_web_result_is_thin(tool_result)
        for tool_result in tool_results
    ):
        lines.append(
            "One or more latest tool outputs are thin or partial. Say that they are limited and give only the supported fragment."
        )
    repair_fact_lines = [
        _build_model_native_search_repair_fact_line(tool_result)
        for tool_result in tool_results
        if _tool_result_needs_model_native_search_repair(tool_result)
    ]
    if repair_fact_lines:
        lines.append(
            (
                f"{_SEARCH_REPAIR_FACTS_NOTE_PREFIX} "
                "use the latest structured search facts plus the user request to decide the next step."
            )
        )
        lines.extend(f"- {fact_line}" for fact_line in repair_fact_lines)
        lines.append(
            "If another available tool call or a better-grounded retry is needed, emit it now. Otherwise answer only with the supported fragment and say the evidence is limited."
        )
    if any(tool_result.success is False for tool_result in tool_results):
        lines.append(
            "A failed or timed-out tool call does not confirm the requested fact. If no successful tool result supports it, say you could not confirm it."
        )
    if len(tool_results) > 1:
        lines.append(
            "Keep multiple tool results separate. If quality differs across entities or sources, say which parts are solid and which remain weak."
        )
    if any(
        tool_result.tool_name in {
            "system_info",
            "read_text_file",
            "list_directory",
            "run_terminal_command",
            "fetch_url_text",
        }
        for tool_result in tool_results
    ):
        lines.append(
            "For local file, terminal, system, or fetched-page output, summarize only what the tool output actually shows."
        )

    if "fast_web_search" in latest_tool_names and "search_web" in available_tool_names:
        lines.append(
            "search_web is the deeper grounded web path — it fetches pages, condenses "
            "evidence, and builds a richer answer. Call search_web next if the user "
            "wants a full biography, complete research, or a file output from web research."
        )
        lines.append(
            "Do not expand a fast_web_search grounding note into a full biography — "
            "call search_web to ground additional details before writing a complete answer."
        )
    if any(_tool_result_timed_out(tr) for tr in tool_results):
        lines.append(
            "A tool timed out: reuse any valid grounding from earlier successful steps. "
            "Do not invent missing facts. If enough grounded evidence exists for a partial "
            "answer, give it clearly labeled as partial. Otherwise say the search timed "
            "out and the detail could not be confirmed."
        )
    if "list_directory" in latest_tool_names and "read_text_file" in available_tool_names:
        lines.append(
            "If the directory listing surfaced a relevant supported text file and "
            "the user needs its contents, call read_text_file next instead of "
            "stopping at the listing."
        )
    has_thin_search = any(
        _fast_web_search_result_is_thin(tr) or _search_web_result_is_thin(tr)
        for tr in tool_results
    )
    if "write_text_file" in available_tool_names and has_thin_search:
        lines.append(
            "IMPORTANT: search evidence for this turn is thin or partial. "
            "If the next step is writing a file, the file content MUST stay bounded "
            "to the facts that the search evidence actually supports. Do not expand "
            "thin evidence into detailed biographies, timelines, or unsupported claims. "
            "If evidence is partial, the written file must note which details could "
            "not be confirmed."
        )

    return "\n".join(lines)


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


def _build_model_failure_reply(error: ModelCallFailedError) -> str:
    message = error.error.strip()
    if message:
        return message
    return RUNTIME_ERROR_REPLY


def _format_goal_state_note_value(value: str | None) -> str:
    if value is None:
        return "none"
    return json.dumps(value, ensure_ascii=False)


def _build_progress_detail_from_failed_tool_result(tool_result: ToolResult) -> str:
    raw_detail = tool_result.error or tool_result.output_text
    if not raw_detail:
        return "tool failed"

    normalized_detail = " ".join(raw_detail.split()).strip()
    if not normalized_detail:
        return "tool failed"

    tool_prefix = f"Tool '{tool_result.tool_name}' "
    if normalized_detail.startswith(tool_prefix):
        stripped_detail = normalized_detail[len(tool_prefix) :].strip()
        if stripped_detail:
            return stripped_detail
    return normalized_detail
