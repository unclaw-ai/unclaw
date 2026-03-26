"""Structural helpers for grounded reply finalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from unclaw.constants import EMPTY_RESPONSE_REPLY
from unclaw.tools.contracts import ToolDefinition, ToolPermissionLevel, ToolResult

if TYPE_CHECKING:
    from unclaw.core.session_manager import SessionGoalState, SessionProgressEntry

_DEFAULT_TURN_CANCELLED_REPLY = (
    "This request was cancelled before tool work completed."
)
_WRITE_SUCCESS_TOOL_NAME = "write_text_file"
_DELETE_SUCCESS_TOOL_NAME = "delete_file"
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


def _tool_result_payload_dict(tool_result: ToolResult) -> dict[str, Any]:
    return tool_result.payload if isinstance(tool_result.payload, dict) else {}


def _tool_result_action_performed(tool_result: ToolResult) -> bool | None:
    payload = _tool_result_payload_dict(tool_result)
    action_performed = payload.get("action_performed")
    if isinstance(action_performed, bool):
        return action_performed
    if tool_result.success is True:
        return True
    return None


def _tool_result_timed_out(tool_result: ToolResult) -> bool:
    if tool_result.failure_kind == "timeout":
        return True
    payload = _tool_result_payload_dict(tool_result)
    execution_state = payload.get("execution_state")
    if isinstance(execution_state, str) and execution_state == "timed_out":
        return True

    haystack = " ".join(
        part for part in (tool_result.error, tool_result.output_text) if part
    )
    return "timed out" in haystack.casefold()


def _tool_names_with_permission_level(
    tool_definitions: Sequence[ToolDefinition],
    *,
    permission_levels: frozenset[ToolPermissionLevel],
) -> tuple[str, ...]:
    return tuple(
        tool_definition.name
        for tool_definition in tool_definitions
        if tool_definition.permission_level in permission_levels
    )


def _ordered_unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _tool_definition_by_name(
    tool_definitions: Sequence[ToolDefinition],
) -> dict[str, ToolDefinition]:
    return {
        tool_definition.name: tool_definition for tool_definition in tool_definitions
    }


def _tool_result_has_local_path_observation(tool_result: ToolResult) -> bool:
    payload = _tool_result_payload_dict(tool_result)
    for key in (
        "path",
        "resolved_path",
        "requested_path",
        "source_path",
        "destination_path",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _normalize_no_tool_execution_claim_risks(
    no_tool_execution_claim_risks: Mapping[str, Any] | None,
) -> dict[str, bool]:
    if not isinstance(no_tool_execution_claim_risks, Mapping):
        return {
            "unsupported_execution_claim_risk": False,
            "completion_without_execution_risk": False,
            "multi_deliverable_request_risk": False,
            "no_tool_honesty_rescue_used": False,
        }

    return {
        "unsupported_execution_claim_risk": (
            no_tool_execution_claim_risks.get("unsupported_execution_claim_risk")
            is True
        ),
        "completion_without_execution_risk": (
            no_tool_execution_claim_risks.get("completion_without_execution_risk")
            is True
        ),
        "multi_deliverable_request_risk": (
            no_tool_execution_claim_risks.get("multi_deliverable_request_risk")
            is True
        ),
        "no_tool_honesty_rescue_used": (
            no_tool_execution_claim_risks.get("no_tool_honesty_rescue_used") is True
        ),
    }


def _normalize_tool_result(tool_result: ToolResult) -> dict[str, Any]:
    payload = _tool_result_payload_dict(tool_result)
    return {
        "tool_name": tool_result.tool_name,
        "success": tool_result.success,
        "output_text": tool_result.output_text,
        "error": tool_result.error,
        "failure_kind": tool_result.failure_kind,
        "payload": payload,
        "execution_state": payload.get("execution_state"),
        "action_performed": _tool_result_action_performed(tool_result),
    }


def _normalize_session_goal_state(
    session_goal_state: SessionGoalState | None,
) -> dict[str, Any] | None:
    if session_goal_state is None:
        return None

    return {
        "goal": session_goal_state.goal,
        "status": session_goal_state.status,
        "current_step": session_goal_state.current_step,
        "last_blocker": session_goal_state.last_blocker,
        "updated_at": session_goal_state.updated_at,
    }


def _normalize_session_progress_ledger(
    session_progress_ledger: Sequence[SessionProgressEntry],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "status": entry.status,
            "step": entry.step,
            "detail": entry.detail,
            "updated_at": entry.updated_at,
        }
        for entry in session_progress_ledger
    )


def _tool_result_has_thin_search_evidence(tool_result: ToolResult) -> bool:
    """Return True when a search tool result has thin or partial evidence."""
    if tool_result.success is not True:
        return True
    payload = _tool_result_payload_dict(tool_result)
    if tool_result.tool_name == "fast_web_search":
        match_quality = payload.get("match_quality")
        if isinstance(match_quality, str) and match_quality in {"mismatch", "no_results"}:
            return True
        result_count = payload.get("result_count")
        if isinstance(result_count, int) and result_count <= 1:
            return True
        supported_point_count = payload.get("supported_point_count")
        if isinstance(supported_point_count, int) and supported_point_count <= 1:
            return True
    if tool_result.tool_name == "search_web":
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


def _build_grounded_reply_facts(
    *,
    user_input: str,
    assistant_draft_reply: str,
    tool_results: Sequence[ToolResult],
    turn_cancelled_reply: str = _DEFAULT_TURN_CANCELLED_REPLY,
    session_goal_state: SessionGoalState | None = None,
    session_progress_ledger: Sequence[SessionProgressEntry] = (),
    available_tool_definitions: Sequence[ToolDefinition] = (),
    no_tool_execution_claim_risks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tool_definitions_by_name = _tool_definition_by_name(available_tool_definitions)
    normalized_tool_results = tuple(
        _normalize_tool_result(tool_result) for tool_result in tool_results
    )
    successful_tool_names = tuple(
        tool_result.tool_name
        for tool_result in tool_results
        if tool_result.success is True
    )
    failed_tool_names = tuple(
        tool_result.tool_name
        for tool_result in tool_results
        if tool_result.success is False
    )
    latest_tool_result = tool_results[-1] if tool_results else None

    thin_evidence_tool_names = tuple(
        tr.tool_name
        for tr in tool_results
        if tr.tool_name in _SEARCH_TOOL_NAMES
        and _tool_result_has_thin_search_evidence(tr)
    )
    has_thin_search = bool(thin_evidence_tool_names)
    write_succeeded = _WRITE_SUCCESS_TOOL_NAME in successful_tool_names
    delete_succeeded = _DELETE_SUCCESS_TOOL_NAME in successful_tool_names
    state_conflict_tool_names = tuple(
        tool_result.tool_name
        for tool_result in tool_results
        if tool_result.success is False
        and tool_result.failure_kind in _STATE_CONFLICT_FAILURE_KINDS
    )
    action_not_performed_tool_names = tuple(
        tool_result.tool_name
        for tool_result in tool_results
        if _tool_result_action_performed(tool_result) is False
    )
    persisted_goal_state_status = (
        session_goal_state.status if session_goal_state is not None else "none"
    )
    side_effect_tool_names = _tool_names_with_permission_level(
        available_tool_definitions,
        permission_levels=frozenset(
            {
                ToolPermissionLevel.LOCAL_WRITE,
                ToolPermissionLevel.LOCAL_EXECUTE,
            }
        ),
    )
    evidence_gathering_tool_names = _tool_names_with_permission_level(
        available_tool_definitions,
        permission_levels=frozenset(
            {
                ToolPermissionLevel.LOCAL_READ,
                ToolPermissionLevel.NETWORK,
            }
        ),
    )
    observed_permission_levels = _ordered_unique_strings(
        [
            tool_definitions_by_name[tool_result.tool_name].permission_level.value
            for tool_result in tool_results
            if tool_result.tool_name in tool_definitions_by_name
        ]
    )
    side_effect_tool_names_ran = _ordered_unique_strings(
        [
            tool_result.tool_name
            for tool_result in tool_results
            if tool_result.tool_name in tool_definitions_by_name
            and tool_definitions_by_name[tool_result.tool_name].permission_level
            in {
                ToolPermissionLevel.LOCAL_WRITE,
                ToolPermissionLevel.LOCAL_EXECUTE,
            }
        ]
    )
    network_tool_names_ran = _ordered_unique_strings(
        [
            tool_result.tool_name
            for tool_result in tool_results
            if tool_result.tool_name in tool_definitions_by_name
            and tool_definitions_by_name[tool_result.tool_name].permission_level
            is ToolPermissionLevel.NETWORK
        ]
    )
    local_path_observation_tool_names = _ordered_unique_strings(
        [
            tool_result.tool_name
            for tool_result in tool_results
            if _tool_result_has_local_path_observation(tool_result)
        ]
    )
    local_path_action_tool_names_ran = _ordered_unique_strings(
        [
            tool_result.tool_name
            for tool_result in tool_results
            if tool_result.tool_name in tool_definitions_by_name
            and tool_definitions_by_name[tool_result.tool_name].permission_level
            in {
                ToolPermissionLevel.LOCAL_WRITE,
                ToolPermissionLevel.LOCAL_EXECUTE,
            }
            and _tool_result_has_local_path_observation(tool_result)
        ]
    )
    side_effect_tool_succeeded = any(
        tool_result.success is True
        and tool_result.tool_name in side_effect_tool_names_ran
        for tool_result in tool_results
    )
    preparatory_tool_turn = (
        bool(network_tool_names_ran) or bool(local_path_observation_tool_names)
    ) and bool(side_effect_tool_names) and not side_effect_tool_succeeded
    phase_checkpoint_required = any(
        (
            bool(state_conflict_tool_names),
            bool(action_not_performed_tool_names),
            bool(local_path_action_tool_names_ran),
            has_thin_search,
            preparatory_tool_turn,
        )
    )
    normalized_no_tool_execution_claim_risks = (
        _normalize_no_tool_execution_claim_risks(no_tool_execution_claim_risks)
    )
    execution_claim_risks = {
        "no_tools_ran_this_turn": not tool_results,
        "side_effect_tools_available": bool(side_effect_tool_names),
        "side_effect_tool_names": side_effect_tool_names,
        "evidence_gathering_tools_available": bool(evidence_gathering_tool_names),
        "evidence_gathering_tool_names": evidence_gathering_tool_names,
        "persisted_goal_state_status": persisted_goal_state_status,
        "persisted_progress_entry_count": len(session_progress_ledger),
        "completion_without_execution_risk": (
            normalized_no_tool_execution_claim_risks[
                "completion_without_execution_risk"
            ]
        ),
        "unsupported_execution_claim_risk": (
            normalized_no_tool_execution_claim_risks[
                "unsupported_execution_claim_risk"
            ]
            or (
                not tool_results
                and persisted_goal_state_status in {"completed", "blocked"}
            )
        ),
        "multi_deliverable_request_risk": (
            normalized_no_tool_execution_claim_risks[
                "multi_deliverable_request_risk"
            ]
        ),
        "no_tool_honesty_rescue_used": (
            normalized_no_tool_execution_claim_risks[
                "no_tool_honesty_rescue_used"
            ]
        ),
    }
    execution_claim_check_required = any(
        (
            execution_claim_risks["completion_without_execution_risk"],
            execution_claim_risks["unsupported_execution_claim_risk"],
            execution_claim_risks["multi_deliverable_request_risk"],
        )
    )

    return {
        "user_input": user_input,
        "assistant_draft_reply": assistant_draft_reply,
        "turn_cancelled": assistant_draft_reply.strip() == turn_cancelled_reply.strip(),
        "current_turn_tool_results": normalized_tool_results,
        "current_turn_tool_summary": {
            "tool_count": len(tool_results),
            "success_count": len(successful_tool_names),
            "failure_count": len(failed_tool_names),
            "successful_tool_names": successful_tool_names,
            "failed_tool_names": failed_tool_names,
            "all_tools_failed": bool(tool_results) and len(failed_tool_names) == len(tool_results),
            "write_text_file_succeeded": write_succeeded,
            "delete_file_succeeded": delete_succeeded,
            "grounded_search_succeeded": any(
                tool_name in _SEARCH_TOOL_NAMES for tool_name in successful_tool_names
            ),
            "latest_tool_name": (
                latest_tool_result.tool_name if latest_tool_result is not None else None
            ),
            "latest_tool_success": (
                latest_tool_result.success if latest_tool_result is not None else None
            ),
            "latest_tool_failure_kind": (
                latest_tool_result.failure_kind
                if latest_tool_result is not None
                else None
            ),
        },
        "evidence_quality": {
            "has_thin_search_evidence": has_thin_search,
            "thin_evidence_tool_names": thin_evidence_tool_names,
            "write_after_thin_search": write_succeeded and has_thin_search,
        },
        "tool_permission_summary": {
            "observed_permission_levels": observed_permission_levels,
            "side_effect_tool_names_ran": side_effect_tool_names_ran,
            "side_effect_tool_succeeded": side_effect_tool_succeeded,
            "network_tool_names_ran": network_tool_names_ran,
            "local_path_observation_tool_names": (
                local_path_observation_tool_names
            ),
            "preparatory_tool_turn": preparatory_tool_turn,
        },
        "execution_claim_risks": execution_claim_risks,
        "completion_risks": {
            "has_blocking_failures": bool(failed_tool_names),
            "state_conflict_tool_names": state_conflict_tool_names,
            "action_not_performed_tool_names": action_not_performed_tool_names,
            "multi_step_tool_turn": len(tool_results) > 1,
            "phase_checkpoint_required": phase_checkpoint_required,
            "deliverable_check_required": (
                phase_checkpoint_required
                or write_succeeded
                or delete_succeeded
                or execution_claim_check_required
            ),
        },
        "persisted_goal_state_before_turn": _normalize_session_goal_state(
            session_goal_state
        ),
        "persisted_progress_ledger_before_turn": _normalize_session_progress_ledger(
            session_progress_ledger
        ),
    }


def _build_all_failed_tool_reply(
    *,
    tool_results: Sequence[ToolResult],
) -> str:
    if any(_tool_result_action_performed(tool_result) is False for tool_result in tool_results):
        latest_tool_result = tool_results[-1]
        if latest_tool_result.failure_kind == "confirmation_required":
            return "The requested action was not performed because confirmation was required."
        return "The requested action was blocked and was not performed."
    if any(_tool_result_timed_out(tool_result) for tool_result in tool_results):
        return (
            "The tool step timed out, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
    return (
        "The tool step failed, so I couldn't confirm the requested details "
        "from retrieved tool evidence."
    )


def _build_grounded_task_status_reply(
    *,
    session_goal_state: SessionGoalState | None,
    session_progress_ledger: Sequence[SessionProgressEntry],
) -> str:
    if session_goal_state is None:
        return "There is no persisted mission progress for this session."

    if session_goal_state.status == "completed":
        return f"The persisted mission is completed: {session_goal_state.goal}"

    if session_goal_state.status == "blocked":
        blocker = session_goal_state.last_blocker or "no blocker detail was persisted"
        step = session_goal_state.current_step or "the current step"
        return f"The persisted mission is blocked on {step}: {blocker}"

    latest_entry = session_progress_ledger[-1] if session_progress_ledger else None
    step = session_goal_state.current_step or (
        latest_entry.step if latest_entry is not None else "the current step"
    )
    if latest_entry is None:
        return f"The persisted mission is active on {step}."
    return (
        f"The persisted mission is active on {step}. "
        f"Latest persisted progress: {latest_entry.detail}"
    )


def _build_structural_finalization_fallback(
    *,
    reply: str,
    tool_results: Sequence[ToolResult],
    turn_cancelled_reply: str = _DEFAULT_TURN_CANCELLED_REPLY,
    session_goal_state: SessionGoalState | None = None,
    session_progress_ledger: Sequence[SessionProgressEntry] = (),
    finalization_required: bool = False,
) -> str:
    stripped_reply = reply.strip()
    if stripped_reply == turn_cancelled_reply.strip():
        return stripped_reply

    if session_goal_state is not None and not tool_results:
        return _build_grounded_task_status_reply(
            session_goal_state=session_goal_state,
            session_progress_ledger=session_progress_ledger,
        )

    if finalization_required and not tool_results:
        return (
            "I did not execute any tools in this turn, so I cannot confirm "
            "additional completed mission work."
        )

    if tool_results and all(tool_result.success is False for tool_result in tool_results):
        return _build_all_failed_tool_reply(tool_results=tool_results)

    if stripped_reply:
        return stripped_reply

    if tool_results:
        successful_tool_names = tuple(
            tool_result.tool_name
            for tool_result in tool_results
            if tool_result.success is True
        )
        if len(successful_tool_names) == 1:
            return f"The {successful_tool_names[0]} tool succeeded for this turn."
        if successful_tool_names:
            return "The available tool steps succeeded for this turn."

    return EMPTY_RESPONSE_REPLY
