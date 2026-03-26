"""Structural helpers for grounded reply finalization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from unclaw.constants import EMPTY_RESPONSE_REPLY
from unclaw.tools.contracts import ToolResult

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
) -> dict[str, Any]:
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
        "completion_risks": {
            "has_blocking_failures": bool(failed_tool_names),
            "state_conflict_tool_names": state_conflict_tool_names,
            "action_not_performed_tool_names": action_not_performed_tool_names,
            "multi_step_tool_turn": len(tool_results) > 1,
            "deliverable_check_required": (
                bool(failed_tool_names)
                or len(tool_results) > 1
                or write_succeeded
                or delete_succeeded
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
        return "There is no persisted task progress for this session."

    if session_goal_state.status == "completed":
        return f"The persisted task is completed: {session_goal_state.goal}"

    if session_goal_state.status == "blocked":
        blocker = session_goal_state.last_blocker or "no blocker detail was persisted"
        step = session_goal_state.current_step or "the current step"
        return f"The persisted task is blocked on {step}: {blocker}"

    latest_entry = session_progress_ledger[-1] if session_progress_ledger else None
    step = session_goal_state.current_step or (
        latest_entry.step if latest_entry is not None else "the current step"
    )
    if latest_entry is None:
        return f"The persisted task is active on {step}."
    return (
        f"The persisted task is active on {step}. "
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
            "additional completed actions."
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
