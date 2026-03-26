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
_SEARCH_TOOL_NAMES = frozenset({"fast_web_search", "search_web"})


def _tool_result_payload_dict(tool_result: ToolResult) -> dict[str, Any]:
    return tool_result.payload if isinstance(tool_result.payload, dict) else {}


def _tool_result_timed_out(tool_result: ToolResult) -> bool:
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
        "payload": payload,
        "execution_state": payload.get("execution_state"),
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
            "write_text_file_succeeded": _WRITE_SUCCESS_TOOL_NAME in successful_tool_names,
            "grounded_search_succeeded": any(
                tool_name in _SEARCH_TOOL_NAMES for tool_name in successful_tool_names
            ),
            "latest_tool_name": (
                latest_tool_result.tool_name if latest_tool_result is not None else None
            ),
            "latest_tool_success": (
                latest_tool_result.success if latest_tool_result is not None else None
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
