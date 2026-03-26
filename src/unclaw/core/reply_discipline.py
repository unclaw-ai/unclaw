"""Post-tool reply-discipline helpers."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import TYPE_CHECKING, Any

from unclaw.constants import EMPTY_RESPONSE_REPLY
from unclaw.core.routing import (
    _looks_like_deep_search_request,
)
from unclaw.tools.contracts import ToolResult

if TYPE_CHECKING:
    from unclaw.core.session_manager import SessionGoalState, SessionProgressEntry

_DEFAULT_TURN_CANCELLED_REPLY = (
    "This request was cancelled before tool work completed."
)
_TASK_STATUS_REQUEST_PATTERN = re.compile(
    r"\b(?:where does this task stand|where is this task|"
    r"what are we working on in this session|what task are we working on|"
    r"task status|current task|task progress|progress on this task)\b",
    flags=re.IGNORECASE,
)
_LIMITATION_ACK_PATTERN = re.compile(
    r"\b(?:could not|couldn't|cannot|can't|did not|didn't|not confirmed|"
    r"unconfirmed|limited|partial|sparse|insufficient|failed|timed out|timeout|"
    r"je n'ai pas pu|je ne peux pas|je ne pouvais pas|non confirme|"
    r"pas confirme|limite|limitee|partiel|partielle|echec|a expire|expire)\b",
    flags=re.IGNORECASE,
)
_FILE_SUCCESS_CLAIM_PATTERN = re.compile(
    r"\b(?:saved|wrote|written|created|updated|modified)\b(?=.*\b(?:file|note|"
    r"document|locally)\b)",
    flags=re.IGNORECASE | re.DOTALL,
)
_SEARCH_ACTION_CLAIM_PATTERN = re.compile(
    r"\b(?:searched|looked up|researched|verified online)\b",
    flags=re.IGNORECASE,
)
_RESEARCH_COMPLETION_CLAIM_PATTERN = re.compile(
    r"\b(?:complete(?:d)?|finished|done)\b(?=.*\b(?:research|search|biograph(?:y|ies)|"
    r"profile)\b)|\b(?:research|search|biograph(?:y|ies)|profile)\b(?=.*\b(?:complete(?:d)?|"
    r"finished|done)\b)",
    flags=re.IGNORECASE | re.DOTALL,
)
_TASK_COMPLETION_CLAIM_PATTERN = re.compile(
    r"\b(?:task|request|work)\b(?=.*\b(?:complete(?:d)?|done|finished)\b)|"
    r"\b(?:complete(?:d)?|done|finished)\b(?=.*\b(?:task|request|work)\b)",
    flags=re.IGNORECASE | re.DOTALL,
)
_ACTIVE_STATUS_CLAIM_PATTERN = re.compile(
    r"\b(?:in progress|working on|researching|currently working|currently researching|"
    r"still working|still researching|you are writing|we are writing|you are reading|"
    r"we are reading|you are verifying|we are verifying)\b",
    flags=re.IGNORECASE,
)
_BLOCKED_STATUS_CLAIM_PATTERN = re.compile(
    r"\b(?:blocked|stuck|waiting on)\b",
    flags=re.IGNORECASE,
)
_PROMISED_FILE_STEP_PATTERN = re.compile(
    r"(?P<sentence>\b(?:i|we)\s+(?:will(?: now)?|shall(?: now)?|am going to|are going to|"
    r"am now|are now|['’]ll now)\s+(?:create|write|save)\b[^.!?]*\b(?:file|note|document)\b"
    r"[^.!?]*[.!?]?)$",
    flags=re.IGNORECASE,
)
_PROMISED_FILE_VERIFY_PATTERN = re.compile(
    r"(?P<sentence>\b(?:i|we)\s+(?:will(?: now)?|shall(?: now)?|am going to|are going to|"
    r"am now|are now|['’]ll now)\s+(?:verify|check|read|inspect)\b[^.!?]*\bfile\b[^.!?]*"
    r"[.!?]?)$",
    flags=re.IGNORECASE,
)
_PROMISED_SEARCH_STEP_PATTERN = re.compile(
    r"(?P<sentence>\b(?:i|we)\s+(?:will(?: now)?|shall(?: now)?|am going to|are going to|"
    r"am now|are now|['’]ll now)\s+(?:search|look up|research)\b[^.!?]*[.!?]?)$",
    flags=re.IGNORECASE,
)


def _tool_result_payload_dict(tool_result: ToolResult) -> dict[str, Any]:
    return tool_result.payload if isinstance(tool_result.payload, dict) else {}


def _tool_result_timed_out(tool_result: ToolResult) -> bool:
    haystack = " ".join(
        part for part in (tool_result.error, tool_result.output_text) if part
    )
    return "timed out" in haystack.casefold()


def _extract_fast_grounding_points(tool_result: ToolResult) -> tuple[str, ...]:
    grounding_note = _tool_result_payload_dict(tool_result).get("grounding_note")
    note = grounding_note if isinstance(grounding_note, str) else tool_result.output_text
    return tuple(
        line[2:].strip()
        for line in note.splitlines()
        if line.strip().startswith("- ")
    )


def _fast_web_search_result_is_mismatch(tool_result: ToolResult) -> bool:
    if tool_result.success is not True:
        return False
    note = tool_result.output_text.casefold()
    return (
        "different entity" in note
        or "no exact top match" in note
        or "no web results found" in note
    )


def _fast_web_search_result_is_thin(tool_result: ToolResult) -> bool:
    if tool_result.success is not True:
        return True

    payload = _tool_result_payload_dict(tool_result)
    result_count = payload.get("result_count")
    supported_points = _extract_fast_grounding_points(tool_result)
    if _fast_web_search_result_is_mismatch(tool_result):
        return True
    if isinstance(result_count, int) and result_count <= 1:
        return True
    return len(supported_points) <= 1


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


def _reply_acknowledges_limitations(reply: str) -> bool:
    return _LIMITATION_ACK_PATTERN.search(reply) is not None


def _reply_is_effectively_empty(reply: str) -> bool:
    stripped_reply = reply.strip()
    return not stripped_reply or stripped_reply == EMPTY_RESPONSE_REPLY


def _reply_sentence_count(text: str) -> int:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    return len(sentences) if sentences else (1 if text.strip() else 0)


def _reply_expands_primary_point(
    *,
    candidate_reply: str,
    primary_point: str,
) -> bool:
    normalized_reply = _normalized_reply_text(candidate_reply).rstrip(".!?")
    normalized_point = _normalized_reply_text(primary_point).rstrip(".!?")
    if not normalized_point or normalized_point not in normalized_reply:
        return False
    extra_text = normalized_reply.replace(normalized_point, "", 1).strip(" ,;:-")
    return bool(extra_text)


def _build_all_failed_tool_reply(
    *,
    user_input: str,
    tool_results: Sequence[ToolResult],
) -> str:
    del user_input
    timeout_failed = any(_tool_result_timed_out(tool_result) for tool_result in tool_results)

    if timeout_failed:
        return (
            "The tool step timed out, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
    return (
        "The tool step failed, so I couldn't confirm the requested details "
        "from retrieved tool evidence."
    )


def _build_fast_grounding_guarded_reply(
    *,
    candidate_reply: str,
    user_input: str,
    fast_results: Sequence[ToolResult],
) -> str | None:
    if not fast_results:
        return None

    supported_points = [
        point
        for tool_result in fast_results
        for point in _extract_fast_grounding_points(tool_result)[:1]
    ]
    any_mismatch = any(
        _fast_web_search_result_is_mismatch(tool_result) for tool_result in fast_results
    )
    all_thin = all(_fast_web_search_result_is_thin(tool_result) for tool_result in fast_results)
    full_bio_requested = _looks_like_deep_search_request(user_input)
    reply_is_effectively_empty = _reply_is_effectively_empty(candidate_reply)
    reply_is_minimal = reply_is_effectively_empty or _reply_sentence_count(candidate_reply) <= 1

    if len(fast_results) > 1 and (any_mismatch or all_thin):
        return (
            "The quick grounding results were limited or mixed, so I can only "
            "confirm partial fragments for these entities. I couldn't confirm "
            "full biographies from those probes alone."
        )

    primary_point = supported_points[0] if supported_points else ""
    if any_mismatch and not primary_point:
        return (
            "The quick web grounding appeared to match a different entity, so I "
            "couldn't confirm the requested details from that result alone."
        )

    if primary_point and (
        reply_is_effectively_empty
        or full_bio_requested
        or (
            all_thin
            and not reply_is_minimal
            and primary_point.casefold() not in candidate_reply.casefold()
        )
        or (
            all_thin
            and _reply_expands_primary_point(
                candidate_reply=candidate_reply,
                primary_point=primary_point,
            )
        )
        or (all_thin and _reply_sentence_count(candidate_reply) > 1)
    ):
        return (
            f"{primary_point} I couldn't confirm a fuller biography from that "
            "quick grounding probe alone."
        )

    return None


def _looks_like_task_status_request(user_input: str) -> bool:
    return _TASK_STATUS_REQUEST_PATTERN.search(user_input) is not None


def _find_latest_failed_non_write_tool_result(
    tool_results: Sequence[ToolResult],
) -> ToolResult | None:
    for tool_result in reversed(tool_results):
        if tool_result.success is False and tool_result.tool_name != "write_text_file":
            return tool_result
    return None


def _find_latest_completion_blocking_web_tool_result(
    tool_results: Sequence[ToolResult],
) -> ToolResult | None:
    for tool_result in reversed(tool_results):
        if tool_result.success is not True:
            continue
        if tool_result.tool_name == "fast_web_search":
            return tool_result if _fast_web_search_result_is_thin(tool_result) else None
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


def _normalized_reply_text(reply: str) -> str:
    return " ".join(reply.casefold().split())


def _reply_claims_file_write_success(reply: str) -> bool:
    normalized = _normalized_reply_text(reply)
    if any(
        phrase in normalized
        for phrase in (
            "saved it locally",
            "saved the note locally",
            "saved the file",
            "saved the note",
            "created the file",
            "created the note",
            "wrote the file",
            "wrote the note",
            "written the file",
            "written the note",
            "file was created",
            "file has been created",
        )
    ):
        return True
    return _FILE_SUCCESS_CLAIM_PATTERN.search(reply) is not None


def _reply_claims_search_action(reply: str) -> bool:
    return _SEARCH_ACTION_CLAIM_PATTERN.search(reply) is not None


def _reply_claims_complete_research(reply: str) -> bool:
    normalized = _normalized_reply_text(reply)
    if any(
        phrase in normalized
        for phrase in (
            "complete research",
            "completed the research",
            "finished the research",
            "research is complete",
            "research is done",
            "complete biography",
            "completed the biography",
            "finished the biography",
            "biography is complete",
            "biography is done",
        )
    ):
        return True
    return _RESEARCH_COMPLETION_CLAIM_PATTERN.search(reply) is not None


def _reply_claims_task_completion(reply: str) -> bool:
    normalized = _normalized_reply_text(reply)
    if any(
        phrase in normalized
        for phrase in (
            "task is complete",
            "task is completed",
            "task is done",
            "request is complete",
            "request is done",
            "work is complete",
            "work is done",
            "completed the task",
            "finished the task",
        )
    ):
        return True
    return _TASK_COMPLETION_CLAIM_PATTERN.search(reply) is not None


def _reply_claims_active_task_status(reply: str) -> bool:
    return _ACTIVE_STATUS_CLAIM_PATTERN.search(reply) is not None


def _reply_claims_blocked_task_status(reply: str) -> bool:
    return _BLOCKED_STATUS_CLAIM_PATTERN.search(reply) is not None


def _ensure_terminal_punctuation(text: str) -> str:
    stripped_text = text.strip()
    if not stripped_text:
        return stripped_text
    if stripped_text[-1] in ".!?":
        return stripped_text
    return f"{stripped_text}."


def _build_missing_write_claim_reply() -> str:
    return "I haven't created or saved the requested file yet."


def _build_missing_search_claim_reply() -> str:
    return "I haven't run grounded search for that yet."


def _build_partial_grounded_research_reply(*, wrote_file: bool) -> str:
    if wrote_file:
        return (
            "I wrote the file, but I only have partial grounded research so far, "
            "so I couldn't confirm a complete biography or finished research task yet."
        )
    return (
        "I only have partial grounded research so far, so I couldn't confirm "
        "a complete biography or finished research task yet."
    )


def _build_incomplete_task_reply(
    *,
    blocker_detail: str,
    wrote_file: bool,
) -> str:
    normalized_detail = _ensure_terminal_punctuation(" ".join(blocker_detail.split()))
    if wrote_file:
        return f"I wrote the file, but the task is not complete because {normalized_detail}"
    return f"The task is not complete because {normalized_detail}"


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


def _task_status_reply_conflicts_with_persisted_state(
    *,
    reply: str,
    session_goal_state: SessionGoalState,
) -> bool:
    claims_completion = (
        _reply_claims_task_completion(reply)
        or _reply_claims_file_write_success(reply)
        or _reply_claims_complete_research(reply)
    )
    claims_active = _reply_claims_active_task_status(reply)
    claims_blocked = _reply_claims_blocked_task_status(reply)

    if session_goal_state.status == "completed":
        return claims_active or claims_blocked
    if session_goal_state.status == "active":
        return claims_completion or claims_blocked
    return claims_completion or claims_active


def _maybe_ground_task_status_reply(
    *,
    reply: str,
    user_input: str,
    session_goal_state: SessionGoalState | None,
    session_progress_ledger: Sequence[SessionProgressEntry],
) -> str | None:
    if not _looks_like_task_status_request(user_input):
        return None
    if session_goal_state is None:
        return _build_grounded_task_status_reply(
            session_goal_state=None,
            session_progress_ledger=session_progress_ledger,
        )
    if _task_status_reply_conflicts_with_persisted_state(
        reply=reply,
        session_goal_state=session_goal_state,
    ):
        return _build_grounded_task_status_reply(
            session_goal_state=session_goal_state,
            session_progress_ledger=session_progress_ledger,
        )
    return reply


def _replace_trailing_sentence(
    *,
    reply: str,
    pattern: re.Pattern[str],
    replacement: str,
) -> str:
    match = pattern.search(reply.strip())
    if match is None:
        return reply
    sentence = match.group("sentence")
    prefix = reply[: match.start("sentence")].rstrip()
    if prefix:
        return f"{prefix} {replacement}"
    del sentence
    return replacement


def _maybe_rewrite_unexecuted_next_step_claim(
    *,
    reply: str,
    tool_results: Sequence[ToolResult],
) -> str:
    successful_write = any(
        tool_result.tool_name == "write_text_file" and tool_result.success is True
        for tool_result in tool_results
    )
    successful_read = any(
        tool_result.tool_name == "read_text_file" and tool_result.success is True
        for tool_result in tool_results
    )
    successful_search = any(
        tool_result.tool_name in {"fast_web_search", "search_web"}
        and tool_result.success is True
        for tool_result in tool_results
    )

    updated_reply = reply
    if not successful_write:
        updated_reply = _replace_trailing_sentence(
            reply=updated_reply,
            pattern=_PROMISED_FILE_STEP_PATTERN,
            replacement="I haven't created or saved that file yet.",
        )
    if not successful_read:
        updated_reply = _replace_trailing_sentence(
            reply=updated_reply,
            pattern=_PROMISED_FILE_VERIFY_PATTERN,
            replacement="I haven't verified that file yet.",
        )
    if not successful_search:
        updated_reply = _replace_trailing_sentence(
            reply=updated_reply,
            pattern=_PROMISED_SEARCH_STEP_PATTERN,
            replacement="I haven't completed that search step yet.",
        )
    return updated_reply


def _apply_post_tool_reply_discipline(
    *,
    reply: str,
    user_input: str,
    tool_results: Sequence[ToolResult],
    turn_cancelled_reply: str = _DEFAULT_TURN_CANCELLED_REPLY,
    session_goal_state: SessionGoalState | None = None,
    session_progress_ledger: Sequence[SessionProgressEntry] = (),
) -> str:
    stripped_reply = reply.strip()
    grounded_task_status_reply = _maybe_ground_task_status_reply(
        reply=stripped_reply,
        user_input=user_input,
        session_goal_state=session_goal_state,
        session_progress_ledger=session_progress_ledger,
    )
    if grounded_task_status_reply is not None:
        return grounded_task_status_reply

    if not tool_results:
        if (
            stripped_reply
            and not _reply_acknowledges_limitations(stripped_reply)
            and _reply_claims_file_write_success(stripped_reply)
        ):
            return _build_missing_write_claim_reply()
        if (
            stripped_reply
            and not _reply_acknowledges_limitations(stripped_reply)
            and (
                _reply_claims_search_action(stripped_reply)
                or _reply_claims_complete_research(stripped_reply)
            )
        ):
            return _build_missing_search_claim_reply()
        return _maybe_rewrite_unexecuted_next_step_claim(
            reply=stripped_reply,
            tool_results=tool_results,
        )
    if stripped_reply == turn_cancelled_reply:
        return stripped_reply

    if all(tool_result.success is False for tool_result in tool_results):
        if not stripped_reply:
            return stripped_reply
        if _reply_acknowledges_limitations(stripped_reply):
            return stripped_reply
        return _build_all_failed_tool_reply(
            user_input=user_input,
            tool_results=tool_results,
        )

    successful_write = any(
        tool_result.tool_name == "write_text_file" and tool_result.success is True
        for tool_result in tool_results
    )
    successful_search_web = any(
        tool_result.tool_name == "search_web" and tool_result.success
        for tool_result in tool_results
    )
    successful_fast_web = [
        tool_result
        for tool_result in tool_results
        if tool_result.tool_name == "fast_web_search" and tool_result.success
    ]
    if successful_fast_web and not successful_search_web:
        reply_is_effectively_empty = _reply_is_effectively_empty(stripped_reply)
        reply_acknowledges_limitations = _reply_acknowledges_limitations(stripped_reply)
        reply_is_honest_write_only = (
            successful_write
            and _reply_claims_file_write_success(stripped_reply)
            and not _reply_claims_complete_research(stripped_reply)
            and not _reply_claims_task_completion(stripped_reply)
            and not _reply_claims_search_action(stripped_reply)
        )
        guarded_reply = _build_fast_grounding_guarded_reply(
            candidate_reply=stripped_reply,
            user_input=user_input,
            fast_results=successful_fast_web,
        )
        if guarded_reply is not None and not reply_is_honest_write_only:
            # Preserve already-minimal honest replies when they are shorter than
            # our fallback and clearly acknowledge the result limits.
            if reply_acknowledges_limitations and not reply_is_effectively_empty and (
                len(stripped_reply) <= len(guarded_reply)
                or _reply_sentence_count(stripped_reply) <= 2
            ):
                return stripped_reply
            return guarded_reply

    strong_grounded_search = any(
        tool_result.tool_name == "search_web"
        and tool_result.success is True
        and not _search_web_result_is_thin(tool_result)
        for tool_result in tool_results
    )
    latest_failed_non_write_tool_result = _find_latest_failed_non_write_tool_result(
        tool_results
    )
    latest_completion_blocking_web_tool_result = (
        _find_latest_completion_blocking_web_tool_result(tool_results)
    )
    if not _reply_acknowledges_limitations(stripped_reply):
        if not successful_write and _reply_claims_file_write_success(stripped_reply):
            return _build_missing_write_claim_reply()

        if (
            latest_failed_non_write_tool_result is not None
            and (
                _reply_claims_task_completion(stripped_reply)
                or _reply_claims_complete_research(stripped_reply)
            )
        ):
            blocker_detail = (
                latest_failed_non_write_tool_result.error
                or latest_failed_non_write_tool_result.output_text
                or "the earlier tool step failed"
            )
            return _build_incomplete_task_reply(
                blocker_detail=blocker_detail,
                wrote_file=successful_write,
            )

        if (
            latest_completion_blocking_web_tool_result is not None
            and (
                _reply_claims_task_completion(stripped_reply)
                or _reply_claims_complete_research(stripped_reply)
            )
        ):
            return _build_incomplete_task_reply(
                blocker_detail=_build_completion_blocking_web_detail(
                    latest_completion_blocking_web_tool_result
                ),
                wrote_file=successful_write,
            )

        if (
            (_reply_claims_search_action(stripped_reply) or _reply_claims_complete_research(stripped_reply))
            and not any(
                tool_result.tool_name in {"fast_web_search", "search_web"}
                and tool_result.success is True
                for tool_result in tool_results
            )
        ):
            return _build_missing_search_claim_reply()

        if _reply_claims_complete_research(stripped_reply) and not strong_grounded_search:
            return _build_partial_grounded_research_reply(
                wrote_file=successful_write,
            )

    return _maybe_rewrite_unexecuted_next_step_claim(
        reply=stripped_reply,
        tool_results=tool_results,
    )
