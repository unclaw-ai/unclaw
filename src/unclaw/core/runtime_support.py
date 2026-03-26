"""Runtime support helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from typing import TYPE_CHECKING, Any

from unclaw.core.command_handler import CommandHandler
from unclaw.core.orchestrator import ModelCallFailedError
from unclaw.core.reply_discipline import (
    _fast_web_search_result_is_mismatch,
    _fast_web_search_result_is_thin,
    _search_web_result_is_thin,
    _tool_result_timed_out,
)
from unclaw.core.session_manager import SessionManager
from unclaw.constants import RUNTIME_ERROR_REPLY
from unclaw.memory.protocols import SessionMemoryContextProvider
from unclaw.tools.contracts import ToolDefinition, ToolResult
from unclaw.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from unclaw.core.routing import _EntityAnchor

_ENTITY_RECENTERING_NOTE_PREFIX = "Entity recentering hint:"
_POST_TOOL_GROUNDING_NOTE_PREFIX = "Post-tool grounding note:"
_SESSION_GOAL_STATE_NOTE_PREFIX = "Session goal state:"
_SESSION_PROGRESS_LEDGER_NOTE_PREFIX = "Session progress ledger:"
_SESSION_TASK_CONTINUITY_NOTE_PREFIX = "Session task continuity:"
_WRITE_SUCCESS_TOOL_NAME = "write_text_file"
_BLOCKED_GOAL_CONTINUATION_MAX_TOKENS = 2
_BLOCKED_GOAL_CONTINUATION_MAX_TOKEN_ALNUM_CHARS = 3
_BLOCKED_GOAL_CONTINUATION_ALLOWED_PUNCTUATION = ".,!?;:'\"-()[]{}"


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
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> bool:
    if session_manager.get_session_goal_state(session_id) is None:
        return False
    return _turn_has_task_like_runtime_facts(
        tool_results=tool_results,
        assistant_reply=assistant_reply,
        turn_cancelled_reply=turn_cancelled_reply,
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


def _persist_session_progress_ledger_from_runtime_facts(
    *,
    session_manager: SessionManager,
    session_id: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> None:
    if not _turn_qualifies_for_session_goal_state_persistence(
        session_manager=session_manager,
        session_id=session_id,
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
