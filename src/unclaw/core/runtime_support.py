"""Runtime support helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from typing import TYPE_CHECKING, Any

from unclaw.core.command_handler import CommandHandler
from unclaw.core.orchestrator import ModelCallFailedError
from unclaw.core.reply_discipline import (
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


def _persist_session_goal_state_from_runtime_facts(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
    tool_results: Sequence[ToolResult],
    assistant_reply: str,
    turn_cancelled_reply: str,
) -> None:
    if not tool_results:
        return

    latest_tool_result = tool_results[-1]
    last_blocker: str | None = None
    status = "active"
    if assistant_reply.strip() == turn_cancelled_reply.strip():
        status = "blocked"
        last_blocker = turn_cancelled_reply
    elif latest_tool_result.success is False:
        status = "blocked"
        last_blocker = latest_tool_result.error or latest_tool_result.output_text

    session_manager.persist_session_goal_state(
        session_id=session_id,
        goal=user_input,
        status=status,
        current_step=latest_tool_result.tool_name,
        last_blocker=last_blocker,
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
