"""Typed built-in capability fragments for runtime prompt composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from types import MappingProxyType
from typing import Protocol

from unclaw.tools.file_tools import (
    COPY_FILE_DEFINITION,
    DELETE_FILE_DEFINITION,
    LIST_DIRECTORY_DEFINITION,
    MOVE_FILE_DEFINITION,
    READ_TEXT_FILE_DEFINITION,
    RENAME_FILE_DEFINITION,
    WRITE_TEXT_FILE_DEFINITION,
)
from unclaw.tools.long_term_memory_tools import (
    FORGET_LONG_TERM_MEMORY_DEFINITION,
    LIST_LONG_TERM_MEMORY_DEFINITION,
    REMEMBER_LONG_TERM_MEMORY_DEFINITION,
    SEARCH_LONG_TERM_MEMORY_DEFINITION,
)
from unclaw.tools.session_tools import INSPECT_SESSION_HISTORY_DEFINITION
from unclaw.tools.system_tools import SYSTEM_INFO_DEFINITION
from unclaw.tools.terminal_tools import RUN_TERMINAL_COMMAND_DEFINITION
from unclaw.tools.web_tools import (
    FAST_WEB_SEARCH_DEFINITION,
    FETCH_URL_TEXT_DEFINITION,
    SEARCH_WEB_DEFINITION,
)

_MODULE_REFERENCE = "unclaw.core.capability_fragments"
_LONG_TERM_MEMORY_TOOL_NAMES = (
    REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
    SEARCH_LONG_TERM_MEMORY_DEFINITION.name,
    LIST_LONG_TERM_MEMORY_DEFINITION.name,
    FORGET_LONG_TERM_MEMORY_DEFINITION.name,
)
_LOCAL_FILE_MUTATION_TOOL_NAMES = (
    DELETE_FILE_DEFINITION.name,
    MOVE_FILE_DEFINITION.name,
    RENAME_FILE_DEFINITION.name,
    COPY_FILE_DEFINITION.name,
)


class CapabilitySummaryLike(Protocol):
    """Structural typing contract for runtime capability fragment resolution."""

    available_builtin_tool_names: tuple[str, ...]
    local_file_read_available: bool
    local_directory_listing_available: bool
    url_fetch_available: bool
    web_search_available: bool
    system_info_available: bool
    shell_command_execution_available: bool
    memory_summary_available: bool
    model_can_call_tools: bool
    local_file_write_available: bool
    session_history_recall_available: bool
    long_term_memory_available: bool
    fast_web_search_available: bool


class CapabilityFragmentKind(StrEnum):
    """Semantic role of one built-in capability fragment."""

    AVAILABLE_TOOL = "available_tool"
    AVAILABLE_RUNTIME = "available_runtime"
    UNAVAILABLE_CAPABILITY = "unavailable_capability"
    TOOL_MODE = "tool_mode"
    GUIDANCE = "guidance"


class CapabilityPromptSourceKind(StrEnum):
    """Where the current prompt text for a fragment is authored."""

    FUNCTION = "function"
    INLINE = "inline"


class CapabilityToolModeRelevance(StrEnum):
    """Whether a fragment applies in shared, native/model-callable, or slash-only turns."""

    SHARED = "shared"
    MODEL_CALLABLE_ONLY = "model_callable_only"
    USER_INITIATED_ONLY = "user_initiated_only"


class CapabilitySummaryFlag(StrEnum):
    """Summary flags that built-in capability fragments can depend on."""

    LOCAL_FILE_READ_AVAILABLE = "local_file_read_available"
    LOCAL_DIRECTORY_LISTING_AVAILABLE = "local_directory_listing_available"
    URL_FETCH_AVAILABLE = "url_fetch_available"
    WEB_SEARCH_AVAILABLE = "web_search_available"
    SYSTEM_INFO_AVAILABLE = "system_info_available"
    SHELL_COMMAND_EXECUTION_AVAILABLE = "shell_command_execution_available"
    MEMORY_SUMMARY_AVAILABLE = "memory_summary_available"
    MODEL_CAN_CALL_TOOLS = "model_can_call_tools"
    LOCAL_FILE_WRITE_AVAILABLE = "local_file_write_available"
    SESSION_HISTORY_RECALL_AVAILABLE = "session_history_recall_available"
    LONG_TERM_MEMORY_AVAILABLE = "long_term_memory_available"
    FAST_WEB_SEARCH_AVAILABLE = "fast_web_search_available"


@dataclass(frozen=True, slots=True)
class CapabilityPromptSource:
    """Current prompt-text source location for one built-in fragment."""

    kind: CapabilityPromptSourceKind
    reference: str


CapabilityFragmentPromptRenderer = Callable[
    [CapabilitySummaryLike],
    tuple[str, ...],
]


@dataclass(frozen=True, slots=True)
class CapabilityFragmentPrompt:
    """Live prompt content for one built-in capability fragment."""

    source: CapabilityPromptSource
    lines: tuple[str, ...] = ()
    renderer: CapabilityFragmentPromptRenderer | None = None

    def __post_init__(self) -> None:
        has_static_lines = bool(self.lines)
        has_renderer = self.renderer is not None
        if has_static_lines == has_renderer:
            raise ValueError(
                "CapabilityFragmentPrompt must define exactly one of lines or renderer."
            )

    def render_lines(self, summary: CapabilitySummaryLike) -> tuple[str, ...]:
        if self.renderer is not None:
            return self.renderer(summary)
        return self.lines


@dataclass(frozen=True, slots=True)
class CapabilityAvailability:
    """Typed availability semantics for a capability fragment."""

    required_summary_flags: tuple[CapabilitySummaryFlag, ...] = ()
    forbidden_summary_flags: tuple[CapabilitySummaryFlag, ...] = ()
    required_builtin_tool_names: tuple[str, ...] = ()
    forbidden_builtin_tool_names: tuple[str, ...] = ()
    present_any_builtin_tool_names: tuple[str, ...] = ()
    missing_any_builtin_tool_names: tuple[str, ...] = ()

    def matches(self, summary: CapabilitySummaryLike) -> bool:
        available_tool_names = frozenset(summary.available_builtin_tool_names)

        if any(
            getattr(summary, summary_flag.value) is not True
            for summary_flag in self.required_summary_flags
        ):
            return False
        if any(
            getattr(summary, summary_flag.value) is True
            for summary_flag in self.forbidden_summary_flags
        ):
            return False
        if any(
            tool_name not in available_tool_names
            for tool_name in self.required_builtin_tool_names
        ):
            return False
        if any(
            tool_name in available_tool_names
            for tool_name in self.forbidden_builtin_tool_names
        ):
            return False
        if self.present_any_builtin_tool_names and not any(
            tool_name in available_tool_names
            for tool_name in self.present_any_builtin_tool_names
        ):
            return False
        if self.missing_any_builtin_tool_names and not any(
            tool_name not in available_tool_names
            for tool_name in self.missing_any_builtin_tool_names
        ):
            return False
        return True


@dataclass(frozen=True, slots=True)
class CapabilityFragment:
    """Typed metadata for one durable built-in capability fragment."""

    fragment_id: str
    capability_id: str
    name: str
    kind: CapabilityFragmentKind
    prompt: CapabilityFragmentPrompt
    availability: CapabilityAvailability = field(default_factory=CapabilityAvailability)
    tool_mode_relevance: CapabilityToolModeRelevance = (
        CapabilityToolModeRelevance.SHARED
    )
    related_builtin_tool_names: tuple[str, ...] = ()
    related_summary_flags: tuple[CapabilitySummaryFlag, ...] = ()
    description: str | None = None

    @property
    def prompt_source(self) -> CapabilityPromptSource:
        return self.prompt.source

    def matches(self, summary: CapabilitySummaryLike) -> bool:
        return self.availability.matches(summary)

    def render_lines(self, summary: CapabilitySummaryLike) -> tuple[str, ...]:
        return self.prompt.render_lines(summary)


@dataclass(frozen=True, slots=True)
class RenderedCapabilityFragment:
    """Resolved built-in capability fragment plus its live rendered lines."""

    fragment: CapabilityFragment
    rendered_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuiltinCapabilityFragmentRegistry:
    """Deterministic registry of built-in capability fragments."""

    fragments: tuple[CapabilityFragment, ...]
    _fragments_by_id: MappingProxyType = field(init=False, repr=False)
    _fragments_by_capability_id: MappingProxyType = field(init=False, repr=False)
    _fragments_by_tool_name: MappingProxyType = field(init=False, repr=False)
    _fragments_by_summary_flag: MappingProxyType = field(init=False, repr=False)
    _capability_ids: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        fragments_by_id: dict[str, CapabilityFragment] = {}
        fragments_by_capability_id: dict[str, list[CapabilityFragment]] = {}
        fragments_by_tool_name: dict[str, list[CapabilityFragment]] = {}
        fragments_by_summary_flag: dict[CapabilitySummaryFlag, list[CapabilityFragment]] = {}
        capability_ids: list[str] = []

        for fragment in self.fragments:
            if fragment.fragment_id in fragments_by_id:
                raise ValueError(
                    f"Duplicate built-in capability fragment id: {fragment.fragment_id}"
                )

            fragments_by_id[fragment.fragment_id] = fragment

            if fragment.capability_id not in fragments_by_capability_id:
                capability_ids.append(fragment.capability_id)
                fragments_by_capability_id[fragment.capability_id] = []
            fragments_by_capability_id[fragment.capability_id].append(fragment)

            for tool_name in fragment.related_builtin_tool_names:
                fragments_by_tool_name.setdefault(tool_name, []).append(fragment)

            for summary_flag in fragment.related_summary_flags:
                fragments_by_summary_flag.setdefault(summary_flag, []).append(fragment)

        object.__setattr__(
            self,
            "_fragments_by_id",
            MappingProxyType(fragments_by_id),
        )
        object.__setattr__(
            self,
            "_fragments_by_capability_id",
            MappingProxyType(
                {
                    capability_id: tuple(capability_fragments)
                    for capability_id, capability_fragments in fragments_by_capability_id.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "_fragments_by_tool_name",
            MappingProxyType(
                {
                    tool_name: tuple(tool_fragments)
                    for tool_name, tool_fragments in fragments_by_tool_name.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "_fragments_by_summary_flag",
            MappingProxyType(
                {
                    summary_flag: tuple(summary_fragments)
                    for summary_flag, summary_fragments in fragments_by_summary_flag.items()
                }
            ),
        )
        object.__setattr__(self, "_capability_ids", tuple(capability_ids))

    def list_fragments(self) -> tuple[CapabilityFragment, ...]:
        return self.fragments

    def list_capability_ids(self) -> tuple[str, ...]:
        return self._capability_ids

    def get_fragment(self, fragment_id: str) -> CapabilityFragment:
        try:
            return self._fragments_by_id[fragment_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown built-in capability fragment id: {fragment_id}"
            ) from exc

    def get_fragments_for_capability(self, capability_id: str) -> tuple[CapabilityFragment, ...]:
        return self._fragments_by_capability_id.get(capability_id, ())

    def get_fragments_for_tool_name(self, tool_name: str) -> tuple[CapabilityFragment, ...]:
        return self._fragments_by_tool_name.get(tool_name, ())

    def get_fragments_for_summary_flag(
        self,
        summary_flag: CapabilitySummaryFlag,
    ) -> tuple[CapabilityFragment, ...]:
        return self._fragments_by_summary_flag.get(summary_flag, ())

    def resolve_fragments(
        self,
        summary: CapabilitySummaryLike,
    ) -> tuple[CapabilityFragment, ...]:
        return tuple(
            fragment for fragment in self.fragments if fragment.matches(summary)
        )


def _inline_source(fragment_id: str) -> CapabilityPromptSource:
    return CapabilityPromptSource(
        kind=CapabilityPromptSourceKind.INLINE,
        reference=f"{_MODULE_REFERENCE}:{fragment_id}",
    )


def _function_source(function_name: str) -> CapabilityPromptSource:
    return CapabilityPromptSource(
        kind=CapabilityPromptSourceKind.FUNCTION,
        reference=f"{_MODULE_REFERENCE}.{function_name}",
    )


def _static_prompt(
    fragment_id: str,
    *lines: str,
) -> CapabilityFragmentPrompt:
    return CapabilityFragmentPrompt(
        source=_inline_source(fragment_id),
        lines=tuple(lines),
    )


def _dynamic_prompt(
    function_name: str,
    renderer: CapabilityFragmentPromptRenderer,
) -> CapabilityFragmentPrompt:
    return CapabilityFragmentPrompt(
        source=_function_source(function_name),
        renderer=renderer,
    )


def _missing_local_file_actions(summary: CapabilitySummaryLike) -> tuple[str, ...]:
    available_tool_names = frozenset(summary.available_builtin_tool_names)
    actions: list[str] = []

    if DELETE_FILE_DEFINITION.name not in available_tool_names:
        actions.append("delete")
    if MOVE_FILE_DEFINITION.name not in available_tool_names:
        actions.append("move")
    if RENAME_FILE_DEFINITION.name not in available_tool_names:
        actions.append("rename")
    if COPY_FILE_DEFINITION.name not in available_tool_names:
        actions.append("copy")

    return tuple(actions)


def _format_local_file_action_list(actions: tuple[str, ...]) -> str | None:
    if not actions:
        return None
    if len(actions) == 1:
        return actions[0]
    if len(actions) == 2:
        return f"{actions[0]} or {actions[1]}"
    return f"{', '.join(actions[:-1])}, or {actions[-1]}"


def _render_unavailable_local_file_actions_summary(
    summary: CapabilitySummaryLike,
) -> tuple[str, ...]:
    actions = _format_local_file_action_list(_missing_local_file_actions(summary))
    if actions is None:
        return ()
    return (
        f"{actions.capitalize()} local files or directories (no such tool is registered).",
    )


def _render_unavailable_local_file_actions_warning(
    summary: CapabilitySummaryLike,
) -> tuple[str, ...]:
    actions = _format_local_file_action_list(_missing_local_file_actions(summary))
    if actions is None:
        return ()
    return (
        f"If the user asks to {actions} a file or directory, say clearly this "
        "capability does not exist. Do not suggest it might work after confirmation.",
    )


_BUILTIN_CAPABILITY_PROMPTS = MappingProxyType(
    {
        "available.local_file_read": _static_prompt(
            "available.local_file_read",
            "/read <path>: read local .txt, .md, .json, or .csv files inside "
            "allowed roots. Other formats (pdf, docx, xlsx, etc.) are not "
            "supported in V1.",
        ),
        "available.local_directory_listing": _static_prompt(
            "available.local_directory_listing",
            "/ls [path]: list local directories inside allowed roots.",
        ),
        "available.url_fetch": _static_prompt(
            "available.url_fetch",
            "/fetch <url>: fetch one public URL and extract text.",
        ),
        "available.web_search": _static_prompt(
            "available.web_search",
            "/search <query>: search the public web, read a few relevant pages, "
            "and answer naturally from grounded web context with compact sources.",
        ),
        "available.system_info": _static_prompt(
            "available.system_info",
            "system_info: return current local machine and runtime facts, "
            "including local date/time, day, OS, CPU core count, total RAM, "
            "hostname, and locale.",
        ),
        "available.shell_command_execution": _static_prompt(
            "available.shell_command_execution",
            "run_terminal_command <command>: execute one bounded local shell command "
            "in a validated working directory. Use it for local inspections, scripts, "
            "and stepwise automation. Captures stdout/stderr, enforces a timeout, "
            "and does not provide an interactive terminal.",
        ),
        "available.local_file_write": _static_prompt(
            "available.local_file_write",
            "write_text_file <path>: write a local file. "
            "Relative paths are created inside data/files/ by default. "
            "Use collision_policy to control collision behaviour: "
            "'fail' (default) — refuse if the file already exists; "
            "'version' — write to a new timestamped sibling path "
            "(same directory, same basename, same extension, e.g. note_20260322_185430.txt); "
            "'overwrite' — replace the existing file (only allowed in dev mode). "
            "When a write is refused, the tool payload contains suggested_version_path: "
            "a ready-to-use sibling path derived from the same basename. "
            "If the user wants a safe alternative, retry with that suggested path and "
            "collision_policy='version' — do not invent a different filename. "
            "Only use 'overwrite' when the user explicitly asked to replace the file "
            "and dev mode is enabled — do not infer overwrite intent from context alone.",
        ),
        "available.local_file_delete": _static_prompt(
            "available.local_file_delete",
            "delete_file <path>: delete one local file. "
            "Relative paths resolve inside data/files/ by default. "
            "Requires confirm=true and only deletes files, not directories.",
        ),
        "available.local_file_move": _static_prompt(
            "available.local_file_move",
            "move_file <source_path> <destination_path>: move one local file. "
            "Relative paths resolve inside data/files/ by default. "
            "Default is overwrite=false — fails if the destination already exists. "
            "Only moves files, not directories.",
        ),
        "available.local_file_rename": _static_prompt(
            "available.local_file_rename",
            "rename_file <source_path> <destination_path>: rename one local file. "
            "Relative paths resolve inside data/files/ by default. "
            "Default is overwrite=false — fails if the destination already exists. "
            "Only renames files, not directories.",
        ),
        "available.local_file_copy": _static_prompt(
            "available.local_file_copy",
            "copy_file <source_path> <destination_path>: copy one local file. "
            "Relative paths resolve inside data/files/ by default. "
            "Default is overwrite=false — fails if the destination already exists. "
            "Only copies files, not directories.",
        ),
        "available.session_history_recall": _static_prompt(
            "available.session_history_recall",
            "inspect_session_history: return an exact, deterministic list of "
            "persisted messages for the current session. "
            "Supports filter_role (user/assistant/tool), nth (1-indexed lookup), "
            "and limit. Use this tool for exact questions about prior prompts, "
            "their order, or message counts.",
        ),
        "available.long_term_memory": _static_prompt(
            "available.long_term_memory",
            "remember_long_term_memory / search_long_term_memory / "
            "list_long_term_memory / forget_long_term_memory: "
            "store, search, list, or delete persistent cross-session facts "
            "and preferences. "
            "Not injected automatically — call tools explicitly. "
            "Use search_long_term_memory for targeted recall and "
            "list_long_term_memory for broad recall. "
            "Not for session message history — use inspect_session_history for that.",
        ),
        "available.session_memory_summary": _static_prompt(
            "available.session_memory_summary",
            "Session memory and summary access.",
        ),
        "available.fast_web_search": _static_prompt(
            "available.fast_web_search",
            "fast_web_search <query>: quick lightweight web grounding probe. "
            "Use before full search_web when unsure about a person, place, "
            "organization, or product name. Use the user's literal wording first. "
            "Returns a tiny grounding note, "
            "not a full answer.",
        ),
        "unavailable.local_file_read": _static_prompt(
            "unavailable.local_file_read",
            "Local file read via /read <path>.",
        ),
        "unavailable.local_directory_listing": _static_prompt(
            "unavailable.local_directory_listing",
            "Local directory listing via /ls [path].",
        ),
        "unavailable.url_fetch": _static_prompt(
            "unavailable.url_fetch",
            "Direct URL fetch via /fetch <url>.",
        ),
        "unavailable.web_search": _static_prompt(
            "unavailable.web_search",
            "Web search via /search <query>.",
        ),
        "unavailable.system_info": _static_prompt(
            "unavailable.system_info",
            "Local machine and runtime information via system_info.",
        ),
        "unavailable.shell_command_execution": _static_prompt(
            "unavailable.shell_command_execution",
            "Local shell or terminal command execution via run_terminal_command.",
        ),
        "unavailable.local_file_write": _static_prompt(
            "unavailable.local_file_write",
            "Local file write via write_text_file.",
        ),
        "unavailable.session_memory_summary": _static_prompt(
            "unavailable.session_memory_summary",
            "Session memory and summary access.",
        ),
        "unavailable.local_file_actions_summary": _dynamic_prompt(
            "_render_unavailable_local_file_actions_summary",
            _render_unavailable_local_file_actions_summary,
        ),
        "unavailable.any_non_listed_capability": _static_prompt(
            "unavailable.any_non_listed_capability",
            "Any capability that is not listed as available above.",
        ),
        "tool_mode.model_callable": _static_prompt(
            "tool_mode.model_callable",
            "Tool invocation mode: model-callable (you may call tools directly this turn).",
        ),
        "tool_mode.user_initiated": _static_prompt(
            "tool_mode.user_initiated",
            "Tool invocation mode: user-initiated slash commands only "
            "(you cannot call tools directly this turn).",
        ),
        "guidance.shared_core_rules": _static_prompt(
            "guidance.shared_core_rules",
            "Do not claim you have no tool access when one or more built-in tools are available.",
            "You may say you can use Unclaw built-in tools that are listed as available.",
        ),
        "guidance.model_callable.core_rules": _static_prompt(
            "guidance.model_callable.core_rules",
            "Use only the listed built-in tools and base the final answer on their results.",
            "If the answer is complete and safe from the conversation alone, reply directly without tools.",
            "For compound requests, decompose into the minimum useful sub-tasks.",
            "Use tools only where they add needed runtime or external information.",
            "Preserve the user's requested order when practical.",
            "Do not call tools for parts already answerable from the conversation.",
            "Combine tool results into one coherent final answer.",
            "When a listed tool can directly verify the user's request, prefer "
            "the tool over guessing, reinterpretation, or premature clarification.",
            "If the first tool result is useful but incomplete and one obvious "
            "next tool step remains, take that step before giving a vague answer.",
            "Do not ask for clarification when the current request already gives "
            "enough information for the first obvious tool call.",
            "Only claim a file was written, created, or modified if write_text_file "
            "output is present in this conversation. If no such tool ran, say the "
            "action has not happened yet.",
            "If the user explicitly asked to write, save, or create a file and the "
            "task involves research or content generation, the task is not complete "
            "until write_text_file has been called and its output confirms success. "
            "Do not stop after research alone when a file output was requested.",
            "If a person, organization, or entity name in the user's request is "
            "ambiguous or could match multiple real entities, do not silently "
            "substitute a different entity. Ask for clarification or explicitly "
            "state which entity you identified and why before proceeding.",
            "When calling fast_web_search or search_web, pass the entity name "
            "exactly as the user wrote it. Do not 'correct', normalize, or "
            "substitute the user-supplied name with a more famous alternative "
            "before evidence from tool output confirms a correction is warranted.",
        ),
        "guidance.user_initiated.core_rules": _static_prompt(
            "guidance.user_initiated.core_rules",
            "Tools listed above are available only when the user types the slash "
            "command. You cannot invoke them yourself in this turn.",
            "Do not say \"let me search\" or \"I will look that up\" as if you can "
            "perform the action right now.",
            "You cannot write, modify, or create any file in this turn. If "
            "the user asks you to do so, say the action was not performed — you have "
            "not written or changed anything.",
        ),
        "guidance.model_callable.local_choice.full": _static_prompt(
            "guidance.model_callable.local_choice.full",
            "Use list_directory for local directories and read_text_file for supported local text files.",
            "For explicit local inspection requests with a concrete path or scope, "
            "try the obvious local tool before asking for clarification.",
        ),
        "guidance.model_callable.local_choice.list_only": _static_prompt(
            "guidance.model_callable.local_choice.list_only",
            "Use list_directory for local directories.",
            "For explicit local directory inspection requests with a concrete path "
            "or scope, call list_directory before asking for clarification.",
        ),
        "guidance.model_callable.local_choice.read_only": _static_prompt(
            "guidance.model_callable.local_choice.read_only",
            "Use read_text_file for supported local text files.",
            "For explicit local file inspection requests with a concrete path, "
            "call read_text_file before asking for clarification.",
        ),
        "guidance.model_callable.web_choice.full": _static_prompt(
            "guidance.model_callable.web_choice.full",
            "Use search_web for up-to-date external information. Use fetch_url_text "
            "for a specific public URL.",
            "For live public facts (events, people, news, weather, scores), prefer "
            "search_web over run_terminal_command with curl/wget.",
            "When the user explicitly asks you to look something up online, do the "
            "lookup before answering from weak memory or asking unnecessary clarifications.",
        ),
        "guidance.model_callable.web_choice.search_only": _static_prompt(
            "guidance.model_callable.web_choice.search_only",
            "Use search_web for up-to-date external information.",
            "For live public facts (events, people, news, weather, scores), prefer "
            "search_web over run_terminal_command with curl/wget.",
            "When the user explicitly asks you to look something up online, do the "
            "lookup before answering from weak memory or asking unnecessary clarifications.",
        ),
        "guidance.model_callable.web_choice.fetch_only": _static_prompt(
            "guidance.model_callable.web_choice.fetch_only",
            "Use fetch_url_text for a specific public URL.",
        ),
        "guidance.model_callable.fast_web_grounding": _static_prompt(
            "guidance.model_callable.fast_web_grounding",
            "For identity or biography requests such as 'who is', 'qui est', "
            "'c'est qui', or 'bio de', use fast_web_search first with the exact "
            "entity wording from the user before deciding whether search_web is needed.",
            "When unsure about a person, place, organization, or product name, "
            "use fast_web_search first with the exact name as written by the user. "
            "This prevents silently substituting an unknown entity with a more "
            "famous wrong one.",
            "If you are tempted to correct, normalize, reinterpret, infer the "
            "domain, or answer from weak memory, do a literal fast_web_search "
            "first instead of guessing.",
            "For recent or current topics, keep the first lookup literal as written "
            "by the user before adding any cautious news-oriented expansion.",
            "If fast_web_search results are clearly for a different entity, say "
            "so explicitly instead of presenting the wrong entity's information.",
            "After the user corrects you, reuse the corrected entity or context "
            "exactly on the next search step and do not drift back to the earlier mistake.",
            "A fast_web_search note is only a quick grounding probe. Do not "
            "expand it into a full biography or background profile unless a "
            "later search_web step confirms those details.",
            "fast_web_search is very fast and cheap — use it liberally for "
            "entity resolution before committing to a full research flow.",
        ),
        "guidance.model_callable.session_history": _static_prompt(
            "guidance.model_callable.session_history",
            "Use inspect_session_history to answer exact questions about prior "
            "prompts, message order, or counts. Do not guess from memory — call "
            "the tool for exact recall. The tool supports filter_role "
            "(user/assistant/tool), nth (1-indexed lookup), and limit. Reply to "
            "the user in their language after reading the tool result.",
        ),
        "guidance.model_callable.system_info": _static_prompt(
            "guidance.model_callable.system_info",
            "Use system_info for current local machine facts and runtime facts such "
            "as local date/time, day, OS, CPU, RAM, hostname, and locale.",
            "Do not claim you cannot know the current date, time, or OS when "
            "system_info is available — call the tool to get accurate local facts.",
            "For obvious local date, time, day, OS, hostname, or machine-spec "
            "questions, call system_info before answering or clarifying.",
        ),
        "guidance.model_callable.shell_command_execution": _static_prompt(
            "guidance.model_callable.shell_command_execution",
            "Use run_terminal_command for local shell inspections, scripts, and "
            "stepwise automation when direct terminal execution is genuinely needed.",
            "Never invent terminal output, exit codes, or side effects. If the "
            "command did not run successfully, say so clearly.",
            "After execution, summarize the important result instead of dumping raw "
            "stdout/stderr unless the user asks for the exact output.",
        ),
        "guidance.model_callable.long_term_memory": _static_prompt(
            "guidance.model_callable.long_term_memory",
            "Long-term memory tools are for previously stored persistent "
            "cross-session facts or preferences. They are not injected automatically.",
            "Use them for stable personal facts, preferences, and hardware or setup information.",
            "Use search_long_term_memory for targeted recall of a stored fact.",
            "For search_long_term_memory, pass concise semantic query terms for "
            "the topic, even if the user asked in another language.",
            "Use list_long_term_memory for broad recall of stored memories.",
            "Use remember_long_term_memory only when the user explicitly wants a "
            "fact saved for later or explicitly corrects a stored fact.",
            "Use forget_long_term_memory only when the user explicitly wants a "
            "stored memory removed.",
            "Use system_info for current local machine facts, not long-term memory.",
            "Do not use long-term memory tools for current session message history "
            "or message order; use inspect_session_history for that.",
            "Do not auto-store facts that were not explicitly requested for storage.",
        ),
        "guidance.shared_tool_output_honesty": _static_prompt(
            "guidance.shared_tool_output_honesty",
            "Do not claim you already searched, fetched, read, ran a terminal "
            "command, wrote, created, modified, or deleted anything unless actual "
            "tool output for that action is present in this conversation.",
            "If tool output is already present in the conversation, treat it as "
            "retrieved context that you may summarize, compare, extract, or "
            "analyze. Do not say you cannot access it, and do not ask the user "
            "to paste it again.",
            "After a tool call, if the result is sparse, partial, mixed, or "
            "failed, say that and give only the facts it directly supports.",
            "If a capability is unavailable, say so clearly instead of implying it exists.",
        ),
        "guidance.unavailable_local_file_actions_warning": _dynamic_prompt(
            "_render_unavailable_local_file_actions_warning",
            _render_unavailable_local_file_actions_warning,
        ),
    }
)


def _fragment(
    *,
    fragment_id: str,
    capability_id: str,
    name: str,
    kind: CapabilityFragmentKind,
    prompt_symbol: str,
    prompt_detail: str,
    availability: CapabilityAvailability | None = None,
    tool_mode_relevance: CapabilityToolModeRelevance = (
        CapabilityToolModeRelevance.SHARED
    ),
    related_builtin_tool_names: tuple[str, ...] = (),
    related_summary_flags: tuple[CapabilitySummaryFlag, ...] = (),
    description: str | None = None,
) -> CapabilityFragment:
    del prompt_symbol, prompt_detail
    return CapabilityFragment(
        fragment_id=fragment_id,
        capability_id=capability_id,
        name=name,
        kind=kind,
        prompt=_BUILTIN_CAPABILITY_PROMPTS[fragment_id],
        availability=availability or CapabilityAvailability(),
        tool_mode_relevance=tool_mode_relevance,
        related_builtin_tool_names=related_builtin_tool_names,
        related_summary_flags=related_summary_flags,
        description=description,
    )


_BUILTIN_CAPABILITY_FRAGMENTS: tuple[CapabilityFragment, ...] = (
    _fragment(
        fragment_id="available.local_file_read",
        capability_id="local_file_read",
        name="Available local file read tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="local_file_read",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.LOCAL_FILE_READ_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=(READ_TEXT_FILE_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.LOCAL_FILE_READ_AVAILABLE,),
        description="Expose /read when read_text_file is registered.",
    ),
    _fragment(
        fragment_id="available.local_directory_listing",
        capability_id="local_directory_listing",
        name="Available local directory listing tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="local_directory_listing",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.LOCAL_DIRECTORY_LISTING_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=(LIST_DIRECTORY_DEFINITION.name,),
        related_summary_flags=(
            CapabilitySummaryFlag.LOCAL_DIRECTORY_LISTING_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="available.url_fetch",
        capability_id="url_fetch",
        name="Available URL fetch tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="url_fetch",
        availability=CapabilityAvailability(
            required_summary_flags=(CapabilitySummaryFlag.URL_FETCH_AVAILABLE,),
        ),
        related_builtin_tool_names=(FETCH_URL_TEXT_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.URL_FETCH_AVAILABLE,),
    ),
    _fragment(
        fragment_id="available.web_search",
        capability_id="web_search",
        name="Available web search tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="web_search",
        availability=CapabilityAvailability(
            required_summary_flags=(CapabilitySummaryFlag.WEB_SEARCH_AVAILABLE,),
        ),
        related_builtin_tool_names=(SEARCH_WEB_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.WEB_SEARCH_AVAILABLE,),
    ),
    _fragment(
        fragment_id="available.system_info",
        capability_id="system_info",
        name="Available system_info tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="system_info",
        availability=CapabilityAvailability(
            required_summary_flags=(CapabilitySummaryFlag.SYSTEM_INFO_AVAILABLE,),
        ),
        related_builtin_tool_names=(SYSTEM_INFO_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.SYSTEM_INFO_AVAILABLE,),
    ),
    _fragment(
        fragment_id="available.shell_command_execution",
        capability_id="shell_command_execution",
        name="Available run_terminal_command tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="shell_command_execution",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.SHELL_COMMAND_EXECUTION_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=(RUN_TERMINAL_COMMAND_DEFINITION.name,),
        related_summary_flags=(
            CapabilitySummaryFlag.SHELL_COMMAND_EXECUTION_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="available.local_file_write",
        capability_id="local_file_write",
        name="Available local file write tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="local_file_write",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.LOCAL_FILE_WRITE_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=(WRITE_TEXT_FILE_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.LOCAL_FILE_WRITE_AVAILABLE,),
    ),
    _fragment(
        fragment_id="available.local_file_delete",
        capability_id="local_file_delete",
        name="Available delete_file tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="local_file_delete",
        availability=CapabilityAvailability(
            required_builtin_tool_names=(DELETE_FILE_DEFINITION.name,),
        ),
        related_builtin_tool_names=(DELETE_FILE_DEFINITION.name,),
    ),
    _fragment(
        fragment_id="available.local_file_move",
        capability_id="local_file_move",
        name="Available move_file tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="local_file_move",
        availability=CapabilityAvailability(
            required_builtin_tool_names=(MOVE_FILE_DEFINITION.name,),
        ),
        related_builtin_tool_names=(MOVE_FILE_DEFINITION.name,),
    ),
    _fragment(
        fragment_id="available.local_file_rename",
        capability_id="local_file_rename",
        name="Available rename_file tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="local_file_rename",
        availability=CapabilityAvailability(
            required_builtin_tool_names=(RENAME_FILE_DEFINITION.name,),
        ),
        related_builtin_tool_names=(RENAME_FILE_DEFINITION.name,),
    ),
    _fragment(
        fragment_id="available.local_file_copy",
        capability_id="local_file_copy",
        name="Available copy_file tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="local_file_copy",
        availability=CapabilityAvailability(
            required_builtin_tool_names=(COPY_FILE_DEFINITION.name,),
        ),
        related_builtin_tool_names=(COPY_FILE_DEFINITION.name,),
    ),
    _fragment(
        fragment_id="available.session_history_recall",
        capability_id="session_history_recall",
        name="Available inspect_session_history tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="session_history_recall",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.SESSION_HISTORY_RECALL_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=(INSPECT_SESSION_HISTORY_DEFINITION.name,),
        related_summary_flags=(
            CapabilitySummaryFlag.SESSION_HISTORY_RECALL_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="available.long_term_memory",
        capability_id="long_term_memory",
        name="Available long-term memory tool family line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="long_term_memory",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.LONG_TERM_MEMORY_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=_LONG_TERM_MEMORY_TOOL_NAMES,
        related_summary_flags=(CapabilitySummaryFlag.LONG_TERM_MEMORY_AVAILABLE,),
    ),
    _fragment(
        fragment_id="available.fast_web_search",
        capability_id="fast_web_search",
        name="Available fast_web_search tool line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="fast_web_search",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.FAST_WEB_SEARCH_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=(FAST_WEB_SEARCH_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.FAST_WEB_SEARCH_AVAILABLE,),
        description="Expose fast_web_search when registered.",
    ),
    _fragment(
        fragment_id="available.session_memory_summary",
        capability_id="session_memory_summary",
        name="Available session memory summary runtime line",
        kind=CapabilityFragmentKind.AVAILABLE_RUNTIME,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="session_memory_summary_available",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MEMORY_SUMMARY_AVAILABLE,
            ),
        ),
        related_summary_flags=(CapabilitySummaryFlag.MEMORY_SUMMARY_AVAILABLE,),
    ),
    _fragment(
        fragment_id="unavailable.local_file_read",
        capability_id="local_file_read",
        name="Unavailable local file read line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="local_file_read",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(
                CapabilitySummaryFlag.LOCAL_FILE_READ_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=(READ_TEXT_FILE_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.LOCAL_FILE_READ_AVAILABLE,),
    ),
    _fragment(
        fragment_id="unavailable.local_directory_listing",
        capability_id="local_directory_listing",
        name="Unavailable local directory listing line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="local_directory_listing",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(
                CapabilitySummaryFlag.LOCAL_DIRECTORY_LISTING_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=(LIST_DIRECTORY_DEFINITION.name,),
        related_summary_flags=(
            CapabilitySummaryFlag.LOCAL_DIRECTORY_LISTING_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="unavailable.url_fetch",
        capability_id="url_fetch",
        name="Unavailable URL fetch line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="url_fetch",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(CapabilitySummaryFlag.URL_FETCH_AVAILABLE,),
        ),
        related_builtin_tool_names=(FETCH_URL_TEXT_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.URL_FETCH_AVAILABLE,),
    ),
    _fragment(
        fragment_id="unavailable.web_search",
        capability_id="web_search",
        name="Unavailable web search line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="web_search",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(CapabilitySummaryFlag.WEB_SEARCH_AVAILABLE,),
        ),
        related_builtin_tool_names=(SEARCH_WEB_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.WEB_SEARCH_AVAILABLE,),
    ),
    _fragment(
        fragment_id="unavailable.system_info",
        capability_id="system_info",
        name="Unavailable system_info line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="system_info",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(CapabilitySummaryFlag.SYSTEM_INFO_AVAILABLE,),
        ),
        related_builtin_tool_names=(SYSTEM_INFO_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.SYSTEM_INFO_AVAILABLE,),
    ),
    _fragment(
        fragment_id="unavailable.shell_command_execution",
        capability_id="shell_command_execution",
        name="Shell command execution unavailable line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="shell_command_execution",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(
                CapabilitySummaryFlag.SHELL_COMMAND_EXECUTION_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=(RUN_TERMINAL_COMMAND_DEFINITION.name,),
        related_summary_flags=(
            CapabilitySummaryFlag.SHELL_COMMAND_EXECUTION_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="unavailable.local_file_write",
        capability_id="local_file_write",
        name="Unavailable local file write line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="local_file_write",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(
                CapabilitySummaryFlag.LOCAL_FILE_WRITE_AVAILABLE,
            ),
        ),
        related_builtin_tool_names=(WRITE_TEXT_FILE_DEFINITION.name,),
        related_summary_flags=(CapabilitySummaryFlag.LOCAL_FILE_WRITE_AVAILABLE,),
    ),
    _fragment(
        fragment_id="unavailable.session_memory_summary",
        capability_id="session_memory_summary",
        name="Unavailable session memory summary line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="session_memory_summary",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(
                CapabilitySummaryFlag.MEMORY_SUMMARY_AVAILABLE,
            ),
        ),
        related_summary_flags=(CapabilitySummaryFlag.MEMORY_SUMMARY_AVAILABLE,),
    ),
    _fragment(
        fragment_id="unavailable.local_file_actions_summary",
        capability_id="unavailable_capabilities",
        name="Unavailable delete/move/rename/copy aggregate line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="local_file_actions_summary",
        availability=CapabilityAvailability(
            missing_any_builtin_tool_names=_LOCAL_FILE_MUTATION_TOOL_NAMES,
        ),
        related_builtin_tool_names=_LOCAL_FILE_MUTATION_TOOL_NAMES,
    ),
    _fragment(
        fragment_id="unavailable.any_non_listed_capability",
        capability_id="unavailable_capabilities",
        name="Catch-all unavailable capability line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="any_non_listed_capability",
    ),
    _fragment(
        fragment_id="tool_mode.model_callable",
        capability_id="tool_invocation_mode",
        name="Model-callable tool mode line",
        kind=CapabilityFragmentKind.TOOL_MODE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="tool_mode_model_callable",
        availability=CapabilityAvailability(
            required_summary_flags=(CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_summary_flags=(CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,),
    ),
    _fragment(
        fragment_id="tool_mode.user_initiated",
        capability_id="tool_invocation_mode",
        name="Slash-command-only tool mode line",
        kind=CapabilityFragmentKind.TOOL_MODE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="tool_mode_user_initiated",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.USER_INITIATED_ONLY,
        related_summary_flags=(CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,),
    ),
    _fragment(
        fragment_id="guidance.shared_core_rules",
        capability_id="shared_rules",
        name="Shared capability honesty rules",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="shared_core_rules",
    ),
    _fragment(
        fragment_id="guidance.model_callable.core_rules",
        capability_id="tool_invocation_mode",
        name="Model-callable core guidance block",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="model_callable_core_rules",
        availability=CapabilityAvailability(
            required_summary_flags=(CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_summary_flags=(CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,),
    ),
    _fragment(
        fragment_id="guidance.user_initiated.core_rules",
        capability_id="tool_invocation_mode",
        name="Slash-command-only core guidance block",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="user_initiated_core_rules",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.USER_INITIATED_ONLY,
        related_summary_flags=(CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,),
    ),
    _fragment(
        fragment_id="guidance.model_callable.local_choice.full",
        capability_id="tool_invocation_mode",
        name="Model-callable local file choice guidance (list + read)",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="_build_native_tool_choice_lines",
        prompt_detail="local_choice_full",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.LOCAL_DIRECTORY_LISTING_AVAILABLE,
                CapabilitySummaryFlag.LOCAL_FILE_READ_AVAILABLE,
            ),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=(
            LIST_DIRECTORY_DEFINITION.name,
            READ_TEXT_FILE_DEFINITION.name,
        ),
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.LOCAL_DIRECTORY_LISTING_AVAILABLE,
            CapabilitySummaryFlag.LOCAL_FILE_READ_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.model_callable.local_choice.list_only",
        capability_id="tool_invocation_mode",
        name="Model-callable local file choice guidance (list only)",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="_build_native_tool_choice_lines",
        prompt_detail="local_choice_list_only",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.LOCAL_DIRECTORY_LISTING_AVAILABLE,
            ),
            forbidden_summary_flags=(
                CapabilitySummaryFlag.LOCAL_FILE_READ_AVAILABLE,
            ),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=(
            LIST_DIRECTORY_DEFINITION.name,
            READ_TEXT_FILE_DEFINITION.name,
        ),
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.LOCAL_DIRECTORY_LISTING_AVAILABLE,
            CapabilitySummaryFlag.LOCAL_FILE_READ_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.model_callable.local_choice.read_only",
        capability_id="tool_invocation_mode",
        name="Model-callable local file choice guidance (read only)",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="_build_native_tool_choice_lines",
        prompt_detail="local_choice_read_only",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.LOCAL_FILE_READ_AVAILABLE,
            ),
            forbidden_summary_flags=(
                CapabilitySummaryFlag.LOCAL_DIRECTORY_LISTING_AVAILABLE,
            ),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=(
            LIST_DIRECTORY_DEFINITION.name,
            READ_TEXT_FILE_DEFINITION.name,
        ),
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.LOCAL_DIRECTORY_LISTING_AVAILABLE,
            CapabilitySummaryFlag.LOCAL_FILE_READ_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.model_callable.web_choice.full",
        capability_id="tool_invocation_mode",
        name="Model-callable web choice guidance (search + fetch)",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="_build_native_tool_choice_lines",
        prompt_detail="web_choice_full",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.WEB_SEARCH_AVAILABLE,
                CapabilitySummaryFlag.URL_FETCH_AVAILABLE,
            ),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=(
            SEARCH_WEB_DEFINITION.name,
            FETCH_URL_TEXT_DEFINITION.name,
        ),
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.WEB_SEARCH_AVAILABLE,
            CapabilitySummaryFlag.URL_FETCH_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.model_callable.web_choice.search_only",
        capability_id="tool_invocation_mode",
        name="Model-callable web choice guidance (search only)",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="_build_native_tool_choice_lines",
        prompt_detail="web_choice_search_only",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.WEB_SEARCH_AVAILABLE,
            ),
            forbidden_summary_flags=(CapabilitySummaryFlag.URL_FETCH_AVAILABLE,),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=(
            SEARCH_WEB_DEFINITION.name,
            FETCH_URL_TEXT_DEFINITION.name,
        ),
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.WEB_SEARCH_AVAILABLE,
            CapabilitySummaryFlag.URL_FETCH_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.model_callable.web_choice.fetch_only",
        capability_id="tool_invocation_mode",
        name="Model-callable web choice guidance (fetch only)",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="_build_native_tool_choice_lines",
        prompt_detail="web_choice_fetch_only",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.URL_FETCH_AVAILABLE,
            ),
            forbidden_summary_flags=(CapabilitySummaryFlag.WEB_SEARCH_AVAILABLE,),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=(
            SEARCH_WEB_DEFINITION.name,
            FETCH_URL_TEXT_DEFINITION.name,
        ),
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.WEB_SEARCH_AVAILABLE,
            CapabilitySummaryFlag.URL_FETCH_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.model_callable.fast_web_grounding",
        capability_id="fast_web_search",
        name="Model-callable fast web grounding guidance block",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="fast_web_grounding_guidance",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.FAST_WEB_SEARCH_AVAILABLE,
            ),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=(FAST_WEB_SEARCH_DEFINITION.name,),
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.FAST_WEB_SEARCH_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.model_callable.session_history",
        capability_id="session_history_recall",
        name="Model-callable session history guidance block",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="session_history_guidance",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.SESSION_HISTORY_RECALL_AVAILABLE,
            ),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=(INSPECT_SESSION_HISTORY_DEFINITION.name,),
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.SESSION_HISTORY_RECALL_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.model_callable.system_info",
        capability_id="system_info",
        name="Model-callable system_info guidance block",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="system_info_guidance",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.SYSTEM_INFO_AVAILABLE,
            ),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=(SYSTEM_INFO_DEFINITION.name,),
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.SYSTEM_INFO_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.model_callable.shell_command_execution",
        capability_id="shell_command_execution",
        name="Model-callable terminal command guidance block",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="shell_command_execution_guidance",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.SHELL_COMMAND_EXECUTION_AVAILABLE,
            ),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=(RUN_TERMINAL_COMMAND_DEFINITION.name,),
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.SHELL_COMMAND_EXECUTION_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.model_callable.long_term_memory",
        capability_id="long_term_memory",
        name="Model-callable long-term memory guidance block",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="long_term_memory_guidance",
        availability=CapabilityAvailability(
            required_summary_flags=(
                CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
                CapabilitySummaryFlag.LONG_TERM_MEMORY_AVAILABLE,
            ),
        ),
        tool_mode_relevance=CapabilityToolModeRelevance.MODEL_CALLABLE_ONLY,
        related_builtin_tool_names=_LONG_TERM_MEMORY_TOOL_NAMES,
        related_summary_flags=(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS,
            CapabilitySummaryFlag.LONG_TERM_MEMORY_AVAILABLE,
        ),
    ),
    _fragment(
        fragment_id="guidance.shared_tool_output_honesty",
        capability_id="shared_rules",
        name="Shared tool output honesty block",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="shared_tool_output_honesty",
    ),
    _fragment(
        fragment_id="guidance.unavailable_local_file_actions_warning",
        capability_id="unavailable_capabilities",
        name="Unavailable local file action warning block",
        kind=CapabilityFragmentKind.GUIDANCE,
        prompt_symbol="build_runtime_capability_context",
        prompt_detail="unavailable_local_file_actions_warning",
        availability=CapabilityAvailability(
            missing_any_builtin_tool_names=_LOCAL_FILE_MUTATION_TOOL_NAMES,
        ),
        related_builtin_tool_names=_LOCAL_FILE_MUTATION_TOOL_NAMES,
    ),
)


@lru_cache(maxsize=1)
def load_builtin_capability_fragment_registry() -> BuiltinCapabilityFragmentRegistry:
    """Load the deterministic built-in capability fragment registry."""
    return BuiltinCapabilityFragmentRegistry(_BUILTIN_CAPABILITY_FRAGMENTS)


def list_builtin_capability_fragments() -> tuple[CapabilityFragment, ...]:
    """List built-in capability fragments in deterministic registry order."""
    return load_builtin_capability_fragment_registry().list_fragments()


def resolve_builtin_capability_fragments(
    summary: CapabilitySummaryLike,
) -> tuple[CapabilityFragment, ...]:
    """Resolve active built-in capability fragments for one runtime summary."""
    return load_builtin_capability_fragment_registry().resolve_fragments(summary)


def render_builtin_capability_fragment(
    fragment: CapabilityFragment,
    summary: CapabilitySummaryLike,
) -> RenderedCapabilityFragment:
    """Render one resolved built-in capability fragment into prompt lines."""
    return RenderedCapabilityFragment(
        fragment=fragment,
        rendered_lines=fragment.render_lines(summary),
    )


def resolve_rendered_builtin_capability_fragments(
    summary: CapabilitySummaryLike,
) -> tuple[RenderedCapabilityFragment, ...]:
    """Resolve and render built-in capability fragments in registry order."""
    rendered_fragments: list[RenderedCapabilityFragment] = []

    for fragment in resolve_builtin_capability_fragments(summary):
        rendered_fragment = render_builtin_capability_fragment(fragment, summary)
        if rendered_fragment.rendered_lines:
            rendered_fragments.append(rendered_fragment)

    return tuple(rendered_fragments)


__all__ = [
    "BuiltinCapabilityFragmentRegistry",
    "CapabilityAvailability",
    "CapabilityFragment",
    "CapabilityFragmentPrompt",
    "CapabilityFragmentKind",
    "CapabilityPromptSource",
    "CapabilityPromptSourceKind",
    "RenderedCapabilityFragment",
    "CapabilitySummaryFlag",
    "CapabilityToolModeRelevance",
    "list_builtin_capability_fragments",
    "load_builtin_capability_fragment_registry",
    "render_builtin_capability_fragment",
    "resolve_builtin_capability_fragments",
    "resolve_rendered_builtin_capability_fragments",
]
