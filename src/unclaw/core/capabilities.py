"""Runtime capability summaries for prompt honesty and future CLI UX."""

from __future__ import annotations

from dataclasses import dataclass

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
from unclaw.tools.notes_tools import CREATE_NOTE_DEFINITION
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.session_tools import INSPECT_SESSION_HISTORY_DEFINITION
from unclaw.tools.system_tools import SYSTEM_INFO_DEFINITION
from unclaw.tools.web_tools import FETCH_URL_TEXT_DEFINITION, SEARCH_WEB_DEFINITION

_LONG_TERM_MEMORY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
        SEARCH_LONG_TERM_MEMORY_DEFINITION.name,
        LIST_LONG_TERM_MEMORY_DEFINITION.name,
        FORGET_LONG_TERM_MEMORY_DEFINITION.name,
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeCapabilitySummary:
    """Compact summary of the runtime's real user-facing capabilities."""

    available_builtin_tool_names: tuple[str, ...]
    local_file_read_available: bool
    local_directory_listing_available: bool
    url_fetch_available: bool
    web_search_available: bool
    system_info_available: bool
    memory_summary_available: bool
    model_can_call_tools: bool = False
    notes_available: bool = False
    local_file_write_available: bool = False
    session_history_recall_available: bool = False
    long_term_memory_available: bool = False

    @property
    def enabled_builtin_tool_count(self) -> int:
        return len(self.available_builtin_tool_names)

    @property
    def has_builtin_tools(self) -> bool:
        return self.enabled_builtin_tool_count > 0


def build_runtime_capability_summary(
    *,
    tool_registry: ToolRegistry,
    memory_summary_available: bool,
    model_can_call_tools: bool = False,
) -> RuntimeCapabilitySummary:
    """Summarize the currently enabled built-in tools and related runtime features."""
    available_tool_names = tuple(
        tool_definition.name for tool_definition in tool_registry.list_tools()
    )
    available_tool_name_set = frozenset(available_tool_names)

    return RuntimeCapabilitySummary(
        available_builtin_tool_names=available_tool_names,
        local_file_read_available=READ_TEXT_FILE_DEFINITION.name in available_tool_name_set,
        local_directory_listing_available=(
            LIST_DIRECTORY_DEFINITION.name in available_tool_name_set
        ),
        url_fetch_available=FETCH_URL_TEXT_DEFINITION.name in available_tool_name_set,
        web_search_available=SEARCH_WEB_DEFINITION.name in available_tool_name_set,
        system_info_available=SYSTEM_INFO_DEFINITION.name in available_tool_name_set,
        memory_summary_available=memory_summary_available,
        model_can_call_tools=model_can_call_tools,
        notes_available=CREATE_NOTE_DEFINITION.name in available_tool_name_set,
        local_file_write_available=WRITE_TEXT_FILE_DEFINITION.name in available_tool_name_set,
        session_history_recall_available=(
            INSPECT_SESSION_HISTORY_DEFINITION.name in available_tool_name_set
        ),
        long_term_memory_available=bool(
            _LONG_TERM_MEMORY_TOOL_NAMES & available_tool_name_set
        ),
    )


def build_runtime_capability_context(summary: RuntimeCapabilitySummary) -> str:
    """Build the system-context note that keeps the model honest about tools."""
    available_tool_lines = _build_available_tool_lines(summary)
    unavailable_lines = _build_unavailable_lines(summary)
    unavailable_local_file_actions = _format_unavailable_local_file_actions(summary)

    lines = [
        "Runtime capability status:",
        f"Enabled built-in tools: {summary.enabled_builtin_tool_count}",
        "Available built-in tools:",
    ]

    if available_tool_lines:
        lines.extend(f"- {line}" for line in available_tool_lines)
    else:
        lines.append("- None.")

    if summary.memory_summary_available:
        lines.append("Other available runtime capabilities:")
        lines.append("- Session memory and summary access.")

    lines.append("Unavailable capabilities:")
    lines.extend(f"- {line}" for line in unavailable_lines)
    if summary.model_can_call_tools:
        lines.append("Tool invocation mode: model-callable (you may call tools directly this turn).")
    else:
        lines.append(
            "Tool invocation mode: user-initiated slash commands only "
            "(you cannot call tools directly this turn)."
        )

    lines.extend(
        (
            "Behavior rules:",
            (
                "- Do not claim you have no tool access when one or more built-in "
                "tools are available."
            ),
            (
                "- You may say you can use Unclaw built-in tools that are listed "
                "as available."
            ),
        )
    )

    if summary.model_can_call_tools:
        lines.extend(
            (
                "- Use only the listed built-in tools and base the final answer "
                "on their results.",
                (
                    "- If the answer is complete and safe from the conversation "
                    "alone, reply directly without tools."
                ),
                "- For compound requests, decompose into the minimum useful sub-tasks.",
                "- Use tools only where they add needed runtime or external information.",
                "- Preserve the user's requested order when practical.",
                "- Do not call tools for parts already answerable from the conversation.",
                "- Combine tool results into one coherent final answer.",
                *_build_native_tool_choice_lines(summary),
                (
                    "- Only claim a file was written, created, or modified if "
                    "write_text_file or a notes write tool (create_note, update_note) "
                    "output is present in this conversation. If no such tool ran, "
                    "say the action has not happened yet."
                ),
            )
        )
        if summary.session_history_recall_available:
            lines.extend(
                (
                    "- Use inspect_session_history to answer exact questions about "
                    "prior prompts, message order, or counts. "
                    "Do not guess from memory — call the tool for exact recall. "
                    "The tool supports filter_role (user/assistant/tool), "
                    "nth (1-indexed lookup), and limit. "
                    "Reply to the user in their language after reading the tool result.",
                )
            )
        if summary.system_info_available:
            lines.extend(
                (
                    "- Use system_info for current local machine facts and runtime facts "
                    "such as local date/time, day, OS, CPU, RAM, hostname, and locale.",
                )
            )
        if summary.long_term_memory_available:
            lines.extend(
                (
                    "- Long-term memory tools are for previously stored persistent "
                    "cross-session facts or preferences. They are not injected "
                    "automatically.",
                    "- Use them for stable personal facts, preferences, and hardware "
                    "or setup information.",
                    "- Use search_long_term_memory for targeted recall of a stored fact.",
                    "- For search_long_term_memory, pass concise semantic query terms "
                    "for the topic, even if the user asked in another language.",
                    "- Use list_long_term_memory for broad recall of stored memories.",
                    "- Use remember_long_term_memory only when the user explicitly wants "
                    "a fact saved for later or explicitly corrects a stored fact.",
                    "- Use forget_long_term_memory only when the user explicitly wants a "
                    "stored memory removed.",
                    "- Use system_info for current local machine facts, not long-term "
                    "memory.",
                    "- Do not use long-term memory tools for current session message "
                    "history or message order; use inspect_session_history for that.",
                    "- Do not auto-store facts that were not explicitly requested for "
                    "storage.",
                )
            )
    else:
        lines.extend(
            (
                "- Tools listed above are available only when the user types the "
                "slash command. You cannot invoke them yourself in this turn.",
                "- Do not say \"let me search\" or \"I will look that up\" as if "
                "you can perform the action right now.",
                (
                    "- You cannot write, modify, or create any file or note in this "
                    "turn. If the user asks you to do so, say the action was not "
                    "performed — you have not written or changed anything."
                ),
            )
        )

    lines.extend(
        (
            (
                "- Do not claim you already searched, fetched, read, wrote, created, "
                "modified, or deleted anything unless actual tool output for that "
                "action is present in this conversation."
            ),
            (
                "- If tool output is already present in the conversation, treat it "
                "as retrieved context that you may summarize, compare, extract, "
                "or analyze. Do not say you cannot access it, and do not ask the "
                "user to paste it again."
            ),
            (
                "- If a capability is unavailable, say so clearly instead of "
                "implying it exists."
            ),
        )
    )
    if unavailable_local_file_actions is not None:
        lines.append(
            f"- If the user asks to {unavailable_local_file_actions} a file or "
            "directory, say clearly this capability does not exist. Do not "
            "suggest it might work after confirmation."
        )
    return "\n".join(lines)


def _build_native_tool_choice_lines(
    summary: RuntimeCapabilitySummary,
) -> tuple[str, ...]:
    lines: list[str] = []

    if summary.local_directory_listing_available and summary.local_file_read_available:
        lines.append(
            "Use list_directory for local directories and read_text_file for "
            "supported local text files."
        )
    elif summary.local_directory_listing_available:
        lines.append("Use list_directory for local directories.")
    elif summary.local_file_read_available:
        lines.append("Use read_text_file for supported local text files.")

    if summary.web_search_available and summary.url_fetch_available:
        lines.append(
            "Use search_web for up-to-date external information. "
            "Use fetch_url_text for a specific public URL."
        )
    elif summary.web_search_available:
        lines.append("Use search_web for up-to-date external information.")
    elif summary.url_fetch_available:
        lines.append("Use fetch_url_text for a specific public URL.")

    return tuple(f"- {line}" for line in lines)


def _build_available_tool_lines(summary: RuntimeCapabilitySummary) -> tuple[str, ...]:
    lines: list[str] = []
    if summary.local_file_read_available:
        lines.append(
            "/read <path>: read local .txt, .md, .json, or .csv files inside "
            "allowed roots. Other formats (pdf, docx, xlsx, etc.) are not "
            "supported in V1."
        )
    if summary.local_directory_listing_available:
        lines.append("/ls [path]: list local directories inside allowed roots.")
    if summary.url_fetch_available:
        lines.append("/fetch <url>: fetch one public URL and extract text.")
    if summary.web_search_available:
        lines.append(
            "/search <query>: search the public web, read a few relevant pages, "
            "and answer naturally from grounded web context with compact sources."
        )
    if summary.system_info_available:
        lines.append(
            "system_info: return current local machine and runtime facts, "
            "including local date/time, day, OS, CPU core count, total RAM, "
            "hostname, and locale."
        )
    if summary.notes_available:
        lines.append(
            "create_note / read_note / list_notes / update_note: "
            "create, read, list, or overwrite local markdown notes."
        )
    if summary.local_file_write_available:
        lines.append(
            "write_text_file <path>: write a new local file. "
            "Relative paths are created inside data/files/ by default. "
            "Default is overwrite=false — fails if the file already exists. "
            "Only use overwrite=true when the user explicitly intends to replace an existing file."
        )
    if DELETE_FILE_DEFINITION.name in summary.available_builtin_tool_names:
        lines.append(
            "delete_file <path>: delete one local file. "
            "Relative paths resolve inside data/files/ by default. "
            "Requires confirm=true and only deletes files, not directories."
        )
    if MOVE_FILE_DEFINITION.name in summary.available_builtin_tool_names:
        lines.append(
            "move_file <source_path> <destination_path>: move one local file. "
            "Relative paths resolve inside data/files/ by default. "
            "Default is overwrite=false — fails if the destination already exists. "
            "Only moves files, not directories."
        )
    if RENAME_FILE_DEFINITION.name in summary.available_builtin_tool_names:
        lines.append(
            "rename_file <source_path> <destination_path>: rename one local file. "
            "Relative paths resolve inside data/files/ by default. "
            "Default is overwrite=false — fails if the destination already exists. "
            "Only renames files, not directories."
        )
    if COPY_FILE_DEFINITION.name in summary.available_builtin_tool_names:
        lines.append(
            "copy_file <source_path> <destination_path>: copy one local file. "
            "Relative paths resolve inside data/files/ by default. "
            "Default is overwrite=false — fails if the destination already exists. "
            "Only copies files, not directories."
        )
    if summary.session_history_recall_available:
        lines.append(
            "inspect_session_history: return an exact, deterministic list of "
            "persisted messages for the current session. "
            "Supports filter_role (user/assistant/tool), nth (1-indexed lookup), "
            "and limit. Use this tool for exact questions about prior prompts, "
            "their order, or message counts."
        )
    if summary.long_term_memory_available:
        lines.append(
            "remember_long_term_memory / search_long_term_memory / "
            "list_long_term_memory / forget_long_term_memory: "
            "store, search, list, or delete persistent cross-session facts "
            "and preferences. "
            "Not injected automatically — call tools explicitly. "
            "Use search_long_term_memory for targeted recall and "
            "list_long_term_memory for broad recall. "
            "Not for session message history — use inspect_session_history for that."
        )
    return tuple(lines)


def _build_unavailable_lines(summary: RuntimeCapabilitySummary) -> tuple[str, ...]:
    lines = [
        "Shell command execution.",
        "Any capability that is not listed as available above.",
    ]
    unavailable_local_file_actions = _format_unavailable_local_file_actions(summary)
    if unavailable_local_file_actions is not None:
        lines.insert(
            1,
            (
                f"{unavailable_local_file_actions.capitalize()} local files "
                "or directories (no such tool is registered)."
            ),
        )

    if not summary.local_file_read_available:
        lines.insert(0, "Local file read via /read <path>.")
    if not summary.local_directory_listing_available:
        lines.insert(0, "Local directory listing via /ls [path].")
    if not summary.url_fetch_available:
        lines.insert(0, "Direct URL fetch via /fetch <url>.")
    if not summary.web_search_available:
        lines.insert(0, "Web search via /search <query>.")
    if not summary.system_info_available:
        lines.insert(0, "Local machine and runtime information via system_info.")
    if not summary.notes_available:
        lines.insert(0, "Local notes (create_note, read_note, list_notes, update_note).")
    if not summary.local_file_write_available:
        lines.insert(0, "Local file write via write_text_file.")
    if not summary.memory_summary_available:
        lines.insert(0, "Session memory and summary access.")

    return tuple(lines)


def _format_unavailable_local_file_actions(
    summary: RuntimeCapabilitySummary,
) -> str | None:
    actions: list[str] = []
    if DELETE_FILE_DEFINITION.name not in summary.available_builtin_tool_names:
        actions.append("delete")
    if MOVE_FILE_DEFINITION.name not in summary.available_builtin_tool_names:
        actions.append("move")
    if RENAME_FILE_DEFINITION.name not in summary.available_builtin_tool_names:
        actions.append("rename")
    if COPY_FILE_DEFINITION.name not in summary.available_builtin_tool_names:
        actions.append("copy")

    if not actions:
        return None
    if len(actions) == 1:
        return actions[0]
    if len(actions) == 2:
        return f"{actions[0]} or {actions[1]}"
    return f"{', '.join(actions[:-1])}, or {actions[-1]}"


__all__ = [
    "RuntimeCapabilitySummary",
    "build_runtime_capability_context",
    "build_runtime_capability_summary",
]
