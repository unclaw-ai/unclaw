"""Runtime capability summaries and fragment-driven capability context rendering."""

from __future__ import annotations

from dataclasses import dataclass

from unclaw.core.capability_fragments import (
    CapabilityFragmentKind,
    RenderedCapabilityFragment,
    resolve_rendered_builtin_capability_fragments,
)
from unclaw.tools.file_tools import (
    LIST_DIRECTORY_DEFINITION,
    READ_TEXT_FILE_DEFINITION,
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
    rendered_fragments = resolve_rendered_builtin_capability_fragments(summary)
    available_tool_lines = _collect_rendered_lines(
        rendered_fragments,
        CapabilityFragmentKind.AVAILABLE_TOOL,
    )
    available_runtime_lines = _collect_rendered_lines(
        rendered_fragments,
        CapabilityFragmentKind.AVAILABLE_RUNTIME,
    )
    unavailable_lines = _collect_rendered_lines(
        rendered_fragments,
        CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
    )
    tool_mode_lines = _collect_rendered_lines(
        rendered_fragments,
        CapabilityFragmentKind.TOOL_MODE,
    )
    guidance_lines = _collect_rendered_lines(
        rendered_fragments,
        CapabilityFragmentKind.GUIDANCE,
    )

    lines = [
        "Runtime capability status:",
        f"Enabled built-in tools: {summary.enabled_builtin_tool_count}",
        "Available built-in tools:",
    ]

    if available_tool_lines:
        lines.extend(f"- {line}" for line in available_tool_lines)
    else:
        lines.append("- None.")

    if available_runtime_lines:
        lines.append("Other available runtime capabilities:")
        lines.extend(f"- {line}" for line in available_runtime_lines)

    lines.append("Unavailable capabilities:")
    lines.extend(f"- {line}" for line in unavailable_lines)
    lines.extend(tool_mode_lines)
    lines.append("Behavior rules:")
    lines.extend(f"- {line}" for line in guidance_lines)

    return "\n".join(lines)


def _collect_rendered_lines(
    rendered_fragments: tuple[RenderedCapabilityFragment, ...],
    kind: CapabilityFragmentKind,
) -> tuple[str, ...]:
    lines: list[str] = []

    for rendered_fragment in rendered_fragments:
        if rendered_fragment.fragment.kind is kind:
            lines.extend(rendered_fragment.rendered_lines)

    return tuple(lines)


__all__ = [
    "RuntimeCapabilitySummary",
    "build_runtime_capability_context",
    "build_runtime_capability_summary",
]
