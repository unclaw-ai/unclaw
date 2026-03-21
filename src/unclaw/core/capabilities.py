"""Runtime capability summaries and fragment-driven capability context rendering."""

from __future__ import annotations

from dataclasses import dataclass

from unclaw.core.capability_budget import (
    CapabilityBudgetPolicy,
    CapabilityGuidanceDetail,
    CapabilitySectionDetail,
    STANDARD_CAPABILITY_BUDGET_POLICY,
)
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
from unclaw.tools.terminal_tools import RUN_TERMINAL_COMMAND_DEFINITION
from unclaw.tools.web_tools import FETCH_URL_TEXT_DEFINITION, SEARCH_WEB_DEFINITION

_LONG_TERM_MEMORY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
        SEARCH_LONG_TERM_MEMORY_DEFINITION.name,
        LIST_LONG_TERM_MEMORY_DEFINITION.name,
        FORGET_LONG_TERM_MEMORY_DEFINITION.name,
    }
)

_COMPACT_UNAVAILABLE_FRAGMENT_IDS: frozenset[str] = frozenset(
    {
        "unavailable.local_file_actions_summary",
        "unavailable.any_non_listed_capability",
    }
)
_MINIMAL_GUIDANCE_FRAGMENT_IDS: frozenset[str] = frozenset(
    {
        "guidance.model_callable.core_rules",
        "guidance.user_initiated.core_rules",
        "guidance.shared_tool_output_honesty",
    }
)
_COMPACT_GUIDANCE_LINE_SELECTIONS: dict[str, tuple[int, ...]] = {
    "guidance.model_callable.core_rules": (0, 1, 3, 7),
    "guidance.model_callable.long_term_memory": (0, 2, 4, 5, 8, 9),
}
_COMPACT_TOOL_LABELS_PER_LINE = 4


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
    shell_command_execution_available: bool = False

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
        shell_command_execution_available=(
            RUN_TERMINAL_COMMAND_DEFINITION.name in available_tool_name_set
        ),
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


def build_runtime_capability_context(
    summary: RuntimeCapabilitySummary,
    *,
    budget_policy: CapabilityBudgetPolicy | None = None,
) -> str:
    """Build the system-context note that keeps the model honest about tools."""
    active_budget_policy = budget_policy or STANDARD_CAPABILITY_BUDGET_POLICY
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
    ]

    lines.extend(
        _build_available_tool_section(
            available_tool_lines=available_tool_lines,
            summary=summary,
            detail=active_budget_policy.available_tool_detail,
        )
    )

    if active_budget_policy.include_available_runtime_section and available_runtime_lines:
        lines.append("Other available runtime capabilities:")
        lines.extend(f"- {line}" for line in available_runtime_lines)

    lines.extend(
        _build_unavailable_capability_section(
            rendered_fragments=rendered_fragments,
            unavailable_lines=unavailable_lines,
            detail=active_budget_policy.unavailable_capability_detail,
        )
    )
    lines.extend(tool_mode_lines)
    budgeted_guidance_lines = _collect_budgeted_guidance_lines(
        rendered_fragments=rendered_fragments,
        guidance_lines=guidance_lines,
        detail=active_budget_policy.guidance_detail,
    )
    if budgeted_guidance_lines:
        lines.append("Behavior rules:")
        lines.extend(f"- {line}" for line in budgeted_guidance_lines)

    return "\n".join(lines)


def _build_available_tool_section(
    *,
    available_tool_lines: tuple[str, ...],
    summary: RuntimeCapabilitySummary,
    detail: CapabilitySectionDetail,
) -> tuple[str, ...]:
    if detail is CapabilitySectionDetail.NONE:
        if summary.has_builtin_tools:
            return ("Built-in tools are available in this turn.",)
        return ()

    if detail is CapabilitySectionDetail.COMPACT:
        if not available_tool_lines:
            return ("Available built-in tools (compact):", "- None.")
        compact_tool_lines = _compact_capability_labels(available_tool_lines)
        return (
            "Available built-in tools (compact):",
            *(f"- {line}" for line in compact_tool_lines),
        )

    lines = ["Available built-in tools:"]
    if available_tool_lines:
        lines.extend(f"- {line}" for line in available_tool_lines)
    else:
        lines.append("- None.")
    return tuple(lines)


def _build_unavailable_capability_section(
    *,
    rendered_fragments: tuple[RenderedCapabilityFragment, ...],
    unavailable_lines: tuple[str, ...],
    detail: CapabilitySectionDetail,
) -> tuple[str, ...]:
    if detail is CapabilitySectionDetail.NONE:
        return ()

    if detail is CapabilitySectionDetail.COMPACT:
        compact_unavailable_lines = tuple(
            line
            for rendered_fragment in rendered_fragments
            if rendered_fragment.fragment.kind is CapabilityFragmentKind.UNAVAILABLE_CAPABILITY
            and rendered_fragment.fragment.fragment_id in _COMPACT_UNAVAILABLE_FRAGMENT_IDS
            for line in rendered_fragment.rendered_lines
        )
        if not compact_unavailable_lines:
            return ()
        return (
            "Unavailable capabilities (compact):",
            *(f"- {line}" for line in compact_unavailable_lines),
        )

    return (
        "Unavailable capabilities:",
        *(f"- {line}" for line in unavailable_lines),
    )


def _collect_budgeted_guidance_lines(
    *,
    rendered_fragments: tuple[RenderedCapabilityFragment, ...],
    guidance_lines: tuple[str, ...],
    detail: CapabilityGuidanceDetail,
) -> tuple[str, ...]:
    if detail is CapabilityGuidanceDetail.FULL:
        return guidance_lines

    collected_lines: list[str] = []

    for rendered_fragment in rendered_fragments:
        if rendered_fragment.fragment.kind is not CapabilityFragmentKind.GUIDANCE:
            continue

        fragment_id = rendered_fragment.fragment.fragment_id
        if (
            detail is CapabilityGuidanceDetail.MINIMAL
            and fragment_id not in _MINIMAL_GUIDANCE_FRAGMENT_IDS
        ):
            continue

        selected_lines = _select_guidance_fragment_lines(
            fragment_id=fragment_id,
            lines=rendered_fragment.rendered_lines,
            detail=detail,
        )
        collected_lines.extend(selected_lines)

    return tuple(collected_lines)


def _select_guidance_fragment_lines(
    *,
    fragment_id: str,
    lines: tuple[str, ...],
    detail: CapabilityGuidanceDetail,
) -> tuple[str, ...]:
    if detail is CapabilityGuidanceDetail.FULL:
        return lines

    selected_indexes = _COMPACT_GUIDANCE_LINE_SELECTIONS.get(fragment_id)
    if selected_indexes is None:
        return lines

    return tuple(
        lines[index]
        for index in selected_indexes
        if 0 <= index < len(lines)
    )


def _compact_capability_labels(lines: tuple[str, ...]) -> tuple[str, ...]:
    labels = tuple(_extract_capability_label(line) for line in lines)
    compact_lines: list[str] = []

    for start_index in range(0, len(labels), _COMPACT_TOOL_LABELS_PER_LINE):
        chunk = labels[start_index : start_index + _COMPACT_TOOL_LABELS_PER_LINE]
        compact_lines.append(", ".join(chunk))

    return tuple(compact_lines)


def _extract_capability_label(line: str) -> str:
    prefix, separator, _remainder = line.partition(":")
    if separator:
        return prefix.strip()
    return line.rstrip(".").strip()


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
