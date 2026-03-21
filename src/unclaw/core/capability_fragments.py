"""Typed built-in capability fragments for future prompt composition.

Phase 1 keeps runtime behaviour unchanged: the live capability prompt still
renders from ``unclaw.core.capabilities``. This module introduces the durable
typed metadata layer that future phases can compose from safely.
"""

from __future__ import annotations

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
from unclaw.tools.notes_tools import (
    CREATE_NOTE_DEFINITION,
    LIST_NOTES_DEFINITION,
    READ_NOTE_DEFINITION,
    UPDATE_NOTE_DEFINITION,
)
from unclaw.tools.session_tools import INSPECT_SESSION_HISTORY_DEFINITION
from unclaw.tools.system_tools import SYSTEM_INFO_DEFINITION
from unclaw.tools.web_tools import FETCH_URL_TEXT_DEFINITION, SEARCH_WEB_DEFINITION

_CAPABILITIES_MODULE_REFERENCE = "unclaw.core.capabilities"
_NOTES_TOOL_NAMES = (
    CREATE_NOTE_DEFINITION.name,
    READ_NOTE_DEFINITION.name,
    LIST_NOTES_DEFINITION.name,
    UPDATE_NOTE_DEFINITION.name,
)
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
    memory_summary_available: bool
    model_can_call_tools: bool
    notes_available: bool
    local_file_write_available: bool
    session_history_recall_available: bool
    long_term_memory_available: bool


class CapabilityFragmentKind(StrEnum):
    """Semantic role of one built-in capability fragment."""

    AVAILABLE_TOOL = "available_tool"
    AVAILABLE_RUNTIME = "available_runtime"
    UNAVAILABLE_CAPABILITY = "unavailable_capability"
    TOOL_MODE = "tool_mode"
    GUIDANCE = "guidance"


class CapabilityPromptSourceKind(StrEnum):
    """Where the current prompt text for a fragment comes from."""

    FUNCTION = "function"
    FILE = "file"
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
    MEMORY_SUMMARY_AVAILABLE = "memory_summary_available"
    MODEL_CAN_CALL_TOOLS = "model_can_call_tools"
    NOTES_AVAILABLE = "notes_available"
    LOCAL_FILE_WRITE_AVAILABLE = "local_file_write_available"
    SESSION_HISTORY_RECALL_AVAILABLE = "session_history_recall_available"
    LONG_TERM_MEMORY_AVAILABLE = "long_term_memory_available"


@dataclass(frozen=True, slots=True)
class CapabilityPromptSource:
    """Current prompt-text source location for one built-in fragment."""

    kind: CapabilityPromptSourceKind
    reference: str


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
    prompt_source: CapabilityPromptSource
    availability: CapabilityAvailability = field(default_factory=CapabilityAvailability)
    tool_mode_relevance: CapabilityToolModeRelevance = (
        CapabilityToolModeRelevance.SHARED
    )
    related_builtin_tool_names: tuple[str, ...] = ()
    related_summary_flags: tuple[CapabilitySummaryFlag, ...] = ()
    description: str | None = None

    def matches(self, summary: CapabilitySummaryLike) -> bool:
        return self.availability.matches(summary)


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


def _capabilities_source(
    symbol_name: str,
    detail: str,
) -> CapabilityPromptSource:
    return CapabilityPromptSource(
        kind=CapabilityPromptSourceKind.FUNCTION,
        reference=f"{_CAPABILITIES_MODULE_REFERENCE}.{symbol_name}:{detail}",
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
    return CapabilityFragment(
        fragment_id=fragment_id,
        capability_id=capability_id,
        name=name,
        kind=kind,
        prompt_source=_capabilities_source(prompt_symbol, prompt_detail),
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
        fragment_id="available.notes",
        capability_id="notes",
        name="Available notes tool family line",
        kind=CapabilityFragmentKind.AVAILABLE_TOOL,
        prompt_symbol="_build_available_tool_lines",
        prompt_detail="notes",
        availability=CapabilityAvailability(
            required_summary_flags=(CapabilitySummaryFlag.NOTES_AVAILABLE,),
        ),
        related_builtin_tool_names=_NOTES_TOOL_NAMES,
        related_summary_flags=(CapabilitySummaryFlag.NOTES_AVAILABLE,),
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
        fragment_id="unavailable.notes",
        capability_id="notes",
        name="Unavailable notes tool family line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="notes",
        availability=CapabilityAvailability(
            forbidden_summary_flags=(CapabilitySummaryFlag.NOTES_AVAILABLE,),
        ),
        related_builtin_tool_names=_NOTES_TOOL_NAMES,
        related_summary_flags=(CapabilitySummaryFlag.NOTES_AVAILABLE,),
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
        fragment_id="unavailable.shell_command_execution",
        capability_id="unavailable_capabilities",
        name="Shell command execution unavailable line",
        kind=CapabilityFragmentKind.UNAVAILABLE_CAPABILITY,
        prompt_symbol="_build_unavailable_lines",
        prompt_detail="shell_command_execution",
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


__all__ = [
    "BuiltinCapabilityFragmentRegistry",
    "CapabilityAvailability",
    "CapabilityFragment",
    "CapabilityFragmentKind",
    "CapabilityPromptSource",
    "CapabilityPromptSourceKind",
    "CapabilitySummaryFlag",
    "CapabilityToolModeRelevance",
    "list_builtin_capability_fragments",
    "load_builtin_capability_fragment_registry",
    "resolve_builtin_capability_fragments",
]
