from __future__ import annotations

import pytest

from unclaw.core.capabilities import build_runtime_capability_summary
from unclaw.core.capability_fragments import (
    CapabilityFragmentKind,
    CapabilityPromptSourceKind,
    CapabilitySummaryFlag,
    CapabilityToolModeRelevance,
    load_builtin_capability_fragment_registry,
    resolve_builtin_capability_fragments,
)
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.session_manager import SessionManager
from unclaw.settings import load_settings
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.system_tools import SYSTEM_INFO_DEFINITION
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

pytestmark = pytest.mark.unit


def _fragment_ids(fragments) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
    return tuple(fragment.fragment_id for fragment in fragments)


def test_builtin_capability_fragment_registry_has_stable_fragment_and_capability_order() -> None:
    registry = load_builtin_capability_fragment_registry()

    assert registry.list_capability_ids() == (
        "local_file_read",
        "local_directory_listing",
        "url_fetch",
        "web_search",
        "system_info",
        "notes",
        "local_file_write",
        "local_file_delete",
        "local_file_move",
        "local_file_rename",
        "local_file_copy",
        "session_history_recall",
        "long_term_memory",
        "session_memory_summary",
        "unavailable_capabilities",
        "tool_invocation_mode",
        "shared_rules",
    )

    assert _fragment_ids(registry.list_fragments()) == (
        "available.local_file_read",
        "available.local_directory_listing",
        "available.url_fetch",
        "available.web_search",
        "available.system_info",
        "available.notes",
        "available.local_file_write",
        "available.local_file_delete",
        "available.local_file_move",
        "available.local_file_rename",
        "available.local_file_copy",
        "available.session_history_recall",
        "available.long_term_memory",
        "available.session_memory_summary",
        "unavailable.local_file_read",
        "unavailable.local_directory_listing",
        "unavailable.url_fetch",
        "unavailable.web_search",
        "unavailable.system_info",
        "unavailable.notes",
        "unavailable.local_file_write",
        "unavailable.session_memory_summary",
        "unavailable.local_file_actions_summary",
        "unavailable.shell_command_execution",
        "unavailable.any_non_listed_capability",
        "tool_mode.model_callable",
        "tool_mode.user_initiated",
        "guidance.shared_core_rules",
        "guidance.model_callable.core_rules",
        "guidance.user_initiated.core_rules",
        "guidance.model_callable.local_choice.full",
        "guidance.model_callable.local_choice.list_only",
        "guidance.model_callable.local_choice.read_only",
        "guidance.model_callable.web_choice.full",
        "guidance.model_callable.web_choice.search_only",
        "guidance.model_callable.web_choice.fetch_only",
        "guidance.model_callable.session_history",
        "guidance.model_callable.system_info",
        "guidance.model_callable.long_term_memory",
        "guidance.shared_tool_output_honesty",
        "guidance.unavailable_local_file_actions_warning",
    )


def test_builtin_capability_fragment_registry_exposes_typed_metadata_and_indexes() -> None:
    registry = load_builtin_capability_fragment_registry()

    system_info_fragment = registry.get_fragment("available.system_info")
    assert system_info_fragment.capability_id == "system_info"
    assert system_info_fragment.kind is CapabilityFragmentKind.AVAILABLE_TOOL
    assert system_info_fragment.prompt_source.kind is CapabilityPromptSourceKind.FUNCTION
    assert (
        system_info_fragment.prompt_source.reference
        == "unclaw.core.capabilities._build_available_tool_lines:system_info"
    )
    assert system_info_fragment.related_builtin_tool_names == (
        SYSTEM_INFO_DEFINITION.name,
    )
    assert system_info_fragment.related_summary_flags == (
        CapabilitySummaryFlag.SYSTEM_INFO_AVAILABLE,
    )
    assert system_info_fragment.tool_mode_relevance is CapabilityToolModeRelevance.SHARED

    assert _fragment_ids(registry.get_fragments_for_capability("system_info")) == (
        "available.system_info",
        "unavailable.system_info",
        "guidance.model_callable.system_info",
    )
    assert _fragment_ids(
        registry.get_fragments_for_summary_flag(
            CapabilitySummaryFlag.SYSTEM_INFO_AVAILABLE
        )
    ) == (
        "available.system_info",
        "unavailable.system_info",
        "guidance.model_callable.system_info",
    )


def test_builtin_capability_fragment_registry_maps_current_tool_concepts_to_fragments() -> None:
    registry = load_builtin_capability_fragment_registry()

    assert _fragment_ids(
        registry.get_fragments_for_tool_name(SEARCH_WEB_DEFINITION.name)
    ) == (
        "available.web_search",
        "unavailable.web_search",
        "guidance.model_callable.web_choice.full",
        "guidance.model_callable.web_choice.search_only",
        "guidance.model_callable.web_choice.fetch_only",
    )

    assert _fragment_ids(
        registry.get_fragments_for_summary_flag(
            CapabilitySummaryFlag.MODEL_CAN_CALL_TOOLS
        )
    ) == (
        "tool_mode.model_callable",
        "tool_mode.user_initiated",
        "guidance.model_callable.core_rules",
        "guidance.user_initiated.core_rules",
        "guidance.model_callable.local_choice.full",
        "guidance.model_callable.local_choice.list_only",
        "guidance.model_callable.local_choice.read_only",
        "guidance.model_callable.web_choice.full",
        "guidance.model_callable.web_choice.search_only",
        "guidance.model_callable.web_choice.fetch_only",
        "guidance.model_callable.session_history",
        "guidance.model_callable.system_info",
        "guidance.model_callable.long_term_memory",
    )


def test_resolve_builtin_capability_fragments_for_empty_non_native_runtime() -> None:
    summary = build_runtime_capability_summary(
        tool_registry=ToolRegistry(),
        memory_summary_available=False,
        model_can_call_tools=False,
    )

    assert _fragment_ids(resolve_builtin_capability_fragments(summary)) == (
        "unavailable.local_file_read",
        "unavailable.local_directory_listing",
        "unavailable.url_fetch",
        "unavailable.web_search",
        "unavailable.system_info",
        "unavailable.notes",
        "unavailable.local_file_write",
        "unavailable.session_memory_summary",
        "unavailable.local_file_actions_summary",
        "unavailable.shell_command_execution",
        "unavailable.any_non_listed_capability",
        "tool_mode.user_initiated",
        "guidance.shared_core_rules",
        "guidance.user_initiated.core_rules",
        "guidance.shared_tool_output_honesty",
        "guidance.unavailable_local_file_actions_warning",
    )


def test_resolve_builtin_capability_fragments_tracks_native_and_non_native_default_runtime(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        tool_registry = create_default_tool_registry(
            settings,
            session_manager=session_manager,
        )

        native_summary = build_runtime_capability_summary(
            tool_registry=tool_registry,
            memory_summary_available=True,
            model_can_call_tools=True,
        )
        native_ids = set(_fragment_ids(resolve_builtin_capability_fragments(native_summary)))

        assert {
            "available.local_file_read",
            "available.local_directory_listing",
            "available.url_fetch",
            "available.web_search",
            "available.system_info",
            "available.notes",
            "available.local_file_write",
            "available.local_file_delete",
            "available.local_file_move",
            "available.local_file_rename",
            "available.local_file_copy",
            "available.session_history_recall",
            "available.long_term_memory",
            "available.session_memory_summary",
            "tool_mode.model_callable",
            "guidance.model_callable.core_rules",
            "guidance.model_callable.local_choice.full",
            "guidance.model_callable.web_choice.full",
            "guidance.model_callable.session_history",
            "guidance.model_callable.system_info",
            "guidance.model_callable.long_term_memory",
        } <= native_ids
        assert "tool_mode.user_initiated" not in native_ids
        assert "guidance.user_initiated.core_rules" not in native_ids
        assert "unavailable.local_file_read" not in native_ids
        assert "unavailable.local_file_actions_summary" not in native_ids
        assert "guidance.unavailable_local_file_actions_warning" not in native_ids

        non_native_summary = build_runtime_capability_summary(
            tool_registry=tool_registry,
            memory_summary_available=True,
            model_can_call_tools=False,
        )
        non_native_ids = set(
            _fragment_ids(resolve_builtin_capability_fragments(non_native_summary))
        )

        assert {
            "available.local_file_read",
            "available.local_directory_listing",
            "available.url_fetch",
            "available.web_search",
            "available.system_info",
            "available.notes",
            "available.local_file_write",
            "available.local_file_delete",
            "available.local_file_move",
            "available.local_file_rename",
            "available.local_file_copy",
            "available.session_history_recall",
            "available.long_term_memory",
            "available.session_memory_summary",
            "tool_mode.user_initiated",
            "guidance.user_initiated.core_rules",
        } <= non_native_ids
        assert "tool_mode.model_callable" not in non_native_ids
        assert "guidance.model_callable.core_rules" not in non_native_ids
        assert "guidance.model_callable.local_choice.full" not in non_native_ids
        assert "guidance.model_callable.web_choice.full" not in non_native_ids
        assert "guidance.model_callable.session_history" not in non_native_ids
        assert "guidance.model_callable.system_info" not in non_native_ids
        assert "guidance.model_callable.long_term_memory" not in non_native_ids
    finally:
        session_manager.close()
