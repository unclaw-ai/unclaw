"""Targeted tests for P3-4 — capability context for local actions.

Proves:
- removed notes tools stay absent from the runtime capability context
- write_text_file appears in the capability context when it is registered
- write_text_file description explicitly states overwrite=false as the default
- system_info appears in the capability context (regression guard)
- build_runtime_capability_summary populates local_file_write_available
"""

from __future__ import annotations

import pytest

from unclaw.core.capabilities import (
    RuntimeCapabilitySummary,
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.executor import create_default_tool_registry
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_capability_summary(*, model_can_call_tools: bool = False) -> RuntimeCapabilitySummary:
    """Build a summary using the default tool registry."""
    registry = create_default_tool_registry()
    return build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=model_can_call_tools,
    )


def _empty_capability_summary() -> RuntimeCapabilitySummary:
    """Build a summary from an empty registry (no tools registered)."""
    registry = ToolRegistry()
    return build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=False,
    )


# ---------------------------------------------------------------------------
# P3-4: local_file_write_available flag and removed notes runtime surface
# ---------------------------------------------------------------------------


def test_local_file_write_available_true_when_write_tool_registered() -> None:
    summary = _full_capability_summary()
    assert summary.local_file_write_available is True


def test_local_file_write_available_false_when_no_tools_registered() -> None:
    summary = _empty_capability_summary()
    assert summary.local_file_write_available is False


# ---------------------------------------------------------------------------
# P3-4: removed notes tools are no longer exposed
# ---------------------------------------------------------------------------


def test_default_registry_does_not_register_removed_notes_tools() -> None:
    registry = create_default_tool_registry()
    for name in ("create_note", "read_note", "list_notes", "update_note"):
        assert registry.get(name) is None


def test_capability_context_does_not_advertise_removed_notes_tools() -> None:
    for summary in (_full_capability_summary(), _empty_capability_summary()):
        context = build_runtime_capability_context(summary)
        assert "create_note" not in context
        assert "read_note" not in context
        assert "list_notes" not in context
        assert "update_note" not in context
        assert "Local notes" not in context


# ---------------------------------------------------------------------------
# P3-4: capability context includes write_text_file when available
# ---------------------------------------------------------------------------


def test_capability_context_includes_write_text_file_when_available() -> None:
    summary = _full_capability_summary()
    context = build_runtime_capability_context(summary)
    assert "write_text_file" in context


def test_capability_context_write_text_file_states_overwrite_false_default() -> None:
    """The write_text_file capability line must explicitly state overwrite=false default.

    This is the key P3-4 honesty requirement: the model must not assume overwrite=true
    is safe without explicit user intent.
    """
    summary = _full_capability_summary()
    context = build_runtime_capability_context(summary)
    # The context must mention that overwrite defaults to false
    assert "overwrite=false" in context
    # And that overwrite=true requires explicit user intent
    assert "explicitly" in context


def test_capability_context_write_text_file_omitted_from_tools_section_when_unavailable() -> None:
    summary = _empty_capability_summary()
    context = build_runtime_capability_context(summary)
    # listed as unavailable
    assert "write_text_file" in context
    assert "Local file write" in context


# ---------------------------------------------------------------------------
# P3-4: system_info still appears (regression guard)
# ---------------------------------------------------------------------------


def test_capability_context_includes_system_info_when_available() -> None:
    summary = _full_capability_summary()
    context = build_runtime_capability_context(summary)
    assert "system_info" in context
