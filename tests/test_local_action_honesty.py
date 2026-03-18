"""Targeted tests — local-action honesty corrective patch.

Proves:
1. The capability context prohibits claiming file write success without tool output
   (both in slash-command/json_plan mode and in model-callable/native mode).
2. The capability context explicitly lists delete/move/rename/copy as unavailable.
3. The capability context forbids suggesting delete/move/copy might work after confirmation.
4. The existing overwrite refusal short-circuit path still works (regression guard).
"""

from __future__ import annotations

import pytest

from unclaw.core.capabilities import (
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.runtime import _build_overwrite_refusal_reply
from unclaw.tools.contracts import ToolResult
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capability_context(*, model_can_call_tools: bool) -> str:
    """Build the full capability context string using all 10 registered tools."""
    registry = create_default_tool_registry()
    summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=model_can_call_tools,
    )
    return build_runtime_capability_context(summary)


def _empty_capability_context() -> str:
    """Build the capability context with no tools registered."""
    registry = ToolRegistry()
    summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=False,
    )
    return build_runtime_capability_context(summary)


# ---------------------------------------------------------------------------
# 1. Fake write-success prohibition — slash-command / json_plan mode
# ---------------------------------------------------------------------------


def test_capability_context_prohibits_claiming_write_in_slash_command_mode() -> None:
    """In non-callable mode the context must explicitly say write was not performed.

    This prevents a json_plan-mode model from saying 'done' or 'file modified'
    when no write_text_file call was ever dispatched.
    Required by the local-action honesty corrective patch.
    """
    context = _capability_context(model_can_call_tools=False)

    # Must say the action was not performed / nothing was written or changed.
    assert "You cannot write, modify, or create any file or note in this turn" in context
    # Must instruct model to say so honestly.
    assert "say the action was not performed" in context or "you have not written" in context


def test_capability_context_covers_all_state_changing_local_actions_in_shared_rules() -> None:
    """The shared 'do not claim' rule must cover write/create/modify/delete, not only read.

    Applies to both slash-command and native modes.
    """
    for callable_tools in (False, True):
        context = _capability_context(model_can_call_tools=callable_tools)
        # Original rule only covered searched/fetched/read — now must include write.
        assert "wrote" in context, (
            f"'wrote' missing from capability context (model_can_call_tools={callable_tools})"
        )
        assert "created" in context, (
            f"'created' missing from capability context (model_can_call_tools={callable_tools})"
        )
        assert "modified" in context, (
            f"'modified' missing from capability context (model_can_call_tools={callable_tools})"
        )


# ---------------------------------------------------------------------------
# 2. Fake write-success prohibition — native / model-callable mode
# ---------------------------------------------------------------------------


def test_capability_context_requires_tool_output_before_claiming_write_in_native_mode() -> None:
    """In native/callable mode the context must say only claim write if tool output exists.

    This prevents a native-mode model from producing 'I have written the file'
    when it generated a text reply without invoking write_text_file.
    Required by the local-action honesty corrective patch.
    """
    context = _capability_context(model_can_call_tools=True)

    # Must refer to the tool output being required before claiming success.
    assert "Only claim a file was written, created, or modified" in context
    # Must mention the relevant tools.
    assert "write_text_file" in context
    assert "create_note" in context or "update_note" in context
    # Must instruct model to say action has not happened if tool did not run.
    assert "has not happened yet" in context


# ---------------------------------------------------------------------------
# 3. Delete / move / rename / copy explicitly listed as unavailable
# ---------------------------------------------------------------------------


def test_capability_context_explicitly_lists_delete_move_copy_as_unavailable() -> None:
    """Delete, move, rename, copy must be explicitly in the unavailable section.

    The generic 'Any capability not listed above' catch-all is too weak for
    small models. The capability must be named explicitly.
    Required by the local-action honesty corrective patch.
    """
    # Check with all tools registered.
    context_full = _capability_context(model_can_call_tools=False)
    assert "Delete, move, rename, or copy local files or directories" in context_full

    # Check with no tools registered (empty registry).
    context_empty = _empty_capability_context()
    assert "Delete, move, rename, or copy local files or directories" in context_empty


def test_capability_context_delete_explicit_in_native_mode_too() -> None:
    """Delete/move/copy must be explicitly unavailable regardless of tool mode."""
    context = _capability_context(model_can_call_tools=True)
    assert "Delete, move, rename, or copy local files or directories" in context


# ---------------------------------------------------------------------------
# 4. No 'might work after confirmation' suggestion for unavailable actions
# ---------------------------------------------------------------------------


def test_capability_context_forbids_suggesting_delete_after_confirmation() -> None:
    """The context must explicitly say: do not suggest delete might work after confirmation.

    This closes the observed bug where the model said 'deletion might be possible
    with confirmation' when no delete tool exists.
    Required by the local-action honesty corrective patch.
    """
    context = _capability_context(model_can_call_tools=False)
    # Must tell model NOT to suggest the capability might work after confirmation.
    assert "Do not suggest it might work after confirmation" in context


# ---------------------------------------------------------------------------
# 5. Existing overwrite refusal path still works (regression guard)
# ---------------------------------------------------------------------------


def test_overwrite_refusal_short_circuit_still_fires_on_file_exists() -> None:
    """The P3-4 overwrite refusal path must not be broken by this patch.

    Regression guard: _build_overwrite_refusal_reply must still return a
    deterministic reply for the 'already exists' write_text_file failure.
    """
    result = ToolResult.failure(
        tool_name="write_text_file",
        error="File already exists: /tmp/test.txt. Pass overwrite=true to replace it.",
    )
    reply = _build_overwrite_refusal_reply((result,))
    assert reply is not None
    assert "already exists" in reply
    assert "not overwritten" in reply.lower()
    assert "confirm" in reply.lower() or "replace" in reply.lower()


def test_overwrite_refusal_returns_none_for_other_failures() -> None:
    """Non-overwrite write failures must not trigger the short-circuit."""
    result = ToolResult.failure(
        tool_name="write_text_file",
        error="Access to '/etc/passwd' is outside the allowed local roots.",
    )
    assert _build_overwrite_refusal_reply((result,)) is None


def test_overwrite_refusal_returns_none_for_empty_results() -> None:
    """Empty result set must not crash or short-circuit."""
    assert _build_overwrite_refusal_reply(()) is None
