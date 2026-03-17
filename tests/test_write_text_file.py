"""Targeted tests for write_text_file — P3-3 and P3-4 corrective patch."""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.executor import create_default_tool_registry
from unclaw.core.runtime import (
    _build_overwrite_refusal_reply,
    _guard_write_overwrite_intent,
    _has_explicit_overwrite_intent,
)
from unclaw.tools.contracts import ToolCall, ToolPermissionLevel, ToolResult
from unclaw.tools.file_tools import (
    WRITE_TEXT_FILE_DEFINITION,
    _MAX_WRITE_FILE_CHARS,
    register_file_tools,
    write_text_file,
)
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Honest permission classification
# ---------------------------------------------------------------------------


def test_write_text_file_permission_level_is_local_write() -> None:
    assert WRITE_TEXT_FILE_DEFINITION.permission_level is ToolPermissionLevel.LOCAL_WRITE


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_write_text_file_registered_in_default_registry() -> None:
    registry = create_default_tool_registry()
    assert registry.get("write_text_file") is not None


def test_register_file_tools_includes_write_text_file(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_file_tools(registry, project_root=tmp_path)
    assert registry.get("write_text_file") is not None


# ---------------------------------------------------------------------------
# Write success in allowed area
# ---------------------------------------------------------------------------


def test_write_text_file_creates_new_file(tmp_path: Path) -> None:
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(tmp_path / "output.txt"), "content": "hello world"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "hello world"


def test_write_text_file_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "dir" / "file.txt"
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "nested"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    assert target.read_text(encoding="utf-8") == "nested"


def test_write_text_file_sets_0o600_permissions(tmp_path: Path) -> None:
    target = tmp_path / "secret.txt"
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "private"},
    )
    write_text_file(call, allowed_roots=(tmp_path,))
    mode = target.stat().st_mode & 0o777
    assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


def test_write_text_file_payload_contains_path_and_size(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "abc"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    assert result.payload is not None
    assert result.payload["size_chars"] == 3
    assert result.payload["overwrite"] is False


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------


def test_write_text_file_fails_without_overwrite_flag(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "new content"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert "already exists" in result.error  # type: ignore[operator]
    # Original content must be untouched
    assert target.read_text(encoding="utf-8") == "original"


def test_write_text_file_overwrites_with_explicit_flag(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "replaced", "overwrite": True},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    assert target.read_text(encoding="utf-8") == "replaced"


# ---------------------------------------------------------------------------
# Size limit enforcement
# ---------------------------------------------------------------------------


def test_write_text_file_rejects_content_above_size_limit(tmp_path: Path) -> None:
    big_content = "x" * (_MAX_WRITE_FILE_CHARS + 1)
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(tmp_path / "big.txt"), "content": big_content},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert "maximum" in result.error  # type: ignore[operator]
    assert not (tmp_path / "big.txt").exists()


def test_write_text_file_accepts_content_at_size_limit(tmp_path: Path) -> None:
    exact_content = "x" * _MAX_WRITE_FILE_CHARS
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(tmp_path / "exact.txt"), "content": exact_content},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True


# ---------------------------------------------------------------------------
# Path traversal and out-of-scope path blocking
# ---------------------------------------------------------------------------


def test_write_text_file_blocks_path_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside" / "file.txt"

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(outside), "content": "bad"},
    )
    result = write_text_file(call, allowed_roots=(allowed,))
    assert result.success is False
    assert not outside.exists()


def test_write_text_file_blocks_path_traversal_via_dotdot(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    traversal = str(allowed / ".." / "escape.txt")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": traversal, "content": "bad"},
    )
    result = write_text_file(call, allowed_roots=(allowed,))
    assert result.success is False


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_write_text_file_rejects_empty_path(tmp_path: Path) -> None:
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "   ", "content": "x"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert "path" in result.error.lower()  # type: ignore[union-attr]


def test_write_text_file_rejects_non_string_content(tmp_path: Path) -> None:
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(tmp_path / "f.txt"), "content": 123},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert "content" in result.error.lower()  # type: ignore[union-attr]


def test_write_text_file_rejects_non_bool_overwrite(tmp_path: Path) -> None:
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(tmp_path / "f.txt"), "content": "x", "overwrite": "yes"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert "overwrite" in result.error.lower()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


def test_write_text_file_dispatched_via_registry_handler(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_file_tools(registry, project_root=tmp_path)

    registered = registry.get("write_text_file")
    assert registered is not None

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(tmp_path / "via-handler.txt"), "content": "dispatched"},
    )
    result = registered.handler(call)
    assert result.success is True
    assert (tmp_path / "via-handler.txt").read_text(encoding="utf-8") == "dispatched"


# ---------------------------------------------------------------------------
# P3-4 corrective: default_write_dir — relative paths go to data/files/
# ---------------------------------------------------------------------------


def test_relative_path_writes_to_default_write_dir(tmp_path: Path) -> None:
    """Relative path 'hello.txt' must land in default_write_dir/hello.txt."""
    write_dir = tmp_path / "files"
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "hello.txt", "content": "hi"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,), default_write_dir=write_dir)
    assert result.success is True
    assert (write_dir / "hello.txt").read_text(encoding="utf-8") == "hi"
    # Must NOT land at tmp_path/hello.txt
    assert not (tmp_path / "hello.txt").exists()


def test_nested_relative_path_writes_under_default_write_dir(tmp_path: Path) -> None:
    """Nested relative path 'drafts/hello.txt' resolves under default_write_dir."""
    write_dir = tmp_path / "files"
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "drafts/hello.txt", "content": "draft"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,), default_write_dir=write_dir)
    assert result.success is True
    assert (write_dir / "drafts" / "hello.txt").read_text(encoding="utf-8") == "draft"


def test_absolute_path_bypasses_default_write_dir(tmp_path: Path) -> None:
    """Absolute allowed path must bypass the default_write_dir redirect."""
    write_dir = tmp_path / "files"
    target = tmp_path / "explicit.txt"
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "explicit"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,), default_write_dir=write_dir)
    assert result.success is True
    assert target.read_text(encoding="utf-8") == "explicit"
    # Must NOT be in files/
    assert not (write_dir / "explicit.txt").exists()


def test_relative_path_respects_allowed_roots_traversal_block(tmp_path: Path) -> None:
    """A relative path trying to escape write_dir via .. must be blocked."""
    write_dir = tmp_path / "files"
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "../../escape.txt", "content": "bad"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,), default_write_dir=write_dir)
    assert result.success is False


def test_register_file_tools_passes_default_write_dir_to_handler(tmp_path: Path) -> None:
    """register_file_tools with default_write_dir wires the redirect into the handler."""
    write_dir = tmp_path / "files"
    registry = ToolRegistry()
    register_file_tools(registry, project_root=tmp_path, default_write_dir=write_dir)

    registered = registry.get("write_text_file")
    assert registered is not None

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "via-handler.txt", "content": "via handler"},
    )
    result = registered.handler(call)
    assert result.success is True
    assert (write_dir / "via-handler.txt").read_text(encoding="utf-8") == "via handler"


# ---------------------------------------------------------------------------
# P3-4 corrective: overwrite intent guard
# ---------------------------------------------------------------------------


def test_has_explicit_overwrite_intent_detects_english_keywords() -> None:
    assert _has_explicit_overwrite_intent("please overwrite the file")
    assert _has_explicit_overwrite_intent("replace existing content")
    assert _has_explicit_overwrite_intent("Overwrite it")


def test_has_explicit_overwrite_intent_detects_french_keywords() -> None:
    assert _has_explicit_overwrite_intent("écrase ce fichier")
    assert _has_explicit_overwrite_intent("tu peux écraser")
    assert _has_explicit_overwrite_intent("remplace le fichier")
    assert _has_explicit_overwrite_intent("remplacer ce contenu")


def test_has_explicit_overwrite_intent_returns_false_for_neutral_input() -> None:
    assert not _has_explicit_overwrite_intent("write hello.txt with 'hi'")
    assert not _has_explicit_overwrite_intent("create a note about my tasks")
    assert not _has_explicit_overwrite_intent("save my draft to drafts/draft.txt")


def test_guard_strips_overwrite_when_no_intent(tmp_path: Path) -> None:
    """overwrite=True is stripped when user_input contains no overwrite intent."""
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "file.txt", "content": "hello", "overwrite": True},
    )
    guarded = _guard_write_overwrite_intent([call], user_input="write file.txt")
    assert len(guarded) == 1
    assert guarded[0].arguments.get("overwrite") is None


def test_guard_passes_overwrite_through_when_intent_present(tmp_path: Path) -> None:
    """overwrite=True is preserved when user_input contains explicit intent."""
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "file.txt", "content": "new", "overwrite": True},
    )
    guarded = _guard_write_overwrite_intent([call], user_input="please overwrite the file")
    assert len(guarded) == 1
    assert guarded[0].arguments.get("overwrite") is True


def test_guard_strips_string_true_overwrite_when_no_intent() -> None:
    """overwrite='true' (string from LLM) is also stripped when no intent."""
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "file.txt", "content": "hello", "overwrite": "true"},
    )
    guarded = _guard_write_overwrite_intent([call], user_input="write file.txt")
    assert len(guarded) == 1
    assert guarded[0].arguments.get("overwrite") is None


def test_guard_does_not_affect_other_tools() -> None:
    """The guard only touches write_text_file; other tools pass through unchanged."""
    call = ToolCall(
        tool_name="search_web",
        arguments={"query": "hello", "overwrite": True},
    )
    guarded = _guard_write_overwrite_intent([call], user_input="search something")
    assert len(guarded) == 1
    assert guarded[0].arguments.get("overwrite") is True


def test_existing_file_not_overwritten_without_intent(tmp_path: Path) -> None:
    """End-to-end: existing file + overwrite=True + no intent → tool fails."""
    write_dir = tmp_path / "files"
    write_dir.mkdir()
    existing = write_dir / "file.txt"
    existing.write_text("original", encoding="utf-8")

    # Guard strips overwrite, then tool returns "already exists" error.
    call_with_overwrite = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "file.txt", "content": "new", "overwrite": True},
    )
    guarded = _guard_write_overwrite_intent(
        [call_with_overwrite], user_input="update file.txt with new content"
    )
    guarded_call = guarded[0]
    result = write_text_file(
        guarded_call, allowed_roots=(tmp_path,), default_write_dir=write_dir
    )
    assert result.success is False
    assert "already exists" in result.error  # type: ignore[operator]
    assert existing.read_text(encoding="utf-8") == "original"


def test_existing_file_overwritten_with_explicit_intent(tmp_path: Path) -> None:
    """End-to-end: existing file + overwrite=True + explicit intent → succeeds."""
    write_dir = tmp_path / "files"
    write_dir.mkdir()
    existing = write_dir / "file.txt"
    existing.write_text("original", encoding="utf-8")

    call_with_overwrite = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "file.txt", "content": "replaced", "overwrite": True},
    )
    guarded = _guard_write_overwrite_intent(
        [call_with_overwrite], user_input="please overwrite the file with new content"
    )
    guarded_call = guarded[0]
    result = write_text_file(
        guarded_call, allowed_roots=(tmp_path,), default_write_dir=write_dir
    )
    assert result.success is True
    assert existing.read_text(encoding="utf-8") == "replaced"


# ---------------------------------------------------------------------------
# P3-4 corrective follow-up: _build_overwrite_refusal_reply
# ---------------------------------------------------------------------------


def test_build_overwrite_refusal_reply_fires_on_file_exists_failure() -> None:
    """Returns a deterministic reply for the specific 'already exists' failure."""
    result = ToolResult.failure(
        tool_name="write_text_file",
        error="File already exists: /tmp/hello.txt. Pass overwrite=true to replace it.",
    )
    reply = _build_overwrite_refusal_reply((result,))
    assert reply is not None
    assert "already exists" in reply
    assert "not overwritten" in reply.lower()
    assert "confirm" in reply.lower() or "replace" in reply.lower()


def test_build_overwrite_refusal_reply_includes_error_details() -> None:
    """The reply must surface the tool error text so the user sees the path."""
    target_path = "/home/user/data/files/notes.txt"
    error = f"File already exists: {target_path}. Pass overwrite=true to replace it."
    result = ToolResult.failure(tool_name="write_text_file", error=error)
    reply = _build_overwrite_refusal_reply((result,))
    assert reply is not None
    assert target_path in reply


def test_build_overwrite_refusal_reply_returns_none_for_other_write_failures() -> None:
    """Non-'already exists' write failures do not trigger the short-circuit."""
    result = ToolResult.failure(
        tool_name="write_text_file",
        error="Access to '/etc/passwd' is outside the allowed local roots.",
    )
    assert _build_overwrite_refusal_reply((result,)) is None


def test_build_overwrite_refusal_reply_returns_none_for_write_success() -> None:
    """Successful writes must not trigger the short-circuit."""
    result = ToolResult.ok(
        tool_name="write_text_file",
        output_text="File written: /tmp/new.txt",
    )
    assert _build_overwrite_refusal_reply((result,)) is None


def test_build_overwrite_refusal_reply_returns_none_for_other_tools() -> None:
    """Only write_text_file triggers the short-circuit, not other tools."""
    result = ToolResult.failure(
        tool_name="read_text_file",
        error="File already exists: /tmp/hello.txt.",
    )
    assert _build_overwrite_refusal_reply((result,)) is None


def test_build_overwrite_refusal_reply_returns_none_for_empty_results() -> None:
    """Empty tool result set must return None, not crash."""
    assert _build_overwrite_refusal_reply(()) is None


def test_build_overwrite_refusal_reply_file_content_untouched(tmp_path: Path) -> None:
    """Refusal path: the original file content is never mutated."""
    target = tmp_path / "hello.txt"
    target.write_text("original", encoding="utf-8")

    # Simulate the tool result that fires when overwrite protection activates.
    result = ToolResult.failure(
        tool_name="write_text_file",
        error=f"File already exists: {target}. Pass overwrite=true to replace it.",
    )
    reply = _build_overwrite_refusal_reply((result,))
    assert reply is not None
    # The guard operates only on ToolResult; the real file must be untouched.
    assert target.read_text(encoding="utf-8") == "original"
