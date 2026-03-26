"""Tests for write_text_file — collision policy and structured outcome."""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.executor import create_default_tool_registry
from unclaw.tools.contracts import ToolCall, ToolPermissionLevel, ToolResult
from unclaw.tools.dispatcher import ToolDispatcher
from unclaw.tools.file_tools import (
    WRITE_TEXT_FILE_DEFINITION,
    _MAX_WRITE_FILE_CHARS,
    _generate_versioned_path,
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


def test_write_text_file_payload_contains_structured_outcome_fields(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "abc"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    assert result.payload is not None
    assert result.payload["size_chars"] == 3
    assert result.payload["requested_path"] == str(target)
    assert result.payload["resolved_path"] == str(target)
    assert result.payload["created_new_file"] is True
    assert result.payload["created_versioned_file"] is False
    assert result.payload["overwrite_applied"] is False
    assert result.payload["file_already_exists"] is False
    assert result.payload["collision_policy_applied"] == "fail"


# ---------------------------------------------------------------------------
# Collision policy — fail (default)
# ---------------------------------------------------------------------------


def test_write_text_file_fails_by_default_when_file_exists(tmp_path: Path) -> None:
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


def test_write_text_file_fail_policy_returns_structured_outcome(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "new content", "collision_policy": "fail"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert result.failure_kind == "collision_conflict"
    assert result.payload is not None
    assert result.payload["file_already_exists"] is True
    assert result.payload["created_new_file"] is False
    assert result.payload["created_versioned_file"] is False
    assert result.payload["overwrite_applied"] is False
    assert result.payload["collision_policy_applied"] == "fail"
    assert result.payload["action_performed"] is False


def test_write_text_file_fail_refusal_includes_suggested_version_policy(tmp_path: Path) -> None:
    target = tmp_path / "marine_leleu.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "new"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert result.payload is not None
    assert result.payload["suggested_collision_policy"] == "version"


def test_write_text_file_fail_refusal_suggested_version_path_preserves_basename(
    tmp_path: Path,
) -> None:
    target = tmp_path / "marine_leleu.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "new"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.payload is not None
    suggested = Path(result.payload["suggested_version_path"])
    # Must be in the same directory
    assert suggested.parent == target.parent
    # Must keep the same extension
    assert suggested.suffix == ".txt"
    # Stem must start with the original stem
    assert suggested.stem.startswith("marine_leleu_")
    # Must not be the exact same path
    assert suggested != target


def test_write_text_file_fail_refusal_suggested_version_path_no_extension(
    tmp_path: Path,
) -> None:
    target = tmp_path / "report"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "new"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.payload is not None
    suggested = Path(result.payload["suggested_version_path"])
    assert suggested.parent == target.parent
    assert suggested.suffix == ""
    assert suggested.name.startswith("report_")


def test_write_text_file_overwrite_refusal_includes_suggested_version_fields(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "new", "collision_policy": "overwrite"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert result.payload is not None
    assert result.payload["suggested_collision_policy"] == "version"
    suggested = Path(result.payload["suggested_version_path"])
    assert suggested.parent == target.parent
    assert suggested.suffix == ".txt"
    assert suggested.stem.startswith("notes_")


# ---------------------------------------------------------------------------
# Collision policy — version
# ---------------------------------------------------------------------------


def test_write_text_file_version_policy_creates_new_file_when_target_exists(
    tmp_path: Path,
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "versioned", "collision_policy": "version"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    # Original file must be untouched
    assert target.read_text(encoding="utf-8") == "original"
    # A new versioned file must exist
    assert result.payload is not None
    resolved = Path(result.payload["resolved_path"])
    assert resolved != target
    assert resolved.exists()
    assert resolved.read_text(encoding="utf-8") == "versioned"


def test_write_text_file_version_policy_timestamped_name_format(tmp_path: Path) -> None:
    target = tmp_path / "report.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "v2", "collision_policy": "version"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    resolved = Path(result.payload["resolved_path"])  # type: ignore[index]
    # stem must be report_YYYYMMDD_HHMMSS
    assert resolved.stem.startswith("report_")
    assert resolved.suffix == ".txt"
    assert len(resolved.stem) == len("report_20260322_185430")


def test_write_text_file_version_policy_no_extension(tmp_path: Path) -> None:
    target = tmp_path / "report"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "v2", "collision_policy": "version"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    resolved = Path(result.payload["resolved_path"])  # type: ignore[index]
    assert resolved.suffix == ""
    assert resolved.name.startswith("report_")


def test_write_text_file_version_policy_structured_outcome(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "versioned", "collision_policy": "version"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    assert result.payload is not None
    assert result.payload["requested_path"] == str(target)
    assert result.payload["resolved_path"] != str(target)
    assert result.payload["created_new_file"] is True
    assert result.payload["created_versioned_file"] is True
    assert result.payload["overwrite_applied"] is False
    assert result.payload["file_already_exists"] is True
    assert result.payload["collision_policy_applied"] == "version"


def test_write_text_file_version_policy_new_file_writes_normally(tmp_path: Path) -> None:
    """version policy on a new file should just write it at the requested path."""
    target = tmp_path / "new.txt"

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "hello", "collision_policy": "version"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    assert result.payload is not None
    assert result.payload["resolved_path"] == str(target)
    assert result.payload["created_versioned_file"] is False


# ---------------------------------------------------------------------------
# Collision policy — overwrite (dev mode)
# ---------------------------------------------------------------------------


def test_write_text_file_overwrite_policy_refused_without_dev_mode(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "replaced", "collision_policy": "overwrite"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert "dev mode" in result.error  # type: ignore[operator]
    assert target.read_text(encoding="utf-8") == "original"


def test_write_text_file_overwrite_policy_refused_without_dev_mode_structured_outcome(
    tmp_path: Path,
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "replaced", "collision_policy": "overwrite"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.payload is not None
    assert result.payload["file_already_exists"] is True
    assert result.payload["overwrite_applied"] is False
    assert result.payload["collision_policy_applied"] == "overwrite"


def test_write_text_file_overwrite_policy_succeeds_with_dev_mode(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "replaced", "collision_policy": "overwrite"},
    )
    result = write_text_file(
        call,
        allowed_roots=(tmp_path,),
        allow_destructive_file_overwrite=True,
    )
    assert result.success is True
    assert target.read_text(encoding="utf-8") == "replaced"


def test_write_text_file_overwrite_policy_dev_mode_structured_outcome(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "replaced", "collision_policy": "overwrite"},
    )
    result = write_text_file(
        call,
        allowed_roots=(tmp_path,),
        allow_destructive_file_overwrite=True,
    )
    assert result.success is True
    assert result.payload is not None
    assert result.payload["requested_path"] == str(target)
    assert result.payload["resolved_path"] == str(target)
    assert result.payload["created_new_file"] is False
    assert result.payload["created_versioned_file"] is False
    assert result.payload["overwrite_applied"] is True
    assert result.payload["file_already_exists"] is True
    assert result.payload["collision_policy_applied"] == "overwrite"


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


def test_write_text_file_rejects_invalid_collision_policy(tmp_path: Path) -> None:
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(tmp_path / "f.txt"), "content": "x", "collision_policy": "clobber"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert "collision_policy" in result.error.lower()  # type: ignore[union-attr]


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


def test_write_text_file_dispatcher_recovers_text_alias_to_content(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_file_tools(registry, project_root=tmp_path)

    result = ToolDispatcher(registry).dispatch(
        ToolCall(
            tool_name="write_text_file",
            arguments={"path": str(tmp_path / "alias.txt"), "text": "aliased"},
        )
    )

    assert result.success is True
    assert (tmp_path / "alias.txt").read_text(encoding="utf-8") == "aliased"


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


def test_project_relative_path_is_not_prefixed_twice(tmp_path: Path) -> None:
    write_dir = tmp_path / "data" / "files"
    target = tmp_path / "data" / "hello.txt"
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": "data/hello.txt", "content": "explicit"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,), default_write_dir=write_dir)
    assert result.success is True
    assert target.read_text(encoding="utf-8") == "explicit"
    assert result.payload is not None
    assert result.payload["resolved_path"] == str(target)
    assert not (write_dir / "data" / "hello.txt").exists()


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

# (P3-4 overwrite intent guard and refusal reply tests removed — these
# deterministic runtime keyword behaviors were removed in the anti-determinism
# cleanup. Overwrite protection is enforced by the tool layer; the model
# synthesises replies when the tool returns a file-exists error.)
