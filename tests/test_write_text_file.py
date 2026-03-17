"""Targeted tests for write_text_file — P3-3."""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.executor import create_default_tool_registry
from unclaw.tools.contracts import ToolCall, ToolPermissionLevel
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
