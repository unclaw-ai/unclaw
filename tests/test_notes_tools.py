"""Targeted tests for the notes tool family — P3-2."""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.executor import create_default_tool_registry
from unclaw.tools.contracts import ToolCall, ToolPermissionLevel
from unclaw.tools.notes_tools import (
    CREATE_NOTE_DEFINITION,
    LIST_NOTES_DEFINITION,
    READ_NOTE_DEFINITION,
    UPDATE_NOTE_DEFINITION,
    _MAX_LIST_ENTRIES,
    _MAX_NOTE_CHARS,
    _MAX_NOTE_NAME_CHARS,
    create_note,
    list_notes,
    read_note,
    register_notes_tools,
    update_note,
)
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Permission level
# ---------------------------------------------------------------------------


def test_create_note_permission_level_is_local_write() -> None:
    assert CREATE_NOTE_DEFINITION.permission_level is ToolPermissionLevel.LOCAL_WRITE


def test_update_note_permission_level_is_local_write() -> None:
    assert UPDATE_NOTE_DEFINITION.permission_level is ToolPermissionLevel.LOCAL_WRITE


def test_read_note_permission_level_is_local_read() -> None:
    assert READ_NOTE_DEFINITION.permission_level is ToolPermissionLevel.LOCAL_READ


def test_list_notes_permission_level_is_local_read() -> None:
    assert LIST_NOTES_DEFINITION.permission_level is ToolPermissionLevel.LOCAL_READ


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_notes_tools_adds_all_four(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    registry = ToolRegistry()
    register_notes_tools(registry, notes_dir=notes_dir)

    for name in ("create_note", "read_note", "list_notes", "update_note"):
        registered = registry.get(name)
        assert registered is not None, f"Missing tool: {name}"
        assert registered.definition.name == name


def test_register_notes_tools_creates_notes_dir(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    assert not notes_dir.exists()
    registry = ToolRegistry()
    register_notes_tools(registry, notes_dir=notes_dir)
    assert notes_dir.exists()


def test_notes_tools_in_default_registry() -> None:
    registry = create_default_tool_registry()
    for name in ("create_note", "read_note", "list_notes", "update_note"):
        assert registry.get(name) is not None, f"Missing from default registry: {name}"


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_create_note_happy_path(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    call = ToolCall(
        tool_name="create_note",
        arguments={"title": "my-note", "content": "Hello world."},
    )
    result = create_note(call, notes_dir=notes_dir)
    assert result.success is True
    assert result.tool_name == "create_note"
    assert "my-note" in result.output_text
    assert (notes_dir / "my-note.md").read_text(encoding="utf-8") == "Hello world."


def test_read_note_happy_path(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "hello.md").write_text("content here", encoding="utf-8")

    call = ToolCall(tool_name="read_note", arguments={"title": "hello"})
    result = read_note(call, notes_dir=notes_dir)
    assert result.success is True
    assert "content here" in result.output_text
    assert result.payload is not None
    assert result.payload["title"] == "hello"


def test_list_notes_empty(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    call = ToolCall(tool_name="list_notes", arguments={})
    result = list_notes(call, notes_dir=notes_dir)
    assert result.success is True
    assert result.payload is not None
    assert result.payload["titles"] == []
    assert result.payload["count"] == 0


def test_list_notes_with_entries(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    for name in ("alpha", "beta", "gamma"):
        (notes_dir / f"{name}.md").write_text(f"{name} content", encoding="utf-8")

    call = ToolCall(tool_name="list_notes", arguments={})
    result = list_notes(call, notes_dir=notes_dir)
    assert result.success is True
    assert result.payload is not None
    assert set(result.payload["titles"]) == {"alpha", "beta", "gamma"}
    assert result.payload["count"] == 3
    assert result.payload["truncated"] is False


def test_update_note_creates_when_missing(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    call = ToolCall(
        tool_name="update_note",
        arguments={"title": "new-note", "content": "initial"},
    )
    result = update_note(call, notes_dir=notes_dir)
    assert result.success is True
    assert result.payload is not None
    assert result.payload["created"] is True
    assert (notes_dir / "new-note.md").read_text(encoding="utf-8") == "initial"


def test_update_note_overwrites_existing(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "existing.md").write_text("old content", encoding="utf-8")

    call = ToolCall(
        tool_name="update_note",
        arguments={"title": "existing", "content": "new content"},
    )
    result = update_note(call, notes_dir=notes_dir)
    assert result.success is True
    assert result.payload is not None
    assert result.payload["created"] is False
    assert (notes_dir / "existing.md").read_text(encoding="utf-8") == "new content"


def test_create_note_fails_on_duplicate(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "dup.md").write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="create_note",
        arguments={"title": "dup", "content": "new"},
    )
    result = create_note(call, notes_dir=notes_dir)
    assert result.success is False
    assert "already exists" in result.error  # type: ignore[operator]


def test_read_note_fails_on_missing(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    call = ToolCall(tool_name="read_note", arguments={"title": "ghost"})
    result = read_note(call, notes_dir=notes_dir)
    assert result.success is False
    assert "does not exist" in result.error  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Path traversal / note name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_title",
    [
        "../escape",
        "../../etc/passwd",
        "sub/dir",
        "sub\\dir",
        "/absolute",
        "title\x00null",
    ],
)
def test_create_note_rejects_path_traversal(tmp_path: Path, bad_title: str) -> None:
    notes_dir = tmp_path / "notes"
    call = ToolCall(
        tool_name="create_note",
        arguments={"title": bad_title, "content": "x"},
    )
    result = create_note(call, notes_dir=notes_dir)
    assert result.success is False


@pytest.mark.parametrize(
    "bad_title",
    [
        "../escape",
        "sub/dir",
        "sub\\dir",
    ],
)
def test_read_note_rejects_path_traversal(tmp_path: Path, bad_title: str) -> None:
    notes_dir = tmp_path / "notes"
    call = ToolCall(tool_name="read_note", arguments={"title": bad_title})
    result = read_note(call, notes_dir=notes_dir)
    assert result.success is False


def test_create_note_rejects_empty_title(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    call = ToolCall(
        tool_name="create_note",
        arguments={"title": "   ", "content": "x"},
    )
    result = create_note(call, notes_dir=notes_dir)
    assert result.success is False
    assert "non-empty" in result.error  # type: ignore[operator]


def test_create_note_rejects_title_too_long(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    long_title = "a" * (_MAX_NOTE_NAME_CHARS + 1)
    call = ToolCall(
        tool_name="create_note",
        arguments={"title": long_title, "content": "x"},
    )
    result = create_note(call, notes_dir=notes_dir)
    assert result.success is False
    assert str(_MAX_NOTE_NAME_CHARS) in result.error  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Size bounds
# ---------------------------------------------------------------------------


def test_create_note_rejects_content_too_large(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    big_content = "x" * (_MAX_NOTE_CHARS + 1)
    call = ToolCall(
        tool_name="create_note",
        arguments={"title": "big", "content": big_content},
    )
    result = create_note(call, notes_dir=notes_dir)
    assert result.success is False
    assert "maximum size" in result.error  # type: ignore[operator]


def test_update_note_rejects_content_too_large(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    big_content = "x" * (_MAX_NOTE_CHARS + 1)
    call = ToolCall(
        tool_name="update_note",
        arguments={"title": "big", "content": big_content},
    )
    result = update_note(call, notes_dir=notes_dir)
    assert result.success is False
    assert "maximum size" in result.error  # type: ignore[operator]


def test_list_notes_truncates_at_max_entries(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    for i in range(_MAX_LIST_ENTRIES + 5):
        (notes_dir / f"note-{i:04d}.md").write_text("x", encoding="utf-8")

    call = ToolCall(tool_name="list_notes", arguments={})
    result = list_notes(call, notes_dir=notes_dir)
    assert result.success is True
    assert result.payload is not None
    assert result.payload["truncated"] is True
    assert result.payload["count"] == _MAX_LIST_ENTRIES


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


def test_notes_tools_dispatched_via_registry_handlers(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    registry = ToolRegistry()
    register_notes_tools(registry, notes_dir=notes_dir)

    create_call = ToolCall(
        tool_name="create_note",
        arguments={"title": "dispatch-test", "content": "via handler"},
    )
    registered = registry.get("create_note")
    assert registered is not None
    result = registered.handler(create_call)
    assert result.success is True
    assert result.tool_name == "create_note"

    read_call = ToolCall(tool_name="read_note", arguments={"title": "dispatch-test"})
    registered = registry.get("read_note")
    assert registered is not None
    result = registered.handler(read_call)
    assert result.success is True
    assert "via handler" in result.output_text


# ---------------------------------------------------------------------------
# File permissions
# ---------------------------------------------------------------------------


def test_create_note_sets_0o600_permissions(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    call = ToolCall(
        tool_name="create_note",
        arguments={"title": "perms", "content": "check perms"},
    )
    create_note(call, notes_dir=notes_dir)
    note_path = notes_dir / "perms.md"
    mode = note_path.stat().st_mode & 0o777
    assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"
