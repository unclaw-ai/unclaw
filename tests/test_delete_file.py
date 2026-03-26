"""Targeted tests for delete_file local file operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.executor import ToolExecutor, create_default_tool_registry
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolPermissionLevel
from unclaw.tools.file_tools import DELETE_FILE_DEFINITION, delete_file

pytestmark = pytest.mark.unit


def _make_call(path: str, **extra: object) -> ToolCall:
    return ToolCall(
        tool_name="delete_file",
        arguments={
            "path": path,
            **extra,
        },
    )


def test_delete_file_permission_level_is_local_write() -> None:
    assert DELETE_FILE_DEFINITION.permission_level is ToolPermissionLevel.LOCAL_WRITE


def test_delete_file_registered_in_default_registry() -> None:
    registry = create_default_tool_registry()
    assert registry.get("delete_file") is not None


def test_delete_file_deletes_file_with_explicit_confirmation(tmp_path: Path) -> None:
    target = tmp_path / "remove-me.txt"
    target.write_text("bye", encoding="utf-8")

    result = delete_file(
        _make_call(str(target), confirm=True),
        allowed_roots=(tmp_path,),
    )

    assert result.success is True
    assert not target.exists()
    assert result.payload == {
        "path": str(target),
        "confirmed": True,
        "action_performed": True,
    }


def test_delete_file_missing_confirmation_blocks_deletion(tmp_path: Path) -> None:
    target = tmp_path / "keep-me.txt"
    target.write_text("stay", encoding="utf-8")

    result = delete_file(_make_call(str(target)), allowed_roots=(tmp_path,))

    assert result.success is False
    assert result.failure_kind == "confirmation_required"
    assert result.error == "Deletion was not performed. Pass confirm=true to delete the file."
    assert result.payload == {
        "requested_path": str(target),
        "resolved_path": str(target),
        "confirmed": False,
        "confirm_required": True,
        "action_performed": False,
    }
    assert target.exists()


def test_delete_file_rejects_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "missing.txt"

    result = delete_file(
        _make_call(str(target), confirm=True),
        allowed_roots=(tmp_path,),
    )

    assert result.success is False
    assert result.error == f"File does not exist: {target}"


def test_delete_file_rejects_directory_target(tmp_path: Path) -> None:
    target = tmp_path / "directory"
    target.mkdir()

    result = delete_file(
        _make_call(str(target), confirm=True),
        allowed_roots=(tmp_path,),
    )

    assert result.success is False
    assert result.error == f"Path is not a file: {target}"
    assert target.exists()


def test_delete_file_rejects_path_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")

    result = delete_file(
        _make_call(str(outside_file), confirm=True),
        allowed_roots=(allowed_root,),
    )

    assert result.success is False
    assert result.error is not None
    assert "outside the allowed local roots" in result.error
    assert outside_file.exists()


def test_delete_file_dispatched_via_tool_executor_uses_default_files_dir(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    files_dir = settings.paths.files_dir
    files_dir.mkdir(parents=True, exist_ok=True)
    target = files_dir / "draft.txt"
    target.write_text("executor delete", encoding="utf-8")
    executor = ToolExecutor.with_default_tools(settings)

    result = executor.execute(
        ToolCall(
            tool_name="delete_file",
            arguments={
                "path": "draft.txt",
                "confirm": True,
            },
        )
    )

    assert result.success is True
    assert not target.exists()
    assert result.payload == {
        "path": str(target),
        "confirmed": True,
        "action_performed": True,
    }
