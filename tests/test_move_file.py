"""Targeted tests for move_file local file operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.executor import ToolExecutor
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolPermissionLevel
from unclaw.tools.file_tools import MOVE_FILE_DEFINITION, move_file

pytestmark = pytest.mark.unit


def _make_call(source_path: str, destination_path: str, **extra: object) -> ToolCall:
    return ToolCall(
        tool_name="move_file",
        arguments={
            "source_path": source_path,
            "destination_path": destination_path,
            **extra,
        },
    )


def test_move_file_permission_level_is_local_write() -> None:
    assert MOVE_FILE_DEFINITION.permission_level is ToolPermissionLevel.LOCAL_WRITE


def test_move_file_moves_file_within_allowed_roots(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "moved.txt"
    source.write_text("hello move", encoding="utf-8")

    result = move_file(_make_call(str(source), str(destination)), allowed_roots=(tmp_path,))

    assert result.success is True
    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "hello move"
    assert result.payload == {
        "source_path": str(source),
        "destination_path": str(destination),
        "overwrite": False,
    }


def test_move_file_rejects_missing_source(tmp_path: Path) -> None:
    source = tmp_path / "missing.txt"
    destination = tmp_path / "moved.txt"

    result = move_file(_make_call(str(source), str(destination)), allowed_roots=(tmp_path,))

    assert result.success is False
    assert result.error == f"Source file does not exist: {source}"
    assert not destination.exists()


def test_move_file_rejects_existing_destination_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "existing.txt"
    source.write_text("source", encoding="utf-8")
    destination.write_text("destination", encoding="utf-8")

    result = move_file(_make_call(str(source), str(destination)), allowed_roots=(tmp_path,))

    assert result.success is False
    assert (
        result.error
        == f"Destination file already exists: {destination}. Pass overwrite=true to replace it."
    )
    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "destination"


def test_move_file_overwrites_existing_destination_when_explicitly_allowed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "existing.txt"
    source.write_text("source", encoding="utf-8")
    destination.write_text("destination", encoding="utf-8")

    result = move_file(
        _make_call(str(source), str(destination), overwrite=True),
        allowed_roots=(tmp_path,),
    )

    assert result.success is True
    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "source"
    assert result.payload is not None
    assert result.payload["overwrite"] is True


def test_move_file_rejects_directory_sources(tmp_path: Path) -> None:
    source = tmp_path / "directory"
    destination = tmp_path / "moved.txt"
    source.mkdir()

    result = move_file(_make_call(str(source), str(destination)), allowed_roots=(tmp_path,))

    assert result.success is False
    assert result.error == f"Source path is not a file: {source}"


def test_move_file_rejects_source_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    source = tmp_path / "outside-source.txt"
    destination = allowed_root / "moved.txt"
    source.write_text("secret", encoding="utf-8")

    result = move_file(
        _make_call(str(source), str(destination)),
        allowed_roots=(allowed_root,),
    )

    assert result.success is False
    assert result.error is not None
    assert "outside the allowed local roots" in result.error
    assert source.exists()
    assert not destination.exists()


def test_move_file_rejects_destination_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    source = allowed_root / "source.txt"
    destination = tmp_path / "outside-destination.txt"
    source.write_text("secret", encoding="utf-8")

    result = move_file(
        _make_call(str(source), str(destination)),
        allowed_roots=(allowed_root,),
    )

    assert result.success is False
    assert result.error is not None
    assert "outside the allowed local roots" in result.error
    assert source.exists()
    assert not destination.exists()


def test_move_file_project_relative_paths_are_not_prefixed_twice(tmp_path: Path) -> None:
    files_dir = tmp_path / "data" / "files"
    files_dir.mkdir(parents=True)
    source = files_dir / "hello.txt"
    destination = tmp_path / "data" / "hello.txt"
    source.write_text("hello move", encoding="utf-8")

    result = move_file(
        _make_call("data/files/hello.txt", "data/hello.txt"),
        allowed_roots=(tmp_path,),
        default_read_dir=files_dir,
        default_write_dir=files_dir,
    )

    assert result.success is True
    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "hello move"
    assert result.payload == {
        "source_path": str(source),
        "destination_path": str(destination),
        "overwrite": False,
    }
    assert not (files_dir / "data" / "hello.txt").exists()


def test_move_file_dispatched_via_tool_executor_uses_default_files_dir(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    files_dir = settings.paths.files_dir
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / "archive").mkdir()
    (files_dir / "draft.txt").write_text("executor move", encoding="utf-8")
    executor = ToolExecutor.with_default_tools(settings)

    result = executor.execute(
        ToolCall(
            tool_name="move_file",
            arguments={
                "source_path": "draft.txt",
                "destination_path": "archive/final.txt",
            },
        )
    )

    assert result.success is True
    assert not (files_dir / "draft.txt").exists()
    assert (files_dir / "archive" / "final.txt").read_text(encoding="utf-8") == "executor move"
