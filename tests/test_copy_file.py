"""Targeted tests for copy_file local file operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.executor import ToolExecutor
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolPermissionLevel
from unclaw.tools.file_tools import COPY_FILE_DEFINITION, copy_file

pytestmark = pytest.mark.unit


def _make_call(source_path: str, destination_path: str, **extra: object) -> ToolCall:
    return ToolCall(
        tool_name="copy_file",
        arguments={
            "source_path": source_path,
            "destination_path": destination_path,
            **extra,
        },
    )


def test_copy_file_permission_level_is_local_write() -> None:
    assert COPY_FILE_DEFINITION.permission_level is ToolPermissionLevel.LOCAL_WRITE


def test_copy_file_copies_file_within_allowed_roots(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "copied.txt"
    source.write_text("hello copy", encoding="utf-8")

    result = copy_file(_make_call(str(source), str(destination)), allowed_roots=(tmp_path,))

    assert result.success is True
    assert source.read_text(encoding="utf-8") == "hello copy"
    assert destination.read_text(encoding="utf-8") == "hello copy"
    assert result.payload == {
        "source_path": str(source),
        "destination_path": str(destination),
        "overwrite": False,
    }


def test_copy_file_rejects_missing_source(tmp_path: Path) -> None:
    source = tmp_path / "missing.txt"
    destination = tmp_path / "copied.txt"

    result = copy_file(_make_call(str(source), str(destination)), allowed_roots=(tmp_path,))

    assert result.success is False
    assert result.error == f"Source file does not exist: {source}"
    assert not destination.exists()


def test_copy_file_rejects_existing_destination_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "existing.txt"
    source.write_text("source", encoding="utf-8")
    destination.write_text("destination", encoding="utf-8")

    result = copy_file(_make_call(str(source), str(destination)), allowed_roots=(tmp_path,))

    assert result.success is False
    assert (
        result.error
        == f"Destination file already exists: {destination}. Pass overwrite=true to replace it."
    )
    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "destination"


def test_copy_file_overwrites_existing_destination_when_explicitly_allowed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "existing.txt"
    source.write_text("source", encoding="utf-8")
    destination.write_text("destination", encoding="utf-8")

    result = copy_file(
        _make_call(str(source), str(destination), overwrite=True),
        allowed_roots=(tmp_path,),
    )

    assert result.success is True
    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "source"
    assert result.payload is not None
    assert result.payload["overwrite"] is True


def test_copy_file_rejects_directory_sources(tmp_path: Path) -> None:
    source = tmp_path / "directory"
    destination = tmp_path / "copied.txt"
    source.mkdir()

    result = copy_file(_make_call(str(source), str(destination)), allowed_roots=(tmp_path,))

    assert result.success is False
    assert result.error == f"Source path is not a file: {source}"


def test_copy_file_rejects_source_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    source = tmp_path / "outside-source.txt"
    destination = allowed_root / "copied.txt"
    source.write_text("secret", encoding="utf-8")

    result = copy_file(
        _make_call(str(source), str(destination)),
        allowed_roots=(allowed_root,),
    )

    assert result.success is False
    assert result.error is not None
    assert "outside the allowed local roots" in result.error
    assert source.exists()
    assert not destination.exists()


def test_copy_file_rejects_destination_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    source = allowed_root / "source.txt"
    destination = tmp_path / "outside-destination.txt"
    source.write_text("secret", encoding="utf-8")

    result = copy_file(
        _make_call(str(source), str(destination)),
        allowed_roots=(allowed_root,),
    )

    assert result.success is False
    assert result.error is not None
    assert "outside the allowed local roots" in result.error
    assert source.exists()
    assert not destination.exists()


def test_copy_file_dispatched_via_tool_executor_handles_project_relative_paths(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    files_dir = settings.paths.files_dir
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / "example.txt").write_text("executor copy", encoding="utf-8")
    executor = ToolExecutor.with_default_tools(settings)

    result = executor.execute(
        ToolCall(
            tool_name="copy_file",
            arguments={
                "source_path": "data/files/example.txt",
                "destination_path": "data/example-copy.txt",
            },
        )
    )

    destination = settings.paths.data_dir / "example-copy.txt"

    assert result.success is True
    assert (files_dir / "example.txt").read_text(encoding="utf-8") == "executor copy"
    assert destination.read_text(encoding="utf-8") == "executor copy"
    assert result.payload == {
        "source_path": str(files_dir / "example.txt"),
        "destination_path": str(destination),
        "overwrite": False,
    }

