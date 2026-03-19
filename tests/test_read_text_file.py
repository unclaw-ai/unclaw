"""Targeted tests for P4-1: V1 document-reading scope restriction.

Verifies:
- txt / md / json / csv are accepted and read correctly
- pdf / docx / xlsx and files without extension are rejected with an
  explicit V1 unsupported-format error
- The capability context string is honest about the V1 scope
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.capabilities import (
    RuntimeCapabilitySummary,
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.executor import create_default_tool_registry
from unclaw.tools.contracts import ToolCall
from unclaw.tools.file_tools import (
    READABLE_EXTENSIONS,
    read_text_file,
    register_file_tools,
)
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call(path: str, **extra) -> ToolCall:
    return ToolCall(tool_name="read_text_file", arguments={"path": path, **extra})


def _registry_with_root(root: Path) -> ToolRegistry:
    registry = ToolRegistry()
    register_file_tools(registry, project_root=root)
    return registry


# ---------------------------------------------------------------------------
# 1. READABLE_EXTENSIONS constant is exactly the V1 scope
# ---------------------------------------------------------------------------


def test_readable_extensions_constant() -> None:
    assert READABLE_EXTENSIONS == frozenset({".txt", ".md", ".json", ".csv"})


# ---------------------------------------------------------------------------
# 2. Supported formats — still read correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extension", [".txt", ".md", ".json", ".csv"])
def test_supported_extensions_read_ok(tmp_path: Path, extension: str) -> None:
    target = tmp_path / f"sample{extension}"
    target.write_text("hello world", encoding="utf-8")

    result = read_text_file(_make_call(str(target)), allowed_roots=(tmp_path,))

    assert result.success, f"Expected success for {extension}, got: {result.error}"
    assert "hello world" in result.output_text


def test_txt_content_returned_correctly(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("line1\nline2\nline3", encoding="utf-8")

    result = read_text_file(_make_call(str(target)), allowed_roots=(tmp_path,))

    assert result.success
    assert "line1" in result.output_text
    assert "line2" in result.output_text


def test_json_content_returned_correctly(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text('{"key": "value"}', encoding="utf-8")

    result = read_text_file(_make_call(str(target)), allowed_roots=(tmp_path,))

    assert result.success
    assert '"key"' in result.output_text


def test_csv_content_returned_correctly(tmp_path: Path) -> None:
    target = tmp_path / "table.csv"
    target.write_text("a,b,c\n1,2,3", encoding="utf-8")

    result = read_text_file(_make_call(str(target)), allowed_roots=(tmp_path,))

    assert result.success
    assert "a,b,c" in result.output_text


# ---------------------------------------------------------------------------
# 3. Unsupported formats — explicit V1 rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extension", [".pdf", ".docx", ".xlsx"])
def test_unsupported_binary_formats_rejected(tmp_path: Path, extension: str) -> None:
    # File exists on disk but has unsupported extension
    target = tmp_path / f"document{extension}"
    target.write_bytes(b"fake binary content")

    result = read_text_file(_make_call(str(target)), allowed_roots=(tmp_path,))

    assert not result.success
    assert "not supported" in result.error.lower()
    assert "V1" in result.error
    # The error must name the extension the user tried
    assert extension in result.error


def test_unsupported_format_error_lists_supported_types(tmp_path: Path) -> None:
    target = tmp_path / "report.pdf"
    target.write_bytes(b"%PDF fake")

    result = read_text_file(_make_call(str(target)), allowed_roots=(tmp_path,))

    assert not result.success
    # Supported formats must be mentioned in the error
    for ext in (".csv", ".json", ".md", ".txt"):
        assert ext in result.error, f"Expected '{ext}' in error: {result.error}"


def test_no_extension_rejected(tmp_path: Path) -> None:
    target = tmp_path / "Makefile"
    target.write_text("all:\n\techo hi", encoding="utf-8")

    result = read_text_file(_make_call(str(target)), allowed_roots=(tmp_path,))

    assert not result.success
    assert "not supported" in result.error.lower()
    assert "V1" in result.error


@pytest.mark.parametrize("extension", [".odt", ".rtf", ".html", ".py", ".bin"])
def test_other_unsupported_extensions_rejected(tmp_path: Path, extension: str) -> None:
    target = tmp_path / f"file{extension}"
    target.write_text("some content", encoding="utf-8")

    result = read_text_file(_make_call(str(target)), allowed_roots=(tmp_path,))

    assert not result.success
    assert "not supported" in result.error.lower()


# ---------------------------------------------------------------------------
# 4. Extension check fires before existence check (no path disclosure)
# ---------------------------------------------------------------------------


def test_unsupported_extension_nonexistent_file_rejected_before_existence_check(
    tmp_path: Path,
) -> None:
    """The extension error must surface even when the file does not exist.

    This prevents the tool from disclosing whether a non-readable file path
    exists on disk.
    """
    nonexistent_pdf = tmp_path / "ghost.pdf"
    assert not nonexistent_pdf.exists()

    result = read_text_file(_make_call(str(nonexistent_pdf)), allowed_roots=(tmp_path,))

    assert not result.success
    assert "not supported" in result.error.lower()
    assert "V1" in result.error


# ---------------------------------------------------------------------------
# 5. Capability context string is honest about V1 scope
# ---------------------------------------------------------------------------


def _make_full_summary(*, read_available: bool = True) -> RuntimeCapabilitySummary:
    return RuntimeCapabilitySummary(
        available_builtin_tool_names=("read_text_file",) if read_available else (),
        local_file_read_available=read_available,
        local_directory_listing_available=False,
        url_fetch_available=False,
        web_search_available=False,
        system_info_available=False,
        memory_summary_available=False,
        model_can_call_tools=False,
        notes_available=False,
        local_file_write_available=False,
    )


def test_capability_context_mentions_v1_read_scope() -> None:
    context = build_runtime_capability_context(_make_full_summary(read_available=True))
    # Must explicitly mention the V1 limitation
    assert "V1" in context
    # Must list at least the supported formats
    assert ".txt" in context or "txt" in context
    assert ".md" in context or "md" in context


def test_capability_context_mentions_unsupported_formats() -> None:
    context = build_runtime_capability_context(_make_full_summary(read_available=True))
    # Users must be told some formats are NOT supported
    assert "pdf" in context.lower() or "not supported" in context.lower()


# ---------------------------------------------------------------------------
# 6. Default registry still includes read_text_file
# ---------------------------------------------------------------------------


def test_read_text_file_in_default_registry() -> None:
    registry = create_default_tool_registry()
    assert registry.get("read_text_file") is not None


# ---------------------------------------------------------------------------
# 7. default_read_dir: relative paths resolve into data/files/ by default
# ---------------------------------------------------------------------------


def test_relative_read_resolves_into_default_read_dir(tmp_path: Path) -> None:
    """Relative filename resolves inside default_read_dir, not CWD."""
    files_dir = tmp_path / "data" / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "hello.txt").write_text("hello world", encoding="utf-8")

    result = read_text_file(
        _make_call("hello.txt"),
        allowed_roots=(tmp_path,),
        default_read_dir=files_dir,
    )

    assert result.success, f"Expected success, got: {result.error}"
    assert "hello world" in result.output_text
    assert str(files_dir / "hello.txt") in result.output_text


def test_project_relative_read_path_is_not_prefixed_twice(tmp_path: Path) -> None:
    files_dir = tmp_path / "data" / "files"
    files_dir.mkdir(parents=True)
    target = files_dir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")

    result = read_text_file(
        _make_call("data/files/hello.txt"),
        allowed_roots=(tmp_path,),
        default_read_dir=files_dir,
    )

    assert result.success, f"Expected success, got: {result.error}"
    assert result.payload is not None
    assert result.payload["path"] == str(target)
    assert "hello world" in result.output_text


def test_absolute_read_ignores_default_read_dir(tmp_path: Path) -> None:
    """Absolute path bypasses default_read_dir and resolves directly."""
    files_dir = tmp_path / "data" / "files"
    files_dir.mkdir(parents=True)
    target = tmp_path / "absolute.txt"
    target.write_text("absolute content", encoding="utf-8")

    result = read_text_file(
        _make_call(str(target)),
        allowed_roots=(tmp_path,),
        default_read_dir=files_dir,
    )

    assert result.success, f"Expected success, got: {result.error}"
    assert "absolute content" in result.output_text


def test_relative_read_path_traversal_blocked(tmp_path: Path) -> None:
    """A traversal attempt via relative path is blocked by allowed_roots."""
    files_dir = tmp_path / "data" / "files"
    files_dir.mkdir(parents=True)

    # Attempt to escape files_dir and tmp_path with traversal
    result = read_text_file(
        _make_call("../../etc/passwd"),
        allowed_roots=(tmp_path,),
        default_read_dir=files_dir,
    )

    assert not result.success
    assert "outside the allowed" in result.error.lower() or "not supported" in result.error.lower()


def test_relative_read_unsupported_format_still_rejected_in_default_dir(
    tmp_path: Path,
) -> None:
    """Format restriction fires even when relative path resolves inside default_read_dir."""
    files_dir = tmp_path / "data" / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "report.pdf").write_bytes(b"%PDF fake")

    result = read_text_file(
        _make_call("report.pdf"),
        allowed_roots=(tmp_path,),
        default_read_dir=files_dir,
    )

    assert not result.success
    assert "not supported" in result.error.lower()
    assert "V1" in result.error


def test_register_file_tools_wires_default_read_dir(tmp_path: Path) -> None:
    """register_file_tools passes default_read_dir through to the read handler."""
    files_dir = tmp_path / "data" / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "wired.txt").write_text("wired", encoding="utf-8")

    registry = ToolRegistry()
    register_file_tools(
        registry,
        project_root=tmp_path,
        default_read_dir=files_dir,
    )

    registered = registry.get("read_text_file")
    assert registered is not None

    call = _make_call("wired.txt")
    result = registered.handler(call)

    assert result.success, f"Expected success, got: {result.error}"
    assert "wired" in result.output_text
