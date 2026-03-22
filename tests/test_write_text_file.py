"""Targeted tests for write_text_file — P3-3 corrective patch."""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.executor import create_default_tool_registry
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
    assert result.payload["created_new_file"] is True
    assert result.payload["overwrite_applied"] is False


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


def test_write_text_file_overwrite_true_on_existing_is_refused_first(tmp_path: Path) -> None:
    """First call with overwrite=true on an existing file is always refused."""
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "replaced", "overwrite": True},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert result.payload is not None
    assert result.payload["file_already_exists"] is True
    assert result.payload["overwrite_refused"] is True
    assert result.payload["overwrite_retry_allowed"] is True
    assert "overwrite_request_id" in result.payload
    assert result.payload["overwrite_request_id"].startswith("owr_")
    # Original content must be untouched
    assert target.read_text(encoding="utf-8") == "original"


def test_write_text_file_overwrites_with_request_id_from_refused_call(tmp_path: Path) -> None:
    """Round 2: supplying the request id from the refused call executes the overwrite."""
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    # Round 1: refused, get approval handle
    call_1 = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "replaced", "overwrite": True},
    )
    result_1 = write_text_file(call_1, allowed_roots=(tmp_path,))
    assert result_1.success is False
    assert result_1.payload is not None
    request_id = result_1.payload["overwrite_request_id"]

    # Round 2: supply request id — must succeed
    call_2 = ToolCall(
        tool_name="write_text_file",
        arguments={
            "path": str(target),
            "content": "replaced",
            "overwrite": True,
            "overwrite_request_id": request_id,
        },
    )
    result_2 = write_text_file(call_2, allowed_roots=(tmp_path,))
    assert result_2.success is True
    assert target.read_text(encoding="utf-8") == "replaced"
    assert result_2.payload is not None
    assert result_2.payload["overwrite_applied"] is True
    assert result_2.payload["created_new_file"] is False


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
    assert result.payload["path"] == str(target)
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


# ---------------------------------------------------------------------------
# Overwrite contract truthfulness — explicit outcome payload fields
# ---------------------------------------------------------------------------


def test_write_text_file_new_file_payload_marks_created_new_file(tmp_path: Path) -> None:
    """Creating a new file returns created_new_file=True, overwrite_applied=False."""
    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(tmp_path / "new.txt"), "content": "hello"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is True
    assert result.payload is not None
    assert result.payload["created_new_file"] is True
    assert result.payload["overwrite_applied"] is False


def test_write_text_file_overwrite_payload_marks_overwrite_applied(tmp_path: Path) -> None:
    """Successful overwrite (with valid request id) marks overwrite_applied=True."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    # Round 1
    r1 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={"path": str(target), "content": "new", "overwrite": True},
        ),
        allowed_roots=(tmp_path,),
    )
    request_id = r1.payload["overwrite_request_id"]  # type: ignore[index]

    # Round 2
    r2 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={
                "path": str(target),
                "content": "new",
                "overwrite": True,
                "overwrite_request_id": request_id,
            },
        ),
        allowed_roots=(tmp_path,),
    )
    assert r2.success is True
    assert r2.payload is not None
    assert r2.payload["overwrite_applied"] is True
    assert r2.payload["created_new_file"] is False


def test_write_text_file_invalid_request_id_is_refused(tmp_path: Path) -> None:
    """An invalid or fabricated overwrite_request_id must be refused."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={
            "path": str(target),
            "content": "new",
            "overwrite": True,
            "overwrite_request_id": "owr_notreal",
        },
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert result.payload is not None
    assert result.payload["overwrite_refused"] is True
    # File must be untouched
    assert target.read_text(encoding="utf-8") == "old"


def test_write_text_file_request_id_bound_to_content(tmp_path: Path) -> None:
    """Request id from one content value does not authorize a different content."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    # Get a request id for content="A"
    r1 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={"path": str(target), "content": "A", "overwrite": True},
        ),
        allowed_roots=(tmp_path,),
    )
    request_id = r1.payload["overwrite_request_id"]  # type: ignore[index]

    # Try to use that request id for content="B"
    r2 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={
                "path": str(target),
                "content": "B",  # different content
                "overwrite": True,
                "overwrite_request_id": request_id,
            },
        ),
        allowed_roots=(tmp_path,),
    )
    assert r2.success is False
    assert r2.payload is not None
    assert r2.payload["overwrite_refused"] is True
    assert target.read_text(encoding="utf-8") == "old"


def test_write_text_file_overwrite_false_on_existing_returns_file_already_exists(
    tmp_path: Path,
) -> None:
    """overwrite=false on existing file includes file_already_exists in the payload."""
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")

    call = ToolCall(
        tool_name="write_text_file",
        arguments={"path": str(target), "content": "new"},
    )
    result = write_text_file(call, allowed_roots=(tmp_path,))
    assert result.success is False
    assert result.payload is not None
    assert result.payload["file_already_exists"] is True
    assert result.payload["overwrite_refused"] is True
    # No approval handle for overwrite=false case (user did not ask to overwrite)
    assert "overwrite_request_id" not in result.payload


# ---------------------------------------------------------------------------
# Approval handle — expiry, reuse, and path binding
# ---------------------------------------------------------------------------


def test_overwrite_request_id_expired_is_refused(tmp_path: Path) -> None:
    """An expired approval handle must be refused cleanly."""
    import unclaw.tools.file_tools as ft

    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    # Round 1: get approval handle
    r1 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={"path": str(target), "content": "new", "overwrite": True},
        ),
        allowed_roots=(tmp_path,),
    )
    request_id = r1.payload["overwrite_request_id"]  # type: ignore[index]

    # Artificially expire the entry by setting created_at far in the past
    with ft._pending_lock:
        ft._pending_overwrites[request_id].created_at = 0.0

    # Round 2: expired handle must be refused
    r2 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={
                "path": str(target),
                "content": "new",
                "overwrite": True,
                "overwrite_request_id": request_id,
            },
        ),
        allowed_roots=(tmp_path,),
    )
    assert r2.success is False
    assert r2.payload is not None
    assert r2.payload["overwrite_refused"] is True
    assert target.read_text(encoding="utf-8") == "old"


def test_overwrite_request_id_cannot_be_reused(tmp_path: Path) -> None:
    """A consumed approval handle must be refused on a second use."""
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    # Round 1: get approval handle
    r1 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={"path": str(target), "content": "new", "overwrite": True},
        ),
        allowed_roots=(tmp_path,),
    )
    request_id = r1.payload["overwrite_request_id"]  # type: ignore[index]

    # Round 2: first use succeeds and consumes the handle
    r2 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={
                "path": str(target),
                "content": "new",
                "overwrite": True,
                "overwrite_request_id": request_id,
            },
        ),
        allowed_roots=(tmp_path,),
    )
    assert r2.success is True

    # Restore file so overwrite protection triggers again
    target.write_text("old", encoding="utf-8")

    # Round 3: reuse the same consumed handle — must be refused
    r3 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={
                "path": str(target),
                "content": "new",
                "overwrite": True,
                "overwrite_request_id": request_id,
            },
        ),
        allowed_roots=(tmp_path,),
    )
    assert r3.success is False
    assert r3.payload is not None
    assert r3.payload["overwrite_refused"] is True
    assert target.read_text(encoding="utf-8") == "old"


def test_overwrite_request_id_bound_to_path(tmp_path: Path) -> None:
    """Approval handle issued for file A cannot authorize overwrite of file B."""
    target_a = tmp_path / "a.txt"
    target_b = tmp_path / "b.txt"
    target_a.write_text("old_a", encoding="utf-8")
    target_b.write_text("old_b", encoding="utf-8")

    # Get a handle for target_a
    r1 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={"path": str(target_a), "content": "new", "overwrite": True},
        ),
        allowed_roots=(tmp_path,),
    )
    request_id = r1.payload["overwrite_request_id"]  # type: ignore[index]

    # Try to use it against target_b
    r2 = write_text_file(
        ToolCall(
            tool_name="write_text_file",
            arguments={
                "path": str(target_b),
                "content": "new",
                "overwrite": True,
                "overwrite_request_id": request_id,
            },
        ),
        allowed_roots=(tmp_path,),
    )
    assert r2.success is False
    assert r2.payload is not None
    assert r2.payload["overwrite_refused"] is True
    assert target_b.read_text(encoding="utf-8") == "old_b"
