"""Local file tools for the early Unclaw runtime."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path
from typing import Any

from unclaw.tools.contracts import (
    ToolCall,
    ToolArgumentSpec,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry

_DEFAULT_MAX_FILE_CHARS = 8_000
_DEFAULT_DIRECTORY_LIMIT = 200
_MAX_WRITE_FILE_CHARS = 1_000_000  # 1 MB upper bound on text content

# V1 document read scope — only plain text-based formats are supported.
# Binary formats (pdf, docx, xlsx, etc.) are not supported in V1.
READABLE_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".json", ".csv"})
_READABLE_EXTENSIONS_DISPLAY = ", ".join(sorted(READABLE_EXTENSIONS))

READ_TEXT_FILE_DEFINITION = ToolDefinition(
    name="read_text_file",
    description=(
        "Read a local UTF-8 text file. "
        "Relative paths are resolved inside the data/files/ directory by default. "
        f"Supported formats (V1): {_READABLE_EXTENSIONS_DISPLAY}. "
        "Other file types (pdf, docx, xlsx, etc.) are not supported in V1."
    ),
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={
        "path": ToolArgumentSpec(description="Path to a local UTF-8 text file."),
        "max_chars": ToolArgumentSpec(
            description="Optional maximum number of characters to return.",
            value_type="integer",
        ),
    },
)

LIST_DIRECTORY_DEFINITION = ToolDefinition(
    name="list_directory",
    description="List one local directory with depth 1 or 2.",
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={
        "path": ToolArgumentSpec(description="Path to a local directory."),
        "max_depth": ToolArgumentSpec(
            description="Optional depth, allowed values are 1 or 2.",
            value_type="integer",
        ),
        "limit": ToolArgumentSpec(
            description="Optional maximum number of entries to include.",
            value_type="integer",
        ),
    },
)

WRITE_TEXT_FILE_DEFINITION = ToolDefinition(
    name="write_text_file",
    description=(
        "Write plain UTF-8 text content to a local file. "
        "Relative paths are created inside the data/files/ directory by default. "
        "Fails if the file already exists unless overwrite is set to true. "
        "Content is limited to 1 MB. Only writes inside the configured allowed roots."
    ),
    permission_level=ToolPermissionLevel.LOCAL_WRITE,
    arguments={
        "path": ToolArgumentSpec(description="Path to the file to write."),
        "content": ToolArgumentSpec(description="UTF-8 text content to write."),
        "overwrite": ToolArgumentSpec(
            description=(
                "Set to true to allow overwriting an existing file. "
                "Default: false — write fails if the file already exists."
            ),
            value_type="boolean",
        ),
    },
)

MOVE_FILE_DEFINITION = ToolDefinition(
    name="move_file",
    description=(
        "Move one local file to another local path. "
        "Relative paths are resolved inside the data/files/ directory by default. "
        "Fails if the destination already exists unless overwrite is set to true. "
        "Only moves files inside the configured allowed roots."
    ),
    permission_level=ToolPermissionLevel.LOCAL_WRITE,
    arguments={
        "source_path": ToolArgumentSpec(description="Path to the source file to move."),
        "destination_path": ToolArgumentSpec(
            description="Path where the file should be moved."
        ),
        "overwrite": ToolArgumentSpec(
            description=(
                "Set to true to allow replacing an existing destination file. "
                "Default: false — move fails if the destination already exists."
            ),
            value_type="boolean",
        ),
    },
)

COPY_FILE_DEFINITION = ToolDefinition(
    name="copy_file",
    description=(
        "Copy one local file to another local path. "
        "Relative paths are resolved inside the data/files/ directory by default. "
        "Fails if the destination already exists unless overwrite is set to true. "
        "Only copies files inside the configured allowed roots."
    ),
    permission_level=ToolPermissionLevel.LOCAL_WRITE,
    arguments={
        "source_path": ToolArgumentSpec(description="Path to the source file to copy."),
        "destination_path": ToolArgumentSpec(
            description="Path where the copied file should be written."
        ),
        "overwrite": ToolArgumentSpec(
            description=(
                "Set to true to allow replacing an existing destination file. "
                "Default: false — copy fails if the destination already exists."
            ),
            value_type="boolean",
        ),
    },
)

RENAME_FILE_DEFINITION = ToolDefinition(
    name="rename_file",
    description=(
        "Rename one local file to another local path. "
        "Relative paths are resolved inside the data/files/ directory by default. "
        "Fails if the destination already exists unless overwrite is set to true. "
        "Only renames files inside the configured allowed roots."
    ),
    permission_level=ToolPermissionLevel.LOCAL_WRITE,
    arguments={
        "source_path": ToolArgumentSpec(description="Path to the source file to rename."),
        "destination_path": ToolArgumentSpec(
            description="New local path for the renamed file."
        ),
        "overwrite": ToolArgumentSpec(
            description=(
                "Set to true to allow replacing an existing destination file. "
                "Default: false — rename fails if the destination already exists."
            ),
            value_type="boolean",
        ),
    },
)


def register_file_tools(
    registry: ToolRegistry,
    *,
    project_root: Path | None = None,
    configured_roots: tuple[str, ...] = (),
    default_write_dir: Path | None = None,
    default_read_dir: Path | None = None,
) -> None:
    """Register the built-in local file tools."""
    allowed_roots = resolve_allowed_roots(
        project_root=project_root,
        configured_roots=configured_roots,
    )

    def read_handler(call: ToolCall) -> ToolResult:
        return read_text_file(call, allowed_roots=allowed_roots, default_read_dir=default_read_dir)

    def list_handler(call: ToolCall) -> ToolResult:
        return list_directory(call, allowed_roots=allowed_roots)

    def write_handler(call: ToolCall) -> ToolResult:
        return write_text_file(
            call,
            allowed_roots=allowed_roots,
            default_write_dir=default_write_dir,
        )

    def move_handler(call: ToolCall) -> ToolResult:
        return move_file(
            call,
            allowed_roots=allowed_roots,
            default_read_dir=default_read_dir,
            default_write_dir=default_write_dir,
        )

    def copy_handler(call: ToolCall) -> ToolResult:
        return copy_file(
            call,
            allowed_roots=allowed_roots,
            default_read_dir=default_read_dir,
            default_write_dir=default_write_dir,
        )

    def rename_handler(call: ToolCall) -> ToolResult:
        return rename_file(
            call,
            allowed_roots=allowed_roots,
            default_read_dir=default_read_dir,
            default_write_dir=default_write_dir,
        )

    registry.register(READ_TEXT_FILE_DEFINITION, read_handler)
    registry.register(LIST_DIRECTORY_DEFINITION, list_handler)
    registry.register(WRITE_TEXT_FILE_DEFINITION, write_handler)
    registry.register(MOVE_FILE_DEFINITION, move_handler)
    registry.register(COPY_FILE_DEFINITION, copy_handler)
    registry.register(RENAME_FILE_DEFINITION, rename_handler)


def read_text_file(
    call: ToolCall,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
    default_read_dir: Path | None = None,
) -> ToolResult:
    """Read a local text file with a small, safe default output limit."""
    tool_name = READ_TEXT_FILE_DEFINITION.name

    try:
        path_value = _read_string_argument(call.arguments, "path")
        max_chars = _read_positive_int_argument(
            call.arguments,
            "max_chars",
            default=_DEFAULT_MAX_FILE_CHARS,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    normalized_allowed_roots = _normalize_allowed_roots(allowed_roots)
    path = _resolve_file_tool_path(
        path_value,
        default_dir=default_read_dir,
        allowed_roots=normalized_allowed_roots,
    )
    access_error = _restrict_to_allowed_roots(
        tool_name=tool_name,
        path=path,
        allowed_roots=normalized_allowed_roots,
    )
    if access_error is not None:
        return access_error

    extension = path.suffix.lower()
    if extension not in READABLE_EXTENSIONS:
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"File format '{extension or '(no extension)'}' is not supported "
                f"for reading in V1. "
                f"Supported formats: {_READABLE_EXTENSIONS_DISPLAY}."
            ),
        )

    if not path.exists():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"File does not exist: {path}",
        )
    if not path.is_file():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Path is not a file: {path}",
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            content = handle.read(max_chars + 1)
    except UnicodeDecodeError:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"File is not valid UTF-8 text: {path}",
        )
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not read file '{path}': {exc}",
        )

    truncated = len(content) > max_chars
    returned_text = content[:max_chars] if truncated else content
    display_text = returned_text or "[empty file]"
    if truncated:
        display_text = f"{returned_text.rstrip()}\n\n[truncated]"

    output_text = f"File: {path}\n\n{display_text}"
    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload={
            "path": str(path),
            "returned_characters": len(returned_text),
            "truncated": truncated,
        },
    )


def list_directory(
    call: ToolCall,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
) -> ToolResult:
    """List one directory with a compact tree-like output."""
    tool_name = LIST_DIRECTORY_DEFINITION.name

    try:
        path_value = _read_string_argument(call.arguments, "path")
        max_depth = _read_positive_int_argument(call.arguments, "max_depth", default=1)
        limit = _read_positive_int_argument(
            call.arguments,
            "limit",
            default=_DEFAULT_DIRECTORY_LIMIT,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    if max_depth not in (1, 2):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'max_depth' must be 1 or 2.",
        )

    path = _resolve_path(path_value)
    access_error = _restrict_to_allowed_roots(
        tool_name=tool_name,
        path=path,
        allowed_roots=_normalize_allowed_roots(allowed_roots),
    )
    if access_error is not None:
        return access_error
    if not path.exists():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Directory does not exist: {path}",
        )
    if not path.is_dir():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Path is not a directory: {path}",
        )

    lines: list[str] = []
    truncated = _append_directory_lines(
        path=path,
        current_depth=1,
        max_depth=max_depth,
        lines=lines,
        limit=limit,
    )
    displayed_entries = len(lines)

    if not lines:
        lines.append("(empty)")
    if truncated:
        lines.append("... truncated ...")

    output_text = "\n".join([f"Directory: {path}", *lines])
    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload={
            "path": str(path),
            "max_depth": max_depth,
            "displayed_entries": displayed_entries,
            "truncated": truncated,
        },
    )


def write_text_file(
    call: ToolCall,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
    default_write_dir: Path | None = None,
) -> ToolResult:
    """Write plain UTF-8 text to a local file, bounded and permissioned."""
    tool_name = WRITE_TEXT_FILE_DEFINITION.name

    path_value = call.arguments.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'path' must be a non-empty string.",
        )

    content = call.arguments.get("content")
    if not isinstance(content, str):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'content' must be a string.",
        )

    overwrite = call.arguments.get("overwrite", False)
    if not isinstance(overwrite, bool):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'overwrite' must be a boolean.",
        )

    if len(content) > _MAX_WRITE_FILE_CHARS:
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"Content exceeds the maximum allowed size of "
                f"{_MAX_WRITE_FILE_CHARS} characters."
            ),
        )

    path_str = path_value.strip()
    normalized_allowed_roots = _normalize_allowed_roots(allowed_roots)
    path = _resolve_file_tool_path(
        path_str,
        default_dir=default_write_dir,
        allowed_roots=normalized_allowed_roots,
    )

    access_error = _restrict_to_allowed_roots(
        tool_name=tool_name,
        path=path,
        allowed_roots=normalized_allowed_roots,
    )
    if access_error is not None:
        return access_error

    if path.exists() and not overwrite:
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"File already exists: {path}. "
                "Pass overwrite=true to replace it."
            ),
        )

    if path.exists() and not path.is_file():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Path exists but is not a file: {path}",
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        os.chmod(path, 0o600)
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not write file '{path}': {exc}",
        )

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=f"File written: {path}",
        payload={
            "path": str(path),
            "size_chars": len(content),
            "overwrite": overwrite,
        },
    )


def move_file(
    call: ToolCall,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
    default_read_dir: Path | None = None,
    default_write_dir: Path | None = None,
) -> ToolResult:
    """Move one local file to another local path, bounded and permissioned."""
    return _relocate_file(
        call,
        tool_name=MOVE_FILE_DEFINITION.name,
        past_tense="moved",
        present_tense="move",
        allowed_roots=allowed_roots,
        default_read_dir=default_read_dir,
        default_write_dir=default_write_dir,
        allow_cross_device_fallback=True,
    )


def copy_file(
    call: ToolCall,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
    default_read_dir: Path | None = None,
    default_write_dir: Path | None = None,
) -> ToolResult:
    """Copy one local file to another local path, bounded and permissioned."""
    tool_name = COPY_FILE_DEFINITION.name

    try:
        source_path_value = _read_string_argument(call.arguments, "source_path")
        destination_path_value = _read_string_argument(call.arguments, "destination_path")
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    overwrite = call.arguments.get("overwrite", False)
    if not isinstance(overwrite, bool):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'overwrite' must be a boolean.",
        )

    normalized_allowed_roots = _normalize_allowed_roots(allowed_roots)
    source_path = _resolve_file_tool_path(
        source_path_value,
        default_dir=default_read_dir,
        allowed_roots=normalized_allowed_roots,
    )
    destination_path = _resolve_file_tool_path(
        destination_path_value,
        default_dir=default_write_dir,
        allowed_roots=normalized_allowed_roots,
    )
    source_access_error = _restrict_to_allowed_roots(
        tool_name=tool_name,
        path=source_path,
        allowed_roots=normalized_allowed_roots,
    )
    if source_access_error is not None:
        return source_access_error

    destination_access_error = _restrict_to_allowed_roots(
        tool_name=tool_name,
        path=destination_path,
        allowed_roots=normalized_allowed_roots,
    )
    if destination_access_error is not None:
        return destination_access_error

    if not source_path.exists():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Source file does not exist: {source_path}",
        )

    if not source_path.is_file():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Source path is not a file: {source_path}",
        )

    if destination_path.exists() and not overwrite:
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"Destination file already exists: {destination_path}. "
                "Pass overwrite=true to replace it."
            ),
        )

    if destination_path.exists() and not destination_path.is_file():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Destination path exists but is not a file: {destination_path}",
        )

    try:
        shutil.copy2(source_path, destination_path)
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not copy file '{source_path}' to '{destination_path}': {exc}",
        )

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=f"File copied: {source_path} -> {destination_path}",
        payload={
            "source_path": str(source_path),
            "destination_path": str(destination_path),
            "overwrite": overwrite,
        },
    )


def rename_file(
    call: ToolCall,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
    default_read_dir: Path | None = None,
    default_write_dir: Path | None = None,
) -> ToolResult:
    """Rename one local file to another local path, bounded and permissioned."""
    return _relocate_file(
        call,
        tool_name=RENAME_FILE_DEFINITION.name,
        past_tense="renamed",
        present_tense="rename",
        allowed_roots=allowed_roots,
        default_read_dir=default_read_dir,
        default_write_dir=default_write_dir,
        allow_cross_device_fallback=False,
    )


def _relocate_file(
    call: ToolCall,
    *,
    tool_name: str,
    past_tense: str,
    present_tense: str,
    allowed_roots: tuple[Path, ...] | None,
    default_read_dir: Path | None,
    default_write_dir: Path | None,
    allow_cross_device_fallback: bool,
) -> ToolResult:
    try:
        source_path_value = _read_string_argument(call.arguments, "source_path")
        destination_path_value = _read_string_argument(call.arguments, "destination_path")
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    overwrite = call.arguments.get("overwrite", False)
    if not isinstance(overwrite, bool):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'overwrite' must be a boolean.",
        )

    normalized_allowed_roots = _normalize_allowed_roots(allowed_roots)
    source_path = _resolve_file_tool_path(
        source_path_value,
        default_dir=default_read_dir,
        allowed_roots=normalized_allowed_roots,
    )
    destination_path = _resolve_file_tool_path(
        destination_path_value,
        default_dir=default_write_dir,
        allowed_roots=normalized_allowed_roots,
    )
    source_access_error = _restrict_to_allowed_roots(
        tool_name=tool_name,
        path=source_path,
        allowed_roots=normalized_allowed_roots,
    )
    if source_access_error is not None:
        return source_access_error

    destination_access_error = _restrict_to_allowed_roots(
        tool_name=tool_name,
        path=destination_path,
        allowed_roots=normalized_allowed_roots,
    )
    if destination_access_error is not None:
        return destination_access_error

    if not source_path.exists():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Source file does not exist: {source_path}",
        )

    if not source_path.is_file():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Source path is not a file: {source_path}",
        )

    if destination_path.exists() and not overwrite:
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"Destination file already exists: {destination_path}. "
                "Pass overwrite=true to replace it."
            ),
        )

    if destination_path.exists() and not destination_path.is_file():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Destination path exists but is not a file: {destination_path}",
        )

    try:
        if overwrite:
            source_path.replace(destination_path)
        else:
            source_path.rename(destination_path)
    except OSError as exc:
        if not allow_cross_device_fallback or exc.errno != errno.EXDEV:
            return ToolResult.failure(
                tool_name=tool_name,
                error=(
                    f"Could not {present_tense} file '{source_path}' "
                    f"to '{destination_path}': {exc}"
                ),
            )
        try:
            if destination_path.exists():
                destination_path.unlink()
            shutil.move(str(source_path), str(destination_path))
        except OSError as move_exc:
            return ToolResult.failure(
                tool_name=tool_name,
                error=(
                    f"Could not {present_tense} file '{source_path}' "
                    f"to '{destination_path}': {move_exc}"
                ),
            )

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=f"File {past_tense}: {source_path} -> {destination_path}",
        payload={
            "source_path": str(source_path),
            "destination_path": str(destination_path),
            "overwrite": overwrite,
        },
    )


def _append_directory_lines(
    *,
    path: Path,
    current_depth: int,
    max_depth: int,
    lines: list[str],
    limit: int,
) -> bool:
    try:
        entries = sorted(path.iterdir(), key=_directory_sort_key)
    except PermissionError:
        indent = "  " * (current_depth - 1)
        lines.append(f"{indent}- [permission denied]")
        return False
    except OSError as exc:
        indent = "  " * (current_depth - 1)
        lines.append(f"{indent}- [error: {exc}]")
        return False

    for entry in entries:
        if len(lines) >= limit:
            return True

        indent = "  " * (current_depth - 1)
        is_directory = _is_directory(entry)
        lines.append(f"{indent}- {_entry_label(entry, is_directory)}")

        if is_directory and not entry.is_symlink() and current_depth < max_depth:
            truncated = _append_directory_lines(
                path=entry,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                lines=lines,
                limit=limit,
            )
            if truncated:
                return True

    return False


def _directory_sort_key(path: Path) -> tuple[int, str]:
    return (0 if _is_directory(path) else 1, path.name.lower())


def _entry_label(path: Path, is_directory: bool) -> str:
    label = path.name
    if is_directory:
        label = f"{label}/"
    if path.is_symlink():
        label = f"{label} (symlink)"
    return label


def _is_directory(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _resolve_path(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def _resolve_file_tool_path(
    path_value: str,
    *,
    default_dir: Path | None,
    allowed_roots: tuple[Path, ...] | None,
) -> Path:
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute() or default_dir is None:
        return _resolve_path(path_value)

    root_relative_path = _resolve_explicit_root_relative_path(
        raw_path,
        default_dir=default_dir,
        allowed_roots=allowed_roots,
    )
    if root_relative_path is not None:
        return root_relative_path

    return (default_dir.expanduser().resolve() / raw_path).resolve()


def _resolve_explicit_root_relative_path(
    relative_path: Path,
    *,
    default_dir: Path,
    allowed_roots: tuple[Path, ...] | None,
) -> Path | None:
    if len(relative_path.parts) < 2 or ".." in relative_path.parts:
        return None

    normalized_allowed_roots = _normalize_allowed_roots(allowed_roots)
    base_root = normalized_allowed_roots[0]
    try:
        default_dir_relative = default_dir.expanduser().resolve().relative_to(base_root)
    except ValueError:
        return None

    if not default_dir_relative.parts:
        return None
    if relative_path.parts[0] != default_dir_relative.parts[0]:
        return None

    return (base_root / relative_path).resolve()


def resolve_allowed_roots(
    *,
    project_root: Path | None,
    configured_roots: tuple[str, ...],
) -> tuple[Path, ...]:
    base_root = project_root.resolve() if project_root is not None else Path.cwd().resolve()
    roots = [base_root]

    for raw_root in configured_roots:
        candidate = Path(raw_root).expanduser()
        if not candidate.is_absolute():
            candidate = base_root / candidate
        roots.append(candidate.resolve())

    return _normalize_allowed_roots(tuple(roots))


def _normalize_allowed_roots(
    allowed_roots: tuple[Path, ...] | None,
) -> tuple[Path, ...]:
    if not allowed_roots:
        return (Path.cwd().resolve(),)
    return tuple(dict.fromkeys(root.resolve() for root in allowed_roots))


def _restrict_to_allowed_roots(
    *,
    tool_name: str,
    path: Path,
    allowed_roots: tuple[Path, ...],
) -> ToolResult | None:
    if _is_path_allowed(path, allowed_roots):
        return None

    allowed_roots_text = ", ".join(str(root) for root in allowed_roots)
    return ToolResult.failure(
        tool_name=tool_name,
        error=(
            f"Access to '{path}' is outside the allowed local roots. "
            f"Allowed roots: {allowed_roots_text}."
        ),
    )


def _is_path_allowed(path: Path, allowed_roots: tuple[Path, ...]) -> bool:
    for root in allowed_roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _read_string_argument(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Argument '{key}' must be a non-empty string.")
    return value.strip()


def _read_positive_int_argument(
    arguments: dict[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Argument '{key}' must be an integer.")
    if value < 1:
        raise ValueError(f"Argument '{key}' must be greater than zero.")
    return value


__all__ = [
    "COPY_FILE_DEFINITION",
    "LIST_DIRECTORY_DEFINITION",
    "MOVE_FILE_DEFINITION",
    "READABLE_EXTENSIONS",
    "READ_TEXT_FILE_DEFINITION",
    "RENAME_FILE_DEFINITION",
    "WRITE_TEXT_FILE_DEFINITION",
    "_MAX_WRITE_FILE_CHARS",
    "copy_file",
    "list_directory",
    "move_file",
    "read_text_file",
    "rename_file",
    "register_file_tools",
    "write_text_file",
]
