"""Bounded local notes tool family for the Unclaw runtime.

Four tools: create_note, read_note, list_notes, update_note.
All reads and writes are scoped to a single dedicated notes directory.
Path traversal and out-of-scope writes are explicitly blocked.
"""

from __future__ import annotations

import os
from pathlib import Path

from unclaw.tools.contracts import (
    ToolArgumentSpec,
    ToolCall,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry

# Bounds — explicit and auditable.
_MAX_NOTE_CHARS = 32_000
_MAX_NOTE_NAME_CHARS = 128
_MAX_LIST_ENTRIES = 200

CREATE_NOTE_DEFINITION = ToolDefinition(
    name="create_note",
    description=(
        "Create a new local note with the given title and content. "
        "Fails if a note with that title already exists. "
        "Use update_note to overwrite an existing note."
    ),
    permission_level=ToolPermissionLevel.LOCAL_WRITE,
    arguments={
        "title": ToolArgumentSpec(
            description="Note title (used as filename; no path separators allowed).",
        ),
        "content": ToolArgumentSpec(
            description="Text content for the note.",
        ),
    },
)

READ_NOTE_DEFINITION = ToolDefinition(
    name="read_note",
    description="Read a local note by title and return its content.",
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={
        "title": ToolArgumentSpec(description="Note title to read."),
    },
)

LIST_NOTES_DEFINITION = ToolDefinition(
    name="list_notes",
    description="List all available local note titles.",
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={},
)

UPDATE_NOTE_DEFINITION = ToolDefinition(
    name="update_note",
    description=(
        "Overwrite an existing local note with new content. "
        "Creates the note if it does not exist."
    ),
    permission_level=ToolPermissionLevel.LOCAL_WRITE,
    arguments={
        "title": ToolArgumentSpec(
            description="Note title to update (no path separators allowed).",
        ),
        "content": ToolArgumentSpec(
            description="New text content for the note.",
        ),
    },
)


def _validate_note_title(raw: str) -> tuple[str, str | None]:
    """Return (stripped_title, error_message_or_None).

    Validates that the title is safe to use as a filename component:
    - non-empty after stripping
    - within the allowed length
    - free of path separators and null bytes
    """
    title = raw.strip()
    if not title:
        return "", "Note title must be a non-empty string."
    if len(title) > _MAX_NOTE_NAME_CHARS:
        return title, (
            f"Note title must be at most {_MAX_NOTE_NAME_CHARS} characters."
        )
    if "/" in title or "\\" in title or "\x00" in title:
        return title, "Note title must not contain path separators or null bytes."
    return title, None


def _resolve_note_path(notes_dir: Path, title: str) -> Path:
    """Return the resolved absolute path for a note file."""
    return (notes_dir / f"{title}.md").resolve()


def _check_within_notes_dir(
    tool_name: str,
    note_path: Path,
    notes_dir: Path,
) -> ToolResult | None:
    """Return a failure ToolResult if note_path escapes notes_dir, else None.

    This is the primary path-traversal guard. Even if title validation
    passes, we resolve and check containment to be safe.
    """
    try:
        note_path.relative_to(notes_dir.resolve())
        return None
    except ValueError:
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"Note path escapes the notes directory. Access denied."
            ),
        )


def _write_note(notes_dir: Path, note_path: Path, content: str) -> None:
    """Write content to note_path, creating notes_dir if needed.

    Sets file permissions to 0o600 (owner read/write only).
    """
    notes_dir.mkdir(parents=True, exist_ok=True)
    note_path.write_text(content, encoding="utf-8")
    os.chmod(note_path, 0o600)


def create_note(call: ToolCall, *, notes_dir: Path) -> ToolResult:
    """Create a new note. Returns failure if the note already exists."""
    tool_name = CREATE_NOTE_DEFINITION.name

    raw_title = call.arguments.get("title")
    if not isinstance(raw_title, str):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'title' must be a string.",
        )

    raw_content = call.arguments.get("content", "")
    if not isinstance(raw_content, str):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'content' must be a string.",
        )

    title, title_error = _validate_note_title(raw_title)
    if title_error is not None:
        return ToolResult.failure(tool_name=tool_name, error=title_error)

    if len(raw_content) > _MAX_NOTE_CHARS:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Note content exceeds the maximum size of {_MAX_NOTE_CHARS} characters.",
        )

    note_path = _resolve_note_path(notes_dir, title)
    path_error = _check_within_notes_dir(tool_name, note_path, notes_dir)
    if path_error is not None:
        return path_error

    if note_path.exists():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Note '{title}' already exists. Use update_note to overwrite.",
        )

    try:
        _write_note(notes_dir, note_path, raw_content)
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not write note '{title}': {exc}",
        )

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=f"Note '{title}' created.",
        payload={
            "title": title,
            "path": str(note_path),
            "size_chars": len(raw_content),
        },
    )


def read_note(call: ToolCall, *, notes_dir: Path) -> ToolResult:
    """Read an existing note by title."""
    tool_name = READ_NOTE_DEFINITION.name

    raw_title = call.arguments.get("title")
    if not isinstance(raw_title, str):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'title' must be a string.",
        )

    title, title_error = _validate_note_title(raw_title)
    if title_error is not None:
        return ToolResult.failure(tool_name=tool_name, error=title_error)

    note_path = _resolve_note_path(notes_dir, title)
    path_error = _check_within_notes_dir(tool_name, note_path, notes_dir)
    if path_error is not None:
        return path_error

    if not note_path.exists():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Note '{title}' does not exist.",
        )

    try:
        content = note_path.read_text(encoding="utf-8")
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not read note '{title}': {exc}",
        )

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=f"Note: {title}\n\n{content}",
        payload={
            "title": title,
            "path": str(note_path),
            "size_chars": len(content),
        },
    )


def list_notes(call: ToolCall, *, notes_dir: Path) -> ToolResult:
    """List all notes in the notes directory, bounded to _MAX_LIST_ENTRIES."""
    tool_name = LIST_NOTES_DEFINITION.name

    if not notes_dir.exists():
        return ToolResult.ok(
            tool_name=tool_name,
            output_text="No notes found.",
            payload={"titles": [], "count": 0, "truncated": False},
        )

    try:
        entries = sorted(notes_dir.glob("*.md"), key=lambda p: p.name.lower())
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not list notes: {exc}",
        )

    truncated = len(entries) > _MAX_LIST_ENTRIES
    entries = entries[:_MAX_LIST_ENTRIES]
    titles = [p.stem for p in entries]

    if not titles:
        output_text = "No notes found."
    else:
        lines = [f"- {t}" for t in titles]
        if truncated:
            lines.append(f"... (showing first {_MAX_LIST_ENTRIES})")
        output_text = "Notes:\n" + "\n".join(lines)

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload={"titles": titles, "count": len(titles), "truncated": truncated},
    )


def update_note(call: ToolCall, *, notes_dir: Path) -> ToolResult:
    """Overwrite a note with new content, creating it if it does not exist."""
    tool_name = UPDATE_NOTE_DEFINITION.name

    raw_title = call.arguments.get("title")
    if not isinstance(raw_title, str):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'title' must be a string.",
        )

    raw_content = call.arguments.get("content", "")
    if not isinstance(raw_content, str):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'content' must be a string.",
        )

    title, title_error = _validate_note_title(raw_title)
    if title_error is not None:
        return ToolResult.failure(tool_name=tool_name, error=title_error)

    if len(raw_content) > _MAX_NOTE_CHARS:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Note content exceeds the maximum size of {_MAX_NOTE_CHARS} characters.",
        )

    note_path = _resolve_note_path(notes_dir, title)
    path_error = _check_within_notes_dir(tool_name, note_path, notes_dir)
    if path_error is not None:
        return path_error

    existed = note_path.exists()

    try:
        _write_note(notes_dir, note_path, raw_content)
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not write note '{title}': {exc}",
        )

    action = "updated" if existed else "created"
    return ToolResult.ok(
        tool_name=tool_name,
        output_text=f"Note '{title}' {action}.",
        payload={
            "title": title,
            "path": str(note_path),
            "size_chars": len(raw_content),
            "created": not existed,
        },
    )


def register_notes_tools(registry: ToolRegistry, *, notes_dir: Path) -> None:
    """Register the four notes tools, creating the notes directory if needed."""
    notes_dir.mkdir(parents=True, exist_ok=True)

    def create_handler(call: ToolCall) -> ToolResult:
        return create_note(call, notes_dir=notes_dir)

    def read_handler(call: ToolCall) -> ToolResult:
        return read_note(call, notes_dir=notes_dir)

    def list_handler(call: ToolCall) -> ToolResult:
        return list_notes(call, notes_dir=notes_dir)

    def update_handler(call: ToolCall) -> ToolResult:
        return update_note(call, notes_dir=notes_dir)

    registry.register(CREATE_NOTE_DEFINITION, create_handler)
    registry.register(READ_NOTE_DEFINITION, read_handler)
    registry.register(LIST_NOTES_DEFINITION, list_handler)
    registry.register(UPDATE_NOTE_DEFINITION, update_handler)


__all__ = [
    "CREATE_NOTE_DEFINITION",
    "LIST_NOTES_DEFINITION",
    "READ_NOTE_DEFINITION",
    "UPDATE_NOTE_DEFINITION",
    "_MAX_NOTE_CHARS",
    "_MAX_NOTE_NAME_CHARS",
    "_MAX_LIST_ENTRIES",
    "create_note",
    "list_notes",
    "read_note",
    "register_notes_tools",
    "update_note",
]
