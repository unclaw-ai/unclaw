"""Tests for the /memory-status slash command and MemoryDiagnostics helper.

Required coverage:
1. command renders when all three memory layers exist
2. command renders when some paths are missing
3. current session counts are shown correctly
4. legacy data/app.db presence is reported
5. command is read-only (no created/modified memory content)
6. output includes canonical paths under data/memory/
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.memory.diagnostics import (
    MemoryDiagnostics,
    collect_memory_diagnostics,
    render_memory_diagnostics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_session_manager(session_id: str) -> SimpleNamespace:
    return SimpleNamespace(current_session_id=session_id)


# ---------------------------------------------------------------------------
# 1. command renders when all three memory layers exist
# ---------------------------------------------------------------------------

def test_command_renders_all_three_layers_exist(make_temp_project) -> None:
    """The /memory-status command returns OK and shows 'exists' for all layers."""
    from unclaw.bootstrap import bootstrap
    from unclaw.core.command_handler import CommandHandler, CommandStatus
    from unclaw.core.session_manager import SessionManager
    from unclaw.memory.chat_store import ChatMemoryStore
    from unclaw.memory.long_term_store import LongTermStore
    from unclaw.schemas.chat import MessageRole

    project_root = make_temp_project()
    settings = bootstrap(project_root=project_root)

    session_manager = SessionManager.from_settings(settings)
    try:
        session = session_manager.ensure_current_session()
        # Populate conversation JSONL by adding one message.
        session_manager.add_message(
            MessageRole.USER,
            "hello memory-status test",
            session_id=session.id,
        )
        # Initialise the long-term store (creates the DB file).
        LongTermStore(settings.paths.data_dir / "memory" / "long_term.db")

        command_handler = CommandHandler(
            settings=settings,
            session_manager=session_manager,
        )
        result = command_handler.handle("/memory-status")

        assert result.status is CommandStatus.OK
        rendered = "\n".join(result.lines)
        # All three layers must report as existing.
        assert "session DB    : exists" in rendered
        assert "long-term DB  : exists" in rendered
        assert "chats dir     : exists" in rendered
    finally:
        session_manager.close()


# ---------------------------------------------------------------------------
# 2. command renders when some paths are missing
# ---------------------------------------------------------------------------

def test_command_renders_missing_layers(make_temp_project) -> None:
    """The command returns OK and shows 'missing' for absent files — no crash."""
    from unclaw.core.command_handler import CommandHandler, CommandStatus
    from unclaw.settings import load_settings

    project_root = make_temp_project()
    # Load settings without bootstrap so no data files are created.
    settings = load_settings(project_root=project_root)

    command_handler = CommandHandler(
        settings=settings,
        session_manager=_make_stub_session_manager("sess-missing"),
    )
    result = command_handler.handle("/memory-status")

    assert result.status is CommandStatus.OK
    rendered = "\n".join(result.lines)
    # Nothing has been created, so all layers should be missing.
    assert "session DB    : missing" in rendered
    assert "long-term DB  : missing" in rendered
    assert "chats dir     : missing" in rendered


# ---------------------------------------------------------------------------
# 3. current session counts are shown correctly
# ---------------------------------------------------------------------------

def test_current_session_counts_are_correct(tmp_path: Path) -> None:
    """collect_memory_diagnostics reports correct message counts for the session."""
    import sqlite3

    from unclaw.memory.chat_store import ChatMemoryStore

    session_id = "sess-count-test"

    # Build a minimal SQLite DB with the messages table and insert 2 rows.
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    db_path = memory_dir / "app.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE messages (id TEXT, session_id TEXT, role TEXT, content TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
        ("msg_1", session_id, "user", "first", "2026-01-01T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
        ("msg_2", session_id, "assistant", "reply", "2026-01-01T00:00:01Z"),
    )
    conn.commit()
    conn.close()

    # Add 3 lines to the JSONL file.
    chats_dir = memory_dir / "chats"
    chats_dir.mkdir()
    jsonl_path = chats_dir / f"{session_id}.jsonl"
    jsonl_path.write_text(
        '{"role":"user","content":"first","ts":"2026-01-01T00:00:00Z"}\n'
        '{"role":"assistant","content":"reply","ts":"2026-01-01T00:00:01Z"}\n'
        '{"role":"user","content":"second","ts":"2026-01-01T00:00:02Z"}\n',
        encoding="utf-8",
    )

    diagnostics = collect_memory_diagnostics(
        data_dir=tmp_path,
        session_db_path=db_path,
        current_session_id=session_id,
    )

    assert diagnostics.session_db_message_count == 2
    assert diagnostics.session_jsonl_message_count == 3

    rendered = "\n".join(render_memory_diagnostics(diagnostics))
    assert "messages in session DB   : 2" in rendered
    assert "messages in session JSONL: 3" in rendered
    assert session_id in rendered


# ---------------------------------------------------------------------------
# 4. legacy data/app.db presence is reported
# ---------------------------------------------------------------------------

def test_legacy_db_presence_reported(tmp_path: Path) -> None:
    """collect_memory_diagnostics reports legacy_db_exists=True when the file exists."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    db_path = memory_dir / "app.db"
    db_path.touch()

    # Create the legacy file at data/app.db.
    legacy_path = tmp_path / "app.db"
    legacy_path.write_text("legacy", encoding="utf-8")

    diagnostics = collect_memory_diagnostics(
        data_dir=tmp_path,
        session_db_path=db_path,
        current_session_id=None,
    )

    assert diagnostics.legacy_db_exists is True

    rendered = "\n".join(render_memory_diagnostics(diagnostics))
    assert "legacy DB     : exists" in rendered


def test_legacy_db_absent_reported(tmp_path: Path) -> None:
    """collect_memory_diagnostics reports legacy_db_exists=False when absent."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    db_path = memory_dir / "app.db"

    diagnostics = collect_memory_diagnostics(
        data_dir=tmp_path,
        session_db_path=db_path,
        current_session_id=None,
    )

    assert diagnostics.legacy_db_exists is False

    rendered = "\n".join(render_memory_diagnostics(diagnostics))
    assert "legacy DB     : missing" in rendered


# ---------------------------------------------------------------------------
# 5. command is read-only (no created/modified memory content)
# ---------------------------------------------------------------------------

def test_command_is_read_only(make_temp_project) -> None:
    """Calling /memory-status must not add messages, JSONL lines, or DB records."""
    from unclaw.bootstrap import bootstrap
    from unclaw.core.command_handler import CommandHandler, CommandStatus
    from unclaw.core.session_manager import SessionManager
    from unclaw.schemas.chat import MessageRole

    project_root = make_temp_project()
    settings = bootstrap(project_root=project_root)

    session_manager = SessionManager.from_settings(settings)
    try:
        session = session_manager.ensure_current_session()
        # Add one real message so the JSONL file exists.
        session_manager.add_message(
            MessageRole.USER,
            "pre-diagnostic message",
            session_id=session.id,
        )

        # Capture counts before the command.
        messages_before = session_manager.list_messages(session.id)
        chats_dir = settings.paths.data_dir / "memory" / "chats"
        jsonl_path = chats_dir / f"{session.id}.jsonl"
        jsonl_size_before = jsonl_path.stat().st_size if jsonl_path.is_file() else -1

        command_handler = CommandHandler(
            settings=settings,
            session_manager=session_manager,
        )
        result = command_handler.handle("/memory-status")

        assert result.status is CommandStatus.OK

        # No new messages must have been added to the session.
        messages_after = session_manager.list_messages(session.id)
        assert len(messages_after) == len(messages_before)

        # The JSONL file must not have grown.
        jsonl_size_after = jsonl_path.stat().st_size if jsonl_path.is_file() else -1
        assert jsonl_size_after == jsonl_size_before
    finally:
        session_manager.close()


# ---------------------------------------------------------------------------
# 6. output includes canonical paths under data/memory/
# ---------------------------------------------------------------------------

def test_output_includes_canonical_paths_under_data_memory(make_temp_project) -> None:
    """Rendered output must include paths rooted under data/memory/."""
    from unclaw.core.command_handler import CommandHandler, CommandStatus
    from unclaw.settings import load_settings

    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    # Derive expected canonical paths from settings.
    expected_session_db = settings.paths.database_path
    expected_long_term_db = settings.paths.data_dir / "memory" / "long_term.db"
    expected_chats_dir = settings.paths.data_dir / "memory" / "chats"

    command_handler = CommandHandler(
        settings=settings,
        session_manager=_make_stub_session_manager("sess-paths"),
    )
    result = command_handler.handle("/memory-status")

    assert result.status is CommandStatus.OK
    rendered = "\n".join(result.lines)

    # All three canonical paths must appear in the output.
    assert str(expected_session_db) in rendered
    assert str(expected_long_term_db) in rendered
    assert str(expected_chats_dir) in rendered

    # All three paths must be under the memory/ subdirectory.
    assert "memory" in str(expected_session_db)
    assert "memory" in str(expected_long_term_db)
    assert "memory" in str(expected_chats_dir)
