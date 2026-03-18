"""Targeted tests for the session-db storage consolidation under data/memory/.

Requirement: settings.paths.database_path must resolve to data/memory/app.db.
Migration strategy:
  1. Canonical path exists → used as-is.
  2. Legacy data/app.db exists → adopted (copied) into data/memory/app.db.
  3. Neither exists → fresh db created at data/memory/app.db on first open.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

# Import unclaw.memory first so that the package __init__ is primed before
# session_manager loads chat_store; this resolves the pre-existing circular
# import between memory.manager and core.session_manager.
import unclaw.memory  # noqa: F401
from unclaw.bootstrap import bootstrap, prepare_runtime, _migrate_session_db
from unclaw.core.session_manager import SessionManager
from unclaw.db.sqlite import initialize_schema, open_connection
from unclaw.settings import load_settings

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# A. Canonical path
# ---------------------------------------------------------------------------


def test_canonical_db_path_is_under_memory_directory(make_temp_project) -> None:
    """settings.paths.database_path must end with memory/app.db."""
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    assert settings.paths.database_path.name == "app.db"
    assert settings.paths.database_path.parent.name == "memory"
    assert settings.paths.database_path == settings.paths.data_dir / "memory" / "app.db"


# ---------------------------------------------------------------------------
# B. Fresh install
# ---------------------------------------------------------------------------


def test_fresh_install_creates_db_under_memory(make_temp_project) -> None:
    """On a fresh install (no legacy db), bootstrap creates data/memory/app.db."""
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    # Neither legacy nor canonical db exist yet.
    legacy_path = settings.paths.data_dir / "app.db"
    assert not legacy_path.exists()
    assert not settings.paths.database_path.exists()

    session_manager = SessionManager.from_settings(settings)
    session_manager.close()

    assert settings.paths.database_path.exists()
    assert settings.paths.database_path.parent.name == "memory"


# ---------------------------------------------------------------------------
# C. Migration: legacy data/app.db is adopted
# ---------------------------------------------------------------------------


def test_legacy_db_is_adopted_on_prepare_runtime(make_temp_project) -> None:
    """If data/app.db exists but data/memory/app.db does not, it is copied over."""
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    # Create a legacy db with a known session.
    legacy_path = settings.paths.data_dir / "app.db"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_connection(legacy_path)
    initialize_schema(conn)
    from unclaw.db.repositories import SessionRepository
    repo = SessionRepository(conn)
    session = repo.create_session(title="legacy-session", is_active=True)
    legacy_session_id = session.id
    conn.close()

    assert not settings.paths.database_path.exists()

    prepare_runtime(settings)

    # Canonical path must now exist and contain the migrated session.
    assert settings.paths.database_path.exists()
    conn = open_connection(settings.paths.database_path)
    sessions = SessionRepository(conn).list_sessions()
    conn.close()
    assert any(s.id == legacy_session_id for s in sessions)


def test_legacy_db_is_preserved_after_migration(make_temp_project) -> None:
    """The original data/app.db must not be deleted by migration."""
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    legacy_path = settings.paths.data_dir / "app.db"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_connection(legacy_path)
    initialize_schema(conn)
    conn.close()

    prepare_runtime(settings)

    assert legacy_path.exists(), "Legacy db must be left in place for reversibility."


# ---------------------------------------------------------------------------
# D. Existing canonical db is not overwritten
# ---------------------------------------------------------------------------


def test_existing_memory_db_is_not_overwritten(make_temp_project) -> None:
    """If data/memory/app.db already exists, migration must leave it untouched."""
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    # Pre-create canonical db with a sentinel session.
    settings.paths.database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_connection(settings.paths.database_path)
    initialize_schema(conn)
    from unclaw.db.repositories import SessionRepository
    repo = SessionRepository(conn)
    sentinel = repo.create_session(title="sentinel-session", is_active=True)
    sentinel_id = sentinel.id
    conn.close()

    # Also create a legacy db (should be ignored).
    legacy_path = settings.paths.data_dir / "app.db"
    conn = open_connection(legacy_path)
    initialize_schema(conn)
    conn.close()

    prepare_runtime(settings)

    # Sentinel must still be present.
    conn = open_connection(settings.paths.database_path)
    sessions = SessionRepository(conn).list_sessions()
    conn.close()
    assert any(s.id == sentinel_id for s in sessions)


# ---------------------------------------------------------------------------
# E. Session loads correctly after migration
# ---------------------------------------------------------------------------


def test_session_manager_loads_after_migration(make_temp_project) -> None:
    """SessionManager.from_settings works correctly after legacy-db migration."""
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    # Create a legacy db with a session and a message.
    legacy_path = settings.paths.data_dir / "app.db"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_connection(legacy_path)
    initialize_schema(conn)
    from unclaw.db.repositories import MessageRepository, SessionRepository
    repo = SessionRepository(conn)
    session = repo.create_session(title="migrated-session", is_active=True)
    MessageRepository(conn).add_message(
        session_id=session.id,
        role="user",
        content="hello from legacy",
    )
    conn.close()

    # Run bootstrap (triggers migration).
    bootstrap(project_root=project_root)

    # Session manager must see the migrated session and its messages.
    session_manager = SessionManager.from_settings(settings)
    try:
        messages = session_manager.list_messages(session.id)
    finally:
        session_manager.close()

    assert len(messages) == 1
    assert messages[0].content == "hello from legacy"


# ---------------------------------------------------------------------------
# F. _migrate_session_db unit: no-op when neither db exists
# ---------------------------------------------------------------------------


def test_migrate_session_db_is_noop_when_no_legacy_exists(tmp_path: Path) -> None:
    """_migrate_session_db must not raise and must not create any file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "memory" / "app.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _migrate_session_db(data_dir, db_path)

    assert not db_path.exists()
