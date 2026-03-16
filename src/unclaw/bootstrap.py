"""Startup preparation logic."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from unclaw.db.repositories import EventRepository
from unclaw.db.sqlite import initialize_schema, open_connection
from unclaw.errors import BootstrapError
from unclaw.settings import Settings, load_settings


def bootstrap(project_root: Path | None = None) -> Settings:
    settings = load_settings(project_root=project_root)
    prepare_runtime(settings)
    return settings


def prepare_runtime(settings: Settings) -> None:
    try:
        for directory in settings.paths.runtime_directories():
            directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BootstrapError(
            "Could not prepare the Unclaw runtime directories."
        ) from exc

    _apply_trace_retention(settings)


def _apply_trace_retention(settings: Settings) -> None:
    retention_days = settings.app.logging.retention_days
    if retention_days < 1:
        return

    cutoff = _retention_cutoff_iso(retention_days)
    _prune_runtime_events(settings.paths.database_path, created_before=cutoff)
    _prune_runtime_log(settings.paths.log_file_path, created_before=cutoff)


def _retention_cutoff_iso(retention_days: int) -> str:
    cutoff = datetime.now(tz=UTC) - timedelta(days=retention_days)
    return cutoff.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _prune_runtime_events(database_path: Path, *, created_before: str) -> None:
    connection: sqlite3.Connection | None = None
    try:
        connection = open_connection(database_path)
        initialize_schema(connection)
        EventRepository(connection).purge_events_before(created_before=created_before)
    except sqlite3.Error:
        return
    finally:
        if connection is not None:
            connection.close()


def _prune_runtime_log(log_path: Path, *, created_before: str) -> None:
    if not log_path.is_file():
        return

    try:
        raw_lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    kept_lines = [
        line
        for line in raw_lines
        if _should_keep_runtime_log_line(line, created_before=created_before)
    ]
    if kept_lines == raw_lines:
        return

    try:
        if kept_lines:
            log_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        else:
            log_path.unlink(missing_ok=True)
    except OSError:
        return


def _should_keep_runtime_log_line(line: str, *, created_before: str) -> bool:
    normalized_line = line.strip()
    if not normalized_line:
        return False

    try:
        payload = json.loads(normalized_line)
    except json.JSONDecodeError:
        return True

    if not isinstance(payload, dict):
        return True

    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        return True

    return created_at >= created_before
