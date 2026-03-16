"""Telegram channel configuration, authorization, and session persistence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from unclaw.db.sqlite import open_connection
from unclaw.errors import ConfigurationError
from unclaw.llm.base import utc_now_iso
from unclaw.settings import Settings

_TELEGRAM_CONFIG_FILE_NAME = "telegram.yaml"
_LATEST_REJECTED_EVENT_TYPE = "telegram.chat.rejected"


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    """Minimal Telegram channel configuration."""

    bot_token_env_var: str
    polling_timeout_seconds: int
    allowed_chat_ids: frozenset[int]

    def is_chat_allowed(self, chat_id: int) -> bool:
        return chat_id in self.allowed_chat_ids


@dataclass(frozen=True, slots=True)
class TelegramAuthorizationUpdate:
    chat_id: int
    allowed_chat_ids: tuple[int, ...]
    file_changed: bool
    was_authorized: bool


@dataclass(slots=True)
class TelegramChatSessionStore:
    """Persist Telegram chat-to-session bindings in the local SQLite database."""

    connection: sqlite3.Connection

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS telegram_chat_sessions (
                chat_id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_telegram_chat_sessions_session_id
            ON telegram_chat_sessions (session_id);
            """
        )
        self.connection.commit()

    def get_session_id(self, chat_id: int) -> str | None:
        row = self.connection.execute(
            """
            SELECT session_id
            FROM telegram_chat_sessions
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()
        if row is None:
            return None
        value = row["session_id"]
        if not isinstance(value, str) or not value.strip():
            return None
        return value

    def bind_chat(self, *, chat_id: int, session_id: str) -> None:
        timestamp = utc_now_iso()
        self.connection.execute(
            """
            INSERT INTO telegram_chat_sessions (
                chat_id,
                session_id,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                session_id = excluded.session_id,
                updated_at = excluded.updated_at
            """,
            (chat_id, session_id, timestamp, timestamp),
        )
        self.connection.commit()


def load_telegram_config(settings: Settings) -> TelegramConfig:
    """Load and validate the Telegram channel configuration file."""

    config_path = settings.paths.config_dir / _TELEGRAM_CONFIG_FILE_NAME
    payload = _load_yaml_mapping(config_path)

    bot_token_env_var = _validate_telegram_token_env_var(
        _read_str(payload, "bot_token_env_var")
    )
    polling_timeout_seconds = _read_int(payload, "polling_timeout_seconds", minimum=1)
    allowed_chat_ids = _read_allowed_chat_ids(payload.get("allowed_chat_ids"))

    return TelegramConfig(
        bot_token_env_var=bot_token_env_var,
        polling_timeout_seconds=polling_timeout_seconds,
        allowed_chat_ids=allowed_chat_ids,
    )


def allow_telegram_chat(settings: Settings, chat_id: int) -> TelegramAuthorizationUpdate:
    config = load_telegram_config(settings)
    updated_chat_ids = tuple(sorted({*config.allowed_chat_ids, chat_id}))
    file_changed = _write_allowed_chat_ids(
        settings,
        allowed_chat_ids=list(updated_chat_ids),
    )
    return TelegramAuthorizationUpdate(
        chat_id=chat_id,
        allowed_chat_ids=updated_chat_ids,
        file_changed=file_changed,
        was_authorized=config.is_chat_allowed(chat_id),
    )


def revoke_telegram_chat(
    settings: Settings,
    chat_id: int,
) -> TelegramAuthorizationUpdate:
    config = load_telegram_config(settings)
    updated_chat_ids = tuple(
        chat for chat in sorted(config.allowed_chat_ids) if chat != chat_id
    )
    file_changed = _write_allowed_chat_ids(
        settings,
        allowed_chat_ids=list(updated_chat_ids),
    )
    return TelegramAuthorizationUpdate(
        chat_id=chat_id,
        allowed_chat_ids=updated_chat_ids,
        file_changed=file_changed,
        was_authorized=config.is_chat_allowed(chat_id),
    )


def find_latest_rejected_chat_id(settings: Settings) -> int | None:
    database_path = settings.paths.database_path
    if not database_path.exists():
        return None

    connection = open_connection(database_path)
    try:
        rows = connection.execute(
            """
            SELECT payload_json
            FROM events
            WHERE event_type = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 20
            """,
            (_LATEST_REJECTED_EVENT_TYPE,),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        connection.close()

    for row in rows:
        payload_json = row["payload_json"]
        if not isinstance(payload_json, str) or not payload_json.strip():
            continue
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        chat_id = payload.get("chat_id")
        reason = payload.get("reason")
        if isinstance(chat_id, int) and reason == "unauthorized":
            return chat_id
    return None


def format_telegram_access_mode(config: TelegramConfig) -> str:
    authorized_count = len(config.allowed_chat_ids)
    if authorized_count == 0:
        return "deny-by-default (0 chats)"
    if authorized_count == 1:
        return "allowlist (1 chat)"
    return f"allowlist ({authorized_count} chats)"


def print_authorized_chat_list(
    allowed_chat_ids: frozenset[int],
    *,
    latest_rejected_chat_id: int | None,
) -> None:
    if not allowed_chat_ids:
        print("No Telegram chats are authorized yet. Secure deny-by-default mode is active.")
    else:
        print("Authorized Telegram chats:")
        for chat_id in sorted(allowed_chat_ids):
            print(f"- {chat_id}")

    if latest_rejected_chat_id is not None:
        print(f"Latest rejected chat: {latest_rejected_chat_id}")
        print(
            "Authorize it with `unclaw telegram allow-latest` "
            f"or `unclaw telegram allow {latest_rejected_chat_id}` on this machine."
        )
    else:
        print(
            "Tip: send one message to the bot, then run "
            "`unclaw telegram allow-latest` on this machine."
        )


def format_authorized_chat_count(allowed_chat_ids: tuple[int, ...]) -> str:
    count = len(allowed_chat_ids)
    label = "chat is" if count == 1 else "chats are"
    return f"{count} authorized Telegram {label} now configured."


def build_unauthorized_chat_message(chat_id: int) -> str:
    return (
        "This chat is not authorized yet for this Unclaw bot.\n\n"
        "On the machine running Unclaw, run:\n"
        f"unclaw telegram allow {chat_id}\n"
        "or:\n"
        "unclaw telegram allow-latest\n\n"
        "Then send your message again."
    )


# --- Internal helpers ---


def _validate_telegram_token_env_var(value: str) -> str:
    from unclaw.local_secrets import validate_telegram_token_env_var_name

    return validate_telegram_token_env_var_name(value)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Missing Telegram configuration file: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Could not read Telegram configuration file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in Telegram configuration file: {path}") from exc

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"Telegram configuration must contain a mapping: {path}"
        )
    return payload


def _write_allowed_chat_ids(
    settings: Settings,
    *,
    allowed_chat_ids: list[int],
) -> bool:
    config_path = settings.paths.config_dir / _TELEGRAM_CONFIG_FILE_NAME
    payload = _load_yaml_mapping(config_path)
    if payload.get("allowed_chat_ids") == allowed_chat_ids:
        return False

    payload["allowed_chat_ids"] = allowed_chat_ids
    rendered = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=False,
    )
    temp_path = config_path.with_name(f"{config_path.name}.tmp")
    temp_path.write_text(rendered, encoding="utf-8")
    temp_path.replace(config_path)
    return True


def _read_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Telegram setting '{key}' must be a non-empty string.")
    return value.strip()


def _read_int(
    payload: dict[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ConfigurationError(f"Telegram setting '{key}' must be an integer.")
    if minimum is not None and value < minimum:
        raise ConfigurationError(
            f"Telegram setting '{key}' must be greater than or equal to {minimum}."
        )
    return value


def _read_allowed_chat_ids(value: Any) -> frozenset[int]:
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        raise ConfigurationError("Telegram setting 'allowed_chat_ids' must be a list.")

    chat_ids: set[int] = set()
    for raw_chat_id in value:
        if isinstance(raw_chat_id, bool):
            raise ConfigurationError(
                "Telegram chat ids must be integers, not boolean values."
            )
        if isinstance(raw_chat_id, int):
            chat_ids.add(raw_chat_id)
            continue
        if isinstance(raw_chat_id, str):
            stripped_value = raw_chat_id.strip()
            if not stripped_value:
                raise ConfigurationError("Telegram chat ids must not be empty strings.")
            try:
                chat_ids.add(int(stripped_value))
            except ValueError as exc:
                raise ConfigurationError(
                    "Telegram chat ids must be integers."
                ) from exc
            continue
        raise ConfigurationError("Telegram chat ids must be integers.")
    return frozenset(chat_ids)
