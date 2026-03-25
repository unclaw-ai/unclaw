"""Session lifecycle helpers backed by the SQLite repositories."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Self

from unclaw.db.repositories import EventRepository, MessageRepository, SessionRepository
from unclaw.db.sqlite import initialize_schema, open_connection
from unclaw.errors import UnclawError
from unclaw.llm.base import utc_now_iso
from unclaw.memory.chat_store import ChatMemoryStore
from unclaw.schemas.chat import ChatMessage, MessageRole
from unclaw.schemas.events import EventLevel
from unclaw.schemas.session import SessionRecord, SessionSummary
from unclaw.settings import Settings

_SESSION_GOAL_STATE_EVENT_TYPE = "session.goal_state.updated"
_SESSION_GOAL_STATUS_VALUES = frozenset({"active", "blocked", "completed"})
_SESSION_GOAL_MAX_CHARS = 240
_SESSION_GOAL_STEP_MAX_CHARS = 80
_SESSION_GOAL_BLOCKER_MAX_CHARS = 200


class SessionManagerError(UnclawError):
    """Raised when session lifecycle operations fail."""


@dataclass(frozen=True, slots=True)
class SessionGoalState:
    """Compact persisted goal state for one session."""

    goal: str
    status: str
    current_step: str | None
    last_blocker: str | None
    updated_at: str


@dataclass(slots=True)
class SessionManager:
    """Coordinate session state for one running CLI process."""

    settings: Settings
    connection: sqlite3.Connection
    session_repository: SessionRepository
    message_repository: MessageRepository
    event_repository: EventRepository
    current_session_id: str | None = None
    chat_store: ChatMemoryStore | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Create a session manager bound to the configured SQLite database."""
        connection = open_connection(settings.paths.database_path)
        manager = cls(
            settings=settings,
            connection=connection,
            session_repository=SessionRepository(connection),
            message_repository=MessageRepository(connection),
            event_repository=EventRepository(connection),
            chat_store=ChatMemoryStore(
                settings.paths.data_dir / "memory" / "chats"
            ),
        )
        manager.initialize()
        return manager

    def initialize(self) -> None:
        """Ensure the SQLite schema exists and restore the current session."""
        initialize_schema(self.connection)
        self._restore_current_session()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.connection.close()

    def create_session(
        self,
        title: str | None = None,
        *,
        make_current: bool = True,
    ) -> SessionRecord:
        """Create a new session and optionally switch to it."""
        record = self.session_repository.create_session(
            title=title or self._default_session_title(),
            is_active=False,
        )
        if not make_current:
            return record
        return self.switch_session(record.id)

    def list_sessions(self, *, limit: int | None = None) -> list[SessionSummary]:
        """Return recent sessions ordered by last update time."""
        sessions = self.session_repository.list_sessions()
        if limit is None:
            return sessions
        if limit < 1:
            return []
        return sessions[:limit]

    def load_session(self, session_id: str) -> SessionRecord | None:
        """Load one session by id."""
        return self.session_repository.get_session(session_id)

    def get_current_session(self) -> SessionRecord | None:
        """Return the current in-memory session, if available."""
        if self.current_session_id is None:
            return None
        return self.load_session(self.current_session_id)

    def ensure_current_session(self) -> SessionRecord:
        """Return a current session, creating or restoring one when needed."""
        current_session = self.get_current_session()
        if current_session is not None:
            return current_session

        existing_sessions = self.list_sessions(limit=1)
        if existing_sessions:
            return self.switch_session(existing_sessions[0].id)

        return self.create_session()

    def switch_session(self, session_id: str) -> SessionRecord:
        """Switch the active session for the current CLI process."""
        session = self.load_session(session_id)
        if session is None:
            raise SessionManagerError(f"Session '{session_id}' was not found.")

        switched_session = self.session_repository.set_active_session(session_id)
        if switched_session is None:
            raise SessionManagerError(f"Session '{session_id}' could not be reloaded.")
        self.current_session_id = session_id
        self.ensure_jsonl_backfilled(session_id)
        return switched_session

    def ensure_jsonl_backfilled(self, session_id: str) -> None:
        """Backfill the JSONL chat mirror from SQLite if it is empty or missing.

        Must be called from the main thread (SQLite thread-affinity).
        No-op if the JSONL already has content or chat_store is absent.

        This covers sessions created before the JSONL mirror existed, so that
        inspect_session_history always has a complete history to read from.
        """
        if self.chat_store is None:
            return
        messages = self.message_repository.list_messages(session_id)
        if not messages:
            return
        self.chat_store.backfill_from_messages(
            session_id=session_id,
            messages=[(m.role.value, m.content, m.created_at) for m in messages],
        )

    def rename_session(self, session_id: str, title: str) -> SessionRecord:
        """Rename an existing session."""
        updated_session = self.session_repository.update_session(
            session_id,
            title=title,
        )
        if updated_session is None:
            raise SessionManagerError(f"Session '{session_id}' was not found.")
        return updated_session

    def add_message(
        self,
        role: MessageRole | str,
        content: str,
        *,
        session_id: str | None = None,
    ) -> ChatMessage:
        """Persist one chat message for a session."""
        resolved_session_id = session_id or self.current_session_id
        if resolved_session_id is None:
            raise SessionManagerError("No active session is available.")

        if self.load_session(resolved_session_id) is None:
            raise SessionManagerError(
                f"Session '{resolved_session_id}' is not available."
            )

        message = self.message_repository.add_message(
            session_id=resolved_session_id,
            role=role,
            content=content,
        )
        if self.chat_store is not None:
            self.chat_store.append_message(
                session_id=resolved_session_id,
                role=message.role.value,
                content=message.content,
                created_at=message.created_at,
            )
        return message

    def list_messages(self, session_id: str | None = None) -> list[ChatMessage]:
        """Return the stored messages for one session."""
        resolved_session_id = session_id or self.current_session_id
        if resolved_session_id is None:
            raise SessionManagerError("No active session is available.")
        return self.message_repository.list_messages(resolved_session_id)

    def get_session_goal_state(
        self,
        session_id: str | None = None,
    ) -> SessionGoalState | None:
        """Return the latest persisted goal state for one session, if present."""
        resolved_session_id = self._resolve_session_id(session_id)
        row = self.connection.execute(
            """
            SELECT payload_json
            FROM events
            WHERE session_id = ?
              AND event_type = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (resolved_session_id, _SESSION_GOAL_STATE_EVENT_TYPE),
        ).fetchone()
        if row is None:
            return None
        payload_json = row["payload_json"]
        if not isinstance(payload_json, str):
            return None
        return _parse_session_goal_state(payload_json)

    def persist_session_goal_state(
        self,
        *,
        goal: str,
        status: str,
        current_step: str | None = None,
        last_blocker: str | None = None,
        session_id: str | None = None,
    ) -> SessionGoalState:
        """Persist one compact goal-state snapshot for a session."""
        resolved_session_id = self._resolve_session_id(session_id)
        if self.load_session(resolved_session_id) is None:
            raise SessionManagerError(
                f"Session '{resolved_session_id}' is not available."
            )

        timestamp = utc_now_iso()
        goal_state = SessionGoalState(
            goal=_normalize_bounded_text(
                goal,
                field_name="goal",
                max_chars=_SESSION_GOAL_MAX_CHARS,
                required=True,
            ),
            status=_normalize_goal_status(status),
            current_step=_normalize_bounded_text(
                current_step,
                field_name="current_step",
                max_chars=_SESSION_GOAL_STEP_MAX_CHARS,
            ),
            last_blocker=_normalize_bounded_text(
                last_blocker,
                field_name="last_blocker",
                max_chars=_SESSION_GOAL_BLOCKER_MAX_CHARS,
            ),
            updated_at=timestamp,
        )
        self.event_repository.add_event(
            session_id=resolved_session_id,
            event_type=_SESSION_GOAL_STATE_EVENT_TYPE,
            level=EventLevel.INFO,
            message="Session goal state updated.",
            payload_json=_serialize_session_goal_state(goal_state),
            created_at=timestamp,
        )
        return goal_state

    def _restore_current_session(self) -> None:
        sessions = self.session_repository.list_sessions()
        if not sessions:
            self.current_session_id = None
            return

        active_session = next((session for session in sessions if session.is_active), None)
        selected_session = active_session or sessions[0]
        self.current_session_id = selected_session.id
        self.ensure_jsonl_backfilled(selected_session.id)

    def _default_session_title(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"Session {timestamp}"

    def _resolve_session_id(self, session_id: str | None) -> str:
        resolved_session_id = session_id or self.current_session_id
        if resolved_session_id is None:
            raise SessionManagerError("No active session is available.")
        return resolved_session_id


def _serialize_session_goal_state(goal_state: SessionGoalState) -> str:
    return json.dumps(
        {
            "goal": goal_state.goal,
            "status": goal_state.status,
            "current_step": goal_state.current_step,
            "last_blocker": goal_state.last_blocker,
            "updated_at": goal_state.updated_at,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_session_goal_state(payload_json: str) -> SessionGoalState | None:
    try:
        parsed = json.loads(payload_json)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    goal = parsed.get("goal")
    status = parsed.get("status")
    current_step = parsed.get("current_step")
    last_blocker = parsed.get("last_blocker")
    updated_at = parsed.get("updated_at")
    if not isinstance(goal, str) or not isinstance(status, str):
        return None
    if not isinstance(updated_at, str) or not updated_at.strip():
        return None
    if current_step is not None and not isinstance(current_step, str):
        return None
    if last_blocker is not None and not isinstance(last_blocker, str):
        return None

    try:
        return SessionGoalState(
            goal=_normalize_bounded_text(
                goal,
                field_name="goal",
                max_chars=_SESSION_GOAL_MAX_CHARS,
                required=True,
            ),
            status=_normalize_goal_status(status),
            current_step=_normalize_bounded_text(
                current_step,
                field_name="current_step",
                max_chars=_SESSION_GOAL_STEP_MAX_CHARS,
            ),
            last_blocker=_normalize_bounded_text(
                last_blocker,
                field_name="last_blocker",
                max_chars=_SESSION_GOAL_BLOCKER_MAX_CHARS,
            ),
            updated_at=" ".join(updated_at.split()).strip(),
        )
    except ValueError:
        return None


def _normalize_goal_status(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _SESSION_GOAL_STATUS_VALUES:
        raise ValueError(f"Unsupported session goal status: {value!r}.")
    return normalized


def _normalize_bounded_text(
    value: str | None,
    *,
    field_name: str,
    max_chars: int,
    required: bool = False,
) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{field_name} must be a non-empty string.")
        return None

    normalized = " ".join(value.split()).strip()
    if not normalized:
        if required:
            raise ValueError(f"{field_name} must be a non-empty string.")
        return None

    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip(' ,;:.')}..."
