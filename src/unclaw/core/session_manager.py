"""Session lifecycle helpers backed by the SQLite repositories."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Self
from uuid import uuid4

from unclaw.core.mission_state import (
    MissionState,
    parse_mission_state,
)
from unclaw.core.mission_workspace import (
    MissionWorkspacePointer,
    MissionWorkspaceStore,
    parse_mission_workspace_pointer,
    serialize_mission_workspace_pointer,
)
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
_SESSION_PROGRESS_LEDGER_EVENT_TYPE = "session.progress.ledger.updated"
_SESSION_MISSION_STATE_EVENT_TYPE = "session.mission_state.updated"
_SESSION_GOAL_STATUS_VALUES = frozenset({"active", "blocked", "completed"})
_SESSION_GOAL_MAX_CHARS = 240
_SESSION_GOAL_STEP_MAX_CHARS = 80
_SESSION_GOAL_BLOCKER_MAX_CHARS = 200
_SESSION_PROGRESS_LEDGER_MAX_ENTRIES = 3
_SESSION_PROGRESS_STEP_MAX_CHARS = 80
_SESSION_PROGRESS_DETAIL_MAX_CHARS = 120


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


@dataclass(frozen=True, slots=True)
class SessionProgressEntry:
    """Compact persisted progress entry for one session."""

    status: str
    step: str
    detail: str
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
    mission_workspace_store: MissionWorkspaceStore | None = None

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
            mission_workspace_store=MissionWorkspaceStore(
                settings.paths.data_dir / "runtime" / "missions"
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
        mission_row = self._load_latest_event_row(
            resolved_session_id,
            _SESSION_MISSION_STATE_EVENT_TYPE,
        )
        goal_row = self._load_latest_event_row(
            resolved_session_id,
            _SESSION_GOAL_STATE_EVENT_TYPE,
        )
        mission_state = self.get_current_mission_state(resolved_session_id)

        if goal_row is not None and not _mission_projection_should_dominate(mission_state):
            payload_json = goal_row["payload_json"]
            if isinstance(payload_json, str):
                goal_state = _parse_session_goal_state(payload_json)
                if goal_state is not None:
                    return goal_state

        if (
            mission_state is not None
            and (
                goal_row is None
                or (
                    isinstance(mission_state.updated_at, str)
                    and isinstance(goal_row["created_at"], str)
                    and mission_state.updated_at > goal_row["created_at"]
                )
            )
        ):
            return _project_session_goal_state_from_mission(mission_state)

        if goal_row is not None:
            payload_json = goal_row["payload_json"]
            if isinstance(payload_json, str):
                goal_state = _parse_session_goal_state(payload_json)
                if goal_state is not None:
                    return goal_state

        if mission_state is None:
            return None
        return _project_session_goal_state_from_mission(mission_state)

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

    def get_session_progress_ledger(
        self,
        session_id: str | None = None,
    ) -> tuple[SessionProgressEntry, ...]:
        """Return the latest persisted progress ledger snapshot for one session."""
        resolved_session_id = self._resolve_session_id(session_id)
        mission_row = self._load_latest_event_row(
            resolved_session_id,
            _SESSION_MISSION_STATE_EVENT_TYPE,
        )
        ledger_row = self._load_latest_event_row(
            resolved_session_id,
            _SESSION_PROGRESS_LEDGER_EVENT_TYPE,
        )
        mission_state = self.get_current_mission_state(resolved_session_id)

        if ledger_row is not None and not _mission_projection_should_dominate(mission_state):
            payload_json = ledger_row["payload_json"]
            if isinstance(payload_json, str):
                ledger = _parse_session_progress_ledger(payload_json)
                if ledger:
                    return ledger

        if (
            mission_state is not None
            and (
                ledger_row is None
                or (
                    isinstance(mission_state.updated_at, str)
                    and isinstance(ledger_row["created_at"], str)
                    and mission_state.updated_at > ledger_row["created_at"]
                )
            )
        ):
            return _project_session_progress_ledger_from_mission(mission_state)

        if ledger_row is not None:
            payload_json = ledger_row["payload_json"]
            if isinstance(payload_json, str):
                ledger = _parse_session_progress_ledger(payload_json)
                if ledger:
                    return ledger

        if mission_state is None:
            return ()
        return _project_session_progress_ledger_from_mission(mission_state)

    def persist_session_progress_entry(
        self,
        *,
        status: str,
        step: str,
        detail: str,
        session_id: str | None = None,
    ) -> tuple[SessionProgressEntry, ...]:
        """Append one compact progress entry and persist the bounded latest tail."""
        resolved_session_id = self._resolve_session_id(session_id)
        if self.load_session(resolved_session_id) is None:
            raise SessionManagerError(
                f"Session '{resolved_session_id}' is not available."
            )

        timestamp = utc_now_iso()
        next_entry = SessionProgressEntry(
            status=_normalize_goal_status(status),
            step=_normalize_bounded_text(
                step,
                field_name="step",
                max_chars=_SESSION_PROGRESS_STEP_MAX_CHARS,
                required=True,
            ),
            detail=_normalize_bounded_text(
                detail,
                field_name="detail",
                max_chars=_SESSION_PROGRESS_DETAIL_MAX_CHARS,
                required=True,
            ),
            updated_at=timestamp,
        )
        existing_entries = list(
            self._load_legacy_progress_ledger_only(resolved_session_id)
        )
        ledger = tuple(
            (existing_entries + [next_entry])[-_SESSION_PROGRESS_LEDGER_MAX_ENTRIES :]
        )
        self.event_repository.add_event(
            session_id=resolved_session_id,
            event_type=_SESSION_PROGRESS_LEDGER_EVENT_TYPE,
            level=EventLevel.INFO,
            message="Session progress ledger updated.",
            payload_json=_serialize_session_progress_ledger(ledger),
            created_at=timestamp,
        )
        return ledger

    def get_current_mission_state(
        self,
        session_id: str | None = None,
    ) -> MissionState | None:
        """Return the latest persisted mission state for one session, if present."""
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
            (resolved_session_id, _SESSION_MISSION_STATE_EVENT_TYPE),
        ).fetchone()
        if row is None:
            return None
        payload_json = row["payload_json"]
        if not isinstance(payload_json, str):
            return None
        pointer = parse_mission_workspace_pointer(payload_json)
        if pointer is not None and self.mission_workspace_store is not None:
            mission_state = self.mission_workspace_store.load_mission(
                session_id=resolved_session_id,
                mission_id=pointer.mission_id,
                workspace_path=pointer.workspace_path,
            )
            if mission_state is not None:
                return mission_state
        return parse_mission_state(payload_json)

    def persist_mission_state(
        self,
        mission_state: MissionState,
        *,
        session_id: str | None = None,
        message: str = "Session mission state updated.",
    ) -> MissionState:
        """Persist one mission-state snapshot for a session."""
        resolved_session_id = self._resolve_session_id(session_id)
        if self.load_session(resolved_session_id) is None:
            raise SessionManagerError(
                f"Session '{resolved_session_id}' is not available."
            )

        if self.mission_workspace_store is None:
            raise SessionManagerError("Mission workspace store is not available.")

        workspace_path = self.mission_workspace_store.save_mission(
            session_id=resolved_session_id,
            mission_state=mission_state,
        )
        serialized_payload = serialize_mission_workspace_pointer(
            MissionWorkspacePointer(
                mission_id=mission_state.mission_id,
                workspace_path=str(workspace_path),
                updated_at=mission_state.updated_at,
                status=mission_state.status,
                executor_state=mission_state.executor_state,
                active_task_id=mission_state.active_task_id,
            )
        )
        self.event_repository.add_event(
            session_id=resolved_session_id,
            event_type=_SESSION_MISSION_STATE_EVENT_TYPE,
            level=EventLevel.INFO,
            message=message,
            payload_json=serialized_payload,
            created_at=mission_state.updated_at,
        )
        return mission_state

    def create_mission_id(self) -> str:
        """Return a new stable mission identifier."""
        return f"mission-{uuid4().hex[:12]}"

    def persist_legacy_mission_projection(
        self,
        *,
        mission_state: MissionState,
        session_id: str | None = None,
    ) -> None:
        """Emit legacy goal/progress events derived from the mission snapshot."""
        resolved_session_id = self._resolve_session_id(session_id)
        goal_state = _project_session_goal_state_from_mission(mission_state)
        progress_ledger = _project_session_progress_ledger_from_mission(mission_state)
        self.event_repository.add_event(
            session_id=resolved_session_id,
            event_type=_SESSION_GOAL_STATE_EVENT_TYPE,
            level=EventLevel.INFO,
            message="Session mission state projected to legacy goal state.",
            payload_json=_serialize_session_goal_state(goal_state),
            created_at=mission_state.updated_at,
        )
        self.event_repository.add_event(
            session_id=resolved_session_id,
            event_type=_SESSION_PROGRESS_LEDGER_EVENT_TYPE,
            level=EventLevel.INFO,
            message="Session mission state projected to legacy progress ledger.",
            payload_json=_serialize_session_progress_ledger(progress_ledger),
            created_at=mission_state.updated_at,
        )

    def _load_latest_event_row(
        self,
        session_id: str,
        event_type: str,
    ) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT payload_json, created_at
            FROM events
            WHERE session_id = ?
              AND event_type = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (session_id, event_type),
        ).fetchone()

    def _load_legacy_progress_ledger_only(
        self,
        session_id: str,
    ) -> tuple[SessionProgressEntry, ...]:
        row = self._load_latest_event_row(
            session_id,
            _SESSION_PROGRESS_LEDGER_EVENT_TYPE,
        )
        if row is None:
            return ()
        payload_json = row["payload_json"]
        if not isinstance(payload_json, str):
            return ()
        return _parse_session_progress_ledger(payload_json)

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


def _serialize_session_progress_ledger(
    ledger: tuple[SessionProgressEntry, ...],
) -> str:
    return json.dumps(
        [
            {
                "status": entry.status,
                "step": entry.step,
                "detail": entry.detail,
                "updated_at": entry.updated_at,
            }
            for entry in ledger
        ],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_session_progress_ledger(
    payload_json: str,
) -> tuple[SessionProgressEntry, ...]:
    try:
        parsed = json.loads(payload_json)
    except json.JSONDecodeError:
        return ()

    if not isinstance(parsed, list):
        return ()

    entries: list[SessionProgressEntry] = []
    for raw_entry in parsed:
        if not isinstance(raw_entry, dict):
            return ()

        status = raw_entry.get("status")
        step = raw_entry.get("step")
        detail = raw_entry.get("detail")
        updated_at = raw_entry.get("updated_at")
        if (
            not isinstance(status, str)
            or not isinstance(step, str)
            or not isinstance(detail, str)
            or not isinstance(updated_at, str)
            or not updated_at.strip()
        ):
            return ()

        try:
            entries.append(
                SessionProgressEntry(
                    status=_normalize_goal_status(status),
                    step=_normalize_bounded_text(
                        step,
                        field_name="step",
                        max_chars=_SESSION_PROGRESS_STEP_MAX_CHARS,
                        required=True,
                    ),
                    detail=_normalize_bounded_text(
                        detail,
                        field_name="detail",
                        max_chars=_SESSION_PROGRESS_DETAIL_MAX_CHARS,
                        required=True,
                    ),
                    updated_at=" ".join(updated_at.split()).strip(),
                )
            )
        except ValueError:
            return ()

    return tuple(entries[-_SESSION_PROGRESS_LEDGER_MAX_ENTRIES :])


def _project_session_goal_state_from_mission(
    mission_state: MissionState,
) -> SessionGoalState:
    active_deliverable = mission_state.get_deliverable()
    current_step = mission_state.active_task
    if current_step is None and mission_state.status == "blocked":
        blocked_deliverable = next(
            (
                deliverable
                for deliverable in reversed(mission_state.deliverables)
                if deliverable.status == "blocked"
            ),
            None,
        )
        if blocked_deliverable is not None:
            current_step = blocked_deliverable.task
    if current_step is None:
        latest_completed_deliverable = next(
            (
                deliverable
                for deliverable in reversed(mission_state.deliverables)
                if deliverable.status == "completed"
            ),
            None,
        )
        if latest_completed_deliverable is not None:
            current_step = latest_completed_deliverable.task
    last_blocker = mission_state.last_blocker
    if active_deliverable is not None:
        if current_step is None:
            current_step = active_deliverable.task
        if mission_state.status == "blocked" and last_blocker is None:
            last_blocker = active_deliverable.blocker

    return SessionGoalState(
        goal=mission_state.goal,
        status=mission_state.status,
        current_step=current_step,
        last_blocker=last_blocker,
        updated_at=mission_state.updated_at,
    )


def _mission_projection_should_dominate(mission_state: MissionState | None) -> bool:
    if mission_state is None:
        return False
    if mission_state.status != "active":
        return False
    if mission_state.executor_state in {"blocked", "completed"}:
        return False
    return bool(
        mission_state.active_deliverable_id
        or len(mission_state.deliverables) > 1
        or mission_state.retry_history
        or mission_state.repair_history
        or mission_state.pending_repairs
        or mission_state.user_visible_progress
        or mission_state.unresolved_gaps
        or mission_state.artifact_observations
    )


def _project_session_progress_ledger_from_mission(
    mission_state: MissionState,
) -> tuple[SessionProgressEntry, ...]:
    entries: list[SessionProgressEntry] = []
    projected_step = mission_state.active_task
    if projected_step is None and mission_state.status == "blocked":
        blocked_deliverable = next(
            (
                deliverable
                for deliverable in reversed(mission_state.deliverables)
                if deliverable.status == "blocked"
            ),
            None,
        )
        if blocked_deliverable is not None:
            projected_step = blocked_deliverable.task
    if projected_step is None:
        latest_completed_deliverable = next(
            (
                deliverable
                for deliverable in reversed(mission_state.deliverables)
                if deliverable.status == "completed"
            ),
            None,
        )
        if latest_completed_deliverable is not None:
            projected_step = latest_completed_deliverable.task

    def _append_entries(status: str, values: tuple[str, ...]) -> None:
        for value in values[-_SESSION_PROGRESS_LEDGER_MAX_ENTRIES :]:
            entries.append(
                SessionProgressEntry(
                    status=status,
                    step=projected_step or "mission",
                    detail=_normalize_bounded_text(
                        value,
                        field_name="detail",
                        max_chars=_SESSION_PROGRESS_DETAIL_MAX_CHARS,
                        required=True,
                    ),
                    updated_at=mission_state.updated_at,
                )
            )

    _append_entries("active", mission_state.retry_history)
    _append_entries("active", mission_state.repair_history)
    if mission_state.status == "blocked":
        blocker = mission_state.last_blocker or "mission blocked"
        entries.append(
            SessionProgressEntry(
                status="blocked",
                step=projected_step or "mission",
                detail=_normalize_bounded_text(
                    blocker,
                    field_name="detail",
                    max_chars=_SESSION_PROGRESS_DETAIL_MAX_CHARS,
                    required=True,
                ),
                updated_at=mission_state.updated_at,
            )
        )
    elif mission_state.status == "completed":
        entries.append(
            SessionProgressEntry(
                status="completed",
                step=projected_step or "mission",
                detail="mission deliverables verified",
                updated_at=mission_state.updated_at,
            )
        )
    elif projected_step is not None:
        entries.append(
            SessionProgressEntry(
                status="active",
                step=projected_step,
                detail=_normalize_bounded_text(
                    mission_state.user_visible_progress or "mission step in progress",
                    field_name="detail",
                    max_chars=_SESSION_PROGRESS_DETAIL_MAX_CHARS,
                    required=True,
                ),
                updated_at=mission_state.updated_at,
            )
        )

    return tuple(entries[-_SESSION_PROGRESS_LEDGER_MAX_ENTRIES :])


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
