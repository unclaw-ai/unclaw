"""Persisted mission-state model for durable multi-step execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

_MISSION_STATUS_VALUES = frozenset({"active", "blocked", "completed"})
_DELIVERABLE_STATUS_VALUES = frozenset(
    {"pending", "active", "completed", "blocked"}
)
_MAX_MISSION_ID_CHARS = 64
_MAX_MISSION_GOAL_CHARS = 240
_MAX_ACTIVE_TASK_CHARS = 120
_MAX_BLOCKER_CHARS = 200
_MAX_DELIVERABLE_ID_CHARS = 48
_MAX_TASK_CHARS = 120
_MAX_DELIVERABLE_CHARS = 160
_MAX_VERIFICATION_CHARS = 160
_MAX_MISSING_CHARS = 200
_MAX_HISTORY_ITEM_CHARS = 160
_MAX_PATH_CHARS = 240
_MAX_EVIDENCE_CHARS = 200
_MAX_DELIVERABLES = 4
_MAX_HISTORY_ENTRIES = 6
_MAX_ARTIFACT_PATHS = 4
_MAX_EVIDENCE_ITEMS = 4


@dataclass(frozen=True, slots=True)
class MissionDeliverableState:
    """Persisted state for one mission deliverable."""

    deliverable_id: str
    task: str
    deliverable: str
    verification: str
    status: str
    missing: str | None
    blocker: str | None
    attempt_count: int
    repair_count: int
    retry_count: int
    artifact_paths: tuple[str, ...]
    evidence: tuple[str, ...]
    updated_at: str


@dataclass(frozen=True, slots=True)
class MissionState:
    """Persisted mission snapshot for one session."""

    mission_id: str
    goal: str
    status: str
    active_deliverable_id: str | None
    active_task: str | None
    completed_deliverables: tuple[str, ...]
    blocked_deliverables: tuple[str, ...]
    deliverables: tuple[MissionDeliverableState, ...]
    retry_history: tuple[str, ...]
    repair_history: tuple[str, ...]
    last_verified_artifact_paths: tuple[str, ...]
    last_successful_evidence: tuple[str, ...]
    last_blocker: str | None
    updated_at: str

    def get_deliverable(
        self,
        deliverable_id: str | None = None,
    ) -> MissionDeliverableState | None:
        resolved_id = deliverable_id or self.active_deliverable_id
        if resolved_id is None:
            return None
        return next(
            (
                deliverable
                for deliverable in self.deliverables
                if deliverable.deliverable_id == resolved_id
            ),
            None,
        )

    def replace_deliverable(
        self,
        deliverable: MissionDeliverableState,
        *,
        updated_at: str,
    ) -> MissionState:
        next_deliverables = tuple(
            deliverable
            if current.deliverable_id == deliverable.deliverable_id
            else current
            for current in self.deliverables
        )
        completed_ids = tuple(
            item.deliverable_id
            for item in next_deliverables
            if item.status == "completed"
        )
        blocked_ids = tuple(
            item.deliverable_id
            for item in next_deliverables
            if item.status == "blocked"
        )
        active_deliverable = next(
            (
                item
                for item in next_deliverables
                if item.status in {"active", "pending"}
            ),
            None,
        )
        return replace(
            self,
            deliverables=next_deliverables,
            completed_deliverables=completed_ids,
            blocked_deliverables=blocked_ids,
            active_deliverable_id=(
                active_deliverable.deliverable_id
                if active_deliverable is not None
                else None
            ),
            active_task=(
                active_deliverable.task if active_deliverable is not None else None
            ),
            updated_at=updated_at,
            last_blocker=(
                deliverable.blocker
                if deliverable.status == "blocked"
                else self.last_blocker
            ),
        )


def normalize_mission_state(mission_state: MissionState) -> MissionState:
    """Return a normalized copy safe for persistence."""

    normalized_deliverables = tuple(
        normalize_mission_deliverable(deliverable)
        for deliverable in mission_state.deliverables[:_MAX_DELIVERABLES]
    )
    active_deliverable = next(
        (
            deliverable
            for deliverable in normalized_deliverables
            if deliverable.deliverable_id == mission_state.active_deliverable_id
        ),
        None,
    )
    if active_deliverable is None:
        active_deliverable = next(
            (
                deliverable
                for deliverable in normalized_deliverables
                if deliverable.status in {"active", "pending"}
            ),
            None,
        )
    completed_deliverables = tuple(
        deliverable.deliverable_id
        for deliverable in normalized_deliverables
        if deliverable.status == "completed"
    )
    blocked_deliverables = tuple(
        deliverable.deliverable_id
        for deliverable in normalized_deliverables
        if deliverable.status == "blocked"
    )
    normalized_status = _normalize_choice(
        mission_state.status,
        field_name="status",
        allowed_values=_MISSION_STATUS_VALUES,
    )
    if normalized_status == "completed" and any(
        deliverable.status != "completed" for deliverable in normalized_deliverables
    ):
        normalized_status = "active"
    if normalized_status == "blocked" and not blocked_deliverables:
        normalized_status = "active"

    return MissionState(
        mission_id=_normalize_text(
            mission_state.mission_id,
            field_name="mission_id",
            max_chars=_MAX_MISSION_ID_CHARS,
            required=True,
        ),
        goal=_normalize_text(
            mission_state.goal,
            field_name="goal",
            max_chars=_MAX_MISSION_GOAL_CHARS,
            required=True,
        ),
        status=normalized_status,
        active_deliverable_id=(
            active_deliverable.deliverable_id
            if active_deliverable is not None
            else _normalize_text(
                mission_state.active_deliverable_id,
                field_name="active_deliverable_id",
                max_chars=_MAX_DELIVERABLE_ID_CHARS,
            )
        ),
        active_task=(
            active_deliverable.task
            if active_deliverable is not None
            else _normalize_text(
                mission_state.active_task,
                field_name="active_task",
                max_chars=_MAX_ACTIVE_TASK_CHARS,
            )
        ),
        completed_deliverables=completed_deliverables,
        blocked_deliverables=blocked_deliverables,
        deliverables=normalized_deliverables,
        retry_history=_normalize_text_items(
            mission_state.retry_history,
            max_items=_MAX_HISTORY_ENTRIES,
            max_chars=_MAX_HISTORY_ITEM_CHARS,
        ),
        repair_history=_normalize_text_items(
            mission_state.repair_history,
            max_items=_MAX_HISTORY_ENTRIES,
            max_chars=_MAX_HISTORY_ITEM_CHARS,
        ),
        last_verified_artifact_paths=_normalize_text_items(
            mission_state.last_verified_artifact_paths,
            max_items=_MAX_ARTIFACT_PATHS,
            max_chars=_MAX_PATH_CHARS,
        ),
        last_successful_evidence=_normalize_text_items(
            mission_state.last_successful_evidence,
            max_items=_MAX_EVIDENCE_ITEMS,
            max_chars=_MAX_EVIDENCE_CHARS,
        ),
        last_blocker=_normalize_text(
            mission_state.last_blocker,
            field_name="last_blocker",
            max_chars=_MAX_BLOCKER_CHARS,
        ),
        updated_at=_normalize_timestamp(mission_state.updated_at),
    )


def normalize_mission_deliverable(
    deliverable: MissionDeliverableState,
) -> MissionDeliverableState:
    """Return a normalized deliverable snapshot safe for persistence."""

    return MissionDeliverableState(
        deliverable_id=_normalize_text(
            deliverable.deliverable_id,
            field_name="deliverable_id",
            max_chars=_MAX_DELIVERABLE_ID_CHARS,
            required=True,
        ),
        task=_normalize_text(
            deliverable.task,
            field_name="task",
            max_chars=_MAX_TASK_CHARS,
            required=True,
        ),
        deliverable=_normalize_text(
            deliverable.deliverable,
            field_name="deliverable",
            max_chars=_MAX_DELIVERABLE_CHARS,
            required=True,
        ),
        verification=_normalize_text(
            deliverable.verification,
            field_name="verification",
            max_chars=_MAX_VERIFICATION_CHARS,
            required=True,
        ),
        status=_normalize_choice(
            deliverable.status,
            field_name="deliverable.status",
            allowed_values=_DELIVERABLE_STATUS_VALUES,
        ),
        missing=_normalize_text(
            deliverable.missing,
            field_name="missing",
            max_chars=_MAX_MISSING_CHARS,
        ),
        blocker=_normalize_text(
            deliverable.blocker,
            field_name="blocker",
            max_chars=_MAX_BLOCKER_CHARS,
        ),
        attempt_count=_normalize_non_negative_int(
            deliverable.attempt_count,
            field_name="attempt_count",
        ),
        repair_count=_normalize_non_negative_int(
            deliverable.repair_count,
            field_name="repair_count",
        ),
        retry_count=_normalize_non_negative_int(
            deliverable.retry_count,
            field_name="retry_count",
        ),
        artifact_paths=_normalize_text_items(
            deliverable.artifact_paths,
            max_items=_MAX_ARTIFACT_PATHS,
            max_chars=_MAX_PATH_CHARS,
        ),
        evidence=_normalize_text_items(
            deliverable.evidence,
            max_items=_MAX_EVIDENCE_ITEMS,
            max_chars=_MAX_EVIDENCE_CHARS,
        ),
        updated_at=_normalize_timestamp(deliverable.updated_at),
    )


def serialize_mission_state(mission_state: MissionState) -> str:
    """Serialize a mission-state snapshot to compact JSON."""

    normalized = normalize_mission_state(mission_state)
    return json.dumps(
        {
            "mission_id": normalized.mission_id,
            "goal": normalized.goal,
            "status": normalized.status,
            "active_deliverable_id": normalized.active_deliverable_id,
            "active_task": normalized.active_task,
            "completed_deliverables": list(normalized.completed_deliverables),
            "blocked_deliverables": list(normalized.blocked_deliverables),
            "deliverables": [
                {
                    "deliverable_id": deliverable.deliverable_id,
                    "task": deliverable.task,
                    "deliverable": deliverable.deliverable,
                    "verification": deliverable.verification,
                    "status": deliverable.status,
                    "missing": deliverable.missing,
                    "blocker": deliverable.blocker,
                    "attempt_count": deliverable.attempt_count,
                    "repair_count": deliverable.repair_count,
                    "retry_count": deliverable.retry_count,
                    "artifact_paths": list(deliverable.artifact_paths),
                    "evidence": list(deliverable.evidence),
                    "updated_at": deliverable.updated_at,
                }
                for deliverable in normalized.deliverables
            ],
            "retry_history": list(normalized.retry_history),
            "repair_history": list(normalized.repair_history),
            "last_verified_artifact_paths": list(
                normalized.last_verified_artifact_paths
            ),
            "last_successful_evidence": list(normalized.last_successful_evidence),
            "last_blocker": normalized.last_blocker,
            "updated_at": normalized.updated_at,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_mission_state(payload_json: str) -> MissionState | None:
    """Parse one mission-state snapshot from JSON."""

    try:
        parsed = json.loads(payload_json)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    raw_deliverables = parsed.get("deliverables")
    if not isinstance(raw_deliverables, list):
        return None

    deliverables: list[MissionDeliverableState] = []
    for raw_deliverable in raw_deliverables:
        deliverable = _parse_mission_deliverable(raw_deliverable)
        if deliverable is None:
            return None
        deliverables.append(deliverable)

    try:
        return normalize_mission_state(
            MissionState(
                mission_id=_coerce_required_str(parsed.get("mission_id")),
                goal=_coerce_required_str(parsed.get("goal")),
                status=_coerce_required_str(parsed.get("status")),
                active_deliverable_id=_coerce_optional_str(
                    parsed.get("active_deliverable_id")
                ),
                active_task=_coerce_optional_str(parsed.get("active_task")),
                completed_deliverables=_coerce_str_tuple(
                    parsed.get("completed_deliverables")
                ),
                blocked_deliverables=_coerce_str_tuple(
                    parsed.get("blocked_deliverables")
                ),
                deliverables=tuple(deliverables),
                retry_history=_coerce_str_tuple(parsed.get("retry_history")),
                repair_history=_coerce_str_tuple(parsed.get("repair_history")),
                last_verified_artifact_paths=_coerce_str_tuple(
                    parsed.get("last_verified_artifact_paths")
                ),
                last_successful_evidence=_coerce_str_tuple(
                    parsed.get("last_successful_evidence")
                ),
                last_blocker=_coerce_optional_str(parsed.get("last_blocker")),
                updated_at=_coerce_required_str(parsed.get("updated_at")),
            )
        )
    except (TypeError, ValueError):
        return None


def _parse_mission_deliverable(payload: Any) -> MissionDeliverableState | None:
    if not isinstance(payload, dict):
        return None

    try:
        return normalize_mission_deliverable(
            MissionDeliverableState(
                deliverable_id=_coerce_required_str(payload.get("deliverable_id")),
                task=_coerce_required_str(payload.get("task")),
                deliverable=_coerce_required_str(payload.get("deliverable")),
                verification=_coerce_required_str(payload.get("verification")),
                status=_coerce_required_str(payload.get("status")),
                missing=_coerce_optional_str(payload.get("missing")),
                blocker=_coerce_optional_str(payload.get("blocker")),
                attempt_count=_coerce_non_negative_int(payload.get("attempt_count")),
                repair_count=_coerce_non_negative_int(payload.get("repair_count")),
                retry_count=_coerce_non_negative_int(payload.get("retry_count")),
                artifact_paths=_coerce_str_tuple(payload.get("artifact_paths")),
                evidence=_coerce_str_tuple(payload.get("evidence")),
                updated_at=_coerce_required_str(payload.get("updated_at")),
            )
        )
    except (TypeError, ValueError):
        return None


def _coerce_required_str(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("expected string")
    return value


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("expected optional string")
    return value


def _coerce_non_negative_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError("expected non-negative integer")
    return value


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("expected string list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("expected string list")
        result.append(item)
    return tuple(result)


def _normalize_choice(
    value: str,
    *,
    field_name: str,
    allowed_values: frozenset[str],
) -> str:
    normalized = " ".join(value.split()).strip().lower()
    if normalized not in allowed_values:
        raise ValueError(
            f"Field '{field_name}' must be one of {sorted(allowed_values)!r}."
        )
    return normalized


def _normalize_text(
    value: str | None,
    *,
    field_name: str,
    max_chars: int,
    required: bool = False,
) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"Field '{field_name}' is required.")
        return None
    normalized = " ".join(value.split()).strip()
    if not normalized:
        if required:
            raise ValueError(f"Field '{field_name}' is required.")
        return None
    return normalized[:max_chars]


def _normalize_text_items(
    values: tuple[str, ...] | list[str] | None,
    *,
    max_items: int,
    max_chars: int,
) -> tuple[str, ...]:
    if not values:
        return ()
    normalized: list[str] = []
    for value in values[:max_items]:
        item = _normalize_text(
            value,
            field_name="text_item",
            max_chars=max_chars,
            required=True,
        )
        if item is not None:
            normalized.append(item)
    return tuple(normalized)


def _normalize_non_negative_int(value: int, *, field_name: str) -> int:
    if value < 0:
        raise ValueError(f"Field '{field_name}' must be non-negative.")
    return value


def _normalize_timestamp(value: str) -> str:
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise ValueError("updated_at is required.")
    return normalized

