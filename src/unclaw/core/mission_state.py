"""Persisted mission-state model for durable multi-step execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

_MISSION_STATUS_VALUES = frozenset({"active", "blocked", "completed"})
_DELIVERABLE_STATUS_VALUES = frozenset(
    {"pending", "active", "completed", "blocked"}
)
_EXECUTOR_STATE_VALUES = frozenset(
    {
        "planning",
        "ready",
        "executing",
        "awaiting_tool_result",
        "awaiting_verification",
        "repairing",
        "blocked",
        "completed",
    }
)
_DELIVERABLE_MODE_VALUES = frozenset({"artifact", "reply", "mixed"})
_MISSION_RELATION_VALUES = frozenset(
    {
        "same_active_mission",
        "repair_same_mission",
        "status_of_same_mission",
        "new_mission",
        "standalone_direct_reply",
    }
)
_DELIVERABLE_EVIDENCE_KIND_VALUES = frozenset(
    {
        "fast_grounding",
        "full_web_research",
        "artifact_write",
        "artifact_readback",
        "reply_emitted",
        "local_delete",
        "directory_listing",
        "calculation_result",
    }
)
_DELIVERABLE_EXECUTION_STATE_VALUES = frozenset(
    {
        "pending",
        "ready",
        "executing",
        "awaiting_tool_result",
        "awaiting_verification",
        "repairing",
        "completed",
        "blocked",
    }
)
_MAX_MISSION_ID_CHARS = 64
_MAX_MISSION_GOAL_CHARS = 240
_MAX_ACTIVE_TASK_CHARS = 120
_MAX_BLOCKER_CHARS = 200
_MAX_PLANNER_SUMMARY_CHARS = 200
_MAX_DELIVERABLE_ID_CHARS = 48
_MAX_TASK_CHARS = 120
_MAX_DELIVERABLE_CHARS = 160
_MAX_VERIFICATION_CHARS = 160
_MAX_MISSING_CHARS = 200
_MAX_HISTORY_ITEM_CHARS = 160
_MAX_WORKING_MEMORY_ITEM_CHARS = 200
_MAX_PATH_CHARS = 240
_MAX_EVIDENCE_CHARS = 200
_MAX_EXECUTOR_REASON_CHARS = 200
_MAX_WAITING_FOR_CHARS = 200
_MAX_ADVANCE_CONDITION_CHARS = 200
_MAX_VERIFIER_NOTE_CHARS = 240
_MAX_VERIFIED_REPLY_CHARS = 4000
_MAX_DELIVERABLES = 4
_MAX_HISTORY_ENTRIES = 6
_MAX_WORKING_MEMORY_ITEMS = 6
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
    mode: str = "mixed"
    required_evidence: tuple[str, ...] = ()
    execution_state: str = "pending"
    waiting_for: str | None = None
    advance_condition: str | None = None
    verifier_notes: str | None = None


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
    planner_summary: str | None = None
    execution_queue: tuple[str, ...] = ()
    completed_steps: tuple[str, ...] = ()
    failed_steps: tuple[str, ...] = ()
    observed_facts: tuple[str, ...] = ()
    artifact_facts: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    pending_repairs: tuple[str, ...] = ()
    final_deliverables_missing: tuple[str, ...] = ()
    executor_state: str = "planning"
    executor_reason: str | None = None
    waiting_for: str | None = None
    advance_condition: str | None = None
    verifier_outputs: tuple[str, ...] = ()
    final_verified_reply: str | None = None
    last_turn_relation: str | None = None

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

    normalized_state = MissionState(
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
        planner_summary=_normalize_text(
            mission_state.planner_summary,
            field_name="planner_summary",
            max_chars=_MAX_PLANNER_SUMMARY_CHARS,
        ),
        execution_queue=_normalize_text_items(
            mission_state.execution_queue,
            max_items=_MAX_DELIVERABLES,
            max_chars=_MAX_DELIVERABLE_ID_CHARS,
        ),
        completed_steps=_normalize_text_items(
            mission_state.completed_steps,
            max_items=_MAX_DELIVERABLES,
            max_chars=_MAX_DELIVERABLE_ID_CHARS,
        ),
        failed_steps=_normalize_text_items(
            mission_state.failed_steps,
            max_items=_MAX_DELIVERABLES,
            max_chars=_MAX_DELIVERABLE_ID_CHARS,
        ),
        observed_facts=_normalize_text_items(
            mission_state.observed_facts,
            max_items=_MAX_WORKING_MEMORY_ITEMS,
            max_chars=_MAX_WORKING_MEMORY_ITEM_CHARS,
        ),
        artifact_facts=_normalize_text_items(
            mission_state.artifact_facts,
            max_items=_MAX_WORKING_MEMORY_ITEMS,
            max_chars=_MAX_WORKING_MEMORY_ITEM_CHARS,
        ),
        blockers=_normalize_text_items(
            mission_state.blockers,
            max_items=_MAX_WORKING_MEMORY_ITEMS,
            max_chars=_MAX_WORKING_MEMORY_ITEM_CHARS,
        ),
        pending_repairs=_normalize_text_items(
            mission_state.pending_repairs,
            max_items=_MAX_WORKING_MEMORY_ITEMS,
            max_chars=_MAX_WORKING_MEMORY_ITEM_CHARS,
        ),
        final_deliverables_missing=_normalize_text_items(
            mission_state.final_deliverables_missing,
            max_items=_MAX_DELIVERABLES,
            max_chars=_MAX_DELIVERABLE_ID_CHARS,
        ),
        executor_state=_normalize_choice(
            mission_state.executor_state,
            field_name="executor_state",
            allowed_values=_EXECUTOR_STATE_VALUES,
        ),
        executor_reason=_normalize_text(
            mission_state.executor_reason,
            field_name="executor_reason",
            max_chars=_MAX_EXECUTOR_REASON_CHARS,
        ),
        waiting_for=_normalize_text(
            mission_state.waiting_for,
            field_name="waiting_for",
            max_chars=_MAX_WAITING_FOR_CHARS,
        ),
        advance_condition=_normalize_text(
            mission_state.advance_condition,
            field_name="advance_condition",
            max_chars=_MAX_ADVANCE_CONDITION_CHARS,
        ),
        verifier_outputs=_normalize_text_items(
            mission_state.verifier_outputs,
            max_items=_MAX_WORKING_MEMORY_ITEMS,
            max_chars=_MAX_VERIFIER_NOTE_CHARS,
        ),
        final_verified_reply=_normalize_text(
            mission_state.final_verified_reply,
            field_name="final_verified_reply",
            max_chars=_MAX_VERIFIED_REPLY_CHARS,
        ),
        last_turn_relation=_normalize_optional_choice(
            mission_state.last_turn_relation,
            field_name="last_turn_relation",
            allowed_values=_MISSION_RELATION_VALUES,
        ),
    )
    if (
        normalized_state.status == "completed"
        and normalized_state.executor_state == "planning"
        and not normalized_state.active_deliverable_id
        and not normalized_state.pending_repairs
        and not normalized_state.final_deliverables_missing
        and normalized_state.deliverables
        and all(
            deliverable.status == "completed"
            for deliverable in normalized_state.deliverables
        )
    ):
        normalized_state = replace(normalized_state, executor_state="completed")
    if (
        normalized_state.status == "blocked"
        and normalized_state.executor_state == "planning"
        and normalized_state.blocked_deliverables
    ):
        normalized_state = replace(normalized_state, executor_state="blocked")
    if mission_completion_ready(normalized_state):
        return replace(normalized_state, status="completed")
    if normalized_state.status == "completed":
        return replace(normalized_state, status="active")
    if (
        normalized_state.status == "blocked"
        and normalized_state.executor_state != "blocked"
        and not normalized_state.blocked_deliverables
    ):
        return replace(normalized_state, status="active")
    return normalized_state


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
        mode=_normalize_choice(
            deliverable.mode,
            field_name="deliverable.mode",
            allowed_values=_DELIVERABLE_MODE_VALUES,
        ),
        required_evidence=_normalize_choice_items(
            deliverable.required_evidence,
            field_name="deliverable.required_evidence",
            allowed_values=_DELIVERABLE_EVIDENCE_KIND_VALUES,
            max_items=_MAX_EVIDENCE_ITEMS,
        ),
        execution_state=_normalize_choice(
            deliverable.execution_state,
            field_name="deliverable.execution_state",
            allowed_values=_DELIVERABLE_EXECUTION_STATE_VALUES,
        ),
        waiting_for=_normalize_text(
            deliverable.waiting_for,
            field_name="deliverable.waiting_for",
            max_chars=_MAX_WAITING_FOR_CHARS,
        ),
        advance_condition=_normalize_text(
            deliverable.advance_condition,
            field_name="deliverable.advance_condition",
            max_chars=_MAX_ADVANCE_CONDITION_CHARS,
        ),
        verifier_notes=_normalize_text(
            deliverable.verifier_notes,
            field_name="deliverable.verifier_notes",
            max_chars=_MAX_VERIFIER_NOTE_CHARS,
        ),
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
                    "mode": deliverable.mode,
                    "required_evidence": list(deliverable.required_evidence),
                    "execution_state": deliverable.execution_state,
                    "waiting_for": deliverable.waiting_for,
                    "advance_condition": deliverable.advance_condition,
                    "verifier_notes": deliverable.verifier_notes,
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
            "planner_summary": normalized.planner_summary,
            "execution_queue": list(normalized.execution_queue),
            "completed_steps": list(normalized.completed_steps),
            "failed_steps": list(normalized.failed_steps),
            "observed_facts": list(normalized.observed_facts),
            "artifact_facts": list(normalized.artifact_facts),
            "blockers": list(normalized.blockers),
            "pending_repairs": list(normalized.pending_repairs),
            "final_deliverables_missing": list(
                normalized.final_deliverables_missing
            ),
            "executor_state": normalized.executor_state,
            "executor_reason": normalized.executor_reason,
            "waiting_for": normalized.waiting_for,
            "advance_condition": normalized.advance_condition,
            "verifier_outputs": list(normalized.verifier_outputs),
            "final_verified_reply": normalized.final_verified_reply,
            "last_turn_relation": normalized.last_turn_relation,
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
                planner_summary=_coerce_optional_str(parsed.get("planner_summary")),
                execution_queue=_coerce_str_tuple(parsed.get("execution_queue")),
                completed_steps=_coerce_str_tuple(parsed.get("completed_steps")),
                failed_steps=_coerce_str_tuple(parsed.get("failed_steps")),
                observed_facts=_coerce_str_tuple(parsed.get("observed_facts")),
                artifact_facts=_coerce_str_tuple(parsed.get("artifact_facts")),
                blockers=_coerce_str_tuple(parsed.get("blockers")),
                pending_repairs=_coerce_str_tuple(parsed.get("pending_repairs")),
                final_deliverables_missing=_coerce_str_tuple(
                    parsed.get("final_deliverables_missing")
                ),
                executor_state=_coerce_optional_str(parsed.get("executor_state"))
                or "planning",
                executor_reason=_coerce_optional_str(parsed.get("executor_reason")),
                waiting_for=_coerce_optional_str(parsed.get("waiting_for")),
                advance_condition=_coerce_optional_str(
                    parsed.get("advance_condition")
                ),
                verifier_outputs=_coerce_str_tuple(parsed.get("verifier_outputs")),
                final_verified_reply=_coerce_optional_str(
                    parsed.get("final_verified_reply")
                ),
                last_turn_relation=_coerce_optional_str(
                    parsed.get("last_turn_relation")
                ),
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
                mode=_coerce_optional_str(payload.get("mode")) or "mixed",
                required_evidence=_coerce_str_tuple(
                    payload.get("required_evidence")
                ),
                execution_state=(
                    _coerce_optional_str(payload.get("execution_state"))
                    or "pending"
                ),
                waiting_for=_coerce_optional_str(payload.get("waiting_for")),
                advance_condition=_coerce_optional_str(
                    payload.get("advance_condition")
                ),
                verifier_notes=_coerce_optional_str(payload.get("verifier_notes")),
            )
        )
    except (TypeError, ValueError):
        return None


def mission_completion_ready(mission_state: MissionState) -> bool:
    """Return True when mission completion invariants are all satisfied."""

    if mission_state.active_deliverable_id is not None:
        return False
    if mission_state.executor_state != "completed":
        return False
    if mission_state.execution_queue:
        return False
    if mission_state.pending_repairs:
        return False
    if mission_state.final_deliverables_missing:
        return False
    if any(deliverable.status != "completed" for deliverable in mission_state.deliverables):
        return False
    if any(
        deliverable.execution_state not in {"completed", "blocked"}
        and deliverable.status != "completed"
        for deliverable in mission_state.deliverables
    ):
        return False
    return True


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


def _normalize_optional_choice(
    value: str | None,
    *,
    field_name: str,
    allowed_values: frozenset[str],
) -> str | None:
    if value is None:
        return None
    return _normalize_choice(
        value,
        field_name=field_name,
        allowed_values=allowed_values,
    )


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


def _normalize_choice_items(
    values: tuple[str, ...] | list[str] | None,
    *,
    field_name: str,
    allowed_values: frozenset[str],
    max_items: int,
) -> tuple[str, ...]:
    if not values:
        return ()
    normalized_items: list[str] = []
    for value in values[:max_items]:
        normalized = _normalize_choice(
            value,
            field_name=field_name,
            allowed_values=allowed_values,
        )
        if normalized not in normalized_items:
            normalized_items.append(normalized)
    return tuple(normalized_items)


def _normalize_non_negative_int(value: int, *, field_name: str) -> int:
    if value < 0:
        raise ValueError(f"Field '{field_name}' must be non-negative.")
    return value


def _normalize_timestamp(value: str) -> str:
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise ValueError("updated_at is required.")
    return normalized
