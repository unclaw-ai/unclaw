"""Mission working-memory helpers for the local agent kernel."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from unclaw.core.execution_queue import (
    missing_deliverable_ids,
    ordered_execution_queue,
    resolve_active_deliverable_id,
)
from unclaw.core.mission_state import (
    MissionDeliverableState,
    MissionState,
    mission_completion_ready,
    normalize_mission_state,
    parse_mission_state,
    serialize_mission_state,
)
from unclaw.llm.base import utc_now_iso

if TYPE_CHECKING:
    from unclaw.core.session_manager import SessionGoalState, SessionProgressEntry

_MAX_WORKSPACE_ITEMS = 6
_KEEP_VALUE = object()


@dataclass(frozen=True, slots=True)
class MissionWorkspacePointer:
    """Compact DB-stored pointer to the external mission workspace."""

    mission_id: str
    workspace_path: str
    updated_at: str
    status: str
    executor_state: str
    active_deliverable_id: str | None = None


@dataclass(slots=True)
class MissionWorkspaceStore:
    """Persist mission working memory under the local runtime data area."""

    base_dir: Path

    def workspace_path(self, *, session_id: str, mission_id: str) -> Path:
        return self.base_dir / session_id / f"{mission_id}.json"

    def save_mission(self, *, session_id: str, mission_state: MissionState) -> Path:
        normalized_state = normalize_mission_state(mission_state)
        workspace_path = self.workspace_path(
            session_id=session_id,
            mission_id=normalized_state.mission_id,
        )
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = workspace_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            serialize_mission_state(normalized_state),
            encoding="utf-8",
        )
        os.replace(temporary_path, workspace_path)
        return workspace_path

    def load_mission(
        self,
        *,
        session_id: str,
        mission_id: str,
        workspace_path: str | None = None,
    ) -> MissionState | None:
        candidate_path = (
            Path(workspace_path)
            if isinstance(workspace_path, str) and workspace_path.strip()
            else self.workspace_path(session_id=session_id, mission_id=mission_id)
        )
        try:
            payload = candidate_path.read_text(encoding="utf-8")
        except OSError:
            return None
        return parse_mission_state(payload)


def serialize_mission_workspace_pointer(pointer: MissionWorkspacePointer) -> str:
    """Serialize one mission workspace pointer for SQLite event storage."""

    return json.dumps(
        {
            "schema": "mission_workspace_pointer.v1",
            "mission_id": pointer.mission_id,
            "workspace_path": pointer.workspace_path,
            "updated_at": pointer.updated_at,
            "status": pointer.status,
            "executor_state": pointer.executor_state,
            "active_deliverable_id": pointer.active_deliverable_id,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_mission_workspace_pointer(
    payload_json: str,
) -> MissionWorkspacePointer | None:
    """Parse one compact mission workspace pointer from SQLite event storage."""

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != "mission_workspace_pointer.v1":
        return None
    mission_id = payload.get("mission_id")
    workspace_path = payload.get("workspace_path")
    updated_at = payload.get("updated_at")
    status = payload.get("status")
    executor_state = payload.get("executor_state")
    active_deliverable_id = payload.get("active_deliverable_id")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (mission_id, workspace_path, updated_at, status, executor_state)
    ):
        return None
    if active_deliverable_id is not None and not isinstance(active_deliverable_id, str):
        return None
    return MissionWorkspacePointer(
        mission_id=mission_id,
        workspace_path=workspace_path,
        updated_at=updated_at,
        status=status,
        executor_state=executor_state,
        active_deliverable_id=active_deliverable_id,
    )


def build_compatibility_mission_state(
    *,
    legacy_goal_state: SessionGoalState,
    legacy_progress_ledger: tuple[SessionProgressEntry, ...] = (),
    updated_at: str | None = None,
) -> MissionState | None:
    """Project a legacy active or blocked goal into weak mission compatibility."""

    if legacy_goal_state.status not in {"active", "blocked"}:
        return None

    timestamp = updated_at or legacy_goal_state.updated_at or utc_now_iso()
    current_step = legacy_goal_state.current_step or "legacy-mission-step"
    latest_detail = (
        legacy_progress_ledger[-1].detail if legacy_progress_ledger else None
    )
    deliverable = MissionDeliverableState(
        deliverable_id="legacy-d1",
        task=current_step,
        deliverable=legacy_goal_state.goal,
        verification="Verify the legacy mission from runtime facts.",
        status=legacy_goal_state.status,
        missing=latest_detail if legacy_goal_state.status == "active" else None,
        blocker=legacy_goal_state.last_blocker,
        attempt_count=0,
        repair_count=0,
        retry_count=0,
        artifact_paths=(),
        evidence=(),
        updated_at=timestamp,
        mode="mixed",
        execution_state=(
            "ready" if legacy_goal_state.status == "active" else "blocked"
        ),
        waiting_for=latest_detail if legacy_goal_state.status == "active" else None,
        advance_condition="Verify the legacy mission from runtime facts.",
        verifier_notes=latest_detail,
    )
    return synchronize_mission_working_memory(
        MissionState(
            mission_id="legacy-goal-compat",
            goal=legacy_goal_state.goal,
            status=legacy_goal_state.status,
            active_deliverable_id=(
                deliverable.deliverable_id
                if legacy_goal_state.status == "active"
                else None
            ),
            active_task=current_step if legacy_goal_state.status == "active" else None,
            completed_deliverables=(),
            blocked_deliverables=(
                (deliverable.deliverable_id,)
                if legacy_goal_state.status == "blocked"
                else ()
            ),
            deliverables=(deliverable,),
            retry_history=(),
            repair_history=(),
            last_verified_artifact_paths=(),
            last_successful_evidence=(),
            last_blocker=legacy_goal_state.last_blocker,
            updated_at=timestamp,
            planner_summary="legacy compatibility mission",
            blockers=(
                (legacy_goal_state.last_blocker,)
                if legacy_goal_state.last_blocker is not None
                else ()
            ),
            executor_state=(
                "ready" if legacy_goal_state.status == "active" else "blocked"
            ),
            executor_reason="legacy compatibility mission",
            waiting_for=latest_detail if legacy_goal_state.status == "active" else None,
            advance_condition=deliverable.verification,
        ),
        updated_at=timestamp,
        observed_facts=(
            (latest_detail,) if latest_detail is not None else ()
        ),
        pending_repairs=(),
    )


def build_mission_state_from_plan(
    *,
    mission_id: str,
    mission_goal: str,
    deliverables: tuple[MissionDeliverableState, ...],
    active_deliverable_id: str | None,
    execution_queue: tuple[str, ...] = (),
    planner_summary: str | None,
    last_turn_relation: str | None = None,
    updated_at: str,
) -> MissionState:
    """Create a persisted mission working-memory snapshot from a model plan."""

    active_deliverables = tuple(
        replace(
            deliverable,
            status="active",
            updated_at=updated_at,
            execution_state="ready",
            waiting_for=f"start {deliverable.task}",
            advance_condition=deliverable.verification,
        )
        if deliverable.deliverable_id == active_deliverable_id
        and deliverable.status == "pending"
        else replace(
            deliverable,
            execution_state=(
                deliverable.execution_state
                if deliverable.execution_state != "pending"
                else "pending"
            ),
            advance_condition=deliverable.verification,
        )
        for deliverable in deliverables
    )
    active_deliverable = next(
        (
            deliverable
            for deliverable in active_deliverables
            if deliverable.deliverable_id == active_deliverable_id
        ),
        None,
    )
    return synchronize_mission_working_memory(
        MissionState(
            mission_id=mission_id,
            goal=mission_goal,
            status="active",
            active_deliverable_id=active_deliverable_id,
            active_task=active_deliverable.task if active_deliverable is not None else None,
            completed_deliverables=(),
            blocked_deliverables=(),
            deliverables=active_deliverables,
            retry_history=(),
            repair_history=(),
            last_verified_artifact_paths=(),
            last_successful_evidence=(),
            last_blocker=None,
            updated_at=updated_at,
            planner_summary=planner_summary,
            last_turn_relation=last_turn_relation,
            execution_queue=execution_queue,
            executor_state="ready",
            executor_reason="mission planned",
            waiting_for=(
                f"execute {active_deliverable.task}"
                if active_deliverable is not None
                else None
            ),
            advance_condition=(
                active_deliverable.verification
                if active_deliverable is not None
                else None
            ),
        ),
        updated_at=updated_at,
        pending_repairs=(),
        verifier_outputs=(),
    )


def synchronize_mission_working_memory(
    mission_state: MissionState,
    *,
    updated_at: str,
    planner_summary: str | None = None,
    observed_facts: tuple[str, ...] = (),
    artifact_facts: tuple[str, ...] = (),
    blockers: tuple[str, ...] | None = None,
    pending_repairs: tuple[str, ...] | None = None,
    executor_state: str | None = None,
    executor_reason: str | object = _KEEP_VALUE,
    waiting_for: str | object = _KEEP_VALUE,
    advance_condition: str | object = _KEEP_VALUE,
    verifier_outputs: tuple[str, ...] | None = None,
    final_verified_reply: str | object = _KEEP_VALUE,
) -> MissionState:
    """Refresh compact queue and fact memory after a mission state transition."""

    execution_queue = ordered_execution_queue(
        deliverables=mission_state.deliverables,
        preferred_queue=mission_state.execution_queue,
    )
    active_deliverable_id = resolve_active_deliverable_id(
        mission_state=mission_state,
        preferred_active_deliverable_id=mission_state.active_deliverable_id,
    )
    active_deliverable = mission_state.get_deliverable(active_deliverable_id)
    completed_deliverables = tuple(
        deliverable.deliverable_id
        for deliverable in mission_state.deliverables
        if deliverable.status == "completed"
    )
    blocked_deliverables = tuple(
        deliverable.deliverable_id
        for deliverable in mission_state.deliverables
        if deliverable.status == "blocked"
    )
    final_missing = missing_deliverable_ids(mission_state.deliverables)
    resolved_executor_state = executor_state or mission_state.executor_state
    resolved_pending_repairs = _replace_items(
        mission_state.pending_repairs,
        pending_repairs,
    )
    normalized_state = replace(
        mission_state,
        status=mission_state.status,
        active_deliverable_id=active_deliverable_id,
        active_task=active_deliverable.task if active_deliverable is not None else None,
        completed_deliverables=completed_deliverables,
        blocked_deliverables=blocked_deliverables,
        updated_at=updated_at,
        planner_summary=planner_summary or mission_state.planner_summary,
        execution_queue=execution_queue,
        completed_steps=completed_deliverables,
        failed_steps=blocked_deliverables,
        observed_facts=_append_items(mission_state.observed_facts, observed_facts),
        artifact_facts=_append_items(mission_state.artifact_facts, artifact_facts),
        blockers=_replace_items(mission_state.blockers, blockers),
        pending_repairs=resolved_pending_repairs,
        final_deliverables_missing=final_missing,
        executor_state=resolved_executor_state,
        executor_reason=(
            mission_state.executor_reason
            if executor_reason is _KEEP_VALUE
            else executor_reason
        ),
        waiting_for=(
            mission_state.waiting_for
            if waiting_for is _KEEP_VALUE
            else waiting_for
        ),
        advance_condition=(
            mission_state.advance_condition
            if advance_condition is _KEEP_VALUE
            else advance_condition
        ),
        verifier_outputs=_replace_items(
            mission_state.verifier_outputs,
            verifier_outputs,
        ),
        final_verified_reply=(
            mission_state.final_verified_reply
            if final_verified_reply is _KEEP_VALUE
            else final_verified_reply
        ),
    )
    if mission_completion_ready(normalized_state):
        return replace(normalized_state, status="completed")
    if resolved_executor_state == "blocked" or (
        blocked_deliverables and active_deliverable_id is None
    ):
        return replace(normalized_state, status="blocked")
    return replace(normalized_state, status="active")


def _append_items(
    existing: tuple[str, ...],
    new_items: tuple[str, ...],
) -> tuple[str, ...]:
    combined: list[str] = [item for item in existing if item]
    for item in new_items:
        if not item or item in combined:
            continue
        combined.append(item)
    return tuple(combined[-_MAX_WORKSPACE_ITEMS:])


def _replace_items(
    existing: tuple[str, ...],
    new_items: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if new_items is None:
        return existing
    combined: list[str] = []
    for item in new_items:
        if not item or item in combined:
            continue
        combined.append(item)
    return tuple(combined[-_MAX_WORKSPACE_ITEMS:])
