"""Mission working-memory helpers for the local agent kernel."""

from __future__ import annotations

from dataclasses import replace

from unclaw.core.execution_queue import (
    missing_deliverable_ids,
    ordered_execution_queue,
    resolve_active_deliverable_id,
)
from unclaw.core.mission_state import MissionDeliverableState, MissionState
from unclaw.core.session_manager import SessionGoalState, SessionProgressEntry
from unclaw.llm.base import utc_now_iso

_MAX_WORKSPACE_ITEMS = 6


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
        ),
        updated_at=timestamp,
        observed_facts=(
            (latest_detail,) if latest_detail is not None else ()
        ),
    )


def build_mission_state_from_plan(
    *,
    mission_id: str,
    mission_goal: str,
    deliverables: tuple[MissionDeliverableState, ...],
    active_deliverable_id: str | None,
    planner_summary: str | None,
    updated_at: str,
) -> MissionState:
    """Create a persisted mission working-memory snapshot from a model plan."""

    active_deliverables = tuple(
        replace(deliverable, status="active", updated_at=updated_at)
        if deliverable.deliverable_id == active_deliverable_id
        and deliverable.status == "pending"
        else deliverable
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
        ),
        updated_at=updated_at,
    )


def synchronize_mission_working_memory(
    mission_state: MissionState,
    *,
    updated_at: str,
    planner_summary: str | None = None,
    observed_facts: tuple[str, ...] = (),
    artifact_facts: tuple[str, ...] = (),
    blockers: tuple[str, ...] = (),
    pending_repairs: tuple[str, ...] = (),
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
    status = mission_state.status
    if not final_missing and blocked_deliverables:
        status = "blocked"
    elif not final_missing:
        status = "completed"
    elif blocked_deliverables and active_deliverable_id is None:
        status = "blocked"
    elif status != "blocked":
        status = "active"

    return replace(
        mission_state,
        status=status,
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
        blockers=_append_items(mission_state.blockers, blockers),
        pending_repairs=_append_items(
            mission_state.pending_repairs,
            pending_repairs,
        ),
        final_deliverables_missing=final_missing,
    )


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
