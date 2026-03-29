"""Mission workspace helpers for file-backed mission persistence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from unclaw.core.mission_state import (
    MissionState,
    MissionTaskState,
    normalize_mission_state,
    parse_mission_state,
    serialize_mission_state,
)
from unclaw.llm.base import utc_now_iso

if TYPE_CHECKING:
    from unclaw.core.session_manager import SessionGoalState, SessionProgressEntry
    from unclaw.tools.contracts import ToolCall, ToolResult


@dataclass(frozen=True, slots=True)
class MissionWorkspacePointer:
    """Compact DB-stored pointer to the external mission workspace."""

    mission_id: str
    workspace_path: str
    updated_at: str
    status: str
    executor_state: str
    active_task_id: str | None = None

    @property
    def active_deliverable_id(self) -> str | None:
        return self.active_task_id


@dataclass(slots=True)
class MissionWorkspaceStore:
    """Persist mission working memory under the local runtime data area."""

    base_dir: Path

    def workspace_path(self, *, session_id: str, mission_id: str) -> Path:
        return self.base_dir / session_id / f"{mission_id}.json"

    def observation_dir(self, *, session_id: str, mission_id: str) -> Path:
        return self.base_dir / session_id / f"{mission_id}.observations"

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

    def save_tool_observation(
        self,
        *,
        session_id: str,
        mission_id: str,
        observation_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
    ) -> Path:
        """Persist one raw tool result outside the main mission loop context."""

        observation_dir = self.observation_dir(
            session_id=session_id,
            mission_id=mission_id,
        )
        observation_dir.mkdir(parents=True, exist_ok=True)
        observation_path = observation_dir / f"{observation_id}.json"
        temporary_path = observation_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "schema": "mission_tool_observation.v1",
                    "observation_id": observation_id,
                    "tool_call": {
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                    },
                    "tool_result": {
                        "tool_name": tool_result.tool_name,
                        "success": tool_result.success,
                        "output_text": tool_result.output_text,
                        "payload": tool_result.payload,
                        "error": tool_result.error,
                        "failure_kind": tool_result.failure_kind,
                    },
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary_path, observation_path)
        return observation_path


def serialize_mission_workspace_pointer(pointer: MissionWorkspacePointer) -> str:
    """Serialize one mission workspace pointer for SQLite event storage."""

    return json.dumps(
        {
            "schema": "mission_workspace_pointer.v2",
            "mission_id": pointer.mission_id,
            "workspace_path": pointer.workspace_path,
            "updated_at": pointer.updated_at,
            "status": pointer.status,
            "executor_state": pointer.executor_state,
            "active_task_id": pointer.active_task_id,
            "active_deliverable_id": pointer.active_task_id,
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
    schema = payload.get("schema")
    if schema not in {"mission_workspace_pointer.v1", "mission_workspace_pointer.v2"}:
        return None
    mission_id = payload.get("mission_id")
    workspace_path = payload.get("workspace_path")
    updated_at = payload.get("updated_at")
    status = payload.get("status")
    executor_state = payload.get("executor_state")
    active_task_id = payload.get("active_task_id")
    if active_task_id is None:
        active_task_id = payload.get("active_deliverable_id")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (mission_id, workspace_path, updated_at, status, executor_state)
    ):
        return None
    if active_task_id is not None and not isinstance(active_task_id, str):
        return None
    return MissionWorkspacePointer(
        mission_id=mission_id,
        workspace_path=workspace_path,
        updated_at=updated_at,
        status=status,
        executor_state=executor_state,
        active_task_id=active_task_id,
    )


def build_compatibility_mission_state(
    *,
    legacy_goal_state: SessionGoalState,
    legacy_progress_ledger: tuple[SessionProgressEntry, ...] = (),
    updated_at: str | None = None,
) -> MissionState | None:
    """Project a legacy active or blocked goal into a single-task mission."""

    if legacy_goal_state.status not in {"active", "blocked", "completed"}:
        return None
    timestamp = updated_at or legacy_goal_state.updated_at or utc_now_iso()
    title = legacy_goal_state.current_step or legacy_goal_state.goal
    latest_detail = legacy_progress_ledger[-1].detail if legacy_progress_ledger else None
    task_status = (
        "completed"
        if legacy_goal_state.status == "completed"
        else "blocked"
        if legacy_goal_state.status == "blocked"
        else "active"
    )
    task = MissionTaskState(
        id="legacy-d1",
        title=title,
        kind="mixed",
        status=task_status,
        required_evidence=(),
        satisfied_evidence=(() if task_status != "completed" else ("reply_emitted",)),
        latest_error=legacy_goal_state.last_blocker,
        evidence=((latest_detail,) if latest_detail else ()),
        updated_at=timestamp,
    )
    state = MissionState(
        mission_id="legacy-goal-compat",
        mission_goal=legacy_goal_state.goal,
        status=legacy_goal_state.status,
        tasks=(task,),
        active_task_id=None if task_status in {"completed", "blocked"} else task.id,
        updated_at=timestamp,
        reasoning_summary="legacy compatibility mission",
        blocker=legacy_goal_state.last_blocker,
        next_expected_evidence=latest_detail,
        final_reply=None,
        executor_state=(
            "completed"
            if legacy_goal_state.status == "completed"
            else "blocked"
            if legacy_goal_state.status == "blocked"
            else "active"
        ),
        retry_history=(),
        repair_history=((latest_detail,) if latest_detail and task_status == "active" else ()),
        last_blocker=legacy_goal_state.last_blocker,
    )
    return normalize_mission_state(state)


def build_mission_state_from_plan(
    *,
    mission_id: str,
    mission_goal: str,
    deliverables: tuple[MissionTaskState, ...],
    active_deliverable_id: str | None,
    execution_queue: tuple[str, ...] = (),
    planner_summary: str | None,
    last_turn_relation: str | None = None,
    updated_at: str,
) -> MissionState:
    """Compatibility builder used by older callers.

    The single-agent loop now treats every deliverable as a task in the
    persisted mission state.
    """

    del execution_queue, last_turn_relation
    tasks = tuple(
        replace(
            task,
            status=(
                "active"
                if task.id == active_deliverable_id and task.status == "pending"
                else task.status
            ),
            updated_at=updated_at,
        )
        for task in deliverables
    )
    return normalize_mission_state(
        MissionState(
            mission_id=mission_id,
            mission_goal=mission_goal,
            status="active",
            tasks=tasks,
            active_task_id=active_deliverable_id,
            updated_at=updated_at,
            reasoning_summary=planner_summary,
            next_expected_evidence=None,
            executor_state="active",
        )
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
    executor_reason: str | object = None,
    waiting_for: str | object = None,
    advance_condition: str | object = None,
    verifier_outputs: tuple[str, ...] | None = None,
    final_verified_reply: str | object = None,
) -> MissionState:
    """Compatibility refresh helper.

    Older code calls this function to refresh compact mission memory. The new
    single-agent loop already keeps the mission state compact, so this helper
    simply normalizes the mission while allowing a small amount of metadata to
    be updated.
    """

    del executor_reason, advance_condition, pending_repairs
    last_successful_evidence = (
        verifier_outputs
        if verifier_outputs is not None
        else mission_state.last_successful_evidence
    )
    last_artifact_paths = (
        artifact_facts if artifact_facts else mission_state.last_verified_artifact_paths
    )
    blocker = None
    if blockers:
        blocker = blockers[-1]
    elif mission_state.blocker is not None:
        blocker = mission_state.blocker
    state = replace(
        mission_state,
        updated_at=updated_at,
        reasoning_summary=planner_summary or mission_state.reasoning_summary,
        latest_truthful_status=planner_summary or mission_state.latest_truthful_status,
        blocker=blocker,
        next_expected_evidence=(
            waiting_for
            if isinstance(waiting_for, str) and waiting_for.strip()
            else mission_state.next_expected_evidence
        ),
        final_reply=(
            final_verified_reply
            if isinstance(final_verified_reply, str) and final_verified_reply.strip()
            else mission_state.final_reply
        ),
        executor_state=executor_state or mission_state.executor_state,
        repair_history=tuple(
            item
            for item in dict.fromkeys(
                mission_state.repair_history + tuple(observed_facts)
            )
        )[-8:],
        last_successful_evidence=tuple(
            item
            for item in dict.fromkeys(
                mission_state.last_successful_evidence + tuple(last_successful_evidence)
            )
        )[-8:],
        last_verified_artifact_paths=tuple(
            item
            for item in dict.fromkeys(
                mission_state.last_verified_artifact_paths + tuple(last_artifact_paths)
            )
        )[-8:],
        last_blocker=blocker or mission_state.last_blocker,
    )
    return normalize_mission_state(state)


__all__ = [
    "MissionWorkspacePointer",
    "MissionWorkspaceStore",
    "build_compatibility_mission_state",
    "build_mission_state_from_plan",
    "parse_mission_workspace_pointer",
    "serialize_mission_workspace_pointer",
    "synchronize_mission_working_memory",
]
