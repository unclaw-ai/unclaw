"""Persisted mission-state model for the single-agent mission loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from unclaw.llm.base import utc_now_iso

_MISSION_STATUS_VALUES = frozenset({"active", "blocked", "completed"})
_TASK_STATUS_VALUES = frozenset(
    {"pending", "active", "repairing", "completed", "blocked"}
)
_TASK_KIND_VALUES = frozenset(
    {
        "reply",
        "web_grounding",
        "web_research",
        "file_write",
        "file_read",
        "file_delete",
        "directory_list",
        "calc",
        "mixed",
    }
)
_EXECUTOR_STATE_VALUES = frozenset(
    {"active", "repairing", "blocked", "completed"}
)
_DEFAULT_REQUIRED_EVIDENCE_BY_KIND: dict[str, tuple[str, ...]] = {
    "reply": ("reply_emitted",),
    "web_grounding": ("fast_grounding",),
    "web_research": ("full_web_research",),
    "file_write": ("artifact_write", "artifact_readback"),
    "file_read": ("artifact_readback",),
    "file_delete": ("local_delete",),
    "directory_list": ("directory_listing",),
    "calc": ("calculation_result",),
    "mixed": (),
}
_MAX_TASKS = 8
_MAX_EVIDENCE_ITEMS = 12
_MAX_ARTIFACT_OBSERVATIONS = 12
_MAX_TOOL_HISTORY = 24
_MAX_TEXT_ITEMS = 8


@dataclass(frozen=True, slots=True)
class MissionTaskState:
    """Persisted proof-tracked task inside one mission."""

    id: str
    title: str
    kind: str
    status: str
    depends_on: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    satisfied_evidence: tuple[str, ...] = ()
    artifact_paths: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    latest_error: str | None = None
    repair_count: int = 0
    updated_at: str = ""

    @property
    def deliverable_id(self) -> str:
        return self.id

    @property
    def task(self) -> str:
        return self.title

    @property
    def blocker(self) -> str | None:
        return self.latest_error

    @property
    def missing(self) -> str | None:
        return None if self.status == "completed" else self.latest_error


MissionDeliverableState = MissionTaskState


@dataclass(frozen=True, slots=True)
class MissionEvidenceRecord:
    """One structured evidence record appended by the runtime."""

    kind: str
    task_id: str | None
    summary: str
    created_at: str
    id: str = ""
    tool_name: str | None = None
    artifact_paths: tuple[str, ...] = ()
    success: bool = True


@dataclass(frozen=True, slots=True)
class MissionArtifactObservation:
    """One compact observed artifact fact persisted for later turns."""

    path: str
    status: str
    task_id: str | None
    created_at: str
    summary: str | None = None
    tool_name: str | None = None
    evidence_ref: str | None = None


@dataclass(frozen=True, slots=True)
class MissionToolCallRecord:
    """One executed or rejected tool call recorded in mission history."""

    tool_name: str
    task_id: str | None
    arguments: dict[str, Any]
    created_at: str
    executed: bool = True
    success: bool | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MissionState:
    """Persisted mission snapshot for one session."""

    mission_id: str
    mission_goal: str
    status: str
    tasks: tuple[MissionTaskState, ...]
    active_task_id: str | None
    updated_at: str
    reasoning_summary: str | None = None
    user_visible_progress: str | None = None
    reply_to_user: str | None = None
    completion_claim: bool = False
    blocker: str | None = None
    next_expected_evidence: str | None = None
    unresolved_gaps: tuple[str, ...] = ()
    latest_truthful_status: str | None = None
    evidence_log: tuple[MissionEvidenceRecord, ...] = ()
    artifact_observations: tuple[MissionArtifactObservation, ...] = ()
    tool_history: tuple[MissionToolCallRecord, ...] = ()
    final_reply: str | None = None
    last_user_input: str | None = None
    loop_count: int = 0
    executor_state: str = "active"
    retry_history: tuple[str, ...] = ()
    repair_history: tuple[str, ...] = ()
    last_verified_artifact_paths: tuple[str, ...] = ()
    last_successful_evidence: tuple[str, ...] = ()
    last_blocker: str | None = None

    def get_task(
        self,
        task_id: str | None = None,
    ) -> MissionTaskState | None:
        resolved_id = task_id or self.active_task_id
        if resolved_id is None:
            return None
        return next((task for task in self.tasks if task.id == resolved_id), None)

    def get_deliverable(
        self,
        deliverable_id: str | None = None,
    ) -> MissionTaskState | None:
        return self.get_task(deliverable_id)

    def replace_task(
        self,
        task: MissionTaskState,
        *,
        updated_at: str,
    ) -> MissionState:
        next_tasks = tuple(
            task if current.id == task.id else current for current in self.tasks
        )
        return normalize_mission_state(
            replace(
                self,
                tasks=next_tasks,
                updated_at=updated_at,
            )
        )

    def replace_deliverable(
        self,
        deliverable: MissionTaskState,
        *,
        updated_at: str,
    ) -> MissionState:
        return self.replace_task(deliverable, updated_at=updated_at)

    @property
    def goal(self) -> str:
        return self.mission_goal

    @property
    def deliverables(self) -> tuple[MissionTaskState, ...]:
        return self.tasks

    @property
    def active_deliverable_id(self) -> str | None:
        return self.active_task_id

    @property
    def active_task(self) -> str | None:
        active_task = self.get_task()
        return active_task.title if active_task is not None else None

    @property
    def completed_deliverables(self) -> tuple[str, ...]:
        return tuple(task.id for task in self.tasks if task.status == "completed")

    @property
    def blocked_deliverables(self) -> tuple[str, ...]:
        return tuple(task.id for task in self.tasks if task.status == "blocked")

    @property
    def planner_summary(self) -> str | None:
        return self.reasoning_summary

    @property
    def execution_queue(self) -> tuple[str, ...]:
        return tuple(task.id for task in self.tasks)

    @property
    def completed_steps(self) -> tuple[str, ...]:
        return self.completed_deliverables

    @property
    def failed_steps(self) -> tuple[str, ...]:
        return self.blocked_deliverables

    @property
    def observed_facts(self) -> tuple[str, ...]:
        return tuple(record.summary for record in self.evidence_log if record.success)[
            -_MAX_TEXT_ITEMS:
        ]

    @property
    def artifact_facts(self) -> tuple[str, ...]:
        facts: list[str] = []
        for record in self.evidence_log:
            for path in record.artifact_paths:
                if path not in facts:
                    facts.append(path)
        for observation in self.artifact_observations:
            if observation.path not in facts:
                facts.append(observation.path)
        return tuple(facts[-_MAX_TEXT_ITEMS:])

    @property
    def blockers(self) -> tuple[str, ...]:
        if self.last_blocker:
            return (self.last_blocker,)
        if self.blocker:
            return (self.blocker,)
        return ()

    @property
    def pending_repairs(self) -> tuple[str, ...]:
        return tuple(
            task.id
            for task in self.tasks
            if task.status == "repairing" or (task.latest_error and task.status == "active")
        )

    @property
    def final_deliverables_missing(self) -> tuple[str, ...]:
        return tuple(
            task.id for task in self.tasks if task.status not in {"completed", "blocked"}
        )

    @property
    def waiting_for(self) -> str | None:
        return self.next_expected_evidence

    @property
    def advance_condition(self) -> str | None:
        return self.next_expected_evidence

    @property
    def verifier_outputs(self) -> tuple[str, ...]:
        return self.last_successful_evidence

    @property
    def final_verified_reply(self) -> str | None:
        return self.final_reply

    @property
    def last_turn_relation(self) -> str | None:
        return None


def normalize_mission_task(task: MissionTaskState) -> MissionTaskState:
    """Return a normalized task safe for persistence."""

    normalized_id = _normalize_text(task.id, fallback="task")
    normalized_title = _normalize_text(task.title, fallback=normalized_id)
    normalized_kind = _normalize_choice(
        task.kind or "mixed",
        allowed_values=_TASK_KIND_VALUES,
        fallback="mixed",
    )
    default_required = _DEFAULT_REQUIRED_EVIDENCE_BY_KIND.get(normalized_kind, ())
    normalized_required = _normalize_text_items(
        task.required_evidence or default_required
    )
    normalized_satisfied = tuple(
        item
        for item in _normalize_text_items(task.satisfied_evidence)
        if item in normalized_required or item.startswith("tool:")
    )
    normalized_status = _normalize_choice(
        task.status,
        allowed_values=_TASK_STATUS_VALUES,
        fallback="pending",
    )
    if normalized_required and all(
        evidence in normalized_satisfied for evidence in normalized_required
    ):
        normalized_status = "completed"
    if normalized_status == "completed":
        normalized_satisfied = tuple(
            item
            for item in dict.fromkeys(normalized_required + normalized_satisfied)
        )
    return MissionTaskState(
        id=normalized_id,
        title=normalized_title,
        kind=normalized_kind,
        status=normalized_status,
        depends_on=_normalize_text_items(task.depends_on),
        required_evidence=normalized_required,
        satisfied_evidence=normalized_satisfied,
        artifact_paths=_normalize_text_items(task.artifact_paths),
        evidence=_normalize_text_items(task.evidence),
        evidence_refs=_normalize_text_items(task.evidence_refs),
        latest_error=_normalize_optional_text(task.latest_error),
        repair_count=max(task.repair_count, 0),
        updated_at=_normalize_timestamp(task.updated_at),
    )


normalize_mission_deliverable = normalize_mission_task


def normalize_mission_state(mission_state: MissionState) -> MissionState:
    """Return a normalized copy safe for persistence."""

    normalized_tasks = _normalize_tasks(mission_state.tasks)
    completed_task_ids = tuple(
        task.id for task in normalized_tasks if task.status == "completed"
    )
    blocked_task_ids = tuple(
        task.id for task in normalized_tasks if task.status == "blocked"
    )

    active_task_id = _normalize_optional_text(mission_state.active_task_id)
    if active_task_id is not None and not any(
        task.id == active_task_id for task in normalized_tasks
    ):
        active_task_id = None
    if active_task_id is not None:
        current_active_task = next(
            (task for task in normalized_tasks if task.id == active_task_id),
            None,
        )
        if current_active_task is not None and current_active_task.status in {
            "completed",
            "blocked",
        }:
            active_task_id = None
    if active_task_id is None:
        active_task_id = _first_runnable_incomplete_task_id(normalized_tasks)
    active_task = next(
        (task for task in normalized_tasks if task.id == active_task_id),
        None,
    )

    normalized_status = _normalize_choice(
        mission_state.status,
        allowed_values=_MISSION_STATUS_VALUES,
        fallback="active",
    )
    if normalized_tasks and all(task.status == "completed" for task in normalized_tasks):
        normalized_status = "completed"
        active_task_id = None
    elif blocked_task_ids and active_task is None and not any(
        task.status in {"pending", "active", "repairing"} for task in normalized_tasks
    ):
        normalized_status = "blocked"
    else:
        normalized_status = "active"

    normalized_blocker = _normalize_optional_text(
        mission_state.blocker or mission_state.last_blocker
    )
    if normalized_status == "completed":
        normalized_blocker = None

    normalized_executor_state = _normalize_choice(
        mission_state.executor_state,
        allowed_values=_EXECUTOR_STATE_VALUES,
        fallback=(
            "completed"
            if normalized_status == "completed"
            else "blocked"
            if normalized_status == "blocked"
            else "repairing"
            if any(task.status == "repairing" for task in normalized_tasks)
            else "active"
        ),
    )
    if normalized_status == "completed":
        normalized_executor_state = "completed"
    elif normalized_status == "blocked":
        normalized_executor_state = "blocked"
    elif any(task.status == "repairing" for task in normalized_tasks):
        normalized_executor_state = "repairing"
    else:
        normalized_executor_state = "active"

    evidence_log = tuple(
        _normalize_evidence_record(record)
        for record in mission_state.evidence_log[-_MAX_EVIDENCE_ITEMS:]
    )
    artifact_observations = tuple(
        _normalize_artifact_observation(record)
        for record in mission_state.artifact_observations[-_MAX_ARTIFACT_OBSERVATIONS:]
    )
    tool_history = tuple(
        _normalize_tool_call_record(record)
        for record in mission_state.tool_history[-_MAX_TOOL_HISTORY:]
    )

    last_artifact_paths: list[str] = []
    for task in normalized_tasks:
        for path in task.artifact_paths:
            if path not in last_artifact_paths:
                last_artifact_paths.append(path)
    for record in evidence_log:
        for path in record.artifact_paths:
            if path not in last_artifact_paths:
                last_artifact_paths.append(path)
    for observation in artifact_observations:
        if observation.path not in last_artifact_paths:
            last_artifact_paths.append(observation.path)

    last_successful_evidence: list[str] = []
    for task in normalized_tasks:
        for item in task.evidence:
            if item not in last_successful_evidence:
                last_successful_evidence.append(item)
    for record in evidence_log:
        if record.summary and record.summary not in last_successful_evidence and record.success:
            last_successful_evidence.append(record.summary)

    next_expected_evidence = _normalize_optional_text(mission_state.next_expected_evidence)
    if normalized_status == "completed":
        next_expected_evidence = None
    elif next_expected_evidence is None and active_task is not None:
        missing = [
            evidence
            for evidence in active_task.required_evidence
            if evidence not in active_task.satisfied_evidence
        ]
        if missing:
            next_expected_evidence = ", ".join(missing)

    unresolved_gaps = _normalize_text_items(mission_state.unresolved_gaps)
    if normalized_status == "completed":
        unresolved_gaps = ()
    elif not unresolved_gaps:
        unresolved_gaps = _derive_unresolved_gaps(
            tasks=normalized_tasks,
            status=normalized_status,
            blocker=normalized_blocker,
        )

    latest_truthful_status = _normalize_optional_text(mission_state.latest_truthful_status)
    if latest_truthful_status is None:
        latest_truthful_status = _build_truthful_status_text(
            mission_goal=_normalize_text(
                mission_state.mission_goal,
                fallback=mission_state.last_user_input or "mission",
            ),
            status=normalized_status,
            active_task=active_task.title if active_task is not None else None,
            completed_tasks=tuple(
                task.title for task in normalized_tasks if task.status == "completed"
            ),
            blocker=normalized_blocker,
            next_expected_evidence=next_expected_evidence,
            verified_artifacts=tuple(last_artifact_paths[-_MAX_TEXT_ITEMS:]),
            unresolved_gaps=unresolved_gaps,
        )

    return MissionState(
        mission_id=_normalize_text(mission_state.mission_id, fallback="mission"),
        mission_goal=_normalize_text(
            mission_state.mission_goal,
            fallback=mission_state.last_user_input or "mission",
        ),
        status=normalized_status,
        tasks=normalized_tasks,
        active_task_id=active_task_id,
        updated_at=_normalize_timestamp(mission_state.updated_at),
        reasoning_summary=_normalize_optional_text(mission_state.reasoning_summary),
        user_visible_progress=_normalize_optional_text(mission_state.user_visible_progress),
        reply_to_user=_normalize_optional_text(mission_state.reply_to_user),
        completion_claim=bool(mission_state.completion_claim),
        blocker=normalized_blocker,
        next_expected_evidence=next_expected_evidence,
        unresolved_gaps=unresolved_gaps,
        latest_truthful_status=latest_truthful_status,
        evidence_log=evidence_log,
        artifact_observations=artifact_observations,
        tool_history=tool_history,
        final_reply=_normalize_optional_text(mission_state.final_reply),
        last_user_input=_normalize_optional_text(mission_state.last_user_input),
        loop_count=max(mission_state.loop_count, 0),
        executor_state=normalized_executor_state,
        retry_history=_normalize_text_items(mission_state.retry_history),
        repair_history=_normalize_text_items(mission_state.repair_history),
        last_verified_artifact_paths=tuple(last_artifact_paths[-_MAX_TEXT_ITEMS:]),
        last_successful_evidence=tuple(last_successful_evidence[-_MAX_TEXT_ITEMS:]),
        last_blocker=normalized_blocker,
    )


def serialize_mission_state(mission_state: MissionState) -> str:
    """Serialize one mission snapshot to compact JSON."""

    normalized = normalize_mission_state(mission_state)
    return json.dumps(
        {
            "schema": "mission_state.v3",
            "mission_id": normalized.mission_id,
            "mission_goal": normalized.mission_goal,
            "status": normalized.status,
            "tasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "kind": task.kind,
                    "status": task.status,
                    "depends_on": list(task.depends_on),
                    "required_evidence": list(task.required_evidence),
                    "satisfied_evidence": list(task.satisfied_evidence),
                    "artifact_paths": list(task.artifact_paths),
                    "evidence": list(task.evidence),
                    "evidence_refs": list(task.evidence_refs),
                    "latest_error": task.latest_error,
                    "repair_count": task.repair_count,
                    "updated_at": task.updated_at,
                }
                for task in normalized.tasks
            ],
            "active_task_id": normalized.active_task_id,
            "updated_at": normalized.updated_at,
            "reasoning_summary": normalized.reasoning_summary,
            "user_visible_progress": normalized.user_visible_progress,
            "reply_to_user": normalized.reply_to_user,
            "completion_claim": normalized.completion_claim,
            "blocker": normalized.blocker,
            "next_expected_evidence": normalized.next_expected_evidence,
            "unresolved_gaps": list(normalized.unresolved_gaps),
            "latest_truthful_status": normalized.latest_truthful_status,
            "evidence_log": [
                {
                    "id": record.id,
                    "kind": record.kind,
                    "task_id": record.task_id,
                    "summary": record.summary,
                    "created_at": record.created_at,
                    "tool_name": record.tool_name,
                    "artifact_paths": list(record.artifact_paths),
                    "success": record.success,
                }
                for record in normalized.evidence_log
            ],
            "artifact_observations": [
                {
                    "path": record.path,
                    "status": record.status,
                    "task_id": record.task_id,
                    "created_at": record.created_at,
                    "summary": record.summary,
                    "tool_name": record.tool_name,
                    "evidence_ref": record.evidence_ref,
                }
                for record in normalized.artifact_observations
            ],
            "tool_history": [
                {
                    "tool_name": record.tool_name,
                    "task_id": record.task_id,
                    "arguments": record.arguments,
                    "created_at": record.created_at,
                    "executed": record.executed,
                    "success": record.success,
                    "reason": record.reason,
                }
                for record in normalized.tool_history
            ],
            "final_reply": normalized.final_reply,
            "last_user_input": normalized.last_user_input,
            "loop_count": normalized.loop_count,
            "executor_state": normalized.executor_state,
            "retry_history": list(normalized.retry_history),
            "repair_history": list(normalized.repair_history),
            "last_verified_artifact_paths": list(normalized.last_verified_artifact_paths),
            "last_successful_evidence": list(normalized.last_successful_evidence),
            "last_blocker": normalized.last_blocker,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_mission_state(payload_json: str) -> MissionState | None:
    """Parse one mission snapshot from JSON."""

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") in {"mission_state.v2", "mission_state.v3"} or "tasks" in payload:
        return _parse_v2_mission_state(payload)
    if "deliverables" in payload:
        return _parse_legacy_mission_state(payload)
    return None


def mission_completion_ready(mission_state: MissionState) -> bool:
    """Return True when every task is proven complete."""

    normalized = normalize_mission_state(mission_state)
    if not normalized.tasks:
        return False
    if normalized.unresolved_gaps:
        return False
    return all(task.status == "completed" for task in normalized.tasks)


def _parse_v2_mission_state(payload: dict[str, Any]) -> MissionState | None:
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        return None
    tasks: list[MissionTaskState] = []
    for raw_task in raw_tasks[:_MAX_TASKS]:
        if not isinstance(raw_task, dict):
            return None
        task = MissionTaskState(
            id=_coerce_required_str(raw_task.get("id")),
            title=_coerce_required_str(raw_task.get("title")),
            kind=_coerce_required_str(raw_task.get("kind")),
            status=_coerce_required_str(raw_task.get("status")),
            depends_on=_coerce_str_tuple(raw_task.get("depends_on")),
            required_evidence=_coerce_str_tuple(raw_task.get("required_evidence")),
            satisfied_evidence=_coerce_str_tuple(raw_task.get("satisfied_evidence")),
            artifact_paths=_coerce_str_tuple(raw_task.get("artifact_paths")),
            evidence=_coerce_str_tuple(raw_task.get("evidence")),
            evidence_refs=_coerce_str_tuple(raw_task.get("evidence_refs")),
            latest_error=_coerce_optional_str(raw_task.get("latest_error")),
            repair_count=_coerce_non_negative_int(raw_task.get("repair_count")),
            updated_at=_coerce_optional_str(raw_task.get("updated_at")) or utc_now_iso(),
        )
        tasks.append(task)

    try:
        return normalize_mission_state(
            MissionState(
                mission_id=_coerce_required_str(payload.get("mission_id")),
                mission_goal=(
                    _coerce_optional_str(payload.get("mission_goal"))
                    or _coerce_optional_str(payload.get("goal"))
                    or "mission"
                ),
                status=_coerce_optional_str(payload.get("status")) or "active",
                tasks=tuple(tasks),
                active_task_id=_coerce_optional_str(payload.get("active_task_id")),
                updated_at=_coerce_optional_str(payload.get("updated_at")) or utc_now_iso(),
                reasoning_summary=_coerce_optional_str(payload.get("reasoning_summary")),
                user_visible_progress=_coerce_optional_str(
                    payload.get("user_visible_progress")
                ),
                reply_to_user=_coerce_optional_str(payload.get("reply_to_user")),
                completion_claim=payload.get("completion_claim") is True,
                blocker=_coerce_optional_str(payload.get("blocker")),
                next_expected_evidence=_coerce_optional_str(
                    payload.get("next_expected_evidence")
                ),
                unresolved_gaps=_coerce_str_tuple(payload.get("unresolved_gaps")),
                latest_truthful_status=_coerce_optional_str(
                    payload.get("latest_truthful_status")
                ),
                evidence_log=tuple(
                    _parse_evidence_record(item)
                    for item in _coerce_dict_list(payload.get("evidence_log"))
                    if _parse_evidence_record(item) is not None
                ),
                artifact_observations=tuple(
                    _parse_artifact_observation(item)
                    for item in _coerce_dict_list(payload.get("artifact_observations"))
                    if _parse_artifact_observation(item) is not None
                ),
                tool_history=tuple(
                    _parse_tool_call_record(item)
                    for item in _coerce_dict_list(payload.get("tool_history"))
                    if _parse_tool_call_record(item) is not None
                ),
                final_reply=_coerce_optional_str(payload.get("final_reply")),
                last_user_input=_coerce_optional_str(payload.get("last_user_input")),
                loop_count=_coerce_non_negative_int(payload.get("loop_count")),
                executor_state=_coerce_optional_str(payload.get("executor_state"))
                or "active",
                retry_history=_coerce_str_tuple(payload.get("retry_history")),
                repair_history=_coerce_str_tuple(payload.get("repair_history")),
                last_verified_artifact_paths=_coerce_str_tuple(
                    payload.get("last_verified_artifact_paths")
                ),
                last_successful_evidence=_coerce_str_tuple(
                    payload.get("last_successful_evidence")
                ),
                last_blocker=_coerce_optional_str(payload.get("last_blocker")),
            )
        )
    except ValueError:
        return None


def _parse_legacy_mission_state(payload: dict[str, Any]) -> MissionState | None:
    raw_deliverables = payload.get("deliverables")
    if not isinstance(raw_deliverables, list):
        return None
    tasks: list[MissionTaskState] = []
    for raw_deliverable in raw_deliverables[:_MAX_TASKS]:
        if not isinstance(raw_deliverable, dict):
            return None
        task = MissionTaskState(
            id=_coerce_optional_str(raw_deliverable.get("deliverable_id"))
            or _coerce_optional_str(raw_deliverable.get("id"))
            or f"d{len(tasks) + 1}",
            title=_coerce_optional_str(raw_deliverable.get("task"))
            or _coerce_optional_str(raw_deliverable.get("title"))
            or f"Task {len(tasks) + 1}",
            kind=_infer_legacy_kind(raw_deliverable),
            status=_coerce_optional_str(raw_deliverable.get("status")) or "pending",
            depends_on=_coerce_str_tuple(raw_deliverable.get("depends_on")),
            required_evidence=_coerce_str_tuple(
                raw_deliverable.get("required_evidence")
            ),
            satisfied_evidence=(
                _coerce_str_tuple(raw_deliverable.get("required_evidence"))
                if _coerce_optional_str(raw_deliverable.get("status")) == "completed"
                else ()
            ),
            artifact_paths=_coerce_str_tuple(raw_deliverable.get("artifact_paths")),
            evidence=_coerce_str_tuple(raw_deliverable.get("evidence")),
            evidence_refs=(),
            latest_error=(
                _coerce_optional_str(raw_deliverable.get("blocker"))
                or _coerce_optional_str(raw_deliverable.get("missing"))
            ),
            repair_count=_coerce_non_negative_int(raw_deliverable.get("repair_count")),
            updated_at=_coerce_optional_str(raw_deliverable.get("updated_at")) or utc_now_iso(),
        )
        tasks.append(task)

    try:
        return normalize_mission_state(
            MissionState(
                mission_id=_coerce_required_str(payload.get("mission_id")),
                mission_goal=(
                    _coerce_optional_str(payload.get("goal"))
                    or _coerce_optional_str(payload.get("mission_goal"))
                    or "mission"
                ),
                status=_coerce_optional_str(payload.get("status")) or "active",
                tasks=tuple(tasks),
                active_task_id=(
                    _coerce_optional_str(payload.get("active_task_id"))
                    or _coerce_optional_str(payload.get("active_deliverable_id"))
                ),
                updated_at=_coerce_optional_str(payload.get("updated_at")) or utc_now_iso(),
                reasoning_summary=_coerce_optional_str(payload.get("planner_summary")),
                user_visible_progress=None,
                reply_to_user=_coerce_optional_str(payload.get("final_verified_reply")),
                completion_claim=payload.get("status") == "completed",
                blocker=_coerce_optional_str(payload.get("last_blocker")),
                next_expected_evidence=(
                    _coerce_optional_str(payload.get("waiting_for"))
                    or _coerce_optional_str(payload.get("advance_condition"))
                ),
                unresolved_gaps=(),
                latest_truthful_status=None,
                evidence_log=(),
                artifact_observations=(),
                final_reply=_coerce_optional_str(payload.get("final_verified_reply")),
                last_user_input=None,
                loop_count=0,
                executor_state=_coerce_optional_str(payload.get("executor_state"))
                or "active",
                retry_history=_coerce_str_tuple(payload.get("retry_history")),
                repair_history=_coerce_str_tuple(payload.get("repair_history")),
                last_verified_artifact_paths=_coerce_str_tuple(
                    payload.get("last_verified_artifact_paths")
                ),
                last_successful_evidence=_coerce_str_tuple(
                    payload.get("last_successful_evidence")
                ),
                last_blocker=_coerce_optional_str(payload.get("last_blocker")),
            )
        )
    except ValueError:
        return None


def _infer_legacy_kind(raw_deliverable: dict[str, Any]) -> str:
    required_evidence = set(_coerce_str_tuple(raw_deliverable.get("required_evidence")))
    if "full_web_research" in required_evidence:
        return "web_research"
    if "fast_grounding" in required_evidence:
        return "web_grounding"
    if {
        "artifact_write",
        "artifact_readback",
    }.issubset(required_evidence):
        return "file_write"
    if "artifact_readback" in required_evidence:
        return "file_read"
    if "local_delete" in required_evidence:
        return "file_delete"
    if "directory_listing" in required_evidence:
        return "directory_list"
    if "calculation_result" in required_evidence:
        return "calc"
    mode = _coerce_optional_str(raw_deliverable.get("mode"))
    if mode == "reply":
        return "reply"
    return "mixed"


def _normalize_tasks(tasks: tuple[MissionTaskState, ...]) -> tuple[MissionTaskState, ...]:
    normalized: list[MissionTaskState] = []
    seen_ids: set[str] = set()
    for raw_task in tasks[:_MAX_TASKS]:
        task = normalize_mission_task(raw_task)
        task_id = task.id
        if task_id in seen_ids:
            suffix = 2
            while f"{task_id}-{suffix}" in seen_ids:
                suffix += 1
            task = replace(task, id=f"{task_id}-{suffix}")
        seen_ids.add(task.id)
        normalized.append(task)
    return tuple(normalized)


def _first_runnable_incomplete_task_id(
    tasks: tuple[MissionTaskState, ...],
) -> str | None:
    completed_ids = {task.id for task in tasks if task.status == "completed"}
    for task in tasks:
        if task.status in {"completed", "blocked"}:
            continue
        if all(dependency in completed_ids for dependency in task.depends_on):
            return task.id
    return None


def _normalize_evidence_record(
    record: MissionEvidenceRecord,
) -> MissionEvidenceRecord:
    return MissionEvidenceRecord(
        kind=_normalize_text(record.kind, fallback="evidence"),
        task_id=_normalize_optional_text(record.task_id),
        summary=_normalize_text(record.summary, fallback="evidence"),
        created_at=_normalize_timestamp(record.created_at),
        id=_normalize_text(record.id, fallback="evidence"),
        tool_name=_normalize_optional_text(record.tool_name),
        artifact_paths=_normalize_text_items(record.artifact_paths),
        success=bool(record.success),
    )


def _normalize_artifact_observation(
    record: MissionArtifactObservation,
) -> MissionArtifactObservation:
    return MissionArtifactObservation(
        path=_normalize_text(record.path, fallback="artifact"),
        status=_normalize_text(record.status, fallback="observed"),
        task_id=_normalize_optional_text(record.task_id),
        created_at=_normalize_timestamp(record.created_at),
        summary=_normalize_optional_text(record.summary),
        tool_name=_normalize_optional_text(record.tool_name),
        evidence_ref=_normalize_optional_text(record.evidence_ref),
    )


def _normalize_tool_call_record(
    record: MissionToolCallRecord,
) -> MissionToolCallRecord:
    return MissionToolCallRecord(
        tool_name=_normalize_text(record.tool_name, fallback="tool"),
        task_id=_normalize_optional_text(record.task_id),
        arguments=dict(record.arguments),
        created_at=_normalize_timestamp(record.created_at),
        executed=bool(record.executed),
        success=record.success if isinstance(record.success, bool) or record.success is None else None,
        reason=_normalize_optional_text(record.reason),
    )


def _parse_evidence_record(payload: dict[str, Any]) -> MissionEvidenceRecord | None:
    try:
        return MissionEvidenceRecord(
            kind=_coerce_required_str(payload.get("kind")),
            task_id=_coerce_optional_str(payload.get("task_id")),
            summary=_coerce_required_str(payload.get("summary")),
            created_at=_coerce_optional_str(payload.get("created_at")) or utc_now_iso(),
            id=_coerce_optional_str(payload.get("id")) or "evidence",
            tool_name=_coerce_optional_str(payload.get("tool_name")),
            artifact_paths=_coerce_str_tuple(payload.get("artifact_paths")),
            success=payload.get("success") is not False,
        )
    except ValueError:
        return None


def _parse_artifact_observation(
    payload: dict[str, Any],
) -> MissionArtifactObservation | None:
    try:
        return MissionArtifactObservation(
            path=_coerce_required_str(payload.get("path")),
            status=_coerce_optional_str(payload.get("status")) or "observed",
            task_id=_coerce_optional_str(payload.get("task_id")),
            created_at=_coerce_optional_str(payload.get("created_at")) or utc_now_iso(),
            summary=_coerce_optional_str(payload.get("summary")),
            tool_name=_coerce_optional_str(payload.get("tool_name")),
            evidence_ref=_coerce_optional_str(payload.get("evidence_ref")),
        )
    except ValueError:
        return None


def _parse_tool_call_record(payload: dict[str, Any]) -> MissionToolCallRecord | None:
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None
    try:
        success = payload.get("success")
        return MissionToolCallRecord(
            tool_name=_coerce_required_str(payload.get("tool_name")),
            task_id=_coerce_optional_str(payload.get("task_id")),
            arguments=dict(arguments),
            created_at=_coerce_optional_str(payload.get("created_at")) or utc_now_iso(),
            executed=payload.get("executed") is not False,
            success=success if isinstance(success, bool) or success is None else None,
            reason=_coerce_optional_str(payload.get("reason")),
        )
    except ValueError:
        return None


def _coerce_required_str(value: Any) -> str:
    normalized = _coerce_optional_str(value)
    if normalized is None:
        raise ValueError("Expected a non-empty string.")
    return normalized


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        normalized = _coerce_optional_str(item)
        if normalized is not None and normalized not in items:
            items.append(normalized)
    return tuple(items)


def _coerce_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    return 0


def _coerce_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalize_text(value: str | None, *, fallback: str) -> str:
    normalized = _normalize_optional_text(value)
    return normalized or fallback


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _normalize_text_items(items: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not items:
        return ()
    normalized_items: list[str] = []
    for item in items:
        normalized = _normalize_optional_text(item)
        if normalized is not None and normalized not in normalized_items:
            normalized_items.append(normalized)
    return tuple(normalized_items)


def _normalize_choice(
    value: str | None,
    *,
    allowed_values: frozenset[str],
    fallback: str,
) -> str:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return fallback
    lowered = normalized.casefold()
    return lowered if lowered in allowed_values else fallback


def _normalize_timestamp(value: str | None) -> str:
    normalized = _normalize_optional_text(value)
    return normalized or utc_now_iso()


def _derive_unresolved_gaps(
    *,
    tasks: tuple[MissionTaskState, ...],
    status: str,
    blocker: str | None,
) -> tuple[str, ...]:
    if status == "completed":
        return ()
    gaps: list[str] = []
    for task in tasks:
        if task.status == "completed":
            continue
        missing = [
            evidence
            for evidence in task.required_evidence
            if evidence not in task.satisfied_evidence
        ]
        if task.status == "blocked" and task.latest_error:
            gaps.append(f"{task.title}: {task.latest_error}")
            continue
        if missing:
            gaps.append(f"{task.title}: missing {', '.join(missing)}")
            continue
        gaps.append(f"{task.title}: still {task.status}")
    if blocker and blocker not in gaps:
        gaps.append(blocker)
    return tuple(gaps[:_MAX_TEXT_ITEMS])


def _build_truthful_status_text(
    *,
    mission_goal: str,
    status: str,
    active_task: str | None,
    completed_tasks: tuple[str, ...],
    blocker: str | None,
    next_expected_evidence: str | None,
    verified_artifacts: tuple[str, ...],
    unresolved_gaps: tuple[str, ...],
) -> str:
    if status == "completed":
        artifact_note = ""
        if verified_artifacts:
            artifact_note = " Verified artifacts: " + ", ".join(verified_artifacts) + "."
        return f"Mission complete: {mission_goal}.{artifact_note}".strip()
    if status == "blocked":
        return f"Mission blocked: {blocker or mission_goal}."
    parts = [f"Mission in progress: {mission_goal}."]
    if completed_tasks:
        parts.append("Done: " + ", ".join(completed_tasks) + ".")
    if active_task:
        parts.append(f"Current task: {active_task}.")
    if next_expected_evidence:
        parts.append(f"Waiting for: {next_expected_evidence}.")
    elif unresolved_gaps:
        parts.append(f"Unresolved: {unresolved_gaps[0]}.")
    if verified_artifacts:
        parts.append("Observed artifacts: " + ", ".join(verified_artifacts) + ".")
    return " ".join(parts)


__all__ = [
    "MissionArtifactObservation",
    "MissionDeliverableState",
    "MissionEvidenceRecord",
    "MissionState",
    "MissionTaskState",
    "MissionToolCallRecord",
    "mission_completion_ready",
    "normalize_mission_deliverable",
    "normalize_mission_state",
    "normalize_mission_task",
    "parse_mission_state",
    "serialize_mission_state",
]
