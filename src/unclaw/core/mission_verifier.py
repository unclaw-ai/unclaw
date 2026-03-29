"""Single-agent mission action schema and prompt helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any

from unclaw.core.mission_state import (
    MissionEvidenceCapsule,
    MissionState,
    MissionTaskState,
)
from unclaw.llm.base import LLMMessage, LLMRole
from unclaw.tools.contracts import (
    ToolCall,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)

_MISSION_AGENT_SYSTEM_PROMPT = "\n".join(
    (
        "Single mission agent for the Unclaw local runtime.",
        "Every user turn is a mission.",
        "A simple chat answer is a one-step mission.",
        "A complex request is a multi-step mission.",
        "Return JSON only using the mission step schema below.",
        "Do not add markdown, prose, or explanations outside the JSON object.",
        "You are the only controller: there is no router, no planner, and no verifier after you.",
        "Hard rules:",
        "- Read the raw user request directly and decide whether to start a new mission or continue the current one.",
        "- Use mission_action='start_new' only when the user is clearly asking for a different outcome than the current mission.",
        "- Use mission_action='continue_existing' for status, continue, finish, repair, and follow-up turns about the same mission.",
        "- Keep tasks compact and concrete. Use at most 6 tasks.",
        "- Once a task_board exists, keep unfinished tasks visible until they are proven completed or honestly blocked.",
        "- If the user asked to create, read, update, delete, or list a local artifact, include an explicit file_* or directory_list task in task_board.",
        "- Never erase already-completed tasks from the provided mission state. Add a repair task instead if you need new work after a proven completion.",
        "- If a previous turn said the mission was done but the user says work is missing, continue_existing and add repair work to the same mission.",
        "- Use the persisted mission state as your external working memory: current_goal, task_board, unresolved_gaps, evidence_log, evidence_capsules, artifact_observations, tool_observation_refs, and latest_truthful_status.",
        "- Do not claim mission completion unless every task has the required evidence and unresolved_gaps is empty.",
        "- Artifact claims must come only from observed evidence already present in evidence_log, artifact_observations, or recent_tool_results.",
        "- reply_to_user is the exact user-facing message for this step. Never narrate internal progress like 'I am now reading the file'.",
        "- step_mode='final_reply' means reply_to_user is the real final user-facing answer now.",
        "- step_mode='continue' means more work is still needed after this step.",
        "- step_mode='blocked' means the mission is honestly blocked now.",
        "- For a simple fact question, answer the fact directly instead of returning a mission-status summary.",
        "- If the listed tool_calls in this action will finish the mission, still write the final user answer now instead of a placeholder or progress note.",
        "- Never use raw internal scaffolding inside reply_to_user.",
        "- fast_web_search is grounding only. It can satisfy fast_grounding but never full_web_research.",
        "- search_web is the full research path.",
        "- write_text_file alone never completes a file_write task. A successful read_text_file read-back is also required.",
        "- For local machine facts like OS, time, hostname, or locale, use system_info instead of guessing.",
        "- For status requests, base the answer only on the persisted mission state in the payload.",
        "- reply_to_user may be empty while tool work is still needed.",
        "- user_visible_progress should be a short truthful status line suitable for a compact UI.",
        "- completion_claim=true means the full mission is done now.",
        "- blocker must be empty unless the mission is honestly blocked.",
        "Task kind guidance:",
        "- reply, web_grounding, web_research, file_write, file_read, file_delete, directory_list, calc, mixed.",
        "Required evidence guidance:",
        "- reply -> reply_emitted",
        "- web_grounding -> fast_grounding",
        "- web_research -> full_web_research",
        "- file_write -> artifact_write + artifact_readback",
        "- file_read -> artifact_readback",
        "- file_delete -> local_delete",
        "- directory_list -> directory_listing",
        "- calc -> calculation_result",
        "- For tools without a special contract, require tool:<tool_name> when the fact must come from that tool.",
        "Return JSON only with this shape:",
        (
            '{"mission_action":"start_new|continue_existing",'
            '"step_mode":"continue|final_reply|blocked",'
            '"mission_goal":"...",'
            '"current_goal":"...",'
            '"reasoning_summary":"...",'
            '"task_board":[{"id":"t1","label":"...","kind":"reply","status":"pending|active|repairing|completed|blocked",'
            '"depends_on":[],"required_evidence":["reply_emitted"],"artifact_paths":[],"evidence_refs":[],"latest_error":null,"repair_count":0}],'
            '"active_task_id":"t1",'
            '"unresolved_gaps":["..."],'
            '"tool_calls":[{"task_id":"t1","tool_name":"search_web","arguments":{"query":"..."}}],'
            '"user_visible_progress":"...",'
            '"reply_to_user":"...",'
            '"completion_claim":false,'
            '"blocker":null,'
            '"next_expected_evidence":"..."}'
        ),
    )
)

_MISSION_EVIDENCE_REDUCER_SYSTEM_PROMPT = "\n".join(
    (
        "Mission evidence reducer for the Unclaw local runtime.",
        "You are still inside the same single-agent mission loop.",
        "A heavy tool result was persisted externally because it is too large to keep reinjecting.",
        "Return JSON only with a compact mission evidence capsule.",
        "Do not plan the whole mission again.",
        "Do not emit user-facing prose.",
        "Keep the capsule small and continuation-oriented.",
        "Focus only on:",
        "- what was found,",
        "- what facts are usable now,",
        "- what remains unresolved,",
        "- what artifact work is still pending,",
        "- the most useful source refs or artifact paths.",
        "Return JSON only with this shape:",
        (
            '{"summary":"...",'
            '"found":["..."],'
            '"usable_facts":["..."],'
            '"unresolved":["..."],'
            '"pending_artifact_work":["..."],'
            '"artifact_paths":["..."],'
            '"source_refs":["..."]}'
        ),
    )
)

_PROMPT_MAX_LIST_ITEMS = 8
_PROMPT_MAX_DICT_ITEMS = 12
_PROMPT_MAX_VALUE_CHARS = 320
_PROMPT_MAX_OUTPUT_CHARS = 1200


@dataclass(frozen=True, slots=True)
class MissionToolRequest:
    """One model-requested tool call inside the mission action."""

    tool_name: str
    arguments: dict[str, Any]
    task_id: str | None = None

    def as_tool_call(self) -> ToolCall:
        return ToolCall(tool_name=self.tool_name, arguments=dict(self.arguments))


@dataclass(frozen=True, slots=True)
class MissionAgentAction:
    """One structured action emitted by the single-agent model loop."""

    mission_action: str
    step_mode: str
    mission_goal: str
    current_goal: str
    reasoning_summary: str
    tasks: tuple[MissionTaskState, ...]
    active_task_id: str | None
    unresolved_gaps: tuple[str, ...]
    tool_calls: tuple[MissionToolRequest, ...]
    user_visible_progress: str | None
    reply_to_user: str | None
    completion_claim: bool
    blocker: str | None
    next_expected_evidence: str | None


@dataclass(frozen=True, slots=True)
class MissionRelationDecision:
    """Compatibility wrapper retained for older imports."""

    relation: str
    summary: str


@dataclass(frozen=True, slots=True)
class MissionPlanDecision:
    """Compatibility wrapper retained for older imports."""

    mission_action: str
    mission_goal: str | None
    deliverables: tuple[MissionTaskState, ...]
    active_deliverable_id: str | None
    summary: str
    execution_queue: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MissionVerificationDecision:
    """Compatibility wrapper retained for older imports."""

    mission_status: str
    active_deliverable_id: str | None
    current_deliverable_status: str
    missing: str | None
    blocker: str | None
    artifact_paths: tuple[str, ...]
    evidence: tuple[str, ...]
    next_action: str
    notes_for_next_step: str | None
    assistant_reply: str | None
    final_deliverables_missing: tuple[str, ...] = ()
    repair_strategy: str | None = None


def build_agent_action_messages(
    *,
    user_input: str,
    mission_state: MissionState | None,
    recent_tool_results: Sequence[ToolResult],
    available_tools: Sequence[ToolDefinition],
    max_steps: int,
    current_step: int,
) -> tuple[LLMMessage, ...]:
    """Build the single mission-agent prompt."""

    payload = {
        "user_input": user_input,
        "current_mission_state": _serialize_mission_state(mission_state),
        "recent_tool_results": [
            _serialize_tool_result(tool_result) for tool_result in recent_tool_results
        ],
        "available_tools": [
            _serialize_tool_definition(tool_definition)
            for tool_definition in available_tools
        ],
        "runtime_limits": {
            "max_steps": max_steps,
            "current_step": current_step,
            "remaining_steps": max(max_steps - current_step, 0),
        },
    }
    return (
        LLMMessage(role=LLMRole.SYSTEM, content=_MISSION_AGENT_SYSTEM_PROMPT),
        LLMMessage(
            role=LLMRole.USER,
            content=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        ),
    )


def build_evidence_capsule_messages(
    *,
    user_input: str,
    mission_state: MissionState,
    task: MissionTaskState | None,
    tool_call: ToolCall,
    tool_result: ToolResult,
    observation_ref: str | None,
) -> tuple[LLMMessage, ...]:
    """Build the compact-evidence reduction prompt for one heavy tool result."""

    payload = {
        "user_input": user_input,
        "mission_state": _serialize_mission_state(mission_state),
        "task": (
            {
                "id": task.id,
                "label": task.title,
                "kind": task.kind,
                "status": task.status,
                "depends_on": list(task.depends_on),
                "required_evidence": list(task.required_evidence),
                "artifact_paths": list(task.artifact_paths),
                "latest_error": task.latest_error,
            }
            if task is not None
            else None
        ),
        "tool_call": {
            "tool_name": tool_call.tool_name,
            "arguments": _compact_prompt_value(tool_call.arguments),
        },
        "observation_ref": observation_ref,
        "heavy_tool_result": _serialize_tool_result_for_reduction(tool_result),
    }
    return (
        LLMMessage(role=LLMRole.SYSTEM, content=_MISSION_EVIDENCE_REDUCER_SYSTEM_PROMPT),
        LLMMessage(
            role=LLMRole.USER,
            content=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        ),
    )


def parse_agent_action_response(
    response_text: str,
    *,
    fallback_user_input: str,
) -> MissionAgentAction | None:
    """Parse one mission action response."""

    payload = _parse_json_dict(response_text)
    if payload is None:
        stripped = response_text.strip()
        if not stripped:
            return None
        return MissionAgentAction(
            mission_action="start_new",
            step_mode="final_reply",
            mission_goal=fallback_user_input,
            current_goal=fallback_user_input,
            reasoning_summary="plain-text fallback action",
            tasks=(
                MissionTaskState(
                    id="t1",
                    title="Reply to the user",
                    kind="reply",
                    status="active",
                    required_evidence=("reply_emitted",),
                ),
            ),
            active_task_id="t1",
            unresolved_gaps=(),
            tool_calls=(),
            user_visible_progress=None,
            reply_to_user=stripped,
            completion_claim=True,
            blocker=None,
            next_expected_evidence=None,
        )

    mission_action = _read_choice(
        payload.get("mission_action"),
        allowed_values=frozenset({"start_new", "continue_existing"}),
    ) or "continue_existing"
    step_mode = _read_choice(
        payload.get("step_mode"),
        allowed_values=frozenset({"continue", "final_reply", "blocked"}),
    )
    mission_goal = _read_optional_text(payload.get("mission_goal")) or fallback_user_input
    current_goal = (
        _read_optional_text(payload.get("current_goal"))
        or mission_goal
        or fallback_user_input
    )
    reasoning_summary = (
        _read_optional_text(payload.get("reasoning_summary"))
        or "single-agent mission action"
    )
    if step_mode is None:
        if _read_optional_text(payload.get("blocker")) is not None:
            step_mode = "blocked"
        elif payload.get("completion_claim") is True:
            step_mode = "final_reply"
        else:
            step_mode = "continue"

    raw_tasks = payload.get("task_board")
    if raw_tasks is None:
        raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        return None
    tasks: list[MissionTaskState] = []
    for index, raw_task in enumerate(raw_tasks[:6], start=1):
        if not isinstance(raw_task, dict):
            return None
        task_id = _read_optional_text(raw_task.get("id")) or f"t{index}"
        title = (
            _read_optional_text(raw_task.get("label"))
            or _read_optional_text(raw_task.get("title"))
            or task_id
        )
        kind = _read_choice(
            raw_task.get("kind"),
            allowed_values=frozenset(
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
            ),
        ) or "mixed"
        status = _read_choice(
            raw_task.get("status"),
            allowed_values=frozenset(
                {"pending", "active", "repairing", "completed", "blocked"}
            ),
        ) or ("active" if index == 1 else "pending")
        required_evidence = _read_text_list(raw_task.get("required_evidence")) or (
            _default_required_evidence(kind)
        )
        tasks.append(
            MissionTaskState(
                id=task_id,
                title=title,
                kind=kind,
                status=status,
                depends_on=_read_text_list(raw_task.get("depends_on")),
                required_evidence=required_evidence,
                artifact_paths=_read_text_list(raw_task.get("artifact_paths")),
                evidence_refs=_read_text_list(raw_task.get("evidence_refs")),
                latest_error=_read_optional_text(raw_task.get("latest_error")),
                repair_count=_read_non_negative_int(raw_task.get("repair_count")),
            )
        )

    active_task_id = _read_optional_text(payload.get("active_task_id"))
    if active_task_id is None and tasks:
        active_task_id = tasks[0].id

    raw_tool_calls = payload.get("tool_calls")
    tool_calls: list[MissionToolRequest] = []
    if raw_tool_calls is not None:
        if not isinstance(raw_tool_calls, list):
            return None
        for raw_tool_call in raw_tool_calls[:8]:
            if not isinstance(raw_tool_call, dict):
                return None
            tool_name = _read_optional_text(raw_tool_call.get("tool_name"))
            arguments = raw_tool_call.get("arguments")
            if not tool_name or not isinstance(arguments, dict):
                return None
            tool_calls.append(
                MissionToolRequest(
                    tool_name=tool_name,
                    arguments=dict(arguments),
                    task_id=_read_optional_text(raw_tool_call.get("task_id")),
                )
            )

    return MissionAgentAction(
        mission_action=mission_action,
        step_mode=step_mode,
        mission_goal=mission_goal,
        current_goal=current_goal,
        reasoning_summary=reasoning_summary,
        tasks=tuple(tasks),
        active_task_id=active_task_id,
        unresolved_gaps=_read_text_list(payload.get("unresolved_gaps")),
        tool_calls=tuple(tool_calls),
        user_visible_progress=_read_optional_text(payload.get("user_visible_progress")),
        reply_to_user=_read_optional_text(payload.get("reply_to_user")),
        completion_claim=payload.get("completion_claim") is True,
        blocker=_read_optional_text(payload.get("blocker")),
        next_expected_evidence=_read_optional_text(
            payload.get("next_expected_evidence")
        ),
    )


def parse_evidence_capsule_response(
    response_text: str,
    *,
    fallback_tool_name: str,
    task_id: str | None,
    created_at: str,
    observation_ref: str | None,
) -> MissionEvidenceCapsule | None:
    """Parse one compact evidence-capsule response."""

    payload = _parse_json_dict(response_text)
    if payload is None:
        return None
    summary = _read_optional_text(payload.get("summary"))
    if summary is None:
        return None
    return MissionEvidenceCapsule(
        id="capsule",
        task_id=task_id,
        tool_name=fallback_tool_name,
        created_at=created_at,
        summary=summary,
        found=_read_text_list(payload.get("found")),
        usable_facts=_read_text_list(payload.get("usable_facts")),
        unresolved=_read_text_list(payload.get("unresolved")),
        pending_artifact_work=_read_text_list(payload.get("pending_artifact_work")),
        artifact_paths=_read_text_list(payload.get("artifact_paths")),
        source_refs=_read_text_list(payload.get("source_refs")),
        observation_ref=observation_ref,
    )


def build_mission_relation_messages(
    *,
    user_input: str,
    existing_mission_state: MissionState | None,
    compatibility_mission_state: MissionState | None,
) -> tuple[LLMMessage, ...]:
    """Compatibility wrapper for legacy callers."""

    del compatibility_mission_state
    return build_agent_action_messages(
        user_input=user_input,
        mission_state=existing_mission_state,
        recent_tool_results=(),
        available_tools=(),
        max_steps=1,
        current_step=1,
    )


def parse_mission_relation_response(
    response_text: str,
) -> MissionRelationDecision | None:
    payload = _parse_json_dict(response_text)
    if payload is None:
        return None
    relation = _read_choice(
        payload.get("relation"),
        allowed_values=frozenset(
            {
                "same_active_mission",
                "repair_same_mission",
                "status_of_same_mission",
                "new_mission",
                "standalone_direct_reply",
            }
        ),
    )
    if relation is None:
        return None
    return MissionRelationDecision(
        relation=relation,
        summary=_read_optional_text(payload.get("summary")) or "mission relation",
    )


def build_mission_plan_messages(
    *,
    user_input: str,
    mission_relation: MissionRelationDecision | None,
    existing_mission_state: MissionState | None,
    compatibility_mission_state: MissionState | None,
    first_response: Any | None,
    available_tool_names: Sequence[str],
) -> tuple[LLMMessage, ...]:
    """Compatibility wrapper for legacy callers."""

    del mission_relation, compatibility_mission_state, first_response
    available_tools = [
        ToolDefinition(
            name=name,
            description=f"Tool {name}",
            permission_level=ToolPermissionLevel.LOCAL_READ,
            arguments={},
        )
        for name in available_tool_names
    ]
    return build_agent_action_messages(
        user_input=user_input,
        mission_state=existing_mission_state,
        recent_tool_results=(),
        available_tools=available_tools,
        max_steps=1,
        current_step=1,
    )


def parse_mission_plan_response(
    response_text: str,
    *,
    updated_at: str,
) -> MissionPlanDecision | None:
    del updated_at
    action = parse_agent_action_response(response_text, fallback_user_input="mission")
    if action is None:
        return None
    return MissionPlanDecision(
        mission_action=action.mission_action,
        mission_goal=action.mission_goal,
        deliverables=action.tasks,
        active_deliverable_id=action.active_task_id,
        summary=action.reasoning_summary,
        execution_queue=tuple(task.id for task in action.tasks),
    )


def build_mission_verification_messages(
    *,
    user_input: str,
    mission_state: MissionState,
    draft_reply: str,
    latest_tool_results: Sequence[ToolResult],
    turn_tool_results: Sequence[ToolResult],
    blocker_metadata: dict[str, Any] | None,
    retry_budget_remaining: int,
    repair_budget_remaining: int,
) -> tuple[LLMMessage, ...]:
    """Compatibility wrapper for legacy callers."""

    del draft_reply, turn_tool_results, blocker_metadata, retry_budget_remaining, repair_budget_remaining
    return build_agent_action_messages(
        user_input=user_input,
        mission_state=mission_state,
        recent_tool_results=latest_tool_results,
        available_tools=(),
        max_steps=1,
        current_step=1,
    )


def parse_mission_verification_response(
    response_text: str,
) -> MissionVerificationDecision | None:
    payload = _parse_json_dict(response_text)
    if payload is None:
        return None
    mission_status = _read_choice(
        payload.get("mission_status"),
        allowed_values=frozenset({"active", "blocked", "completed"}),
    )
    current_status = _read_choice(
        payload.get("current_deliverable_status"),
        allowed_values=frozenset({"pending", "active", "blocked", "completed"}),
    )
    next_action = _read_choice(
        payload.get("next_action"),
        allowed_values=frozenset({"continue", "final_reply", "blocked_reply"}),
    )
    if mission_status is None or current_status is None or next_action is None:
        return None
    return MissionVerificationDecision(
        mission_status=mission_status,
        active_deliverable_id=_read_optional_text(payload.get("active_deliverable_id")),
        current_deliverable_status=current_status,
        missing=_read_optional_text(payload.get("missing")),
        blocker=_read_optional_text(payload.get("blocker")),
        artifact_paths=_read_text_list(payload.get("artifact_paths")),
        evidence=_read_text_list(payload.get("evidence")),
        next_action=next_action,
        notes_for_next_step=_read_optional_text(payload.get("notes_for_next_step")),
        assistant_reply=_read_optional_text(payload.get("assistant_reply")),
        final_deliverables_missing=_read_text_list(
            payload.get("final_deliverables_missing")
        ),
        repair_strategy=_read_optional_text(payload.get("repair_strategy")),
    )


def build_mission_progress_note(
    *,
    mission_state: MissionState,
    verification_decision: MissionVerificationDecision,
    user_input: str,
    latest_tool_results: Sequence[ToolResult],
) -> str:
    del latest_tool_results
    return "\n".join(
        (
            "Mission progress check:",
            f"User request: {user_input}",
            f"Mission goal: {mission_state.mission_goal}",
            f"Active task: {mission_state.active_task or 'none'}",
            f"Next action: {verification_decision.next_action}",
        )
    )


def build_mission_initialization_note(
    *,
    mission_state: MissionState,
) -> str:
    return "\n".join(
        (
            "Mission planning:",
            f"Mission goal: {mission_state.mission_goal}",
            f"Mission status: {mission_state.status}",
            f"Active task: {mission_state.active_task or 'none'}",
        )
    )


def render_mission_status(mission_state: MissionState | None) -> str:
    """Render persisted mission status without hallucinating missing facts."""

    if mission_state is None:
        return "No active mission state is available."

    if mission_state.latest_truthful_status:
        return mission_state.latest_truthful_status

    active_task = mission_state.get_task()
    completed_tasks = [task.title for task in mission_state.tasks if task.status == "completed"]
    parts = [f"Mission status: {mission_state.status}."]
    parts.append(f"Goal: {mission_state.mission_goal}.")
    if completed_tasks:
        parts.append("Done: " + ", ".join(completed_tasks) + ".")
    if active_task is not None:
        parts.append(f"Current task: {active_task.title}.")
    if mission_state.next_expected_evidence:
        parts.append(f"Waiting for: {mission_state.next_expected_evidence}.")
    if mission_state.unresolved_gaps:
        parts.append(f"Unresolved: {mission_state.unresolved_gaps[0]}.")
    return " ".join(parts)


def _serialize_mission_state(mission_state: MissionState | None) -> dict[str, Any] | None:
    if mission_state is None:
        return None
    return {
        "mission_id": mission_state.mission_id,
        "mission_goal": mission_state.mission_goal,
        "status": mission_state.status,
        "reasoning_summary": mission_state.reasoning_summary,
        "user_visible_progress": mission_state.user_visible_progress,
        "active_task_id": mission_state.active_task_id,
        "reply_to_user": mission_state.reply_to_user,
        "completion_claim": mission_state.completion_claim,
        "blocker": mission_state.blocker,
        "next_expected_evidence": mission_state.next_expected_evidence,
        "unresolved_gaps": mission_state.unresolved_gaps,
        "latest_truthful_status": mission_state.latest_truthful_status,
        "final_reply": mission_state.final_reply,
        "executor_state": mission_state.executor_state,
        "task_board_summary": _summarize_task_board(mission_state),
        "last_verified_artifact_paths": mission_state.last_verified_artifact_paths,
        "last_successful_evidence": mission_state.last_successful_evidence,
        "tasks": [
            {
                "id": task.id,
                "label": task.title,
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
            }
            for task in mission_state.tasks
        ],
        "evidence_log": [
            {
                "id": record.id,
                "kind": record.kind,
                "task_id": record.task_id,
                "summary": record.summary,
                "tool_name": record.tool_name,
                "artifact_paths": list(record.artifact_paths),
                "success": record.success,
            }
            for record in mission_state.evidence_log[-6:]
        ],
        "evidence_capsules": [
            {
                "id": record.id,
                "task_id": record.task_id,
                "tool_name": record.tool_name,
                "summary": record.summary,
                "found": list(record.found),
                "usable_facts": list(record.usable_facts),
                "unresolved": list(record.unresolved),
                "pending_artifact_work": list(record.pending_artifact_work),
                "artifact_paths": list(record.artifact_paths),
                "source_refs": list(record.source_refs),
                "observation_ref": record.observation_ref,
            }
            for record in mission_state.evidence_capsules[-4:]
        ],
        "artifact_observations": [
            {
                "path": record.path,
                "status": record.status,
                "task_id": record.task_id,
                "summary": record.summary,
                "tool_name": record.tool_name,
                "evidence_ref": record.evidence_ref,
            }
            for record in mission_state.artifact_observations[-6:]
        ],
        "tool_history": [
            {
                "tool_name": record.tool_name,
                "task_id": record.task_id,
                "arguments": _compact_prompt_value(record.arguments),
                "executed": record.executed,
                "success": record.success,
                "reason": record.reason,
            }
            for record in mission_state.tool_history[-8:]
        ],
        "tool_observation_refs": [
            {
                "id": record.id,
                "tool_name": record.tool_name,
                "task_id": record.task_id,
                "workspace_path": record.workspace_path,
                "success": record.success,
            }
            for record in mission_state.tool_observation_refs[-6:]
        ],
    }


def _serialize_tool_result(tool_result: ToolResult) -> dict[str, Any]:
    payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
    return {
        "tool_name": tool_result.tool_name,
        "success": tool_result.success,
        "output_text": _truncate_text(tool_result.output_text, _PROMPT_MAX_OUTPUT_CHARS),
        "error": tool_result.error,
        "failure_kind": tool_result.failure_kind,
        "payload": _compact_prompt_value(payload),
    }


def _serialize_tool_result_for_reduction(tool_result: ToolResult) -> dict[str, Any]:
    payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
    return {
        "tool_name": tool_result.tool_name,
        "success": tool_result.success,
        "output_text": _truncate_text(tool_result.output_text, _PROMPT_MAX_OUTPUT_CHARS * 2),
        "error": tool_result.error,
        "failure_kind": tool_result.failure_kind,
        "payload": _compact_prompt_value(payload, max_depth=4),
    }


def _serialize_tool_definition(tool_definition: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool_definition.name,
        "description": tool_definition.description,
        "permission_level": tool_definition.permission_level.value,
        "arguments": tuple(sorted(tool_definition.arguments)),
        "required_arguments": tuple(sorted(tool_definition.required_arguments)),
    }


def _default_required_evidence(kind: str) -> tuple[str, ...]:
    defaults = {
        "reply": ("reply_emitted",),
        "web_grounding": ("fast_grounding",),
        "web_research": ("full_web_research",),
        "file_write": ("artifact_write", "artifact_readback"),
        "file_read": ("artifact_readback",),
        "file_delete": ("local_delete",),
        "directory_list": ("directory_listing",),
        "calc": ("calculation_result",),
    }
    return defaults.get(kind, ())


def _summarize_task_board(mission_state: MissionState) -> dict[str, Any]:
    completed = [task.title for task in mission_state.tasks if task.status == "completed"]
    pending = [
        task.title
        for task in mission_state.tasks
        if task.status in {"pending", "active", "repairing"}
    ]
    return {
        "active_task": mission_state.active_task,
        "completed": completed[-4:],
        "pending": pending[:4],
        "next_expected_evidence": mission_state.next_expected_evidence,
        "unresolved_gaps": list(mission_state.unresolved_gaps[:4]),
    }


def _compact_prompt_value(
    value: Any,
    *,
    max_depth: int = 3,
) -> Any:
    if max_depth <= 0:
        if isinstance(value, str):
            return _truncate_text(value, _PROMPT_MAX_VALUE_CHARS)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return f"[list:{len(value)}]"
        if isinstance(value, dict):
            return f"{{dict:{len(value)}}}"
        return repr(value)
    if isinstance(value, str):
        return _truncate_text(value, _PROMPT_MAX_VALUE_CHARS)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [
            _compact_prompt_value(item, max_depth=max_depth - 1)
            for item in value[:_PROMPT_MAX_LIST_ITEMS]
        ]
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for index, key in enumerate(sorted(value)):
            if index >= _PROMPT_MAX_DICT_ITEMS:
                break
            compacted[str(key)] = _compact_prompt_value(
                value[key],
                max_depth=max_depth - 1,
            )
        return compacted
    return _truncate_text(repr(value), _PROMPT_MAX_VALUE_CHARS)


def _truncate_text(value: str | None, max_chars: int) -> str | None:
    if value is None:
        return None
    compact = " ".join(value.split()).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _parse_json_dict(response_text: str) -> dict[str, Any] | None:
    stripped_response = response_text.strip()
    if not stripped_response:
        return None
    try:
        payload = json.loads(stripped_response)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _read_choice(
    value: Any,
    *,
    allowed_values: frozenset[str],
) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip().lower()
    if normalized not in allowed_values:
        return None
    return normalized


def _read_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _read_text_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        normalized = _read_optional_text(item)
        if normalized is not None and normalized not in items:
            items.append(normalized)
    return tuple(items)


def _read_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    return 0


__all__ = [
    "MissionAgentAction",
    "MissionToolRequest",
    "build_agent_action_messages",
    "build_evidence_capsule_messages",
    "parse_evidence_capsule_response",
    "parse_agent_action_response",
    "render_mission_status",
]
