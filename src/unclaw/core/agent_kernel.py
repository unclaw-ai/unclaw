"""Single-agent mission kernel for compact local execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import unclaw.core.agent_loop as _agent_loop
from unclaw.core.mission_events import MissionEventCallback, emit_mission_event
from unclaw.core.mission_state import (
    MissionDeliverableState,
    MissionEvidenceRecord,
    MissionState,
    MissionTaskState,
    MissionToolCallRecord,
    mission_completion_ready,
    normalize_mission_state,
)
from unclaw.core.mission_verifier import (
    MissionAgentAction,
    MissionToolRequest,
    build_agent_action_messages,
    parse_agent_action_response,
    render_mission_status,
)
from unclaw.core.mission_workspace import build_mission_state_from_plan
from unclaw.core.orchestrator import ModelCallFailedError, Orchestrator
from unclaw.core.reply_discipline import _tool_result_timed_out
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMContentCallback, utc_now_iso
from unclaw.logs.tracer import Tracer
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolResult
from unclaw.tools.registry import ToolRegistry

_ACTION_PARSE_REPAIR_ATTEMPTS = 2
_MISSION_REPAIR_BUDGET_PER_TASK = 2
_MISSION_BLOCKED_REPLY = (
    "The mission is blocked and needs either a repair step or a new instruction."
)
_MISSION_INCOMPLETE_REPLY = (
    "The mission is still active and needs another proven step before it can finish."
)
_STATUS_REPLY_PREFIXES = (
    "mission goal:",
    "current active task:",
    "completed tasks:",
    "blocked task:",
    "next expected action or evidence:",
)
_PROGRESS_REPLY_PREFIXES = (
    "i am now ",
    "i will now ",
    "i'm now ",
    "je vais ",
    "je vais maintenant ",
    "je suis en train de ",
)


@dataclass(frozen=True, slots=True)
class AgentKernelPlan:
    """Compatibility planning wrapper for older callers."""

    should_use_kernel: bool
    plan_decision: MissionAgentAction | None = None
    degraded_reply: str | None = None


@dataclass(frozen=True, slots=True)
class AgentKernelRunResult:
    """Resolved outcome for one mission-kernel turn."""

    assistant_reply: str
    mission_state: MissionState | None
    tool_results: tuple[ToolResult, ...]
    persisted: bool


def should_resume_mission_in_kernel(mission_state: MissionState | None) -> bool:
    """Return True when a persisted mission should stay on the kernel path."""

    if mission_state is None:
        return False
    normalized = normalize_mission_state(mission_state)
    return normalized.status == "active"


def classify_turn_mission_relation(  # noqa: D401 - compatibility shim
    *,
    session_id: str,
    user_input: str,
    orchestrator: Orchestrator,
    tracer: Tracer,
    model_profile_name: str,
    thinking_enabled: bool,
    existing_mission_state: MissionState | None,
    compatibility_mission_state: MissionState | None,
):
    """Compatibility shim: the single-agent kernel no longer pre-classifies turns."""

    del (
        session_id,
        user_input,
        orchestrator,
        tracer,
        model_profile_name,
        thinking_enabled,
        existing_mission_state,
        compatibility_mission_state,
    )
    return None


def plan_mission_turn(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
    orchestrator: Orchestrator,
    tracer: Tracer,
    model_profile_name: str,
    thinking_enabled: bool,
    tool_definitions: Sequence[ToolDefinition],
    first_response,
    relation_decision,
    existing_mission_state: MissionState | None,
    compatibility_mission_state: MissionState | None,
) -> AgentKernelPlan:
    """Compatibility planning entrypoint for older runtime callers."""

    del first_response, relation_decision, compatibility_mission_state
    action = _call_agent_action_with_repair(
        session_id=session_id,
        user_input=user_input,
        mission_state=existing_mission_state,
        recent_tool_results=(),
        orchestrator=orchestrator,
        tracer=tracer,
        model_profile_name=model_profile_name,
        thinking_enabled=thinking_enabled,
        tool_definitions=tool_definitions,
        max_steps=1,
        current_step=1,
    )
    if action is None:
        return AgentKernelPlan(
            should_use_kernel=True,
            degraded_reply="I couldn't build a reliable mission action.",
        )
    del session_manager
    return AgentKernelPlan(should_use_kernel=True, plan_decision=action)


def run_agent_kernel_turn(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
    orchestrator: Orchestrator,
    tracer: Tracer,
    tool_registry: ToolRegistry,
    tool_definitions: Sequence[ToolDefinition],
    model_profile_name: str,
    thinking_enabled: bool,
    capability_summary,
    system_context_notes: tuple[str, ...],
    max_steps: int,
    tool_guard_state: _agent_loop._RuntimeToolGuardState,
    tool_call_callback: Callable[[ToolCall], None] | None,
    mission_event_callback: MissionEventCallback | None,
    content_callback: LLMContentCallback | None,
    first_response=None,
    relation_decision=None,
    existing_mission_state: MissionState | None = None,
    compatibility_mission_state: MissionState | None = None,
    planned_decision: MissionAgentAction | None = None,
) -> AgentKernelRunResult:
    """Execute one mission turn through the single-agent loop."""

    del capability_summary
    del system_context_notes
    del content_callback
    del first_response
    del relation_decision
    del compatibility_mission_state

    mission_state = (
        normalize_mission_state(existing_mission_state)
        if existing_mission_state is not None
        else session_manager.get_current_mission_state(session_id)
    )
    all_tool_results: list[ToolResult] = []
    recent_tool_results: tuple[ToolResult, ...] = ()
    final_reply: str | None = None

    emit_mission_event(mission_event_callback, scope="mission", detail="single-agent loop")

    for step_index in range(max(max_steps, 1)):
        if tool_guard_state.is_cancelled():
            mission_state = _block_mission(
                mission_state=mission_state,
                blocker=_agent_loop._TURN_CANCELLED_REPLY,
                user_input=user_input,
            )
            final_reply = _agent_loop._TURN_CANCELLED_REPLY
            break

        action = planned_decision if step_index == 0 and planned_decision is not None else _call_agent_action_with_repair(
            session_id=session_id,
            user_input=user_input,
            mission_state=mission_state,
            recent_tool_results=recent_tool_results,
            orchestrator=orchestrator,
            tracer=tracer,
            model_profile_name=model_profile_name,
            thinking_enabled=thinking_enabled,
            tool_definitions=tool_definitions,
            max_steps=max_steps,
            current_step=step_index + 1,
        )
        if action is None:
            final_reply = _build_mission_status_reply(mission_state)
            break

        mission_state = _apply_action_metadata(
            session_manager=session_manager,
            user_input=user_input,
            existing_mission_state=mission_state,
            action=action,
        )
        session_manager.persist_mission_state(mission_state, session_id=session_id)
        emit_mission_event(
            mission_event_callback,
            scope="mission",
            detail=f"active task: {mission_state.active_task or 'none'}",
        )

        recent_tool_results = ()
        if action.tool_calls:
            mission_state, recent_tool_results = _execute_action_tool_calls(
                mission_state=mission_state,
                action=action,
                session_manager=session_manager,
                session_id=session_id,
                tracer=tracer,
                tool_registry=tool_registry,
                tool_guard_state=tool_guard_state,
                tool_call_callback=tool_call_callback,
                mission_event_callback=mission_event_callback,
            )
            all_tool_results.extend(recent_tool_results)

        reply_is_final = _reply_is_final_user_message(action.reply_to_user)

        if action.reply_to_user and reply_is_final:
            mission_state = _apply_reply_evidence(
                mission_state=mission_state,
                reply_to_user=action.reply_to_user,
            )

        mission_state = _apply_action_completion_signals(
            mission_state=mission_state,
            action=action,
            reply_is_final=reply_is_final,
        )
        session_manager.persist_mission_state(mission_state, session_id=session_id)

        if mission_completion_ready(mission_state):
            final_reply_candidate = _final_reply_candidate(
                mission_state=mission_state,
                action=action,
                reply_is_final=reply_is_final,
            )
            if final_reply_candidate is None and step_index + 1 < max(max_steps, 1):
                emit_mission_event(
                    mission_event_callback,
                    scope="mission",
                    detail="mission evidence complete; awaiting final reply",
                )
                continue
            mission_state = normalize_mission_state(
                replace(
                    mission_state,
                    status="completed",
                    completion_claim=True,
                    final_reply=(
                        final_reply_candidate or render_mission_status(mission_state)
                    ),
                    reply_to_user=final_reply_candidate or mission_state.reply_to_user,
                )
            )
            session_manager.persist_mission_state(mission_state, session_id=session_id)
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail="mission completed",
            )
            final_reply = mission_state.final_reply or _build_mission_status_reply(mission_state)
            break

        if mission_state.status == "blocked":
            session_manager.persist_mission_state(mission_state, session_id=session_id)
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail="mission blocked",
            )
            final_reply = (
                action.reply_to_user
                or mission_state.final_reply
                or _build_mission_status_reply(mission_state)
            )
            break

        if not action.tool_calls and not action.reply_to_user:
            final_reply = _build_mission_status_reply(mission_state)
            break

        if (
            step_index + 1 >= max(max_steps, 1)
            and final_reply is None
            and mission_state.status != "completed"
        ):
            final_reply = _build_mission_status_reply(mission_state)
            break

    if mission_state is not None:
        session_manager.persist_mission_state(mission_state, session_id=session_id)
    if final_reply is None:
        final_reply = _build_mission_status_reply(mission_state)

    return AgentKernelRunResult(
        assistant_reply=final_reply,
        mission_state=mission_state,
        tool_results=tuple(all_tool_results),
        persisted=mission_state is not None,
    )


def _call_agent_action_with_repair(
    *,
    session_id: str,
    user_input: str,
    mission_state: MissionState | None,
    recent_tool_results: Sequence[ToolResult],
    orchestrator: Orchestrator,
    tracer: Tracer,
    model_profile_name: str,
    thinking_enabled: bool,
    tool_definitions: Sequence[ToolDefinition],
    max_steps: int,
    current_step: int,
) -> MissionAgentAction | None:
    messages = list(
        build_agent_action_messages(
            user_input=user_input,
            mission_state=mission_state,
            recent_tool_results=recent_tool_results,
            available_tools=tool_definitions,
            max_steps=max_steps,
            current_step=current_step,
        )
    )
    repair_note = (
        "Return valid JSON only using the single mission action schema. "
        "Do not add prose outside the JSON object."
    )
    for attempt in range(_ACTION_PARSE_REPAIR_ATTEMPTS):
        try:
            turn_result = orchestrator.call_model(
                session_id=session_id,
                messages=tuple(messages),
                model_profile_name=model_profile_name,
                thinking_enabled=thinking_enabled,
                content_callback=None,
                tools=(),
            )
        except ModelCallFailedError:
            return None
        _trace_model_success(tracer=tracer, session_id=session_id, turn_result=turn_result)
        action = parse_agent_action_response(
            turn_result.response.content,
            fallback_user_input=user_input,
        )
        if action is not None:
            return action
        if attempt + 1 >= _ACTION_PARSE_REPAIR_ATTEMPTS:
            return None
        messages.extend(
            (
                replace(messages[-1], role="assistant", content=turn_result.response.content),
                messages[0].__class__(role="system", content=repair_note),
            )
        )
    return None


def _apply_action_metadata(
    *,
    session_manager: SessionManager,
    user_input: str,
    existing_mission_state: MissionState | None,
    action: MissionAgentAction,
) -> MissionState:
    if (
        existing_mission_state is None
        or existing_mission_state.status == "completed"
        or action.mission_action == "start_new"
    ):
        tasks = action.tasks or (
            MissionTaskState(
                id="t1",
                title="Reply to the user",
                kind="reply",
                status="active",
                required_evidence=("reply_emitted",),
            ),
        )
        state = build_mission_state_from_plan(
            mission_id=session_manager.create_mission_id(),
            mission_goal=action.mission_goal or user_input,
            deliverables=tasks,
            active_deliverable_id=action.active_task_id or tasks[0].id,
            planner_summary=action.reasoning_summary,
            updated_at=utc_now_iso(),
        )
        return normalize_mission_state(
            replace(
                state,
                reply_to_user=action.reply_to_user,
                completion_claim=action.completion_claim,
                blocker=action.blocker,
                next_expected_evidence=action.next_expected_evidence,
                last_user_input=user_input,
                loop_count=1,
            )
        )

    existing_by_id = {task.id: task for task in existing_mission_state.tasks}
    merged_tasks: list[MissionTaskState] = list(existing_mission_state.tasks)
    replaced_ids: set[str] = set()
    for action_task in action.tasks:
        existing_task = existing_by_id.get(action_task.id)
        if existing_task is not None:
            if existing_task.status == "completed":
                replaced_ids.add(action_task.id)
                continue
            merged_task = replace(
                action_task,
                satisfied_evidence=existing_task.satisfied_evidence,
                artifact_paths=(
                    action_task.artifact_paths or existing_task.artifact_paths
                ),
                evidence=existing_task.evidence,
                latest_error=action_task.latest_error or existing_task.latest_error,
                repair_count=max(action_task.repair_count, existing_task.repair_count),
                updated_at=utc_now_iso(),
            )
            merged_tasks = [
                merged_task if task.id == action_task.id else task for task in merged_tasks
            ]
            replaced_ids.add(action_task.id)
            continue
        merged_tasks.append(replace(action_task, updated_at=utc_now_iso()))
        replaced_ids.add(action_task.id)

    state = replace(
        existing_mission_state,
        mission_goal=action.mission_goal or existing_mission_state.mission_goal,
        tasks=tuple(merged_tasks),
        active_task_id=action.active_task_id or existing_mission_state.active_task_id,
        reasoning_summary=action.reasoning_summary,
        reply_to_user=action.reply_to_user,
        completion_claim=action.completion_claim,
        blocker=action.blocker or existing_mission_state.blocker,
        next_expected_evidence=(
            action.next_expected_evidence or existing_mission_state.next_expected_evidence
        ),
        last_user_input=user_input,
        loop_count=existing_mission_state.loop_count + 1,
        updated_at=utc_now_iso(),
        executor_state=(
            "repairing"
            if any(task.status == "repairing" for task in merged_tasks)
            else "active"
        ),
    )
    return normalize_mission_state(state)


def _execute_action_tool_calls(
    *,
    mission_state: MissionState,
    action: MissionAgentAction,
    session_manager: SessionManager,
    session_id: str,
    tracer: Tracer,
    tool_registry: ToolRegistry,
    tool_guard_state: _agent_loop._RuntimeToolGuardState,
    tool_call_callback: Callable[[ToolCall], None] | None,
    mission_event_callback: MissionEventCallback | None,
) -> tuple[MissionState, tuple[ToolResult, ...]]:
    recent_tool_results: list[ToolResult] = []
    state = mission_state
    for tool_request in action.tool_calls:
        target_task_id = tool_request.task_id or state.active_task_id or action.active_task_id
        target_task = state.get_task(target_task_id)
        if target_task is None:
            state = _record_rejected_tool_call(
                state=state,
                tool_request=tool_request,
                reason="unknown task",
                target_task_id=target_task_id,
            )
            continue
        if target_task.status == "completed":
            state = _record_rejected_tool_call(
                state=state,
                tool_request=tool_request,
                reason="task already completed",
                target_task_id=target_task.id,
            )
            continue
        unmet_dependencies = [
            dependency
            for dependency in target_task.depends_on
            if dependency not in state.completed_deliverables
        ]
        if unmet_dependencies:
            state = _record_rejected_tool_call(
                state=replace(
                    state,
                    next_expected_evidence=(
                        "Complete prerequisite task(s): " + ", ".join(unmet_dependencies)
                    ),
                ),
                tool_request=tool_request,
                reason="prerequisite proof missing",
                target_task_id=target_task.id,
            )
            continue

        tool_call = _annotate_tool_call_for_mission(
            tool_request=tool_request,
            mission_state=state,
            target_task_id=target_task.id,
        )
        emit_mission_event(
            mission_event_callback,
            scope="mission",
            detail=f"{target_task.title}: {tool_call.tool_name}",
        )
        stop_reply = _agent_loop._preflight_runtime_tool_batch(
            tool_calls=(tool_call,),
            tool_guard_state=tool_guard_state,
        )
        if stop_reply is not None:
            return (
                _block_mission(
                    mission_state=state,
                    blocker=stop_reply,
                    user_input=state.last_user_input or state.mission_goal,
                    target_task_id=target_task.id,
                ),
                tuple(recent_tool_results),
            )

        tool_results = _agent_loop._execute_runtime_tool_calls(
            session_manager=session_manager,
            session_id=session_id,
            tracer=tracer,
            tool_registry=tool_registry,
            tool_calls=(tool_call,),
            tool_guard_state=tool_guard_state,
            tool_call_callback=tool_call_callback,
        )
        for tool_result in tool_results:
            recent_tool_results.append(tool_result)
            state = _apply_tool_result(
                mission_state=state,
                task_id=target_task.id,
                tool_call=tool_call,
                tool_result=tool_result,
            )
            if state.status == "blocked":
                return state, tuple(recent_tool_results)
    return state, tuple(recent_tool_results)


def _annotate_tool_call_for_mission(
    *,
    tool_request: MissionToolRequest,
    mission_state: MissionState,
    target_task_id: str,
) -> ToolCall:
    arguments = dict(tool_request.arguments)
    arguments.setdefault("mission_id", mission_state.mission_id)
    arguments.setdefault("mission_task_id", target_task_id)
    arguments.setdefault("mission_deliverable_id", target_task_id)
    return ToolCall(tool_name=tool_request.tool_name, arguments=arguments)


def _apply_tool_result(
    *,
    mission_state: MissionState,
    task_id: str,
    tool_call: ToolCall,
    tool_result: ToolResult,
) -> MissionState:
    task = mission_state.get_task(task_id)
    if task is None:
        return mission_state
    timestamp = utc_now_iso()
    payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
    artifact_paths = _collect_artifact_paths(tool_result)
    evidence_kinds = _evidence_kinds_for_tool_result(tool_result)
    evidence_texts = _collect_evidence_texts(tool_result)
    tool_history = mission_state.tool_history + (
        MissionToolCallRecord(
            tool_name=tool_call.tool_name,
            task_id=task_id,
            arguments=dict(tool_call.arguments),
            created_at=timestamp,
            executed=True,
            success=tool_result.success,
            reason=tool_result.error,
        ),
    )

    if tool_result.success is not True:
        repair_count = task.repair_count + 1
        latest_error = tool_result.error or tool_result.output_text
        if _tool_result_timed_out(tool_result) and repair_count <= _MISSION_REPAIR_BUDGET_PER_TASK:
            updated_task = replace(
                task,
                status="repairing",
                latest_error=latest_error,
                repair_count=repair_count,
                updated_at=timestamp,
            )
            state = mission_state.replace_task(updated_task, updated_at=timestamp)
            return normalize_mission_state(
                replace(
                    state,
                    tool_history=tool_history[-24:],
                    next_expected_evidence=_repair_note_for_tool(tool_call.tool_name),
                    repair_history=(
                        state.repair_history + (_repair_note_for_tool(tool_call.tool_name),)
                    )[-8:],
                    executor_state="repairing",
                    blocker=None,
                    last_blocker=None,
                )
            )
        return _block_mission(
            mission_state=replace(
                mission_state,
                tool_history=tool_history[-24:],
            ),
            blocker=latest_error or f"{tool_call.tool_name} failed",
            user_input=mission_state.last_user_input or mission_state.mission_goal,
            target_task_id=task_id,
        )

    satisfied_evidence = tuple(
        item
        for item in dict.fromkeys(task.satisfied_evidence + evidence_kinds)
    )
    updated_task = replace(
        task,
        satisfied_evidence=satisfied_evidence,
        artifact_paths=tuple(
            item for item in dict.fromkeys(task.artifact_paths + artifact_paths)
        )[:8],
        evidence=tuple(
            item for item in dict.fromkeys(task.evidence + evidence_texts)
        )[:8],
        latest_error=None,
        status=(
            "completed"
            if _task_has_required_evidence(task=task, satisfied_evidence=satisfied_evidence)
            else "active"
        ),
        updated_at=timestamp,
    )
    state = mission_state.replace_task(updated_task, updated_at=timestamp)
    next_state = normalize_mission_state(
        replace(
            state,
            evidence_log=(
                state.evidence_log
                + tuple(
                    MissionEvidenceRecord(
                        kind=evidence_kind,
                        task_id=task_id,
                        summary=_evidence_summary(
                            tool_result=tool_result,
                            evidence_kind=evidence_kind,
                        ),
                        created_at=timestamp,
                        tool_name=tool_result.tool_name,
                        artifact_paths=artifact_paths,
                        success=True,
                    )
                    for evidence_kind in evidence_kinds
                )
            )[-12:],
            tool_history=tool_history[-24:],
            last_verified_artifact_paths=tuple(
                item
                for item in dict.fromkeys(
                    state.last_verified_artifact_paths + artifact_paths
                )
            )[-8:],
            last_successful_evidence=tuple(
                item
                for item in dict.fromkeys(
                    state.last_successful_evidence + evidence_texts
                )
            )[-8:],
            next_expected_evidence=_next_expected_evidence(state),
            executor_state=(
                "active"
                if state.status == "active"
                else state.executor_state
            ),
            blocker=None,
            last_blocker=None,
        )
    )
    del payload
    return next_state


def _apply_reply_evidence(
    *,
    mission_state: MissionState,
    reply_to_user: str,
) -> MissionState:
    target_task = mission_state.get_task()
    if target_task is None:
        target_task = next(
            (
                task
                for task in mission_state.tasks
                if task.kind in {"reply", "mixed"}
                and task.status not in {"completed", "blocked"}
                and all(
                    dependency in mission_state.completed_deliverables
                    for dependency in task.depends_on
                )
            ),
            None,
        )
    if target_task is None:
        return normalize_mission_state(
            replace(
                mission_state,
                reply_to_user=reply_to_user,
                final_reply=reply_to_user if mission_completion_ready(mission_state) else mission_state.final_reply,
            )
        )
    timestamp = utc_now_iso()
    satisfied_evidence = tuple(
        item
        for item in dict.fromkeys(target_task.satisfied_evidence + ("reply_emitted",))
    )
    updated_task = replace(
        target_task,
        satisfied_evidence=satisfied_evidence,
        evidence=tuple(
            item
            for item in dict.fromkeys(
                target_task.evidence + (_compact_reply_evidence(reply_to_user),)
            )
        )[:8],
        status=(
            "completed"
            if _task_has_required_evidence(
                task=target_task,
                satisfied_evidence=satisfied_evidence,
            )
            else "active"
        ),
        updated_at=timestamp,
    )
    state = mission_state.replace_task(updated_task, updated_at=timestamp)
    return normalize_mission_state(
        replace(
            state,
            reply_to_user=reply_to_user,
            final_reply=(reply_to_user if mission_completion_ready(state) else state.final_reply),
            evidence_log=(
                state.evidence_log
                + (
                    MissionEvidenceRecord(
                        kind="reply_emitted",
                        task_id=updated_task.id,
                        summary=_compact_reply_evidence(reply_to_user),
                        created_at=timestamp,
                        tool_name=None,
                        artifact_paths=(),
                        success=True,
                    ),
                )
            )[-12:],
            last_successful_evidence=tuple(
                item
                for item in dict.fromkeys(
                    state.last_successful_evidence
                    + (_compact_reply_evidence(reply_to_user),)
                )
            )[-8:],
            next_expected_evidence=_next_expected_evidence(state),
        )
    )


def _apply_action_completion_signals(
    *,
    mission_state: MissionState,
    action: MissionAgentAction,
    reply_is_final: bool,
) -> MissionState:
    state = mission_state
    if action.blocker:
        return _block_mission(
            mission_state=state,
            blocker=action.blocker,
            user_input=state.last_user_input or state.mission_goal,
            target_task_id=state.active_task_id,
        )
    if action.completion_claim and mission_completion_ready(state) and (
        reply_is_final or state.final_reply is not None
    ):
        return normalize_mission_state(
            replace(
                state,
                status="completed",
                completion_claim=True,
                final_reply=(
                    action.reply_to_user if reply_is_final else state.final_reply
                ),
                reply_to_user=(
                    action.reply_to_user if reply_is_final else state.reply_to_user
                ),
                executor_state="completed",
                next_expected_evidence=None,
            )
        )
    if action.next_expected_evidence:
        return normalize_mission_state(
            replace(
                state,
                next_expected_evidence=action.next_expected_evidence,
            )
        )
    return normalize_mission_state(replace(state, next_expected_evidence=_next_expected_evidence(state)))


def _record_rejected_tool_call(
    *,
    state: MissionState,
    tool_request: MissionToolRequest,
    reason: str,
    target_task_id: str | None,
) -> MissionState:
    timestamp = utc_now_iso()
    return normalize_mission_state(
        replace(
            state,
            tool_history=(
                state.tool_history
                + (
                    MissionToolCallRecord(
                        tool_name=tool_request.tool_name,
                        task_id=target_task_id,
                        arguments=dict(tool_request.arguments),
                        created_at=timestamp,
                        executed=False,
                        success=None,
                        reason=reason,
                    ),
                )
            )[-24:],
        )
    )


def _block_mission(
    *,
    mission_state: MissionState | None,
    blocker: str,
    user_input: str,
    target_task_id: str | None = None,
) -> MissionState:
    timestamp = utc_now_iso()
    if mission_state is None:
        task = MissionTaskState(
            id="t1",
            title="Mission blocked",
            kind="mixed",
            status="blocked",
            latest_error=blocker,
            updated_at=timestamp,
        )
        return normalize_mission_state(
            MissionState(
                mission_id="mission-blocked",
                mission_goal=user_input,
                status="blocked",
                tasks=(task,),
                active_task_id=None,
                updated_at=timestamp,
                blocker=blocker,
                next_expected_evidence=None,
                last_user_input=user_input,
                executor_state="blocked",
                last_blocker=blocker,
            )
        )
    blocked_task_id = target_task_id or mission_state.active_task_id
    tasks = list(mission_state.tasks)
    if blocked_task_id is not None:
        tasks = [
            replace(
                task,
                status="blocked" if task.id == blocked_task_id else task.status,
                latest_error=blocker if task.id == blocked_task_id else task.latest_error,
                updated_at=timestamp if task.id == blocked_task_id else task.updated_at,
            )
            for task in tasks
        ]
    return normalize_mission_state(
        replace(
            mission_state,
            status="blocked",
            tasks=tuple(tasks),
            active_task_id=None,
            blocker=blocker,
            next_expected_evidence=None,
            final_reply=None,
            updated_at=timestamp,
            executor_state="blocked",
            last_blocker=blocker,
        )
    )


def _evidence_kinds_for_tool_result(tool_result: ToolResult) -> tuple[str, ...]:
    if tool_result.success is not True:
        return ()
    mapping = {
        "fast_web_search": ("fast_grounding",),
        "search_web": ("full_web_research",),
        "write_text_file": ("artifact_write",),
        "read_text_file": ("artifact_readback",),
        "delete_file": ("local_delete",),
        "list_directory": ("directory_listing",),
        "calc": ("calculation_result",),
    }
    return mapping.get(tool_result.tool_name, (f"tool:{tool_result.tool_name}",))


def _collect_artifact_paths(tool_result: ToolResult) -> tuple[str, ...]:
    payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
    paths: list[str] = []
    for key in ("resolved_path", "path", "requested_path"):
        value = payload.get(key)
        if isinstance(value, str) and value not in paths:
            paths.append(value)
    return tuple(paths[:4])


def _collect_evidence_texts(tool_result: ToolResult) -> tuple[str, ...]:
    payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
    evidence: list[str] = []
    summary_points = payload.get("summary_points")
    if isinstance(summary_points, list):
        for item in summary_points:
            if isinstance(item, str) and item not in evidence:
                evidence.append(item)
    display_sources = payload.get("display_sources")
    if isinstance(display_sources, list):
        for item in display_sources:
            if isinstance(item, dict):
                url = item.get("url")
                if isinstance(url, str) and url not in evidence:
                    evidence.append(url)
    if not evidence and tool_result.output_text.strip():
        evidence.append(tool_result.output_text.strip().splitlines()[0])
    return tuple(evidence[:4])


def _task_has_required_evidence(
    *,
    task: MissionTaskState,
    satisfied_evidence: tuple[str, ...],
) -> bool:
    if not task.required_evidence:
        return bool(satisfied_evidence) or task.kind == "reply"
    return all(evidence in satisfied_evidence for evidence in task.required_evidence)


def _next_expected_evidence(mission_state: MissionState) -> str | None:
    active_task = mission_state.get_task()
    if active_task is None:
        return None
    missing = [
        evidence
        for evidence in active_task.required_evidence
        if evidence not in active_task.satisfied_evidence
    ]
    if missing:
        return ", ".join(missing)
    return None


def _repair_note_for_tool(tool_name: str) -> str:
    if tool_name == "search_web":
        return "Retry search_web with a narrower query or smaller max_results."
    if tool_name == "write_text_file":
        return "Repair the file write and verify it with read_text_file."
    return f"Repair the {tool_name} step with a bounded retry."


def _compact_reply_evidence(reply_to_user: str) -> str:
    compact = " ".join(reply_to_user.split()).strip()
    return compact[:160] if len(compact) > 160 else compact


def _reply_is_final_user_message(reply_to_user: str | None) -> bool:
    if reply_to_user is None:
        return False
    compact = " ".join(reply_to_user.split()).strip()
    if not compact:
        return False
    lowered = compact.casefold()
    if lowered.startswith(_STATUS_REPLY_PREFIXES):
        return False
    if any(lowered.startswith(prefix) for prefix in _PROGRESS_REPLY_PREFIXES):
        return False
    return True


def _final_reply_candidate(
    *,
    mission_state: MissionState,
    action: MissionAgentAction,
    reply_is_final: bool,
) -> str | None:
    if reply_is_final and action.reply_to_user:
        return action.reply_to_user
    if mission_state.final_reply and _reply_is_final_user_message(mission_state.final_reply):
        return mission_state.final_reply
    if mission_state.reply_to_user and _reply_is_final_user_message(mission_state.reply_to_user):
        return mission_state.reply_to_user
    return None


def _evidence_summary(*, tool_result: ToolResult, evidence_kind: str) -> str:
    evidence_texts = _collect_evidence_texts(tool_result)
    if evidence_texts:
        return evidence_texts[0]
    if evidence_kind.startswith("tool:"):
        return f"{tool_result.tool_name} succeeded"
    return tool_result.output_text.strip() or f"{evidence_kind} recorded"


def _build_mission_status_reply(mission_state: MissionState | None) -> str:
    if mission_state is None:
        return _MISSION_INCOMPLETE_REPLY
    if mission_state.status == "completed":
        return mission_state.final_reply or render_mission_status(mission_state)
    if mission_state.status == "blocked":
        blocker = mission_state.last_blocker or mission_state.blocker or "unknown blocker"
        return f"{render_mission_status(mission_state)}\nBlocker: {blocker}"
    return render_mission_status(mission_state)


def _trace_model_success(*, tracer: Tracer, session_id: str, turn_result) -> None:
    tracer.trace_model_succeeded(
        session_id=session_id,
        provider=turn_result.response.provider,
        model_name=turn_result.response.model_name,
        finish_reason=turn_result.response.finish_reason,
        output_length=len(turn_result.response.content),
        model_duration_ms=turn_result.model_duration_ms,
        reasoning=turn_result.response.reasoning,
    )


__all__ = [
    "AgentKernelPlan",
    "AgentKernelRunResult",
    "_build_mission_status_reply",
    "classify_turn_mission_relation",
    "plan_mission_turn",
    "run_agent_kernel_turn",
    "should_resume_mission_in_kernel",
]
