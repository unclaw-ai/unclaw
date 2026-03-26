"""True mission kernel for compact local-agent execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import unclaw.core.agent_loop as _agent_loop
import unclaw.core.runtime_support as _runtime_support
from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.core.execution_queue import resolve_active_deliverable_id
from unclaw.core.mission_events import MissionEventCallback, emit_mission_event
from unclaw.core.mission_state import MissionDeliverableState, MissionState
from unclaw.core.mission_verifier import (
    MissionPlanDecision,
    MissionVerificationDecision,
    build_mission_initialization_note,
    build_mission_plan_messages,
    build_mission_progress_note,
    build_mission_verification_messages,
    parse_mission_plan_response,
    parse_mission_verification_response,
)
from unclaw.core.mission_workspace import (
    build_mission_state_from_plan,
    synchronize_mission_working_memory,
)
from unclaw.core.orchestrator import (
    ModelCallFailedError,
    Orchestrator,
    OrchestratorTurnResult,
)
from unclaw.core.reply_discipline import _tool_result_timed_out
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMContentCallback, LLMMessage, LLMRole, utc_now_iso
from unclaw.logs.tracer import Tracer
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolResult
from unclaw.tools.registry import ToolRegistry

_PLANNER_PARSE_REPAIR_ATTEMPTS = 2
_VERIFIER_PARSE_REPAIR_ATTEMPTS = 2
_MISSION_RETRY_BUDGET_PER_DELIVERABLE = 2
_MISSION_REPAIR_BUDGET_PER_DELIVERABLE = 2
_MISSION_STATUS_BLOCKED_REPLY = (
    "The mission is blocked and needs either a repaired step or a new user instruction."
)
_MISSION_STATUS_INCOMPLETE_REPLY = (
    "The mission is still active and needs another verified step before it can finish."
)


@dataclass(frozen=True, slots=True)
class AgentKernelPlan:
    """Mission-planning outcome used to decide kernel routing."""

    should_use_kernel: bool
    plan_decision: MissionPlanDecision | None
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
    if len(mission_state.deliverables) > 1:
        return True
    if mission_state.status != "active":
        return False
    return bool(
        mission_state.retry_history
        or mission_state.repair_history
        or mission_state.pending_repairs
    )


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
    first_response: OrchestratorTurnResult | None,
    existing_mission_state: MissionState | None,
    compatibility_mission_state: MissionState | None,
) -> AgentKernelPlan:
    """Return whether the current turn should enter the mission kernel."""

    plan_decision = _plan_mission(
        session_id=session_id,
        user_input=user_input,
        orchestrator=orchestrator,
        tracer=tracer,
        model_profile_name=model_profile_name,
        thinking_enabled=thinking_enabled,
        tool_definitions=tool_definitions,
        first_response=first_response,
        existing_mission_state=existing_mission_state,
        compatibility_mission_state=compatibility_mission_state,
    )
    if plan_decision is None:
        if existing_mission_state is not None:
            return AgentKernelPlan(
                should_use_kernel=True,
                plan_decision=_continue_plan_from_mission(existing_mission_state),
            )
        if first_response is not None:
            draft_reply = first_response.response.content.strip()
            return AgentKernelPlan(
                should_use_kernel=False,
                plan_decision=None,
                degraded_reply=draft_reply or None,
            )
        return AgentKernelPlan(should_use_kernel=False, plan_decision=None)

    if first_response is None:
        return AgentKernelPlan(should_use_kernel=True, plan_decision=plan_decision)
    if plan_decision.mission_action == "direct_reply_only":
        return AgentKernelPlan(should_use_kernel=False, plan_decision=plan_decision)

    first_response_has_local_write = _response_has_local_write_tool_call(
        response=first_response,
        tool_definitions=tool_definitions,
    )
    should_use_kernel = (
        existing_mission_state is not None
        or compatibility_mission_state is not None
        or len(plan_decision.deliverables) > 1
        or first_response_has_local_write
    )
    return AgentKernelPlan(
        should_use_kernel=should_use_kernel,
        plan_decision=plan_decision,
    )


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
    capability_summary: RuntimeCapabilitySummary,
    system_context_notes: tuple[str, ...],
    max_steps: int,
    tool_guard_state: _agent_loop._RuntimeToolGuardState,
    tool_call_callback: Callable[[ToolCall], None] | None,
    mission_event_callback: MissionEventCallback | None,
    content_callback: LLMContentCallback | None,
    first_response: OrchestratorTurnResult | None = None,
    existing_mission_state: MissionState | None = None,
    compatibility_mission_state: MissionState | None = None,
    planned_decision: MissionPlanDecision | None = None,
) -> AgentKernelRunResult:
    """Execute one mission turn through the planner/executor/verifier kernel."""

    emit_mission_event(mission_event_callback, scope="mission", detail="planning")
    active_mission_state = (
        existing_mission_state
        if existing_mission_state is not None
        else session_manager.get_current_mission_state(session_id)
    )
    plan_decision = planned_decision or _plan_mission(
        session_id=session_id,
        user_input=user_input,
        orchestrator=orchestrator,
        tracer=tracer,
        model_profile_name=model_profile_name,
        thinking_enabled=thinking_enabled,
        tool_definitions=tool_definitions,
        first_response=first_response,
        existing_mission_state=active_mission_state,
        compatibility_mission_state=compatibility_mission_state,
    )
    if plan_decision is None:
        return AgentKernelRunResult(
            assistant_reply=(
                "I couldn't form a reliable mission plan for this request, so I "
                "stopped instead of guessing."
            ),
            mission_state=None,
            tool_results=(),
            persisted=False,
        )

    if plan_decision.mission_action == "direct_reply_only":
        if first_response is not None:
            direct_reply = first_response.response.content.strip()
            return AgentKernelRunResult(
                assistant_reply=direct_reply or _runtime_support.EMPTY_RESPONSE_REPLY,
                mission_state=None,
                tool_results=(),
                persisted=False,
            )
        direct_response = orchestrator.run_turn(
            session_id=session_id,
            user_message=user_input,
            model_profile_name=model_profile_name,
            capability_summary=capability_summary,
            system_context_notes=system_context_notes,
            thinking_enabled=thinking_enabled,
            content_callback=content_callback,
            tools=tool_definitions,
        )
        _trace_model_success(tracer=tracer, session_id=session_id, turn_result=direct_response)
        return AgentKernelRunResult(
            assistant_reply=(
                direct_response.response.content.strip()
                or _runtime_support.EMPTY_RESPONSE_REPLY
            ),
            mission_state=None,
            tool_results=(),
            persisted=False,
        )

    mission_state = _resolve_mission_state(
        session_manager=session_manager,
        session_id=session_id,
        user_input=user_input,
        existing_mission_state=active_mission_state,
        compatibility_mission_state=compatibility_mission_state,
        plan_decision=plan_decision,
    )
    session_manager.persist_mission_state(mission_state, session_id=session_id)

    context_messages: list[LLMMessage] = (
        list(first_response.context_messages) if first_response is not None else []
    )
    current_response = first_response
    accumulated_tool_results: list[ToolResult] = []
    final_reply: str | None = None
    execution_announced = False

    for step_index in range(max_steps):
        mission_state = synchronize_mission_working_memory(
            mission_state,
            updated_at=utc_now_iso(),
        )
        session_manager.persist_mission_state(mission_state, session_id=session_id)
        active_deliverable = mission_state.get_deliverable()
        active_deliverable_id = active_deliverable.deliverable_id if active_deliverable else None

        if tool_guard_state.is_cancelled():
            mission_state = _mark_blocked_mission(
                mission_state=mission_state,
                blocker=_agent_loop._TURN_CANCELLED_REPLY,
                updated_at=utc_now_iso(),
            )
            final_reply = _agent_loop._TURN_CANCELLED_REPLY
            break

        if current_response is None:
            detail = (
                f"executing {active_deliverable_id}"
                if active_deliverable_id is not None
                else "executing"
            )
            emit_mission_event(mission_event_callback, scope="mission", detail=detail)
            execution_announced = True
            current_response = _call_execution_turn(
                orchestrator=orchestrator,
                tracer=tracer,
                session_id=session_id,
                user_input=user_input,
                capability_summary=capability_summary,
                system_context_notes=system_context_notes,
                context_messages=context_messages,
                mission_state=mission_state,
                model_profile_name=model_profile_name,
                thinking_enabled=thinking_enabled,
                content_callback=content_callback,
                tool_definitions=tool_definitions,
            )

        latest_tool_results: tuple[ToolResult, ...] = ()
        draft_reply = current_response.response.content.strip()
        if not execution_announced:
            detail = (
                f"executing {active_deliverable_id}"
                if active_deliverable_id is not None
                else "executing"
            )
            emit_mission_event(mission_event_callback, scope="mission", detail=detail)
            execution_announced = True

        if current_response.response.tool_calls:
            stop_reply = _agent_loop._preflight_runtime_tool_batch(
                tool_calls=current_response.response.tool_calls,
                tool_guard_state=tool_guard_state,
            )
            if stop_reply is not None:
                mission_state = _mark_blocked_mission(
                    mission_state=mission_state,
                    blocker=stop_reply,
                    updated_at=utc_now_iso(),
                )
                final_reply = stop_reply
                break

            context_messages.append(
                LLMMessage(
                    role=LLMRole.ASSISTANT,
                    content=current_response.response.content,
                    tool_calls_payload=_agent_loop._extract_raw_tool_calls(
                        current_response.response
                    ),
                )
            )
            latest_tool_results = _agent_loop._execute_runtime_tool_calls(
                session_manager=session_manager,
                session_id=session_id,
                tracer=tracer,
                tool_registry=tool_registry,
                tool_calls=_annotate_tool_calls_for_mission(
                    tool_calls=current_response.response.tool_calls,
                    mission_state=mission_state,
                ),
                tool_guard_state=tool_guard_state,
                tool_call_callback=tool_call_callback,
            )
            verification_tool_results = _run_artifact_verification(
                mission_state=mission_state,
                latest_tool_results=latest_tool_results,
                session_manager=session_manager,
                session_id=session_id,
                tracer=tracer,
                tool_registry=tool_registry,
                tool_guard_state=tool_guard_state,
                tool_call_callback=tool_call_callback,
            )
            latest_tool_results = latest_tool_results + verification_tool_results
            accumulated_tool_results.extend(latest_tool_results)
            _agent_loop._append_tool_result_messages(
                context_messages=context_messages,
                tool_results=latest_tool_results,
            )
            draft_reply = ""

        if active_deliverable_id is not None:
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail=f"verifying {active_deliverable_id}",
            )
        verification_decision = _verify_mission(
            orchestrator=orchestrator,
            tracer=tracer,
            session_id=session_id,
            user_input=user_input,
            mission_state=mission_state,
            draft_reply=draft_reply or "",
            latest_tool_results=latest_tool_results,
            turn_tool_results=tuple(accumulated_tool_results),
            model_profile_name=model_profile_name,
            thinking_enabled=thinking_enabled,
        )
        previous_deliverable = mission_state.get_deliverable(active_deliverable_id)
        mission_state = _apply_verification_decision(
            mission_state=mission_state,
            verification_decision=verification_decision,
            latest_tool_results=latest_tool_results,
            updated_at=utc_now_iso(),
        )
        session_manager.persist_mission_state(mission_state, session_id=session_id)
        updated_deliverable = mission_state.get_deliverable(active_deliverable_id)

        if (
            active_deliverable_id is not None
            and previous_deliverable is not None
            and previous_deliverable.status != "completed"
            and updated_deliverable is not None
            and updated_deliverable.status == "completed"
        ):
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail=f"completed {active_deliverable_id}",
            )
        elif (
            active_deliverable_id is not None
            and verification_decision.next_action == "continue"
            and (verification_decision.repair_strategy or verification_decision.notes_for_next_step)
        ):
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail=f"repair {active_deliverable_id}",
            )

        if mission_state.status == "completed":
            if verification_decision.assistant_reply:
                emit_mission_event(
                    mission_event_callback,
                    scope="mission",
                    detail="finalizing",
                )
                final_reply = verification_decision.assistant_reply
                break
            if draft_reply:
                emit_mission_event(
                    mission_event_callback,
                    scope="mission",
                    detail="finalizing",
                )
                final_reply = draft_reply
                break
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail="finalizing",
            )
            context_messages = _prepare_next_execution_messages(
                context_messages=context_messages,
                mission_state=mission_state,
                verification_decision=verification_decision,
                user_input=user_input,
                draft_reply=draft_reply,
                latest_tool_results=latest_tool_results,
                tool_definitions=tool_definitions,
            )
            current_response = None
            continue

        if verification_decision.next_action == "blocked_reply" or mission_state.status == "blocked":
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail="finalizing",
            )
            final_reply = (
                verification_decision.assistant_reply
                or draft_reply
                or _build_mission_status_reply(mission_state)
                or _MISSION_STATUS_BLOCKED_REPLY
            )
            break

        if step_index + 1 >= max_steps:
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail="finalizing",
            )
            final_reply = _MISSION_STATUS_INCOMPLETE_REPLY
            break

        context_messages = _prepare_next_execution_messages(
            context_messages=context_messages,
            mission_state=mission_state,
            verification_decision=verification_decision,
            user_input=user_input,
            draft_reply=draft_reply,
            latest_tool_results=latest_tool_results,
            tool_definitions=tool_definitions,
        )
        current_response = None
        execution_announced = False

    if final_reply is None:
        final_reply = _MISSION_STATUS_INCOMPLETE_REPLY

    return AgentKernelRunResult(
        assistant_reply=final_reply,
        mission_state=mission_state,
        tool_results=tuple(accumulated_tool_results),
        persisted=True,
    )


def _plan_mission(
    *,
    session_id: str,
    user_input: str,
    orchestrator: Orchestrator,
    tracer: Tracer,
    model_profile_name: str,
    thinking_enabled: bool,
    tool_definitions: Sequence[ToolDefinition],
    first_response: OrchestratorTurnResult | None,
    existing_mission_state: MissionState | None,
    compatibility_mission_state: MissionState | None,
) -> MissionPlanDecision | None:
    planning_messages = build_mission_plan_messages(
        user_input=user_input,
        existing_mission_state=existing_mission_state,
        compatibility_mission_state=compatibility_mission_state,
        first_response=first_response,
        available_tool_names=tuple(
            tool_definition.name for tool_definition in tool_definitions
        ),
    )
    return _call_json_decision_with_repair(
        orchestrator=orchestrator,
        tracer=tracer,
        session_id=session_id,
        model_profile_name=model_profile_name,
        thinking_enabled=thinking_enabled,
        messages=planning_messages,
        parse_response=parse_mission_plan_response,
        repair_note=(
            "Mission planner repair: return valid JSON only using the requested "
            "mission planner shape, keep deliverables compact and concrete, and "
            "do not replace deliverables with tool names."
        ),
        parse_kwargs={"updated_at": utc_now_iso()},
        max_attempts=_PLANNER_PARSE_REPAIR_ATTEMPTS,
    )


def _verify_mission(
    *,
    orchestrator: Orchestrator,
    tracer: Tracer,
    session_id: str,
    user_input: str,
    mission_state: MissionState,
    draft_reply: str,
    latest_tool_results: Sequence[ToolResult],
    turn_tool_results: Sequence[ToolResult],
    model_profile_name: str,
    thinking_enabled: bool,
) -> MissionVerificationDecision:
    active_deliverable = mission_state.get_deliverable()
    retry_budget_remaining = max(
        0,
        _MISSION_RETRY_BUDGET_PER_DELIVERABLE
        - (active_deliverable.retry_count if active_deliverable is not None else 0),
    )
    repair_budget_remaining = max(
        0,
        _MISSION_REPAIR_BUDGET_PER_DELIVERABLE
        - (active_deliverable.repair_count if active_deliverable is not None else 0),
    )
    verification_messages = build_mission_verification_messages(
        user_input=user_input,
        mission_state=mission_state,
        draft_reply=draft_reply,
        latest_tool_results=latest_tool_results,
        turn_tool_results=turn_tool_results,
        blocker_metadata=None,
        retry_budget_remaining=retry_budget_remaining,
        repair_budget_remaining=repair_budget_remaining,
    )
    decision = _call_json_decision_with_repair(
        orchestrator=orchestrator,
        tracer=tracer,
        session_id=session_id,
        model_profile_name=model_profile_name,
        thinking_enabled=thinking_enabled,
        messages=verification_messages,
        parse_response=parse_mission_verification_response,
        repair_note=(
            "Mission verifier repair: return valid JSON only using the mission "
            "verifier shape. Do not mark the mission completed while any "
            "deliverable still needs verification."
        ),
        parse_kwargs={},
        max_attempts=_VERIFIER_PARSE_REPAIR_ATTEMPTS,
    )
    if decision is not None:
        return decision
    return _build_safe_verification_fallback(
        mission_state=mission_state,
        draft_reply=draft_reply,
        latest_tool_results=latest_tool_results,
    )


def _call_json_decision_with_repair(
    *,
    orchestrator: Orchestrator,
    tracer: Tracer,
    session_id: str,
    model_profile_name: str,
    thinking_enabled: bool,
    messages: Sequence[LLMMessage],
    parse_response: Callable[..., object | None],
    repair_note: str,
    parse_kwargs: dict[str, object],
    max_attempts: int,
) -> object | None:
    attempt_messages = list(messages)
    for attempt in range(max_attempts):
        try:
            turn_result = orchestrator.call_model(
                session_id=session_id,
                messages=attempt_messages,
                model_profile_name=model_profile_name,
                thinking_enabled=thinking_enabled,
                content_callback=None,
                tools=(),
            )
        except ModelCallFailedError:
            return None
        _trace_model_success(tracer=tracer, session_id=session_id, turn_result=turn_result)
        parsed = parse_response(turn_result.response.content, **parse_kwargs)
        if parsed is not None:
            return parsed
        if attempt + 1 >= max_attempts:
            return None
        attempt_messages.extend(
            (
                LLMMessage(
                    role=LLMRole.ASSISTANT,
                    content=turn_result.response.content,
                ),
                LLMMessage(role=LLMRole.SYSTEM, content=repair_note),
            )
        )
    return None


def _resolve_mission_state(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
    existing_mission_state: MissionState | None,
    compatibility_mission_state: MissionState | None,
    plan_decision: MissionPlanDecision,
) -> MissionState:
    updated_at = utc_now_iso()
    continued_mission = existing_mission_state
    if continued_mission is None and compatibility_mission_state is not None:
        continued_mission = compatibility_mission_state

    if (
        plan_decision.mission_action == "continue_existing"
        and existing_mission_state is not None
    ):
        return synchronize_mission_working_memory(
            replace(
                existing_mission_state,
                goal=plan_decision.mission_goal or existing_mission_state.goal,
                planner_summary=plan_decision.summary,
                updated_at=updated_at,
            ),
            updated_at=updated_at,
        )
    deliverables = plan_decision.deliverables
    mission_goal = plan_decision.mission_goal or user_input
    if (
        plan_decision.mission_action == "continue_existing"
        and continued_mission is not None
        and existing_mission_state is None
    ):
        mission_goal = plan_decision.mission_goal or continued_mission.goal
    return build_mission_state_from_plan(
        mission_id=session_manager.create_mission_id(),
        mission_goal=mission_goal,
        deliverables=deliverables,
        active_deliverable_id=(
            plan_decision.active_deliverable_id
            or (deliverables[0].deliverable_id if deliverables else None)
        ),
        planner_summary=plan_decision.summary,
        updated_at=updated_at,
    )


def _call_execution_turn(
    *,
    orchestrator: Orchestrator,
    tracer: Tracer,
    session_id: str,
    user_input: str,
    capability_summary: RuntimeCapabilitySummary,
    system_context_notes: tuple[str, ...],
    context_messages: list[LLMMessage],
    mission_state: MissionState,
    model_profile_name: str,
    thinking_enabled: bool,
    content_callback: LLMContentCallback | None,
    tool_definitions: Sequence[ToolDefinition],
) -> OrchestratorTurnResult:
    if context_messages:
        turn_result = orchestrator.call_model(
            session_id=session_id,
            messages=context_messages,
            model_profile_name=model_profile_name,
            thinking_enabled=thinking_enabled,
            content_callback=content_callback,
            tools=tool_definitions,
        )
        _trace_model_success(tracer=tracer, session_id=session_id, turn_result=turn_result)
        return turn_result

    turn_result = orchestrator.run_turn(
        session_id=session_id,
        user_message=user_input,
        model_profile_name=model_profile_name,
        capability_summary=capability_summary,
        system_context_notes=system_context_notes + (
            build_mission_initialization_note(mission_state=mission_state),
        ),
        thinking_enabled=thinking_enabled,
        content_callback=content_callback,
        tools=tool_definitions,
    )
    _trace_model_success(tracer=tracer, session_id=session_id, turn_result=turn_result)
    return turn_result


def _prepare_next_execution_messages(
    *,
    context_messages: list[LLMMessage],
    mission_state: MissionState,
    verification_decision: MissionVerificationDecision,
    user_input: str,
    draft_reply: str,
    latest_tool_results: Sequence[ToolResult],
    tool_definitions: Sequence[ToolDefinition],
) -> list[LLMMessage]:
    next_messages = list(context_messages)
    if draft_reply:
        next_messages.append(LLMMessage(role=LLMRole.ASSISTANT, content=draft_reply))
    if latest_tool_results:
        next_messages.append(
            LLMMessage(
                role=LLMRole.SYSTEM,
                content=_runtime_support._build_post_tool_grounding_note(
                    tool_results=latest_tool_results,
                    tool_definitions=tool_definitions,
                ),
            )
        )
    next_messages.append(
        LLMMessage(
            role=LLMRole.SYSTEM,
            content=build_mission_progress_note(
                mission_state=mission_state,
                verification_decision=verification_decision,
                user_input=user_input,
                latest_tool_results=latest_tool_results,
            ),
        )
    )
    return next_messages


def _apply_verification_decision(
    *,
    mission_state: MissionState,
    verification_decision: MissionVerificationDecision,
    latest_tool_results: Sequence[ToolResult],
    updated_at: str,
) -> MissionState:
    current_deliverable = mission_state.get_deliverable()
    next_deliverables = list(mission_state.deliverables)
    retry_history = list(mission_state.retry_history)
    repair_history = list(mission_state.repair_history)
    failed_tool_results = tuple(
        tool_result for tool_result in latest_tool_results if tool_result.success is False
    )
    retry_requested = (
        bool(failed_tool_results) and verification_decision.next_action == "continue"
    )
    if retry_requested:
        retry_note = (
            verification_decision.repair_strategy
            or verification_decision.notes_for_next_step
        )
        if retry_note:
            retry_history.append(retry_note)

    if current_deliverable is not None:
        retry_count = current_deliverable.retry_count
        repair_count = current_deliverable.repair_count
        if retry_requested:
            retry_count += 1
        elif (
            verification_decision.next_action == "continue"
            and verification_decision.current_deliverable_status != "completed"
        ):
            repair_count += 1
            if verification_decision.notes_for_next_step:
                repair_history.append(verification_decision.notes_for_next_step)
        updated_deliverable = replace(
            current_deliverable,
            status=verification_decision.current_deliverable_status,
            missing=verification_decision.missing,
            blocker=verification_decision.blocker,
            attempt_count=current_deliverable.attempt_count + 1,
            retry_count=retry_count,
            repair_count=repair_count,
            artifact_paths=(
                verification_decision.artifact_paths
                or current_deliverable.artifact_paths
            ),
            evidence=verification_decision.evidence or current_deliverable.evidence,
            updated_at=updated_at,
        )
        next_deliverables = [
            updated_deliverable
            if deliverable.deliverable_id == updated_deliverable.deliverable_id
            else deliverable
            for deliverable in next_deliverables
        ]

    active_deliverable_id = verification_decision.active_deliverable_id
    if active_deliverable_id is None and verification_decision.mission_status != "completed":
        active_deliverable_id = next(
            (
                deliverable.deliverable_id
                for deliverable in next_deliverables
                if deliverable.status in {"pending", "active"}
            ),
            None,
        )
    normalized_deliverables = [
        replace(deliverable, status="active", updated_at=updated_at)
        if deliverable.deliverable_id == active_deliverable_id
        and deliverable.status == "pending"
        else deliverable
        for deliverable in next_deliverables
    ]
    artifact_facts = tuple(
        f"artifact={path}" for path in verification_decision.artifact_paths
    )
    blocker_items = (
        (verification_decision.blocker,)
        if verification_decision.blocker is not None
        else ()
    )
    repair_items = ()
    if verification_decision.mission_status != "completed":
        if verification_decision.repair_strategy is not None:
            repair_items = (verification_decision.repair_strategy,)
        elif (
            verification_decision.notes_for_next_step
            and verification_decision.next_action == "continue"
        ):
            repair_items = (verification_decision.notes_for_next_step,)

    updated_state = replace(
        mission_state,
        status=verification_decision.mission_status,
        active_deliverable_id=active_deliverable_id,
        deliverables=tuple(normalized_deliverables),
        retry_history=tuple(retry_history[-6:]),
        repair_history=tuple(repair_history[-6:]),
        last_verified_artifact_paths=(
            verification_decision.artifact_paths
            or mission_state.last_verified_artifact_paths
        ),
        last_successful_evidence=(
            verification_decision.evidence or mission_state.last_successful_evidence
        ),
        last_blocker=verification_decision.blocker or mission_state.last_blocker,
        updated_at=updated_at,
    )
    synchronized = synchronize_mission_working_memory(
        updated_state,
        updated_at=updated_at,
        observed_facts=verification_decision.evidence,
        artifact_facts=artifact_facts,
        blockers=blocker_items,
        pending_repairs=repair_items,
    )
    if verification_decision.final_deliverables_missing:
        synchronized = replace(
            synchronized,
            final_deliverables_missing=verification_decision.final_deliverables_missing,
            status=(
                "completed"
                if not verification_decision.final_deliverables_missing
                else synchronized.status
            ),
        )
        if verification_decision.final_deliverables_missing:
            synchronized = replace(synchronized, status="active")
    return synchronized


def _build_safe_verification_fallback(
    *,
    mission_state: MissionState,
    draft_reply: str,
    latest_tool_results: Sequence[ToolResult],
) -> MissionVerificationDecision:
    active_deliverable = mission_state.get_deliverable()
    active_deliverable_id = mission_state.active_deliverable_id
    if latest_tool_results and any(tool_result.success is False for tool_result in latest_tool_results):
        timed_out = any(_tool_result_timed_out(tool_result) for tool_result in latest_tool_results)
        if timed_out and active_deliverable is not None and active_deliverable.retry_count < _MISSION_RETRY_BUDGET_PER_DELIVERABLE:
            return MissionVerificationDecision(
                mission_status="active",
                active_deliverable_id=active_deliverable_id,
                current_deliverable_status="active",
                missing="A retry is still needed after the timeout.",
                blocker=None,
                artifact_paths=(),
                evidence=(),
                final_deliverables_missing=mission_state.final_deliverables_missing,
                next_action="continue",
                repair_strategy="retry after timeout",
                notes_for_next_step="Retry the active deliverable with a narrower step.",
                assistant_reply=None,
            )
        blocker = latest_tool_results[-1].error or latest_tool_results[-1].output_text
        return MissionVerificationDecision(
            mission_status="blocked",
            active_deliverable_id=active_deliverable_id,
            current_deliverable_status="blocked",
            missing=None,
            blocker=blocker,
            artifact_paths=(),
            evidence=(),
            final_deliverables_missing=mission_state.final_deliverables_missing,
            next_action="blocked_reply",
            repair_strategy=None,
            notes_for_next_step=None,
            assistant_reply=None,
        )

    if mission_state.status == "completed" and draft_reply:
        return MissionVerificationDecision(
            mission_status="completed",
            active_deliverable_id=None,
            current_deliverable_status="completed",
            missing=None,
            blocker=mission_state.last_blocker,
            artifact_paths=mission_state.last_verified_artifact_paths,
            evidence=mission_state.last_successful_evidence,
            final_deliverables_missing=(),
            next_action="final_reply",
            repair_strategy=None,
            notes_for_next_step=None,
            assistant_reply=draft_reply,
        )

    return MissionVerificationDecision(
        mission_status="active",
        active_deliverable_id=active_deliverable_id,
        current_deliverable_status=(
            active_deliverable.status if active_deliverable is not None else "active"
        ),
        missing=(
            active_deliverable.missing
            if active_deliverable is not None
            else "Another verified step is required."
        ),
        blocker=None,
        artifact_paths=mission_state.last_verified_artifact_paths,
        evidence=mission_state.last_successful_evidence,
        final_deliverables_missing=mission_state.final_deliverables_missing,
        next_action="continue",
        repair_strategy="continue mission",
        notes_for_next_step="Continue the mission and verify the remaining deliverables.",
        assistant_reply=None,
    )


def _mark_blocked_mission(
    *,
    mission_state: MissionState,
    blocker: str,
    updated_at: str,
) -> MissionState:
    current_deliverable = mission_state.get_deliverable()
    next_deliverables = list(mission_state.deliverables)
    if current_deliverable is not None:
        blocked_deliverable = replace(
            current_deliverable,
            status="blocked",
            blocker=blocker,
            updated_at=updated_at,
        )
        next_deliverables = [
            blocked_deliverable
            if deliverable.deliverable_id == blocked_deliverable.deliverable_id
            else deliverable
            for deliverable in next_deliverables
        ]
    blocked_state = replace(
        mission_state,
        status="blocked",
        deliverables=tuple(next_deliverables),
        last_blocker=blocker,
        updated_at=updated_at,
    )
    return synchronize_mission_working_memory(
        blocked_state,
        updated_at=updated_at,
        blockers=(blocker,),
    )


def _response_has_local_write_tool_call(
    *,
    response: OrchestratorTurnResult,
    tool_definitions: Sequence[ToolDefinition],
) -> bool:
    permissions = {
        tool_definition.name: tool_definition.permission_level.value
        for tool_definition in tool_definitions
    }
    return any(
        permissions.get(tool_call.tool_name) == "local_write"
        and tool_call.tool_name != "delete_file"
        for tool_call in response.response.tool_calls or ()
    )


def _continue_plan_from_mission(mission_state: MissionState) -> MissionPlanDecision:
    return MissionPlanDecision(
        mission_action="continue_existing",
        mission_goal=mission_state.goal,
        deliverables=mission_state.deliverables,
        execution_queue=mission_state.execution_queue,
        active_deliverable_id=resolve_active_deliverable_id(mission_state=mission_state),
        summary=mission_state.planner_summary or "continue existing mission",
    )


def _annotate_tool_calls_for_mission(
    *,
    tool_calls: Sequence[ToolCall],
    mission_state: MissionState,
) -> tuple[ToolCall, ...]:
    mission_annotated_tool_names = frozenset(
        {"write_text_file", "delete_file", "read_text_file"}
    )
    annotated_calls: list[ToolCall] = []
    for tool_call in tool_calls:
        if tool_call.tool_name not in mission_annotated_tool_names:
            annotated_calls.append(tool_call)
            continue
        annotated_arguments = dict(tool_call.arguments)
        annotated_arguments.setdefault("mission_id", mission_state.mission_id)
        annotated_arguments.setdefault(
            "mission_deliverable_id",
            mission_state.active_deliverable_id,
        )
        annotated_calls.append(
            ToolCall(tool_name=tool_call.tool_name, arguments=annotated_arguments)
        )
    return tuple(annotated_calls)


def _run_artifact_verification(
    *,
    mission_state: MissionState,
    latest_tool_results: Sequence[ToolResult],
    session_manager: SessionManager,
    session_id: str,
    tracer: Tracer,
    tool_registry: ToolRegistry,
    tool_guard_state: _agent_loop._RuntimeToolGuardState,
    tool_call_callback: Callable[[ToolCall], None] | None,
) -> tuple[ToolResult, ...]:
    if any(tool_result.tool_name == "read_text_file" for tool_result in latest_tool_results):
        return ()
    verification_calls: list[ToolCall] = []
    for tool_result in latest_tool_results:
        if tool_result.success is not True or tool_result.tool_name != "write_text_file":
            continue
        payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
        resolved_path = payload.get("resolved_path")
        if not isinstance(resolved_path, str):
            continue
        verification_calls.append(
            ToolCall(
                tool_name="read_text_file",
                arguments={
                    "path": resolved_path,
                    "max_chars": 4000,
                    "mission_id": mission_state.mission_id,
                    "mission_deliverable_id": mission_state.active_deliverable_id,
                },
            )
        )
    if not verification_calls:
        return ()
    return _agent_loop._execute_runtime_tool_calls(
        session_manager=session_manager,
        session_id=session_id,
        tracer=tracer,
        tool_registry=tool_registry,
        tool_calls=tuple(verification_calls),
        tool_guard_state=tool_guard_state,
        tool_call_callback=tool_call_callback,
    )


def _build_mission_status_reply(mission_state: MissionState) -> str | None:
    if mission_state.status == "completed":
        return f"The mission is completed: {mission_state.goal}"
    if mission_state.status == "blocked":
        blocker = mission_state.last_blocker or "no blocker detail was persisted"
        active_task = mission_state.active_task or "the active mission step"
        return f"The mission is blocked on {active_task}: {blocker}"
    if mission_state.active_task is not None:
        return f"The mission is active on {mission_state.active_task}."
    return None


def _trace_model_success(
    *,
    tracer: Tracer,
    session_id: str,
    turn_result: OrchestratorTurnResult,
) -> None:
    tracer.trace_model_succeeded(
        session_id=session_id,
        provider=turn_result.response.provider,
        model_name=turn_result.response.model_name,
        finish_reason=turn_result.response.finish_reason,
        output_length=len(turn_result.response.content),
        model_duration_ms=turn_result.model_duration_ms,
        reasoning=turn_result.response.reasoning,
    )
