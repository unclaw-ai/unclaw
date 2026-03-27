"""True mission kernel for compact local-agent execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import unclaw.core.agent_loop as _agent_loop
import unclaw.core.runtime_support as _runtime_support
from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.core.execution_queue import resolve_active_deliverable_id
from unclaw.core.mission_events import MissionEventCallback, emit_mission_event
from unclaw.core.mission_state import (
    MissionDeliverableState,
    MissionState,
    mission_completion_ready,
)
from unclaw.core.mission_verifier import (
    MissionRelationDecision,
    MissionPlanDecision,
    MissionVerificationDecision,
    build_mission_initialization_note,
    build_mission_relation_messages,
    build_mission_plan_messages,
    build_mission_progress_note,
    build_mission_verification_messages,
    parse_mission_relation_response,
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

_RELATION_PARSE_REPAIR_ATTEMPTS = 2
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
    if mission_state.status != "active":
        return False
    if mission_state.executor_state in {"blocked", "completed"}:
        return False
    return True


def classify_turn_mission_relation(
    *,
    session_id: str,
    user_input: str,
    orchestrator: Orchestrator,
    tracer: Tracer,
    model_profile_name: str,
    thinking_enabled: bool,
    existing_mission_state: MissionState | None,
    compatibility_mission_state: MissionState | None,
) -> MissionRelationDecision | None:
    """Classify how the new turn relates to prior mission context."""

    relation_messages = build_mission_relation_messages(
        user_input=user_input,
        existing_mission_state=existing_mission_state,
        compatibility_mission_state=compatibility_mission_state,
    )
    decision = _call_json_decision_with_repair(
        orchestrator=orchestrator,
        tracer=tracer,
        session_id=session_id,
        model_profile_name=model_profile_name,
        thinking_enabled=thinking_enabled,
        messages=relation_messages,
        parse_response=parse_mission_relation_response,
        repair_note=(
            "Mission relation repair: return valid JSON only using the mission "
            "relation shape and choose exactly one allowed relation."
        ),
        parse_kwargs={},
        max_attempts=_RELATION_PARSE_REPAIR_ATTEMPTS,
    )
    if isinstance(decision, MissionRelationDecision):
        return decision
    if should_resume_mission_in_kernel(existing_mission_state):
        return MissionRelationDecision(
            relation="same_active_mission",
            summary="fallback to the active mission relation",
        )
    if compatibility_mission_state is not None:
        return MissionRelationDecision(
            relation="new_mission",
            summary="fallback to isolated new mission handling",
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
    first_response: OrchestratorTurnResult | None,
    relation_decision: MissionRelationDecision | None,
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
        relation_decision=relation_decision,
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

    if plan_decision.mission_action == "direct_reply_only":
        return AgentKernelPlan(should_use_kernel=False, plan_decision=plan_decision)

    # Kernel-first: any non-direct_reply_only plan enters the kernel
    # immediately. The planner decides whether durable execution is needed;
    # the runtime does not second-guess with narrow heuristics.
    return AgentKernelPlan(should_use_kernel=True, plan_decision=plan_decision)


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
    relation_decision: MissionRelationDecision | None = None,
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
        relation_decision=relation_decision,
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
        relation_decision=relation_decision,
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

    for step_index in range(max_steps):
        mission_state = synchronize_mission_working_memory(
            mission_state,
            updated_at=utc_now_iso(),
        )
        session_manager.persist_mission_state(mission_state, session_id=session_id)
        active_deliverable = mission_state.get_deliverable()
        active_deliverable_id = (
            active_deliverable.deliverable_id if active_deliverable is not None else None
        )

        if tool_guard_state.is_cancelled():
            mission_state = _mark_blocked_mission(
                mission_state=mission_state,
                blocker=_agent_loop._TURN_CANCELLED_REPLY,
                updated_at=utc_now_iso(),
            )
            session_manager.persist_mission_state(mission_state, session_id=session_id)
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail="mission blocked",
            )
            final_reply = _agent_loop._TURN_CANCELLED_REPLY
            break

        if mission_completion_ready(mission_state):
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail="mission completed",
            )
            final_reply = (
                mission_state.final_verified_reply
                or _build_mission_status_reply(mission_state)
                or _MISSION_STATUS_INCOMPLETE_REPLY
            )
            break

        if active_deliverable is None:
            final_reply = (
                _build_mission_status_reply(mission_state)
                or _MISSION_STATUS_INCOMPLETE_REPLY
            )
            break

        if current_response is not None and mission_state.executor_state not in {
            "executing",
            "awaiting_tool_result",
            "awaiting_verification",
        }:
            mission_state = _transition_mission_execution_state(
                mission_state=mission_state,
                executor_state="executing",
                executor_reason="execute the active deliverable",
                waiting_for=f"model action for {active_deliverable.task}",
                advance_condition=active_deliverable.verification,
                deliverable_execution_state="executing",
                verifier_notes=None,
                updated_at=utc_now_iso(),
            )
            session_manager.persist_mission_state(mission_state, session_id=session_id)
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail=_render_mission_transition(
                    active_deliverable_id=active_deliverable_id,
                    executor_state=mission_state.executor_state,
                ),
            )

        if current_response is None:
            mission_state = _transition_mission_execution_state(
                mission_state=mission_state,
                executor_state=(
                    "repairing" if mission_state.pending_repairs else "executing"
                ),
                executor_reason=(
                    "repair the active deliverable"
                    if mission_state.pending_repairs
                    else "execute the active deliverable"
                ),
                waiting_for=f"model action for {active_deliverable.task}",
                advance_condition=active_deliverable.verification,
                deliverable_execution_state=(
                    "repairing" if mission_state.pending_repairs else "executing"
                ),
                verifier_notes=(
                    mission_state.verifier_outputs[-1]
                    if mission_state.verifier_outputs
                    else None
                ),
                updated_at=utc_now_iso(),
            )
            session_manager.persist_mission_state(mission_state, session_id=session_id)
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail=_render_mission_transition(
                    active_deliverable_id=active_deliverable_id,
                    executor_state=mission_state.executor_state,
                ),
            )
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

        if current_response.response.tool_calls:
            mission_state = _transition_mission_execution_state(
                mission_state=mission_state,
                executor_state="awaiting_tool_result",
                executor_reason="wait for tool results",
                waiting_for=f"tool results for {active_deliverable_id}",
                advance_condition=active_deliverable.verification,
                deliverable_execution_state="awaiting_tool_result",
                verifier_notes=None,
                updated_at=utc_now_iso(),
            )
            session_manager.persist_mission_state(mission_state, session_id=session_id)
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail=_render_mission_transition(
                    active_deliverable_id=active_deliverable_id,
                    executor_state=mission_state.executor_state,
                ),
            )
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

        mission_state = _transition_mission_execution_state(
            mission_state=mission_state,
            executor_state="awaiting_verification",
            executor_reason="wait for verification",
            waiting_for=(
                f"verification of tool evidence for {active_deliverable_id}"
                if latest_tool_results
                else f"verification of the reply for {active_deliverable_id}"
            ),
            advance_condition=active_deliverable.verification,
            deliverable_execution_state="awaiting_verification",
            verifier_notes=None,
            updated_at=utc_now_iso(),
        )
        session_manager.persist_mission_state(mission_state, session_id=session_id)

        if active_deliverable_id is not None:
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail=_render_mission_transition(
                    active_deliverable_id=active_deliverable_id,
                    executor_state=mission_state.executor_state,
                ),
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
            draft_reply=draft_reply or "",
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
                detail=f"{active_deliverable_id} completed",
            )
        if (
            active_deliverable_id is not None
            and mission_state.executor_state == "repairing"
            and mission_state.active_deliverable_id == active_deliverable_id
        ):
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail=f"repairing {active_deliverable_id}",
            )

        if mission_completion_ready(mission_state):
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail="mission completed",
            )
            final_reply = (
                mission_state.final_verified_reply
                or verification_decision.assistant_reply
                or _build_mission_status_reply(mission_state)
                or _MISSION_STATUS_INCOMPLETE_REPLY
            )
            break

        if (
            verification_decision.next_action == "blocked_reply"
            or mission_state.status == "blocked"
            or mission_state.executor_state == "blocked"
        ):
            emit_mission_event(
                mission_event_callback,
                scope="mission",
                detail="mission blocked",
            )
            final_reply = (
                verification_decision.assistant_reply
                or _build_mission_status_reply(mission_state)
                or _MISSION_STATUS_BLOCKED_REPLY
            )
            break

        if step_index + 1 >= max_steps:
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
    relation_decision: MissionRelationDecision | None,
    existing_mission_state: MissionState | None,
    compatibility_mission_state: MissionState | None,
) -> MissionPlanDecision | None:
    planning_messages = build_mission_plan_messages(
        user_input=user_input,
        mission_relation=relation_decision,
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
    relation_decision: MissionRelationDecision | None,
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
                last_turn_relation=(
                    relation_decision.relation
                    if relation_decision is not None
                    else existing_mission_state.last_turn_relation
                ),
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
        execution_queue=plan_decision.execution_queue,
        planner_summary=plan_decision.summary,
        last_turn_relation=(
            relation_decision.relation if relation_decision is not None else None
        ),
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
    draft_reply: str,
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

    final_verified_reply = mission_state.final_verified_reply
    active_deliverable_id = mission_state.active_deliverable_id
    executor_state = "ready"
    executor_reason = "continue the mission"
    waiting_for: str | None = None
    advance_condition: str | None = None
    verifier_output_items: tuple[str, ...] = ()
    repair_items: tuple[str, ...] = ()

    if current_deliverable is not None:
        retry_count = current_deliverable.retry_count
        repair_count = current_deliverable.repair_count
        enforced_missing_note: str | None = None
        if retry_requested:
            retry_count += 1
        elif (
            verification_decision.next_action == "continue"
            and verification_decision.current_deliverable_status != "completed"
        ):
            repair_count += 1
            if verification_decision.notes_for_next_step:
                repair_history.append(verification_decision.notes_for_next_step)
        deliverable_status = verification_decision.current_deliverable_status
        deliverable_execution_state = "ready"
        if deliverable_status == "completed":
            deliverable_execution_state = "completed"
        elif (
            verification_decision.mission_status == "blocked"
            or verification_decision.next_action == "blocked_reply"
        ):
            deliverable_execution_state = "blocked"
        elif retry_requested or verification_decision.repair_strategy:
            deliverable_execution_state = "repairing"
        elif verification_decision.notes_for_next_step:
            deliverable_execution_state = "repairing"

        if (
            deliverable_status == "completed"
            and current_deliverable.mode == "artifact"
            and not _has_verified_artifact_evidence(
                mission_state=mission_state,
                current_deliverable=current_deliverable,
                latest_tool_results=latest_tool_results,
                artifact_paths=verification_decision.artifact_paths,
                evidence=verification_decision.evidence,
            )
        ):
            deliverable_status = "active"
            deliverable_execution_state = "repairing"
            verifier_output_items = (
                "Read the artifact back and verify its contents before completion.",
            )
            repair_items = verifier_output_items

        if deliverable_status == "completed":
            satisfied_evidence_kinds = _collect_satisfied_evidence_kinds(
                current_deliverable=current_deliverable,
                verification_decision=verification_decision,
                latest_tool_results=latest_tool_results,
                draft_reply=draft_reply,
            )
            missing_required_evidence = _missing_required_evidence(
                current_deliverable=current_deliverable,
                satisfied_evidence_kinds=satisfied_evidence_kinds,
            )
            if missing_required_evidence:
                deliverable_status = "active"
                deliverable_execution_state = "repairing"
                enforced_missing_note = _build_required_evidence_repair_note(
                    missing_required_evidence=missing_required_evidence,
                    satisfied_evidence_kinds=satisfied_evidence_kinds,
                )
                repair_count += 1
                repair_history.append(enforced_missing_note)
                verifier_output_items = (enforced_missing_note,)
                repair_items = verifier_output_items

        if (
            deliverable_status == "completed"
            and current_deliverable.mode in {"reply", "mixed"}
        ):
            candidate_reply = verification_decision.assistant_reply or draft_reply.strip()
            if candidate_reply:
                final_verified_reply = candidate_reply

        updated_deliverable = replace(
            current_deliverable,
            status=deliverable_status,
            missing=(
                enforced_missing_note
                or verification_decision.missing
                if deliverable_status != "completed"
                else None
            ),
            blocker=(
                verification_decision.blocker
                if deliverable_execution_state == "blocked"
                else None
            ),
            attempt_count=current_deliverable.attempt_count + 1,
            retry_count=retry_count,
            repair_count=repair_count,
            artifact_paths=(
                verification_decision.artifact_paths
                or current_deliverable.artifact_paths
            ),
            evidence=verification_decision.evidence or current_deliverable.evidence,
            updated_at=updated_at,
            execution_state=deliverable_execution_state,
            waiting_for=(
                enforced_missing_note
                or verification_decision.notes_for_next_step
                or verification_decision.repair_strategy
                or (
                    "mission deliverable verified"
                    if deliverable_status == "completed"
                    else current_deliverable.waiting_for
                )
            ),
            advance_condition=(
                None if deliverable_status == "completed" else current_deliverable.verification
            ),
            verifier_notes=(
                enforced_missing_note
                or verification_decision.notes_for_next_step
                or verification_decision.repair_strategy
                or (verifier_output_items[0] if verifier_output_items else None)
            ),
        )
        next_deliverables = [
            updated_deliverable
            if deliverable.deliverable_id == updated_deliverable.deliverable_id
            else deliverable
            for deliverable in next_deliverables
        ]
        if updated_deliverable.status == "completed":
            active_deliverable_id = None
        else:
            active_deliverable_id = updated_deliverable.deliverable_id

    if (
        verification_decision.active_deliverable_id is not None
        and verification_decision.mission_status != "completed"
    ):
        active_deliverable_id = verification_decision.active_deliverable_id

    if verification_decision.mission_status != "blocked":
        active_deliverable_id = _next_unresolved_deliverable_id(
            deliverables=tuple(next_deliverables),
            execution_queue=mission_state.execution_queue,
            preferred_deliverable_id=active_deliverable_id,
        )
    else:
        active_deliverable_id = mission_state.active_deliverable_id

    normalized_deliverables = [
        _normalize_deliverable_activation(
            deliverable=deliverable,
            active_deliverable_id=active_deliverable_id,
            updated_at=updated_at,
        )
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
    if not repair_items and verification_decision.mission_status != "completed":
        if verification_decision.repair_strategy is not None:
            repair_items = (verification_decision.repair_strategy,)
        elif (
            verification_decision.notes_for_next_step
            and verification_decision.next_action == "continue"
            and verification_decision.current_deliverable_status != "completed"
        ):
            repair_items = (verification_decision.notes_for_next_step,)

    if not verifier_output_items:
        verifier_output_items = tuple(
            item
            for item in (
                verification_decision.notes_for_next_step,
                verification_decision.repair_strategy,
                verification_decision.missing,
            )
            if item
        )

    computed_missing = tuple(
        deliverable.deliverable_id
        for deliverable in normalized_deliverables
        if deliverable.status != "completed"
    )
    if verification_decision.next_action == "blocked_reply" or verification_decision.mission_status == "blocked":
        executor_state = "blocked"
        executor_reason = "mission blocked"
    elif not computed_missing and not repair_items:
        executor_state = "completed"
        executor_reason = "mission verified complete"
        active_deliverable_id = None
    elif repair_items:
        executor_state = "repairing"
        executor_reason = "repair the active deliverable before advancing"
        active_target = next(
            (
                deliverable
                for deliverable in normalized_deliverables
                if deliverable.deliverable_id == active_deliverable_id
            ),
            None,
        )
        waiting_for = repair_items[0]
        advance_condition = (
            active_target.verification if active_target is not None else None
        )
    else:
        executor_state = "ready"
        executor_reason = "advance to the next verified step"
        active_target = next(
            (
                deliverable
                for deliverable in normalized_deliverables
                if deliverable.deliverable_id == active_deliverable_id
            ),
            None,
        )
        if active_target is not None:
            waiting_for = f"execute {active_target.task}"
            advance_condition = active_target.verification

    updated_state = replace(
        mission_state,
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
        final_deliverables_missing=tuple(
            dict.fromkeys(computed_missing + verification_decision.final_deliverables_missing)
        ),
    )
    return synchronize_mission_working_memory(
        updated_state,
        updated_at=updated_at,
        observed_facts=verification_decision.evidence,
        artifact_facts=artifact_facts,
        blockers=blocker_items,
        pending_repairs=repair_items,
        executor_state=executor_state,
        executor_reason=executor_reason,
        waiting_for=waiting_for,
        advance_condition=advance_condition,
        verifier_outputs=verifier_output_items,
        final_verified_reply=final_verified_reply,
    )


def _build_safe_verification_fallback(
    *,
    mission_state: MissionState,
    draft_reply: str,
    latest_tool_results: Sequence[ToolResult],
) -> MissionVerificationDecision:
    active_deliverable = mission_state.get_deliverable()
    active_deliverable_id = mission_state.active_deliverable_id
    if latest_tool_results and any(
        tool_result.success is False for tool_result in latest_tool_results
    ):
        timed_out = any(
            _tool_result_timed_out(tool_result) for tool_result in latest_tool_results
        )
        timed_out_search_result = _find_timed_out_search_result(latest_tool_results)
        if (
            timed_out
            and active_deliverable is not None
            and timed_out_search_result is None
            and active_deliverable.retry_count < _MISSION_RETRY_BUDGET_PER_DELIVERABLE
        ):
            retry_note = (
                "Retry search_web once with a narrower step such as max_results=3."
                if timed_out_search_result is not None
                and active_deliverable.retry_count == 0
                else "Retry the active deliverable with a narrower step."
            )
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
                repair_strategy=retry_note,
                notes_for_next_step=retry_note,
                assistant_reply=None,
            )

        if (
            timed_out_search_result is not None
            and active_deliverable is not None
            and active_deliverable.retry_count == 0
        ):
            retry_note = "Retry search_web once with a narrower step such as max_results=3."
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
                repair_strategy=retry_note,
                notes_for_next_step=retry_note,
                assistant_reply=None,
            )

        if (
            timed_out_search_result is not None
            and active_deliverable is not None
            and active_deliverable.repair_count < _MISSION_REPAIR_BUDGET_PER_DELIVERABLE
        ):
            repair_note = _build_search_timeout_repair_note(timed_out_search_result)
            return MissionVerificationDecision(
                mission_status="active",
                active_deliverable_id=active_deliverable_id,
                current_deliverable_status="active",
                missing="A new bounded repair step is needed after repeated search timeout.",
                blocker=None,
                artifact_paths=(),
                evidence=(),
                final_deliverables_missing=mission_state.final_deliverables_missing,
                next_action="continue",
                repair_strategy=repair_note,
                notes_for_next_step=repair_note,
                assistant_reply=None,
            )

        # Structured recovery: collision_conflict is repairable via versioning
        if (
            active_deliverable is not None
            and active_deliverable.repair_count < _MISSION_REPAIR_BUDGET_PER_DELIVERABLE
        ):
            collision_result = _find_collision_conflict_result(latest_tool_results)
            if collision_result is not None:
                payload = (
                    collision_result.payload
                    if isinstance(collision_result.payload, dict)
                    else {}
                )
                suggested_path = payload.get("suggested_version_path")
                repair_note = "Retry write with collision_policy='version'"
                if isinstance(suggested_path, str) and suggested_path:
                    repair_note += f" or use suggested path: {suggested_path}"
                return MissionVerificationDecision(
                    mission_status="active",
                    active_deliverable_id=active_deliverable_id,
                    current_deliverable_status="active",
                    missing="File collision needs resolution via versioning.",
                    blocker=None,
                    artifact_paths=(),
                    evidence=(),
                    final_deliverables_missing=mission_state.final_deliverables_missing,
                    next_action="continue",
                    repair_strategy=repair_note,
                    notes_for_next_step=(
                        "The target file already exists. Retry with "
                        "collision_policy='version' to create a versioned copy."
                        + (
                            f" Suggested path: {suggested_path}"
                            if isinstance(suggested_path, str) and suggested_path
                            else ""
                        )
                    ),
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

    if mission_completion_ready(mission_state):
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
            assistant_reply=mission_state.final_verified_reply or draft_reply or None,
        )

    has_artifact_evidence = (
        active_deliverable is not None
        and _has_verified_artifact_evidence(
            mission_state=mission_state,
            current_deliverable=active_deliverable,
            latest_tool_results=latest_tool_results,
            artifact_paths=mission_state.last_verified_artifact_paths,
            evidence=mission_state.last_successful_evidence,
        )
    )
    has_reply_evidence = bool(draft_reply.strip())
    pending_after_active = _remaining_deliverables_after_active(mission_state=mission_state)

    if (
        active_deliverable is not None
        and active_deliverable.mode == "reply"
        and has_reply_evidence
    ):
        return MissionVerificationDecision(
            mission_status="completed" if not pending_after_active else "active",
            active_deliverable_id=(
                None if not pending_after_active else pending_after_active[0]
            ),
            current_deliverable_status="completed",
            missing=None,
            blocker=None,
            artifact_paths=mission_state.last_verified_artifact_paths,
            evidence=mission_state.last_successful_evidence,
            final_deliverables_missing=pending_after_active,
            next_action="final_reply" if not pending_after_active else "continue",
            repair_strategy=None,
            notes_for_next_step=None,
            assistant_reply=draft_reply.strip(),
        )

    if (
        active_deliverable is not None
        and active_deliverable.mode == "mixed"
        and has_reply_evidence
        and has_artifact_evidence
    ):
        return MissionVerificationDecision(
            mission_status="completed" if not pending_after_active else "active",
            active_deliverable_id=(
                None if not pending_after_active else pending_after_active[0]
            ),
            current_deliverable_status="completed",
            missing=None,
            blocker=None,
            artifact_paths=mission_state.last_verified_artifact_paths,
            evidence=mission_state.last_successful_evidence,
            final_deliverables_missing=pending_after_active,
            next_action="final_reply" if not pending_after_active else "continue",
            repair_strategy=None,
            notes_for_next_step=None,
            assistant_reply=draft_reply.strip(),
        )

    precise_step = _build_precise_verification_step(
        mission_state=mission_state,
        active_deliverable=active_deliverable,
        draft_reply=draft_reply,
        has_artifact_evidence=has_artifact_evidence,
    )
    if _verification_step_already_attempted(
        mission_state=mission_state,
        precise_step=precise_step,
    ):
        blocker = "No new verification evidence was produced after the requested step."
        return MissionVerificationDecision(
            mission_status="blocked",
            active_deliverable_id=active_deliverable_id,
            current_deliverable_status="blocked",
            missing=None,
            blocker=blocker,
            artifact_paths=mission_state.last_verified_artifact_paths,
            evidence=mission_state.last_successful_evidence,
            final_deliverables_missing=mission_state.final_deliverables_missing,
            next_action="blocked_reply",
            repair_strategy=None,
            notes_for_next_step=None,
            assistant_reply=None,
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
        repair_strategy=None,
        notes_for_next_step=precise_step,
        assistant_reply=None,
    )


def _find_collision_conflict_result(
    tool_results: Sequence[ToolResult],
) -> ToolResult | None:
    """Return the first collision_conflict failure from structured tool metadata."""
    for tool_result in tool_results:
        if (
            tool_result.success is False
            and tool_result.failure_kind == "collision_conflict"
        ):
            return tool_result
    return None


def _find_timed_out_search_result(
    tool_results: Sequence[ToolResult],
) -> ToolResult | None:
    for tool_result in tool_results:
        if tool_result.tool_name != "search_web":
            continue
        if tool_result.success is False and _tool_result_timed_out(tool_result):
            return tool_result
    return None


def _build_search_timeout_repair_note(tool_result: ToolResult) -> str:
    payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
    display_sources = payload.get("display_sources")
    if isinstance(display_sources, list) and display_sources:
        first_source = display_sources[0]
        if isinstance(first_source, dict) and isinstance(first_source.get("url"), str):
            return (
                "Use one bounded repair step: fetch_url_text on a returned source or "
                f"decompose the query into a narrower follow-up search. First source: {first_source['url']}"
            )
    query = payload.get("query")
    if isinstance(query, str) and query.strip():
        return (
            "Use one bounded repair step: decompose the query into a narrower "
            f"follow-up search based on {query!r}, or fetch a grounded source directly."
        )
    return (
        "Use one bounded repair step: decompose the search into a narrower query "
        "or fetch and read one already grounded source directly."
    )


def _collect_satisfied_evidence_kinds(
    *,
    current_deliverable: MissionDeliverableState,
    verification_decision: MissionVerificationDecision,
    latest_tool_results: Sequence[ToolResult],
    draft_reply: str,
) -> tuple[str, ...]:
    satisfied: list[str] = []

    def _add(kind: str) -> None:
        if kind not in satisfied:
            satisfied.append(kind)

    if current_deliverable.artifact_paths or verification_decision.artifact_paths:
        _add("artifact_write")

    for tool_result in latest_tool_results:
        if tool_result.success is not True:
            continue
        if tool_result.tool_name == "fast_web_search":
            _add("fast_grounding")
        elif tool_result.tool_name == "search_web":
            if not _runtime_support._search_web_result_is_thin(tool_result):
                _add("full_web_research")
        elif tool_result.tool_name == "fetch_url_text":
            _add("full_web_research")
        elif tool_result.tool_name == "write_text_file":
            _add("artifact_write")
        elif tool_result.tool_name == "read_text_file":
            _add("artifact_readback")
        elif tool_result.tool_name == "delete_file":
            _add("local_delete")
        elif tool_result.tool_name == "list_directory":
            _add("directory_listing")

    if verification_decision.assistant_reply or draft_reply.strip():
        _add("reply_emitted")
    if draft_reply.strip() and not latest_tool_results:
        _add("calculation_result")

    return tuple(satisfied)


def _missing_required_evidence(
    *,
    current_deliverable: MissionDeliverableState,
    satisfied_evidence_kinds: Sequence[str],
) -> tuple[str, ...]:
    satisfied = set(satisfied_evidence_kinds)
    return tuple(
        evidence_kind
        for evidence_kind in current_deliverable.required_evidence
        if evidence_kind not in satisfied
    )


def _build_required_evidence_repair_note(
    *,
    missing_required_evidence: Sequence[str],
    satisfied_evidence_kinds: Sequence[str],
) -> str:
    notes: list[str] = []
    satisfied = set(satisfied_evidence_kinds)
    for evidence_kind in missing_required_evidence:
        if evidence_kind == "full_web_research":
            if "fast_grounding" in satisfied:
                notes.append(
                    "fast_web_search only grounded the topic. Use search_web or grounded source reads for full web research."
                )
            else:
                notes.append(
                    "Use search_web or grounded source reads to complete the required web research."
                )
        elif evidence_kind == "fast_grounding":
            notes.append("Ground the topic first with fast_web_search before advancing.")
        elif evidence_kind == "artifact_write":
            notes.append("Write the requested artifact before advancing.")
        elif evidence_kind == "artifact_readback":
            notes.append("Read the written artifact back and verify it before advancing.")
        elif evidence_kind == "reply_emitted":
            notes.append("Emit the requested user-facing reply before advancing.")
        elif evidence_kind == "local_delete":
            notes.append("Delete the requested local file before advancing.")
        elif evidence_kind == "directory_listing":
            notes.append("List the target directory before advancing.")
        elif evidence_kind == "calculation_result":
            notes.append("Produce the requested calculation result before advancing.")
    if not notes:
        return "Produce the missing required evidence before advancing."
    return " ".join(notes[:2])


def _has_verified_artifact_evidence(
    *,
    mission_state: MissionState,
    current_deliverable: MissionDeliverableState,
    latest_tool_results: Sequence[ToolResult],
    artifact_paths: Sequence[str],
    evidence: Sequence[str],
) -> bool:
    if any(
        tool_result.tool_name == "read_text_file" and tool_result.success is True
        for tool_result in latest_tool_results
    ):
        return True
    if current_deliverable.artifact_paths and current_deliverable.evidence:
        return True
    if artifact_paths and evidence:
        return True
    return bool(
        mission_state.last_verified_artifact_paths
        and mission_state.last_successful_evidence
    )


def _next_unresolved_deliverable_id(
    *,
    deliverables: Sequence[MissionDeliverableState],
    execution_queue: Sequence[str],
    preferred_deliverable_id: str | None,
) -> str | None:
    for deliverable_id in execution_queue:
        deliverable = next(
            (
                item
                for item in deliverables
                if item.deliverable_id == deliverable_id
            ),
            None,
        )
        if deliverable is not None and deliverable.status in {"pending", "active"}:
            return deliverable.deliverable_id
    if preferred_deliverable_id is not None:
        preferred = next(
            (
                deliverable
                for deliverable in deliverables
                if deliverable.deliverable_id == preferred_deliverable_id
            ),
            None,
        )
        if preferred is not None and preferred.status in {"pending", "active"}:
            return preferred_deliverable_id
    next_unresolved = next(
        (
            deliverable.deliverable_id
            for deliverable in deliverables
            if deliverable.status in {"pending", "active"}
        ),
        None,
    )
    return next_unresolved


def _normalize_deliverable_activation(
    *,
    deliverable: MissionDeliverableState,
    active_deliverable_id: str | None,
    updated_at: str,
) -> MissionDeliverableState:
    if deliverable.deliverable_id == active_deliverable_id:
        if deliverable.status == "pending":
            return replace(
                deliverable,
                status="active",
                execution_state=(
                    "ready"
                    if deliverable.execution_state == "pending"
                    else deliverable.execution_state
                ),
                updated_at=updated_at,
            )
        return deliverable
    if deliverable.status == "active":
        return replace(
            deliverable,
            status="pending",
            execution_state=(
                "pending"
                if deliverable.execution_state not in {"completed", "blocked"}
                else deliverable.execution_state
            ),
            updated_at=updated_at,
        )
    return deliverable


def _transition_mission_execution_state(
    *,
    mission_state: MissionState,
    executor_state: str,
    executor_reason: str,
    waiting_for: str | None,
    advance_condition: str | None,
    deliverable_execution_state: str | None,
    verifier_notes: str | None,
    updated_at: str,
) -> MissionState:
    active_deliverable = mission_state.get_deliverable()
    deliverables = list(mission_state.deliverables)
    if active_deliverable is not None and deliverable_execution_state is not None:
        updated_deliverable = replace(
            active_deliverable,
            status=(
                "active"
                if active_deliverable.status not in {"completed", "blocked"}
                else active_deliverable.status
            ),
            execution_state=deliverable_execution_state,
            waiting_for=waiting_for,
            advance_condition=advance_condition,
            verifier_notes=verifier_notes,
            updated_at=updated_at,
        )
        deliverables = [
            updated_deliverable
            if deliverable.deliverable_id == updated_deliverable.deliverable_id
            else deliverable
            for deliverable in deliverables
        ]
    return synchronize_mission_working_memory(
        replace(
            mission_state,
            deliverables=tuple(deliverables),
            updated_at=updated_at,
        ),
        updated_at=updated_at,
        executor_state=executor_state,
        executor_reason=executor_reason,
        waiting_for=waiting_for,
        advance_condition=advance_condition,
    )


def _render_mission_transition(
    *,
    active_deliverable_id: str | None,
    executor_state: str,
) -> str:
    if executor_state in {"planning", "completed", "blocked"}:
        return (
            "planning"
            if executor_state == "planning"
            else f"mission {executor_state}"
        )
    if active_deliverable_id is None:
        return executor_state
    if executor_state == "repairing":
        return f"repairing {active_deliverable_id}"
    return f"{active_deliverable_id} {executor_state}"


def _remaining_deliverables_after_active(
    *,
    mission_state: MissionState,
) -> tuple[str, ...]:
    remaining: list[str] = []
    for deliverable_id in mission_state.execution_queue:
        if deliverable_id == mission_state.active_deliverable_id:
            continue
        deliverable = mission_state.get_deliverable(deliverable_id)
        if deliverable is not None and deliverable.status != "completed":
            remaining.append(deliverable.deliverable_id)
    for deliverable in mission_state.deliverables:
        if (
            deliverable.deliverable_id != mission_state.active_deliverable_id
            and deliverable.status != "completed"
            and deliverable.deliverable_id not in remaining
        ):
            remaining.append(deliverable.deliverable_id)
    return tuple(remaining)


def _build_precise_verification_step(
    *,
    mission_state: MissionState,
    active_deliverable: MissionDeliverableState | None,
    draft_reply: str,
    has_artifact_evidence: bool,
) -> str:
    if active_deliverable is None:
        return "Verify the next unresolved deliverable from persisted mission facts."
    if "full_web_research" in active_deliverable.required_evidence:
        return (
            "Complete one bounded full web research step with search_web or grounded "
            "source reads before advancing."
        )
    if "fast_grounding" in active_deliverable.required_evidence:
        return "Ground the topic first with fast_web_search before advancing."
    if active_deliverable.mode == "artifact":
        if not has_artifact_evidence:
            return (
                "Read the artifact back with read_text_file and confirm it matches "
                "the active deliverable before advancing."
            )
        return (
            "Verify that the persisted artifact evidence satisfies the active "
            "deliverable before advancing."
        )
    if active_deliverable.mode == "reply":
        if draft_reply.strip():
            return (
                "Verify whether the current draft reply satisfies the active "
                "deliverable and then mark it completed or request one concrete repair."
            )
        return "Produce the reply needed for the active deliverable."
    if not has_artifact_evidence:
        return (
            "Verify the artifact side first by reading it back before checking the reply."
        )
    if not draft_reply.strip():
        return "Produce the reply side of the mixed deliverable from verified facts."
    return (
        "Verify both the artifact evidence and the current draft reply for the "
        "active mixed deliverable."
    )


def _verification_step_already_attempted(
    *,
    mission_state: MissionState,
    precise_step: str,
) -> bool:
    active_deliverable = mission_state.get_deliverable()
    if active_deliverable is None:
        return False
    if precise_step in mission_state.verifier_outputs:
        return True
    if active_deliverable.verifier_notes == precise_step:
        return True
    return (
        active_deliverable.attempt_count > 0
        and active_deliverable.execution_state == "awaiting_verification"
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
            execution_state="blocked",
            waiting_for=None,
            verifier_notes=blocker,
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
        pending_repairs=(),
        executor_state="blocked",
        executor_reason="mission blocked",
        waiting_for=None,
        advance_condition=None,
        verifier_outputs=(blocker,),
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
        # Build per-deliverable status for honest reporting
        deliverable_lines: list[str] = []
        for deliverable in mission_state.deliverables:
            deliverable_lines.append(
                f"  - {deliverable.deliverable_id} ({deliverable.task}): {deliverable.status}"
            )
        status_detail = "\n".join(deliverable_lines) if deliverable_lines else ""
        base = f"The mission is active on {mission_state.active_task}."
        if status_detail:
            return f"{base}\nDeliverables:\n{status_detail}"
        return base
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
