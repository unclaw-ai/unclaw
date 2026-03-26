"""Durable mission execution loop for multi-step runtime work."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import unclaw.core.agent_loop as _agent_loop
import unclaw.core.runtime_support as _runtime_support
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
from unclaw.core.orchestrator import (
    ModelCallFailedError,
    Orchestrator,
    OrchestratorTurnResult,
)
from unclaw.core.reply_discipline import _tool_result_timed_out
from unclaw.core.session_manager import (
    SessionGoalState,
    SessionManager,
    SessionProgressEntry,
)
from unclaw.llm.base import LLMMessage, LLMRole, utc_now_iso
from unclaw.logs.tracer import Tracer
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolResult
from unclaw.tools.registry import ToolRegistry

_MISSION_RETRY_BUDGET_PER_DELIVERABLE = 2
_MISSION_REPAIR_BUDGET_PER_DELIVERABLE = 2
_MISSION_STEP_NOTE_MAX_RESULTS = 5


@dataclass(frozen=True, slots=True)
class MissionRunResult:
    """Resolved outcome for one runtime mission turn."""

    assistant_reply: str
    mission_state: MissionState | None
    tool_results: tuple[ToolResult, ...]
    persisted: bool


def run_mission_turn(
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
    max_steps: int,
    tool_guard_state: _agent_loop._RuntimeToolGuardState,
    tool_call_callback: Callable[[ToolCall], None] | None,
    first_response: OrchestratorTurnResult,
    initial_mission_state: MissionState | None = None,
) -> MissionRunResult:
    """Run a bounded mission execution loop for one turn."""

    existing_mission_state = (
        initial_mission_state
        if initial_mission_state is not None
        else session_manager.get_current_mission_state(session_id)
    )
    mission_state, persist_mission_state = _resolve_mission_state(
        session_manager=session_manager,
        session_id=session_id,
        user_input=user_input,
        orchestrator=orchestrator,
        tracer=tracer,
        first_response=first_response,
        existing_mission_state=existing_mission_state,
        tool_definitions=tool_definitions,
        model_profile_name=model_profile_name,
        thinking_enabled=thinking_enabled,
    )
    if mission_state is None:
        draft_reply = first_response.response.content.strip()
        return MissionRunResult(
            assistant_reply=draft_reply or _runtime_support.EMPTY_RESPONSE_REPLY,
            mission_state=None,
            tool_results=(),
            persisted=False,
        )

    if persist_mission_state:
        session_manager.persist_mission_state(mission_state, session_id=session_id)
    context_messages: list[LLMMessage] = list(first_response.context_messages)
    mission_note_appended = False
    current_response = first_response
    accumulated_tool_results: list[ToolResult] = []
    final_reply: str | None = None

    for step_index in range(max_steps):
        if tool_guard_state.is_cancelled():
            final_reply = _agent_loop._TURN_CANCELLED_REPLY
            mission_state = _mark_cancelled_mission(
                mission_state=mission_state,
                updated_at=utc_now_iso(),
            )
            break

        latest_tool_results: tuple[ToolResult, ...] = ()
        draft_reply = current_response.response.content.strip()
        if (
            not current_response.response.tool_calls
            and draft_reply
            and mission_state.status == "completed"
        ):
            final_reply = draft_reply
            break
        if current_response.response.tool_calls:
            stop_reply = _agent_loop._preflight_runtime_tool_batch(
                tool_calls=current_response.response.tool_calls,
                tool_guard_state=tool_guard_state,
            )
            if stop_reply is not None:
                final_reply = stop_reply
                mission_state = _block_mission_from_runtime_reply(
                    mission_state=mission_state,
                    blocker=stop_reply,
                    updated_at=utc_now_iso(),
                )
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
            accumulated_tool_results.extend(latest_tool_results)
            _agent_loop._append_tool_result_messages(
                context_messages=context_messages,
                tool_results=latest_tool_results,
            )
            draft_reply = ""

        updated_at = utc_now_iso()
        verification_decision = _verify_mission_checkpoint(
            orchestrator=orchestrator,
            session_id=session_id,
            model_profile_name=model_profile_name,
            thinking_enabled=thinking_enabled,
            tracer=tracer,
            user_input=user_input,
            mission_state=mission_state,
            draft_reply=draft_reply or "",
            latest_tool_results=latest_tool_results,
            turn_tool_results=tuple(accumulated_tool_results),
            updated_at=updated_at,
        )
        mission_state = _apply_verification_decision(
            mission_state=mission_state,
            verification_decision=verification_decision,
            latest_tool_results=latest_tool_results,
            updated_at=updated_at,
        )
        if (
            not persist_mission_state
            and _should_promote_to_persisted_mission(
                mission_state=mission_state,
                verification_decision=verification_decision,
                latest_tool_results=latest_tool_results,
            )
        ):
            persist_mission_state = True
        if persist_mission_state:
            session_manager.persist_mission_state(mission_state, session_id=session_id)

        if verification_decision.next_action in {"final_reply", "blocked_reply"}:
            final_reply = (
                verification_decision.assistant_reply
                or draft_reply
                or _build_mission_status_reply(mission_state)
            )
            break

        if step_index + 1 >= max_steps:
            final_reply = _build_incomplete_mission_reply(mission_state)
            break

        if not mission_note_appended:
            context_messages.append(
                LLMMessage(
                    role=LLMRole.SYSTEM,
                    content=build_mission_initialization_note(
                        mission_state=mission_state
                    ),
                )
            )
            mission_note_appended = True
        if draft_reply:
            context_messages.append(
                LLMMessage(role=LLMRole.ASSISTANT, content=draft_reply)
            )
        if latest_tool_results:
            context_messages.append(
                LLMMessage(
                    role=LLMRole.SYSTEM,
                    content=_runtime_support._build_post_tool_grounding_note(
                        tool_results=latest_tool_results,
                        tool_definitions=tool_definitions,
                    ),
                )
            )
        context_messages.append(
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
        current_response = _call_model_and_trace(
            orchestrator=orchestrator,
            tracer=tracer,
            session_id=session_id,
            model_profile_name=model_profile_name,
            thinking_enabled=thinking_enabled,
            messages=context_messages,
            tools=tool_definitions,
        )
        inline_tool_response, inline_tool_reply = (
            _agent_loop._recover_inline_native_tool_response(
                current_response.response,
                tool_definitions=tool_definitions,
                max_agent_steps=max_steps,
            )
        )
        if inline_tool_response is not None:
            current_response = replace(current_response, response=inline_tool_response)
        elif inline_tool_reply is not None:
            current_response = replace(
                current_response,
                response=replace(current_response.response, content=inline_tool_reply),
            )

    if final_reply is None:
        final_reply = _build_incomplete_mission_reply(mission_state)
    if mission_state is not None and persist_mission_state:
        session_manager.persist_mission_state(mission_state, session_id=session_id)

    return MissionRunResult(
        assistant_reply=final_reply,
        mission_state=mission_state if persist_mission_state else None,
        tool_results=tuple(accumulated_tool_results),
        persisted=persist_mission_state,
    )


def _resolve_mission_state(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
    orchestrator: Orchestrator,
    tracer: Tracer,
    first_response: OrchestratorTurnResult,
    existing_mission_state: MissionState | None,
    tool_definitions: Sequence[ToolDefinition],
    model_profile_name: str,
    thinking_enabled: bool,
) -> tuple[MissionState | None, bool]:
    updated_at = utc_now_iso()
    planning_messages = build_mission_plan_messages(
        user_input=user_input,
        existing_mission_state=existing_mission_state,
        first_response=first_response,
        available_tool_names=tuple(tool_definition.name for tool_definition in tool_definitions),
    )
    try:
        planning_result = _call_model_and_trace(
            orchestrator=orchestrator,
            tracer=tracer,
            session_id=session_id,
            model_profile_name=model_profile_name,
            thinking_enabled=thinking_enabled,
            messages=planning_messages,
            tools=(),
        )
        plan_decision = parse_mission_plan_response(
            planning_result.response.content,
            updated_at=updated_at,
        )
    except ModelCallFailedError:
        plan_decision = None

    if plan_decision is None:
        plan_decision = _build_fallback_plan_decision(
            user_input=user_input,
            existing_mission_state=existing_mission_state,
            updated_at=updated_at,
        )

    if plan_decision.mission_action == "direct_reply_only":
        return None, False

    if (
        plan_decision.mission_action == "continue_existing"
        and existing_mission_state is not None
    ):
        active_status = existing_mission_state.status
        next_active_deliverable_id = existing_mission_state.active_deliverable_id
        next_active_task = existing_mission_state.active_task
        next_deliverables = existing_mission_state.deliverables
        if first_response.response.tool_calls:
            first_tool_name = first_response.response.tool_calls[0].tool_name
            active_status = "active"
            next_active_deliverable_id = next_active_deliverable_id or (
                existing_mission_state.deliverables[0].deliverable_id
                if existing_mission_state.deliverables
                else None
            )
            next_active_task = first_tool_name
            if next_active_deliverable_id is not None:
                next_deliverables = tuple(
                    replace(
                        deliverable,
                        task=first_tool_name,
                        deliverable=first_tool_name,
                        status="active",
                        updated_at=updated_at,
                    )
                    if deliverable.deliverable_id == next_active_deliverable_id
                    else deliverable
                    for deliverable in existing_mission_state.deliverables
                )
        return (
            replace(
                existing_mission_state,
                status=active_status,
                active_deliverable_id=next_active_deliverable_id,
                active_task=next_active_task,
                deliverables=next_deliverables,
                updated_at=updated_at,
            ),
            True,
        )

    deliverables = plan_decision.deliverables or _build_fallback_plan_decision(
        user_input=user_input,
        existing_mission_state=None,
        updated_at=updated_at,
    ).deliverables
    active_deliverable_id = (
        plan_decision.active_deliverable_id or deliverables[0].deliverable_id
    )
    if len(deliverables) == 1 and first_response.response.tool_calls:
        first_tool_name = first_response.response.tool_calls[0].tool_name
        only_deliverable = deliverables[0]
        if only_deliverable.task == "Complete the requested mission":
            deliverables = (
                replace(
                    only_deliverable,
                    task=first_tool_name,
                    deliverable=first_tool_name,
                ),
            )
    active_deliverables = []
    for deliverable in deliverables:
        if deliverable.deliverable_id == active_deliverable_id:
            active_deliverables.append(replace(deliverable, status="active"))
        else:
            active_deliverables.append(deliverable)
    active_deliverable = next(
        (
            deliverable
            for deliverable in active_deliverables
            if deliverable.deliverable_id == active_deliverable_id
        ),
        None,
    )
    return (
        MissionState(
            mission_id=session_manager.create_mission_id(),
            goal=plan_decision.mission_goal or user_input,
            status="active",
            active_deliverable_id=active_deliverable_id,
            active_task=(
                active_deliverable.task if active_deliverable is not None else None
            ),
            completed_deliverables=(),
            blocked_deliverables=(),
            deliverables=tuple(active_deliverables),
            retry_history=(),
            repair_history=(),
            last_verified_artifact_paths=(),
            last_successful_evidence=(),
            last_blocker=None,
            updated_at=updated_at,
        ),
        (
            session_manager.get_current_mission_state(session_id) is not None
            or len(active_deliverables) > 1
            or _response_has_side_effect_tool_calls(
                response=first_response,
                tool_definitions=tool_definitions,
            )
        ),
    )


def _verify_mission_checkpoint(
    *,
    orchestrator: Orchestrator,
    session_id: str,
    model_profile_name: str,
    thinking_enabled: bool,
    tracer: Tracer,
    user_input: str,
    mission_state: MissionState,
    draft_reply: str,
    latest_tool_results: Sequence[ToolResult],
    turn_tool_results: Sequence[ToolResult],
    updated_at: str,
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
    try:
        verification_result = _call_model_and_trace(
            orchestrator=orchestrator,
            tracer=tracer,
            session_id=session_id,
            model_profile_name=model_profile_name,
            thinking_enabled=thinking_enabled,
            messages=verification_messages,
            tools=(),
        )
        verification_decision = parse_mission_verification_response(
            verification_result.response.content
        )
    except ModelCallFailedError:
        verification_decision = None

    if verification_decision is None:
        verification_decision = _build_fallback_verification_decision(
            mission_state=mission_state,
            draft_reply=draft_reply,
            latest_tool_results=latest_tool_results,
            turn_tool_results=turn_tool_results,
            updated_at=updated_at,
        )
    return verification_decision


def _apply_verification_decision(
    *,
    mission_state: MissionState,
    verification_decision: MissionVerificationDecision,
    latest_tool_results: Sequence[ToolResult],
    updated_at: str,
) -> MissionState:
    current_deliverable = mission_state.get_deliverable()
    next_deliverables: list[MissionDeliverableState] = list(mission_state.deliverables)
    retry_history = list(mission_state.retry_history)
    repair_history = list(mission_state.repair_history)

    if current_deliverable is not None:
        next_attempt_count = current_deliverable.attempt_count + 1
        next_retry_count = current_deliverable.retry_count
        next_repair_count = current_deliverable.repair_count
        if latest_tool_results and any(
            _tool_result_timed_out(tool_result) for tool_result in latest_tool_results
        ):
            next_retry_count += 1
            retry_history.append(
                verification_decision.notes_for_next_step
                or "mission retry requested after timeout"
            )
        elif (
            verification_decision.next_action == "continue"
            and verification_decision.current_deliverable_status != "completed"
        ):
            next_repair_count += 1
            if verification_decision.notes_for_next_step:
                repair_history.append(verification_decision.notes_for_next_step)

        updated_deliverable = replace(
            current_deliverable,
            status=verification_decision.current_deliverable_status,
            missing=verification_decision.missing,
            blocker=verification_decision.blocker,
            attempt_count=next_attempt_count,
            repair_count=next_repair_count,
            retry_count=next_retry_count,
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
    normalized_deliverables: list[MissionDeliverableState] = []
    for deliverable in next_deliverables:
        if deliverable.deliverable_id == active_deliverable_id:
            if deliverable.status == "pending":
                normalized_deliverables.append(
                    replace(deliverable, status="active", updated_at=updated_at)
                )
            else:
                normalized_deliverables.append(deliverable)
        else:
            normalized_deliverables.append(deliverable)

    active_deliverable = next(
        (
            deliverable
            for deliverable in normalized_deliverables
            if deliverable.deliverable_id == active_deliverable_id
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
    mission_status = verification_decision.mission_status
    if mission_status == "completed" and any(
        deliverable.status != "completed" for deliverable in normalized_deliverables
    ):
        mission_status = "active"
    if mission_status == "blocked" and not blocked_deliverables:
        mission_status = "active"
    if mission_status != "completed" and active_deliverable_id is None:
        mission_status = "blocked"

    return MissionState(
        mission_id=mission_state.mission_id,
        goal=mission_state.goal,
        status=mission_status,
        active_deliverable_id=active_deliverable_id,
        active_task=active_deliverable.task if active_deliverable is not None else None,
        completed_deliverables=completed_deliverables,
        blocked_deliverables=blocked_deliverables,
        deliverables=tuple(normalized_deliverables),
        retry_history=tuple(retry_history[-_MISSION_STEP_NOTE_MAX_RESULTS:]),
        repair_history=tuple(repair_history[-_MISSION_STEP_NOTE_MAX_RESULTS:]),
        last_verified_artifact_paths=verification_decision.artifact_paths
        or mission_state.last_verified_artifact_paths,
        last_successful_evidence=verification_decision.evidence
        or mission_state.last_successful_evidence,
        last_blocker=verification_decision.blocker or mission_state.last_blocker,
        updated_at=updated_at,
    )


def _build_fallback_plan_decision(
    *,
    user_input: str,
    existing_mission_state: MissionState | None,
    updated_at: str,
) -> MissionPlanDecision:
    if existing_mission_state is not None:
        return MissionPlanDecision(
            mission_action="continue_existing",
            mission_goal=existing_mission_state.goal,
            deliverables=existing_mission_state.deliverables,
            active_deliverable_id=existing_mission_state.active_deliverable_id,
            summary="continued existing mission",
        )
    return MissionPlanDecision(
        mission_action="start_new",
        mission_goal=user_input,
        deliverables=(
            MissionDeliverableState(
                deliverable_id="d1",
                task="Complete the requested mission",
                deliverable=user_input,
                verification="The user-requested outcome is verified from runtime facts.",
                status="pending",
                missing=None,
                blocker=None,
                attempt_count=0,
                repair_count=0,
                retry_count=0,
                artifact_paths=(),
                evidence=(),
                updated_at=updated_at,
            ),
        ),
        active_deliverable_id="d1",
        summary="started fallback mission plan",
    )


def _build_fallback_verification_decision(
    *,
    mission_state: MissionState,
    draft_reply: str,
    latest_tool_results: Sequence[ToolResult],
    turn_tool_results: Sequence[ToolResult],
    updated_at: str,
) -> MissionVerificationDecision:
    del updated_at
    active_deliverable = mission_state.get_deliverable()
    if latest_tool_results:
        if all(tool_result.success is False for tool_result in latest_tool_results):
            can_retry = any(_tool_result_timed_out(tool_result) for tool_result in latest_tool_results) and (
                active_deliverable is not None
                and active_deliverable.retry_count < _MISSION_RETRY_BUDGET_PER_DELIVERABLE
            )
            if can_retry:
                return MissionVerificationDecision(
                    mission_status="active",
                    active_deliverable_id=mission_state.active_deliverable_id,
                    current_deliverable_status="active",
                    missing="A grounded retry is still needed after the timeout.",
                    blocker=None,
                    artifact_paths=(),
                    evidence=(),
                    next_action="continue",
                    notes_for_next_step=(
                        "Retry the active deliverable with a narrower or lighter tool step."
                    ),
                    assistant_reply=None,
                )
            blocker = latest_tool_results[-1].error or latest_tool_results[-1].output_text
            return MissionVerificationDecision(
                mission_status="blocked",
                active_deliverable_id=mission_state.active_deliverable_id,
                current_deliverable_status="blocked",
                missing=None,
                blocker=blocker,
                artifact_paths=(),
                evidence=(),
                next_action="blocked_reply",
                notes_for_next_step=None,
                assistant_reply=None,
            )
        if _latest_results_include_side_effect_success(latest_tool_results):
            all_verified = all(
                deliverable.status == "completed"
                or deliverable.deliverable_id == mission_state.active_deliverable_id
                for deliverable in mission_state.deliverables
            )
            return MissionVerificationDecision(
                mission_status="completed" if all_verified else "active",
                active_deliverable_id=None,
                current_deliverable_status="completed",
                missing=None,
                blocker=None,
                artifact_paths=_collect_tool_artifact_paths(latest_tool_results),
                evidence=_collect_tool_evidence(latest_tool_results),
                next_action="final_reply" if all_verified else "continue",
                notes_for_next_step=(
                    None
                    if all_verified
                    else "Move to the next pending deliverable in the mission."
                ),
                assistant_reply=None,
            )
        if draft_reply:
            return MissionVerificationDecision(
                mission_status="active",
                active_deliverable_id=mission_state.active_deliverable_id,
                current_deliverable_status="active",
                missing="Another mission deliverable still needs verification.",
                blocker=None,
                artifact_paths=_collect_tool_artifact_paths(turn_tool_results),
                evidence=_collect_tool_evidence(turn_tool_results),
                next_action="continue",
                notes_for_next_step=(
                    "Continue the mission and address any still-unverified deliverables."
                ),
                assistant_reply=None,
            )

    if mission_state.status in {"blocked", "completed"} and draft_reply:
        return MissionVerificationDecision(
            mission_status=mission_state.status,
            active_deliverable_id=mission_state.active_deliverable_id,
            current_deliverable_status=(
                active_deliverable.status if active_deliverable is not None else "active"
            ),
            missing=active_deliverable.missing if active_deliverable is not None else None,
            blocker=mission_state.last_blocker,
            artifact_paths=mission_state.last_verified_artifact_paths,
            evidence=mission_state.last_successful_evidence,
            next_action="final_reply" if mission_state.status == "completed" else "blocked_reply",
            notes_for_next_step=None,
            assistant_reply=draft_reply,
        )

    return MissionVerificationDecision(
        mission_status="active",
        active_deliverable_id=mission_state.active_deliverable_id,
        current_deliverable_status=(
            active_deliverable.status if active_deliverable is not None else "active"
        ),
        missing=active_deliverable.missing if active_deliverable is not None else None,
        blocker=None,
        artifact_paths=mission_state.last_verified_artifact_paths,
        evidence=mission_state.last_successful_evidence,
        next_action=(
            "final_reply"
            if draft_reply and not any(
                deliverable.status in {"pending", "active"}
                for deliverable in mission_state.deliverables
            )
            else "continue"
        ),
        notes_for_next_step=(
            None
            if draft_reply
            and not any(
                deliverable.status in {"pending", "active"}
                for deliverable in mission_state.deliverables
            )
            else "Another tool or repair step is required."
        ),
        assistant_reply=(
            draft_reply
            if draft_reply
            and not any(
                deliverable.status in {"pending", "active"}
                for deliverable in mission_state.deliverables
            )
            else None
        ),
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


def _collect_tool_artifact_paths(tool_results: Sequence[ToolResult]) -> tuple[str, ...]:
    paths: list[str] = []
    for tool_result in tool_results:
        payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
        for key in ("resolved_path", "path", "requested_path"):
            value = payload.get(key)
            if isinstance(value, str) and value not in paths:
                paths.append(value)
    return tuple(paths[:4])


def _collect_tool_evidence(tool_results: Sequence[ToolResult]) -> tuple[str, ...]:
    evidence: list[str] = []
    for tool_result in tool_results:
        payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
        summary_points = payload.get("summary_points")
        if isinstance(summary_points, list):
            for item in summary_points:
                if isinstance(item, str):
                    evidence.append(item)
        display_sources = payload.get("display_sources")
        if isinstance(display_sources, list):
            for item in display_sources:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("url"), str)
                    and item["url"] not in evidence
                ):
                    evidence.append(item["url"])
    return tuple(evidence[:4])


def _latest_results_include_side_effect_success(
    tool_results: Sequence[ToolResult],
) -> bool:
    for tool_result in tool_results:
        if tool_result.success is not True:
            continue
        if tool_result.tool_name in {"write_text_file", "delete_file"}:
            return True
    return False


def _should_promote_to_persisted_mission(
    *,
    mission_state: MissionState,
    verification_decision: MissionVerificationDecision,
    latest_tool_results: Sequence[ToolResult],
) -> bool:
    if len(mission_state.deliverables) > 1:
        return True
    if verification_decision.mission_status == "blocked":
        return True
    if latest_tool_results and any(tool_result.success is False for tool_result in latest_tool_results):
        return True
    return _latest_results_include_side_effect_success(latest_tool_results)


def _response_has_side_effect_tool_calls(
    *,
    response: OrchestratorTurnResult,
    tool_definitions: Sequence[ToolDefinition],
) -> bool:
    tool_permissions = {
        tool_definition.name: tool_definition.permission_level.value
        for tool_definition in tool_definitions
    }
    for tool_call in response.response.tool_calls or ():
        permission_level = tool_permissions.get(tool_call.tool_name)
        if permission_level == "local_write":
            return True
    return False


def build_compat_mission_state_from_legacy_goal_state(
    *,
    legacy_goal_state: SessionGoalState,
    legacy_progress_ledger: Sequence[SessionProgressEntry] = (),
    updated_at: str | None = None,
) -> MissionState | None:
    """Project an older active or blocked goal state into a mission snapshot."""

    if legacy_goal_state.status not in {"active", "blocked"}:
        return None

    timestamp = updated_at or legacy_goal_state.updated_at or utc_now_iso()
    current_step = legacy_goal_state.current_step or "Complete the requested mission"
    latest_detail = (
        legacy_progress_ledger[-1].detail if legacy_progress_ledger else None
    )
    deliverable = MissionDeliverableState(
        deliverable_id="legacy-d1",
        task=current_step,
        deliverable=legacy_goal_state.goal,
        verification="The legacy mission deliverable is verified from runtime facts.",
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

    return MissionState(
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
    )


def _build_mission_status_reply(mission_state: MissionState) -> str:
    if mission_state.status == "completed":
        return f"The persisted mission is completed: {mission_state.goal}"
    if mission_state.status == "blocked":
        blocker = mission_state.last_blocker or "no blocker detail was persisted"
        step = mission_state.active_task or "the current mission step"
        return f"The persisted mission is blocked on {step}: {blocker}"
    step = mission_state.active_task or "the current mission step"
    return f"The persisted mission is active on {step}."


def _build_incomplete_mission_reply(mission_state: MissionState) -> str:
    if mission_state.status == "blocked":
        return _build_mission_status_reply(mission_state)
    return (
        "The mission is still active and needs another verified step before it can be completed."
    )


def _mark_cancelled_mission(
    *,
    mission_state: MissionState,
    updated_at: str,
) -> MissionState:
    current_deliverable = mission_state.get_deliverable()
    next_deliverables = list(mission_state.deliverables)
    if current_deliverable is not None:
        cancelled_deliverable = replace(
            current_deliverable,
            status="blocked",
            blocker=_agent_loop._TURN_CANCELLED_REPLY,
            updated_at=updated_at,
        )
        next_deliverables = [
            cancelled_deliverable
            if deliverable.deliverable_id == cancelled_deliverable.deliverable_id
            else deliverable
            for deliverable in next_deliverables
        ]
    return replace(
        mission_state,
        status="blocked",
        blocked_deliverables=tuple(
            deliverable.deliverable_id
            for deliverable in next_deliverables
            if deliverable.status == "blocked"
        ),
        deliverables=tuple(next_deliverables),
        last_blocker=_agent_loop._TURN_CANCELLED_REPLY,
        updated_at=updated_at,
    )


def _block_mission_from_runtime_reply(
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
    return replace(
        mission_state,
        status="blocked",
        blocked_deliverables=tuple(
            deliverable.deliverable_id
            for deliverable in next_deliverables
            if deliverable.status == "blocked"
        ),
        deliverables=tuple(next_deliverables),
        last_blocker=blocker,
        updated_at=updated_at,
    )


def _call_model_and_trace(
    *,
    orchestrator: Orchestrator,
    tracer: Tracer,
    session_id: str,
    model_profile_name: str,
    thinking_enabled: bool,
    messages: Sequence[LLMMessage],
    tools: Sequence[ToolDefinition],
) -> OrchestratorTurnResult:
    turn_result = orchestrator.call_model(
        session_id=session_id,
        messages=messages,
        model_profile_name=model_profile_name,
        thinking_enabled=thinking_enabled,
        content_callback=None,
        tools=tools,
    )
    tracer.trace_model_succeeded(
        session_id=session_id,
        provider=turn_result.response.provider,
        model_name=turn_result.response.model_name,
        finish_reason=turn_result.response.finish_reason,
        output_length=len(turn_result.response.content),
        model_duration_ms=turn_result.model_duration_ms,
        reasoning=turn_result.response.reasoning,
    )
    return turn_result
