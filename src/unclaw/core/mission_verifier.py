"""Model-native mission planning and verification helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any

from unclaw.core.mission_state import MissionDeliverableState, MissionState
from unclaw.llm.base import LLMMessage, LLMRole
from unclaw.tools.contracts import ToolCall, ToolResult

_MISSION_PLAN_NOTE_PREFIX = "Mission planning:"
_MISSION_PROGRESS_NOTE_PREFIX = "Mission progress check:"
_MISSION_PLANNER_SYSTEM_PROMPT = "\n".join(
    (
        "Mission planner for the Unclaw local agent runtime.",
        "Plan only from the user's request plus the compact persisted mission workspace.",
        "Hard rules:",
        "- Mission means the full user-requested outcome across one or more steps and turns.",
        "- Task means one concrete sub-objective inside that mission.",
        "- Deliverables must be concrete, compact, and verifiable.",
        "- Keep the plan compact for local models: at most 4 deliverables.",
        "- Use the actual existing mission when the user is continuing, repairing, resuming, or asking mission status for the same outcome.",
        "- `compatibility_mission_state` is weak legacy context only. Do not continue it unless the user is clearly asking for that same old mission and no newer actual mission exists.",
        "- Start a new mission only when the user is clearly asking for a different outcome.",
        "- If the request can be answered honestly in one reply with no durable execution, choose direct_reply_only.",
        "- Do not collapse a compound mission into one generic deliverable unless the user's request truly has only one deliverable.",
        "- Do not rewrite a deliverable into a tool name.",
        "Return JSON only with this shape:",
        (
            '{"mission_action":"start_new|continue_existing|direct_reply_only",'
            '"mission_goal":"...",'
            '"deliverables":[{"id":"d1","task":"...","deliverable":"...",'
            '"verification":"...","depends_on":["d0"]}],'
            '"execution_queue":["d1"],'
            '"active_deliverable_id":"d1",'
            '"summary":"..."}'
        ),
    )
)
_MISSION_VERIFIER_SYSTEM_PROMPT = "\n".join(
    (
        "Mission step verifier for the Unclaw local agent runtime.",
        "Judge whether the intended deliverable was actually achieved from the structured facts.",
        "Hard rules:",
        "- Completion requires every mission deliverable to be verified completed.",
        "- Do not trust draft confidence. Verify from tool results, artifact facts, and persisted mission state.",
        "- If a file action did not happen, do not call that deliverable complete.",
        "- If a timeout or failure can still be repaired within budget, keep the mission active and say what is missing.",
        "- If retry budget is exhausted or a blocker is unrecoverable from the given facts, mark the mission blocked.",
        "- If an existing file might already satisfy the active deliverable but it is not verified yet, keep the mission active and request a verification read in notes_for_next_step.",
        "- Do not mark the mission completed while any final_deliverables_missing item remains.",
        "Return JSON only with this shape:",
        (
            '{"mission_status":"active|blocked|completed",'
            '"active_deliverable_id":"...",'
            '"current_deliverable_status":"pending|active|blocked|completed",'
            '"missing":"...",'
            '"blocker":"...",'
            '"artifact_paths":["..."],'
            '"evidence":["..."],'
            '"final_deliverables_missing":["d2"],'
            '"next_action":"continue|final_reply|blocked_reply",'
            '"repair_strategy":"...",'
            '"notes_for_next_step":"...",'
            '"assistant_reply":"..."}'
        ),
    )
)


@dataclass(frozen=True, slots=True)
class MissionPlanDecision:
    """Parsed planner decision for mission initialization or continuation."""

    mission_action: str
    mission_goal: str | None
    deliverables: tuple[MissionDeliverableState, ...]
    active_deliverable_id: str | None
    summary: str
    execution_queue: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MissionVerificationDecision:
    """Parsed verifier decision for one mission checkpoint."""

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


def build_mission_plan_messages(
    *,
    user_input: str,
    existing_mission_state: MissionState | None,
    compatibility_mission_state: MissionState | None,
    first_response: Any | None,
    available_tool_names: Sequence[str],
) -> tuple[LLMMessage, ...]:
    """Build planner messages for mission initialization or continuation."""

    planner_payload = {
        "user_input": user_input,
        "available_tool_names": tuple(available_tool_names),
        "existing_mission_state": (
            _serialize_mission_state(existing_mission_state)
            if existing_mission_state is not None
            else None
        ),
        "compatibility_mission_state": (
            _serialize_mission_state(compatibility_mission_state)
            if compatibility_mission_state is not None
            else None
        ),
        "first_response_summary": _serialize_first_response(first_response),
    }
    return (
        LLMMessage(
            role=LLMRole.SYSTEM,
            content=_MISSION_PLANNER_SYSTEM_PROMPT,
        ),
        LLMMessage(
            role=LLMRole.USER,
            content=_serialize_json_payload(planner_payload),
        ),
    )


def parse_mission_plan_response(
    response_text: str,
    *,
    updated_at: str,
) -> MissionPlanDecision | None:
    """Parse one mission planner response."""

    payload = _parse_json_dict(response_text)
    if payload is None:
        return None

    mission_action = _read_choice(
        payload.get("mission_action"),
        allowed_values=frozenset(
            {"start_new", "continue_existing", "direct_reply_only"}
        ),
    )
    if mission_action is None:
        return None

    mission_goal = _read_optional_text(payload.get("mission_goal"))
    summary = _read_optional_text(payload.get("summary")) or "mission plan prepared"
    execution_queue = _read_text_list(payload.get("execution_queue"))
    deliverables_payload = payload.get("deliverables")
    if not isinstance(deliverables_payload, list):
        return None

    deliverables: list[MissionDeliverableState] = []
    for index, item in enumerate(deliverables_payload[:4]):
        if not isinstance(item, dict):
            return None
        deliverable_id = _read_optional_text(item.get("id")) or f"d{index + 1}"
        task = _read_optional_text(item.get("task"))
        deliverable_text = _read_optional_text(item.get("deliverable"))
        verification = _read_optional_text(item.get("verification"))
        if not task or not deliverable_text or not verification:
            return None
        deliverables.append(
            MissionDeliverableState(
                deliverable_id=deliverable_id,
                task=task,
                deliverable=deliverable_text,
                verification=verification,
                status="pending",
                missing=None,
                blocker=None,
                attempt_count=0,
                repair_count=0,
                retry_count=0,
                artifact_paths=(),
                evidence=(),
                updated_at=updated_at,
            )
        )

    active_deliverable_id = _read_optional_text(payload.get("active_deliverable_id"))
    if active_deliverable_id is None and deliverables:
        active_deliverable_id = deliverables[0].deliverable_id
    if not execution_queue:
        execution_queue = tuple(
            deliverable.deliverable_id for deliverable in deliverables
        )

    return MissionPlanDecision(
        mission_action=mission_action,
        mission_goal=mission_goal,
        deliverables=tuple(deliverables),
        execution_queue=execution_queue,
        active_deliverable_id=active_deliverable_id,
        summary=summary,
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
    """Build verifier messages for one mission checkpoint."""

    verifier_payload = {
        "user_input": user_input,
        "mission_state": _serialize_mission_state(mission_state),
        "draft_reply": draft_reply,
        "latest_tool_results": tuple(
            _serialize_tool_result(tool_result) for tool_result in latest_tool_results
        ),
        "turn_tool_results": tuple(
            _serialize_tool_result(tool_result) for tool_result in turn_tool_results
        ),
        "blocker_metadata": blocker_metadata,
        "retry_budget_remaining": retry_budget_remaining,
        "repair_budget_remaining": repair_budget_remaining,
    }
    return (
        LLMMessage(
            role=LLMRole.SYSTEM,
            content=_MISSION_VERIFIER_SYSTEM_PROMPT,
        ),
        LLMMessage(
            role=LLMRole.USER,
            content=_serialize_json_payload(verifier_payload),
        ),
    )


def parse_mission_verification_response(
    response_text: str,
) -> MissionVerificationDecision | None:
    """Parse one mission verifier response."""

    payload = _parse_json_dict(response_text)
    if payload is None:
        return None

    mission_status = _read_choice(
        payload.get("mission_status"),
        allowed_values=frozenset({"active", "blocked", "completed"}),
    )
    current_deliverable_status = _read_choice(
        payload.get("current_deliverable_status"),
        allowed_values=frozenset({"pending", "active", "blocked", "completed"}),
    )
    next_action = _read_choice(
        payload.get("next_action"),
        allowed_values=frozenset({"continue", "final_reply", "blocked_reply"}),
    )
    if (
        mission_status is None
        or current_deliverable_status is None
        or next_action is None
    ):
        return None

    return MissionVerificationDecision(
        mission_status=mission_status,
        active_deliverable_id=_read_optional_text(payload.get("active_deliverable_id")),
        current_deliverable_status=current_deliverable_status,
        missing=_read_optional_text(payload.get("missing")),
        blocker=_read_optional_text(payload.get("blocker")),
        artifact_paths=_read_text_list(payload.get("artifact_paths")),
        evidence=_read_text_list(payload.get("evidence")),
        final_deliverables_missing=_read_text_list(
            payload.get("final_deliverables_missing")
        ),
        next_action=next_action,
        repair_strategy=_read_optional_text(payload.get("repair_strategy")),
        notes_for_next_step=_read_optional_text(payload.get("notes_for_next_step")),
        assistant_reply=_read_optional_text(payload.get("assistant_reply")),
    )


def build_mission_progress_note(
    *,
    mission_state: MissionState,
    verification_decision: MissionVerificationDecision,
    user_input: str,
    latest_tool_results: Sequence[ToolResult],
) -> str:
    """Build a compact mission checkpoint note for the next execution step."""

    progress_payload = {
        "mission_goal": mission_state.goal,
        "mission_status": verification_decision.mission_status,
        "active_deliverable_id": verification_decision.active_deliverable_id,
        "current_deliverable_status": verification_decision.current_deliverable_status,
        "missing": verification_decision.missing,
        "blocker": verification_decision.blocker,
        "artifact_paths": verification_decision.artifact_paths,
        "evidence": verification_decision.evidence,
        "final_deliverables_missing": verification_decision.final_deliverables_missing,
        "repair_strategy": verification_decision.repair_strategy,
        "notes_for_next_step": verification_decision.notes_for_next_step,
        "latest_tool_results": tuple(
            _serialize_tool_result(tool_result) for tool_result in latest_tool_results
        ),
    }
    return "\n".join(
        (
            (
                f"{_MISSION_PROGRESS_NOTE_PREFIX} "
                "verify the persisted mission state before taking the next step."
            ),
            f"Original user request: {user_input}",
            "Use the structured mission checkpoint facts below.",
            "If a tool call would make meaningful mission progress, emit it now.",
            "If every deliverable is verified complete, answer the user directly.",
            "If the mission is blocked, answer honestly with what completed and what remains blocked.",
            "Structured mission checkpoint facts (JSON):",
            _serialize_json_payload(progress_payload),
        )
    )


def build_mission_initialization_note(
    *,
    mission_state: MissionState,
) -> str:
    """Build a compact persisted-mission note for context injection."""

    deliverable_lines = " | ".join(
        (
            "["
            f"id={json.dumps(deliverable.deliverable_id, ensure_ascii=False)}; "
            f"task={json.dumps(deliverable.task, ensure_ascii=False)}; "
            f"deliverable={json.dumps(deliverable.deliverable, ensure_ascii=False)}; "
            f"status={json.dumps(deliverable.status, ensure_ascii=False)}"
            "]"
        )
        for deliverable in mission_state.deliverables
    )
    return (
        f"{_MISSION_PLAN_NOTE_PREFIX} "
        f"goal={json.dumps(mission_state.goal, ensure_ascii=False)}; "
        f"status={json.dumps(mission_state.status, ensure_ascii=False)}; "
        f"active_deliverable_id={json.dumps(mission_state.active_deliverable_id, ensure_ascii=False)}; "
        f"active_task={json.dumps(mission_state.active_task, ensure_ascii=False)}; "
        f"deliverables={deliverable_lines or '[]'}."
    )


def _serialize_mission_state(mission_state: MissionState | None) -> dict[str, Any] | None:
    if mission_state is None:
        return None
    return {
        "mission_id": mission_state.mission_id,
        "goal": mission_state.goal,
        "status": mission_state.status,
        "active_deliverable_id": mission_state.active_deliverable_id,
        "active_task": mission_state.active_task,
        "completed_deliverables": mission_state.completed_deliverables,
        "blocked_deliverables": mission_state.blocked_deliverables,
        "execution_queue": mission_state.execution_queue,
        "completed_steps": mission_state.completed_steps,
        "failed_steps": mission_state.failed_steps,
        "observed_facts": mission_state.observed_facts,
        "artifact_facts": mission_state.artifact_facts,
        "blockers": mission_state.blockers,
        "pending_repairs": mission_state.pending_repairs,
        "final_deliverables_missing": mission_state.final_deliverables_missing,
        "planner_summary": mission_state.planner_summary,
        "deliverables": tuple(
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
                "artifact_paths": deliverable.artifact_paths,
                "evidence": deliverable.evidence,
                "updated_at": deliverable.updated_at,
            }
            for deliverable in mission_state.deliverables
        ),
        "retry_history": mission_state.retry_history,
        "repair_history": mission_state.repair_history,
        "last_verified_artifact_paths": mission_state.last_verified_artifact_paths,
        "last_successful_evidence": mission_state.last_successful_evidence,
        "last_blocker": mission_state.last_blocker,
        "updated_at": mission_state.updated_at,
    }


def _serialize_first_response(first_response: Any | None) -> dict[str, Any] | None:
    if first_response is None:
        return None
    response = getattr(first_response, "response", None)
    if response is None:
        return None
    tool_calls = getattr(response, "tool_calls", ()) or ()
    return {
        "draft_reply": getattr(response, "content", ""),
        "tool_calls": tuple(
            {
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
            }
            for tool_call in tool_calls
            if isinstance(tool_call, ToolCall)
        ),
    }


def _serialize_tool_result(tool_result: ToolResult) -> dict[str, Any]:
    payload = tool_result.payload if isinstance(tool_result.payload, dict) else {}
    return {
        "tool_name": tool_result.tool_name,
        "success": tool_result.success,
        "output_text": tool_result.output_text,
        "error": tool_result.error,
        "failure_kind": tool_result.failure_kind,
        "payload": payload,
    }


def _serialize_json_payload(payload: dict[str, Any]) -> str:
    normalized_payload = json.loads(
        json.dumps(payload, ensure_ascii=False, default=str)
    )
    return json.dumps(normalized_payload, ensure_ascii=False, indent=2, sort_keys=True)


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
        if normalized is not None:
            items.append(normalized)
    return tuple(items)
