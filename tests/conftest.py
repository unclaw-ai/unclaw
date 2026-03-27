from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from unclaw.llm.base import LLMResponse, LLMRole
from unclaw.model_packs import DEV_MODEL_PACK_NAME, get_model_pack_profiles
from unclaw.tools.contracts import ToolCall


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Ensure the project root is on sys.path so that locally-installed skill bundles
# under ./skills/ are importable as top-level packages (e.g. skills.weather.tool).
_repo_root_str = str(_repo_root())
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)


_REAL_OLLAMA_TOOL_CAPABLE_MODEL_PREFIXES = (
    "qwen",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "mistral",
    "command-r",
    "firefunction",
    "hermes",
)


class ScriptedFakeOllamaProvider:
    """Lightweight fake Ollama provider with sequential scripted responses."""

    provider_name = "ollama"
    _scripted_steps: tuple[Any, ...] = ()
    _captured_messages: list[list[Any]] | None = None
    _captured_calls: list[dict[str, Any]] | None = None
    _auto_stream_text: bool = False
    _call_count: int = 0

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        default_timeout_seconds: float = 60.0,
    ) -> None:
        del base_url, default_timeout_seconds

    @classmethod
    def with_script(
        cls,
        steps: Sequence[LLMResponse | Callable[..., LLMResponse]],
        *,
        captured_messages: list[list[Any]] | None = None,
        captured_calls: list[dict[str, Any]] | None = None,
        auto_stream_text: bool = False,
    ) -> type["ScriptedFakeOllamaProvider"]:
        return type(
            "ConfiguredScriptedFakeOllamaProvider",
            (cls,),
            {
                "_scripted_steps": tuple(steps),
                "_captured_messages": captured_messages,
                "_captured_calls": captured_calls,
                "_auto_stream_text": auto_stream_text,
                "_call_count": 0,
            },
        )

    @classmethod
    def call_count(cls) -> int:
        return cls._call_count

    @staticmethod
    def _is_grounded_finalizer_call(messages: Sequence[Any]) -> bool:
        if not messages:
            return False
        first_content = getattr(messages[0], "content", "")
        return isinstance(first_content, str) and first_content.startswith(
            "Grounded reply finalizer for one runtime turn."
        )

    @staticmethod
    def _is_continuation_check_call(messages: Sequence[Any]) -> bool:
        """Detect agent-loop continuation check."""
        if not messages:
            return False
        last_content = getattr(messages[-1], "content", "")
        return isinstance(last_content, str) and last_content.startswith(
            ("Task completion check:", "Mission continuation check:")
        )

    @staticmethod
    def _is_pre_write_grounding_check_call(messages: Sequence[Any]) -> bool:
        if not messages:
            return False
        last_content = getattr(messages[-1], "content", "")
        return isinstance(last_content, str) and last_content.startswith(
            "Pre-write grounding check:"
        )

    @staticmethod
    def _is_mission_planner_call(messages: Sequence[Any]) -> bool:
        if not messages:
            return False
        first_content = getattr(messages[0], "content", "")
        return isinstance(first_content, str) and first_content.startswith(
            "Mission planner for the Unclaw local agent runtime."
        )

    @staticmethod
    def _is_mission_verifier_call(messages: Sequence[Any]) -> bool:
        if not messages:
            return False
        first_content = getattr(messages[0], "content", "")
        return isinstance(first_content, str) and first_content.startswith(
            "Mission step verifier for the Unclaw local agent runtime."
        )

    @classmethod
    def _build_auto_continuation_response(
        cls, profile: Any, messages: Sequence[Any]
    ) -> LLMResponse:
        """Repeat the draft reply from the preceding assistant message."""
        draft_reply = ""
        for msg in reversed(messages):
            if getattr(msg, "role", None) is LLMRole.ASSISTANT:
                draft_reply = getattr(msg, "content", "") or ""
                break
        return LLMResponse(
            provider="ollama",
            model_name=profile.model_name,
            content=draft_reply,
            created_at="2026-03-26T00:00:01Z",
            finish_reason="stop",
        )

    @classmethod
    def _build_auto_pre_write_grounding_response(
        cls, profile: Any, messages: Sequence[Any]
    ) -> LLMResponse:
        for msg in reversed(messages):
            tool_calls_payload = getattr(msg, "tool_calls_payload", None)
            if not isinstance(tool_calls_payload, tuple) or not tool_calls_payload:
                continue
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="",
                created_at="2026-03-26T00:00:01Z",
                finish_reason="stop",
                tool_calls=tuple(
                    ToolCall(
                        tool_name=item["function"]["name"],
                        arguments=dict(item["function"]["arguments"]),
                    )
                    for item in tool_calls_payload
                    if isinstance(item, dict)
                    and isinstance(item.get("function"), dict)
                    and isinstance(item["function"].get("name"), str)
                    and isinstance(item["function"].get("arguments"), dict)
                ),
                raw_payload={"message": {"content": "", "tool_calls": list(tool_calls_payload)}},
            )

        return LLMResponse(
            provider="ollama",
            model_name=profile.model_name,
            content="I couldn't safely revise the pending write from the available facts.",
            created_at="2026-03-26T00:00:01Z",
            finish_reason="stop",
        )

    @classmethod
    def _build_auto_mission_planner_response(
        cls, profile: Any, messages: Sequence[Any]
    ) -> LLMResponse:
        payload: dict[str, Any] = {}
        if messages:
            raw_content = getattr(messages[-1], "content", "")
            if isinstance(raw_content, str):
                try:
                    parsed = json.loads(raw_content)
                except json.JSONDecodeError:
                    parsed = {}
                if isinstance(parsed, dict):
                    payload = parsed

        existing_mission = payload.get("existing_mission_state")
        compatibility_mission = payload.get("compatibility_mission_state")
        user_input = payload.get("user_input")
        first_response_summary = payload.get("first_response_summary")

        # Determine mission action: continue_existing for active missions,
        # direct_reply_only for simple single-tool turns (no existing
        # mission, single search-family tool call), start_new otherwise.
        _SEARCH_FAMILY = {"search_web", "fast_web_search"}
        first_response_tool_calls = (
            first_response_summary.get("tool_calls", [])
            if isinstance(first_response_summary, dict)
            else []
        )
        _is_single_search_only = (
            len(first_response_tool_calls) == 1
            and isinstance(first_response_tool_calls[0], dict)
            and first_response_tool_calls[0].get("tool_name") in _SEARCH_FAMILY
        )
        if isinstance(existing_mission, dict):
            mission_action = "continue_existing"
        elif _is_single_search_only:
            # A single search-family tool call without an active mission
            # does not need kernel management — let the agent loop handle it.
            mission_action = "direct_reply_only"
        else:
            mission_action = "start_new"
        mission_goal = None
        deliverables: list[dict[str, str]] = []
        active_deliverable_id = None
        if isinstance(existing_mission, dict):
            raw_goal = existing_mission.get("goal")
            if isinstance(raw_goal, str):
                mission_goal = raw_goal
            raw_deliverables = existing_mission.get("deliverables")
            if isinstance(raw_deliverables, list):
                for raw_deliverable in raw_deliverables[:4]:
                    if not isinstance(raw_deliverable, dict):
                        continue
                    deliverable_id = raw_deliverable.get("deliverable_id")
                    task = raw_deliverable.get("task")
                    deliverable = raw_deliverable.get("deliverable")
                    verification = raw_deliverable.get("verification")
                    if all(isinstance(value, str) for value in (deliverable_id, task, deliverable, verification)):
                        deliverables.append(
                            {
                                "id": deliverable_id,
                                "task": task,
                                "deliverable": deliverable,
                                "verification": verification,
                            }
                        )
            raw_active_id = existing_mission.get("active_deliverable_id")
            if isinstance(raw_active_id, str):
                active_deliverable_id = raw_active_id

        if mission_goal is None and isinstance(user_input, str):
            mission_goal = user_input
        if not deliverables and isinstance(user_input, str):
            deliverables.append(
                {
                    "id": "d1",
                    "task": "Complete the requested mission",
                    "deliverable": user_input,
                    "verification": "The requested outcome is verified from runtime facts.",
                }
            )
            active_deliverable_id = "d1"

        return LLMResponse(
            provider="ollama",
            model_name=profile.model_name,
            content=json.dumps(
                {
                    "mission_action": mission_action,
                    "mission_goal": mission_goal or "",
                    "deliverables": deliverables,
                    "active_deliverable_id": active_deliverable_id,
                    "summary": "auto mission plan",
                },
                ensure_ascii=False,
            ),
            created_at="2026-03-26T00:00:01Z",
            finish_reason="stop",
        )

    @classmethod
    def _build_auto_mission_verifier_response(
        cls, profile: Any, messages: Sequence[Any]
    ) -> LLMResponse:
        payload: dict[str, Any] = {}
        if messages:
            raw_content = getattr(messages[-1], "content", "")
            if isinstance(raw_content, str):
                try:
                    parsed = json.loads(raw_content)
                except json.JSONDecodeError:
                    parsed = {}
                if isinstance(parsed, dict):
                    payload = parsed

        mission_state = (
            payload.get("mission_state")
            if isinstance(payload.get("mission_state"), dict)
            else {}
        )
        latest_tool_results = payload.get("latest_tool_results")
        if not isinstance(latest_tool_results, list):
            latest_tool_results = []
        draft_reply = payload.get("draft_reply")
        retry_budget_remaining = payload.get("retry_budget_remaining")
        active_deliverable_id = mission_state.get("active_deliverable_id")
        deliverables = mission_state.get("deliverables")
        has_pending_deliverables = False
        if isinstance(deliverables, list):
            has_pending_deliverables = any(
                isinstance(item, dict)
                and item.get("deliverable_id") != active_deliverable_id
                and item.get("status") in {"pending", "active"}
                for item in deliverables
            )

        def _result_payload(item: dict[str, Any]) -> dict[str, Any]:
            payload_value = item.get("payload")
            return payload_value if isinstance(payload_value, dict) else {}

        if latest_tool_results:
            if all(
                isinstance(item, dict) and item.get("success") is False
                for item in latest_tool_results
            ):
                timed_out = any(
                    isinstance(item, dict)
                    and (
                        item.get("failure_kind") == "timeout"
                        or _result_payload(item).get("execution_state") == "timed_out"
                        or "timed out" in str(item.get("error", "")).casefold()
                    )
                    for item in latest_tool_results
                )
                if timed_out and isinstance(retry_budget_remaining, int) and retry_budget_remaining > 0:
                    decision = {
                        "mission_status": "active",
                        "active_deliverable_id": active_deliverable_id,
                        "current_deliverable_status": "active",
                        "missing": "A retry is still needed after the timeout.",
                        "blocker": None,
                        "artifact_paths": [],
                        "evidence": [],
                        "next_action": "continue",
                        "notes_for_next_step": "Retry the active deliverable with a narrower tool step.",
                        "assistant_reply": None,
                    }
                elif any(
                    isinstance(item, dict)
                    and item.get("tool_name")
                    in {"write_text_file", "delete_file", "read_text_file"}
                    for item in latest_tool_results
                ):
                    latest_error = ""
                    if latest_tool_results and isinstance(latest_tool_results[-1], dict):
                        latest_error = str(
                            latest_tool_results[-1].get("error")
                            or latest_tool_results[-1].get("output_text")
                            or ""
                        )
                    decision = {
                        "mission_status": "active",
                        "active_deliverable_id": active_deliverable_id,
                        "current_deliverable_status": "active",
                        "missing": "A repaired step is still needed after the failed tool result.",
                        "blocker": latest_error or None,
                        "artifact_paths": [],
                        "evidence": [],
                        "next_action": "continue",
                        "notes_for_next_step": "Repair the active deliverable with another bounded tool step.",
                        "assistant_reply": None,
                    }
                else:
                    latest_error = ""
                    if latest_tool_results and isinstance(latest_tool_results[-1], dict):
                        latest_error = str(
                            latest_tool_results[-1].get("error")
                            or latest_tool_results[-1].get("output_text")
                            or ""
                        )
                    decision = {
                        "mission_status": "blocked",
                        "active_deliverable_id": active_deliverable_id,
                        "current_deliverable_status": "blocked",
                        "missing": None,
                        "blocker": latest_error or "mission step failed",
                        "artifact_paths": [],
                        "evidence": [],
                        "next_action": "blocked_reply",
                        "notes_for_next_step": None,
                        "assistant_reply": None,
                    }
            else:
                side_effect_success = any(
                    isinstance(item, dict)
                    and item.get("success") is True
                    and item.get("tool_name") in {"write_text_file", "delete_file"}
                    for item in latest_tool_results
                )
                artifact_paths: list[str] = []
                evidence: list[str] = []
                for item in latest_tool_results:
                    if not isinstance(item, dict):
                        continue
                    item_payload = _result_payload(item)
                    for key in ("resolved_path", "path", "requested_path"):
                        value = item_payload.get(key)
                        if isinstance(value, str) and value not in artifact_paths:
                            artifact_paths.append(value)
                    summary_points = item_payload.get("summary_points")
                    if isinstance(summary_points, list):
                        for summary_point in summary_points:
                            if isinstance(summary_point, str):
                                evidence.append(summary_point)
                    display_sources = item_payload.get("display_sources")
                    if isinstance(display_sources, list):
                        for display_source in display_sources:
                            if (
                                isinstance(display_source, dict)
                                and isinstance(display_source.get("url"), str)
                            ):
                                evidence.append(display_source["url"])

                if side_effect_success:
                    decision = {
                        "mission_status": (
                            "completed" if not has_pending_deliverables else "active"
                        ),
                        "active_deliverable_id": None,
                        "current_deliverable_status": "completed",
                        "missing": None,
                        "blocker": None,
                        "artifact_paths": artifact_paths[:4],
                        "evidence": evidence[:4],
                        "next_action": "continue",
                        "notes_for_next_step": "Answer from the verified side effect and continue only if another deliverable remains.",
                        "assistant_reply": None,
                    }
                else:
                    decision = {
                        "mission_status": "active",
                        "active_deliverable_id": active_deliverable_id,
                        "current_deliverable_status": "active",
                        "missing": "A user-facing answer from the verified tool evidence is still needed.",
                        "blocker": None,
                        "artifact_paths": artifact_paths[:4],
                        "evidence": evidence[:4],
                        "next_action": "continue",
                        "notes_for_next_step": "Use the verified tool evidence to answer the user or continue the next deliverable.",
                        "assistant_reply": None,
                    }
        else:
            has_verified_progress = bool(
                mission_state.get("last_verified_artifact_paths")
                or mission_state.get("last_successful_evidence")
            )
            if mission_state.get("status") == "blocked":
                decision = {
                    "mission_status": "blocked",
                    "active_deliverable_id": active_deliverable_id,
                    "current_deliverable_status": "blocked",
                    "missing": None,
                    "blocker": mission_state.get("last_blocker"),
                    "artifact_paths": list(mission_state.get("last_verified_artifact_paths", []))
                    if isinstance(mission_state.get("last_verified_artifact_paths"), list)
                    else [],
                    "evidence": list(mission_state.get("last_successful_evidence", []))
                    if isinstance(mission_state.get("last_successful_evidence"), list)
                    else [],
                    "next_action": "blocked_reply",
                    "notes_for_next_step": None,
                    "assistant_reply": draft_reply if isinstance(draft_reply, str) and draft_reply.strip() else None,
                }
            elif isinstance(draft_reply, str) and draft_reply.strip() and has_verified_progress:
                decision = {
                    "mission_status": "completed",
                    "active_deliverable_id": None,
                    "current_deliverable_status": "completed",
                    "missing": None,
                    "blocker": mission_state.get("last_blocker"),
                    "artifact_paths": list(mission_state.get("last_verified_artifact_paths", []))
                    if isinstance(mission_state.get("last_verified_artifact_paths"), list)
                    else [],
                    "evidence": list(mission_state.get("last_successful_evidence", []))
                    if isinstance(mission_state.get("last_successful_evidence"), list)
                    else [],
                    "next_action": "final_reply",
                    "notes_for_next_step": None,
                    "assistant_reply": draft_reply,
                }
            elif isinstance(draft_reply, str) and draft_reply.strip():
                decision = {
                    "mission_status": "blocked",
                    "active_deliverable_id": active_deliverable_id,
                    "current_deliverable_status": "blocked",
                    "missing": None,
                    "blocker": mission_state.get("last_blocker"),
                    "artifact_paths": list(mission_state.get("last_verified_artifact_paths", []))
                    if isinstance(mission_state.get("last_verified_artifact_paths"), list)
                    else [],
                    "evidence": list(mission_state.get("last_successful_evidence", []))
                    if isinstance(mission_state.get("last_successful_evidence"), list)
                    else [],
                    "next_action": "blocked_reply",
                    "notes_for_next_step": None,
                    "assistant_reply": draft_reply,
                }
            else:
                decision = {
                    "mission_status": mission_state.get("status", "active"),
                    "active_deliverable_id": active_deliverable_id,
                    "current_deliverable_status": "active",
                    "missing": None,
                    "blocker": mission_state.get("last_blocker"),
                    "artifact_paths": list(mission_state.get("last_verified_artifact_paths", []))
                    if isinstance(mission_state.get("last_verified_artifact_paths"), list)
                    else [],
                    "evidence": list(mission_state.get("last_successful_evidence", []))
                    if isinstance(mission_state.get("last_successful_evidence"), list)
                    else [],
                    "next_action": "continue",
                    "notes_for_next_step": "Another step is needed.",
                    "assistant_reply": None,
                }

        return LLMResponse(
            provider="ollama",
            model_name=profile.model_name,
            content=json.dumps(decision, ensure_ascii=False),
            created_at="2026-03-26T00:00:01Z",
            finish_reason="stop",
        )

    @staticmethod
    def _build_auto_finalizer_response(profile, messages) -> LLMResponse:  # type: ignore[no-untyped-def]
        draft_reply = ""
        tool_results: list[dict[str, Any]] = []
        if messages:
            finalizer_input = getattr(messages[-1], "content", "")
            if isinstance(finalizer_input, str):
                try:
                    payload = json.loads(finalizer_input)
                except json.JSONDecodeError:
                    payload = {}
                if isinstance(payload, dict):
                    raw_draft_reply = payload.get("assistant_draft_reply")
                    if isinstance(raw_draft_reply, str):
                        draft_reply = raw_draft_reply
                    raw_tool_results = payload.get("current_turn_tool_results")
                    if isinstance(raw_tool_results, list):
                        tool_results = [
                            item for item in raw_tool_results if isinstance(item, dict)
                        ]

        final_reply = draft_reply
        if tool_results and all(item.get("success") is False for item in tool_results):
            timed_out = any(
                item.get("execution_state") == "timed_out"
                or "timed out" in str(item.get("error", "")).casefold()
                for item in tool_results
            )
            if timed_out:
                final_reply = (
                    "The tool step timed out, so I couldn't confirm the requested "
                    "details from retrieved tool evidence."
                )
            else:
                final_reply = (
                    "The tool step failed, so I couldn't confirm the requested "
                    "details from retrieved tool evidence."
                )

        return LLMResponse(
            provider="ollama",
            model_name=profile.model_name,
            content=json.dumps({"final_reply": final_reply}),
            created_at="2026-03-26T00:00:00Z",
            finish_reason="stop",
        )

    def chat(  # type: ignore[no-untyped-def]
        self,
        profile,
        messages,
        *,
        timeout_seconds=None,
        thinking_enabled=False,
        content_callback=None,
        tools=None,
    ):
        provider_cls = type(self)
        if provider_cls._captured_messages is not None:
            provider_cls._captured_messages.append(list(messages))
        if provider_cls._captured_calls is not None:
            provider_cls._captured_calls.append(
                {
                    "profile": profile,
                    "messages": list(messages),
                    "timeout_seconds": timeout_seconds,
                    "thinking_enabled": thinking_enabled,
                    "tools": tools,
                }
            )

        if provider_cls._is_grounded_finalizer_call(messages):
            step_index = provider_cls._call_count
            if step_index < len(provider_cls._scripted_steps):
                step = provider_cls._scripted_steps[step_index]
                if callable(step):
                    provider_cls._call_count += 1
                    return step(
                        profile=profile,
                        messages=messages,
                        timeout_seconds=timeout_seconds,
                        thinking_enabled=thinking_enabled,
                        content_callback=content_callback,
                        tools=tools,
                    )
                if isinstance(step, LLMResponse):
                    try:
                        payload = json.loads(step.content)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict) and isinstance(
                        payload.get("final_reply"), str
                    ):
                        provider_cls._call_count += 1
                        return step

            return provider_cls._build_auto_finalizer_response(profile, messages)

        # Auto-handle agent-loop continuation checks without consuming a
        # scripted step — just confirm the draft reply.
        if provider_cls._is_continuation_check_call(messages):
            return provider_cls._build_auto_continuation_response(profile, messages)

        if provider_cls._is_pre_write_grounding_check_call(messages):
            return provider_cls._build_auto_pre_write_grounding_response(
                profile, messages
            )

        if provider_cls._is_mission_planner_call(messages):
            return provider_cls._build_auto_mission_planner_response(
                profile, messages
            )

        if provider_cls._is_mission_verifier_call(messages):
            return provider_cls._build_auto_mission_verifier_response(
                profile, messages
            )

        step_index = provider_cls._call_count
        if step_index >= len(provider_cls._scripted_steps):
            raise AssertionError(
                "ScriptedFakeOllamaProvider received more chat calls than scripted steps."
            )
        provider_cls._call_count += 1
        step = provider_cls._scripted_steps[step_index]
        if callable(step):
            response = step(
                profile=profile,
                messages=messages,
                timeout_seconds=timeout_seconds,
                thinking_enabled=thinking_enabled,
                content_callback=content_callback,
                tools=tools,
            )
        else:
            response = step

        if (
            provider_cls._auto_stream_text
            and content_callback is not None
            and response.content
        ):
            content_callback(response.content)
        return response

    def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
        del timeout_seconds
        return True


def _real_ollama_enabled() -> bool:
    return os.environ.get("REAL_OLLAMA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore[no-untyped-def]
    del config
    if _real_ollama_enabled():
        return

    skip_real_ollama = pytest.mark.skip(
        reason="Set REAL_OLLAMA=1 to run real_ollama tests.",
    )
    for item in items:
        if "real_ollama" in item.keywords:
            item.add_marker(skip_real_ollama)


@pytest.fixture(autouse=True)
def _suppress_continuation_check():
    """Disable the agent-loop continuation check by default in tests.

    Tests that explicitly verify continuation behaviour should re-enable it
    via ``unclaw.core.agent_loop._continuation_check_enabled = True``.
    """
    import unclaw.core.agent_loop as _agent_loop

    original = _agent_loop._continuation_check_enabled
    _agent_loop._continuation_check_enabled = False
    yield
    _agent_loop._continuation_check_enabled = original


@pytest.fixture
def make_temp_project(tmp_path: Path):
    def _make_temp_project(
        *,
        allowed_chat_ids: list[int] | None = None,
        remove_secrets: bool = False,
        enabled_skill_ids: list[str] | None = None,
        install_skill_bundles: dict[str, str] | None = None,
    ) -> Path:
        """Create a temporary project directory for tests.

        Args:
            allowed_chat_ids: Override ``allowed_chat_ids`` in telegram.yaml.
            remove_secrets: Remove secrets.yaml from the temp project.
            enabled_skill_ids: Override ``skills.enabled_skill_ids`` in app.yaml.
                Defaults to ``[]`` so tests are isolated from the developer's
                local config (which may have skills enabled that are not installed
                in the temp project).
            install_skill_bundles: Mapping of ``skill_id → SKILL.md content`` to
                write synthetic skill bundles into ``project/skills/<skill_id>/``.
        """
        project_root = tmp_path / "project"
        shutil.copytree(_repo_root() / "config", project_root / "config")
        for local_override_path in (project_root / "config").glob("*.local.yaml"):
            local_override_path.unlink()

        # Always patch out enabled_skill_ids so tests are not tied to the
        # developer's personal config/app.yaml.  Pass enabled_skill_ids explicitly
        # when a test needs specific skills enabled.
        app_config_path = project_root / "config" / "app.yaml"
        app_payload = yaml.safe_load(app_config_path.read_text(encoding="utf-8"))
        if isinstance(app_payload, dict):
            if not isinstance(app_payload.get("skills"), dict):
                app_payload["skills"] = {}
            app_payload["skills"]["enabled_skill_ids"] = list(
                enabled_skill_ids if enabled_skill_ids is not None else []
            )
            app_config_path.write_text(
                yaml.safe_dump(app_payload, sort_keys=False),
                encoding="utf-8",
            )

        if allowed_chat_ids is not None:
            telegram_config_path = project_root / "config" / "telegram.yaml"
            payload = yaml.safe_load(telegram_config_path.read_text(encoding="utf-8"))
            assert isinstance(payload, dict)
            payload["allowed_chat_ids"] = allowed_chat_ids
            telegram_config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
        if remove_secrets:
            secrets_path = project_root / "config" / "secrets.yaml"
            if secrets_path.exists():
                secrets_path.unlink()

        if install_skill_bundles:
            skills_root = project_root / "skills"
            skills_root.mkdir(exist_ok=True)
            (skills_root / "__init__.py").write_text("", encoding="utf-8")
            for skill_id, skill_md_content in install_skill_bundles.items():
                bundle_dir = skills_root / skill_id
                bundle_dir.mkdir(parents=True, exist_ok=True)
                (bundle_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")

        return project_root

    return _make_temp_project


@pytest.fixture(scope="session")
def real_ollama_model_name() -> str:
    if not _real_ollama_enabled():
        pytest.skip("Set REAL_OLLAMA=1 to run real_ollama tests.")

    from unclaw.llm.ollama_provider import OllamaProvider

    provider = OllamaProvider()
    if not provider.is_available(timeout_seconds=5):
        pytest.fail(
            "REAL_OLLAMA=1 but Ollama is not reachable at http://127.0.0.1:11434."
        )

    available_models = provider.list_models(timeout_seconds=5)
    configured_model = os.environ.get("REAL_OLLAMA_MODEL", "").strip()
    if configured_model:
        if configured_model not in available_models:
            pytest.fail(
                "REAL_OLLAMA_MODEL={!r} is not installed. Available models: {}.".format(
                    configured_model,
                    ", ".join(available_models) or "[none]",
                )
            )
        return configured_model

    for model_name in available_models:
        model_base = model_name.split(":")[0].lower()
        if any(
            model_base.startswith(prefix)
            for prefix in _REAL_OLLAMA_TOOL_CAPABLE_MODEL_PREFIXES
        ):
            return model_name

    pytest.fail(
        "REAL_OLLAMA=1 but no tool-capable Ollama model is installed. "
        "Set REAL_OLLAMA_MODEL to a pulled model such as qwen3.5:4b."
    )


@pytest.fixture
def set_profile_tool_mode():
    def _set_profile_tool_mode(settings, profile_name: str, *, tool_mode: str) -> None:
        profile = settings.models[profile_name]
        settings.models[profile_name] = profile.__class__(
            name=profile.name,
            provider=profile.provider,
            model_name=profile.model_name,
            temperature=profile.temperature,
            thinking_supported=profile.thinking_supported,
            tool_mode=tool_mode,
            num_ctx=profile.num_ctx,
            keep_alive=profile.keep_alive,
            planner_profile=profile.planner_profile,
        )

    return _set_profile_tool_mode


@pytest.fixture
def build_scripted_ollama_provider():
    def _build(
        *steps: LLMResponse | Callable[..., LLMResponse],
        captured_messages: list[list[Any]] | None = None,
        captured_calls: list[dict[str, Any]] | None = None,
        auto_stream_text: bool = False,
    ) -> type[ScriptedFakeOllamaProvider]:
        return ScriptedFakeOllamaProvider.with_script(
            steps,
            captured_messages=captured_messages,
            captured_calls=captured_calls,
            auto_stream_text=auto_stream_text,
        )

    return _build


def _serialize_profile_for_models_config(profile: object) -> dict[str, object]:
    if isinstance(profile, Mapping):
        payload = dict(profile)
    else:
        payload = {
            "provider": getattr(profile, "provider"),
            "model_name": getattr(profile, "model_name"),
            "temperature": getattr(profile, "temperature"),
            "thinking_supported": getattr(profile, "thinking_supported"),
            "tool_mode": getattr(profile, "tool_mode"),
        }
        num_ctx = getattr(profile, "num_ctx", None)
        keep_alive = getattr(profile, "keep_alive", None)
        planner_profile = getattr(profile, "planner_profile", None)
        if num_ctx is not None:
            payload["num_ctx"] = num_ctx
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if planner_profile is not None:
            payload["planner_profile"] = planner_profile

    return payload


@pytest.fixture
def pack_profiles():
    def _pack_profiles(pack_name: str):
        return get_model_pack_profiles(pack_name)

    return _pack_profiles


@pytest.fixture
def write_models_config():
    def _write_models_config(
        project_root: Path,
        *,
        active_pack: str = DEV_MODEL_PACK_NAME,
        dev_profiles: Mapping[str, object] | None = None,
        dev_profile_pack: str = "power",
    ) -> dict[str, object]:
        profiles_source = (
            dev_profiles
            if dev_profiles is not None
            else get_model_pack_profiles(dev_profile_pack)
        )
        payload = {
            "active_pack": active_pack,
            "dev_profiles": {
                profile_name: _serialize_profile_for_models_config(profile)
                for profile_name, profile in profiles_source.items()
            },
        }
        models_path = project_root / "config" / "models.yaml"
        models_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        return payload

    return _write_models_config
