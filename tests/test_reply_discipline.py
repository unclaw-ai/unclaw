from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.reply_discipline import (
    _build_all_failed_tool_reply,
    _build_grounded_reply_facts,
    _build_structural_finalization_fallback,
)
from unclaw.core.session_manager import SessionGoalState, SessionProgressEntry
from unclaw.tools.file_tools import WRITE_TEXT_FILE_DEFINITION
from unclaw.tools.contracts import ToolResult
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

pytestmark = pytest.mark.unit


def test_reply_discipline_module_has_no_semantic_regex_machinery() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "unclaw"
        / "core"
        / "reply_discipline.py"
    )
    source_text = module_path.read_text(encoding="utf-8")

    forbidden_fragments = (
        "import re",
        "_TASK_STATUS_REQUEST_PATTERN",
        "_LIMITATION_ACK_PATTERN",
        "_FILE_SUCCESS_CLAIM_PATTERN",
        "_SEARCH_ACTION_CLAIM_PATTERN",
        "_RESEARCH_COMPLETION_CLAIM_PATTERN",
        "_TASK_COMPLETION_CLAIM_PATTERN",
        "_ACTIVE_STATUS_CLAIM_PATTERN",
        "_BLOCKED_STATUS_CLAIM_PATTERN",
        "_PROMISED_",
        "_looks_like_task_status_request",
        "_reply_claims_",
        "_reply_acknowledges_limitations",
    )

    for fragment in forbidden_fragments:
        assert fragment not in source_text


def test_build_all_failed_tool_reply_distinguishes_timeout_from_other_failures() -> None:
    assert (
        _build_all_failed_tool_reply(
            tool_results=[
                ToolResult.failure(
                    tool_name="search_web",
                    error="Tool 'search_web' timed out after 5 seconds.",
                    payload={"execution_state": "timed_out"},
                ),
            ],
        )
        == "The tool step timed out, so I couldn't confirm the requested details "
        "from retrieved tool evidence."
    )
    assert (
        _build_all_failed_tool_reply(
            tool_results=[
                ToolResult.failure(tool_name="search_web", error="network failure"),
            ],
        )
        == "The tool step failed, so I couldn't confirm the requested details "
        "from retrieved tool evidence."
    )


def test_build_grounded_reply_facts_packages_structural_runtime_evidence() -> None:
    facts = _build_grounded_reply_facts(
        user_input="Etat de la tache ?",
        assistant_draft_reply="C'est termine.",
        tool_results=[
            ToolResult.ok(
                tool_name="write_text_file",
                output_text="Wrote text file: /tmp/note.txt",
                payload={"path": "/tmp/note.txt", "size": 12},
            ),
        ],
        session_goal_state=SessionGoalState(
            goal="Write a short local note file.",
            status="blocked",
            current_step="write_text_file",
            last_blocker="Permission denied: /tmp/note.txt",
            updated_at="2026-03-26T10:00:00Z",
        ),
        session_progress_ledger=(
            SessionProgressEntry(
                status="blocked",
                step="write_text_file",
                detail="Permission denied: /tmp/note.txt",
                updated_at="2026-03-26T10:00:00Z",
            ),
        ),
    )

    assert facts["user_input"] == "Etat de la tache ?"
    assert facts["assistant_draft_reply"] == "C'est termine."
    assert facts["current_turn_tool_summary"]["tool_count"] == 1
    assert facts["current_turn_tool_summary"]["write_text_file_succeeded"] is True
    assert facts["persisted_goal_state_before_turn"] == {
        "goal": "Write a short local note file.",
        "status": "blocked",
        "current_step": "write_text_file",
        "last_blocker": "Permission denied: /tmp/note.txt",
        "updated_at": "2026-03-26T10:00:00Z",
    }
    assert facts["persisted_progress_ledger_before_turn"] == (
        {
            "status": "blocked",
            "step": "write_text_file",
            "detail": "Permission denied: /tmp/note.txt",
            "updated_at": "2026-03-26T10:00:00Z",
        },
    )


def test_build_grounded_reply_facts_includes_no_tool_execution_claim_risks() -> None:
    facts = _build_grounded_reply_facts(
        user_input="Create a note and research the topic.",
        assistant_draft_reply="Done.",
        tool_results=[],
        available_tool_definitions=(
            WRITE_TEXT_FILE_DEFINITION,
            SEARCH_WEB_DEFINITION,
        ),
        no_tool_execution_claim_risks={
            "unsupported_execution_claim_risk": True,
            "completion_without_execution_risk": True,
            "multi_deliverable_request_risk": True,
            "no_tool_honesty_rescue_used": True,
        },
    )

    assert facts["execution_claim_risks"] == {
        "no_tools_ran_this_turn": True,
        "side_effect_tools_available": True,
        "side_effect_tool_names": ("write_text_file",),
        "evidence_gathering_tools_available": True,
        "evidence_gathering_tool_names": ("search_web",),
        "persisted_goal_state_status": "none",
        "persisted_progress_entry_count": 0,
        "completion_without_execution_risk": True,
        "unsupported_execution_claim_risk": True,
        "multi_deliverable_request_risk": True,
        "no_tool_honesty_rescue_used": True,
    }
    assert facts["completion_risks"]["deliverable_check_required"] is True


def test_structural_finalization_fallback_uses_persisted_task_state_summary() -> None:
    result = _build_structural_finalization_fallback(
        reply="The task is completed.",
        tool_results=[],
        session_goal_state=SessionGoalState(
            goal="Write a short local note file.",
            status="blocked",
            current_step="write_text_file",
            last_blocker="Permission denied: /tmp/note.txt",
            updated_at="2026-03-26T10:00:00Z",
        ),
    )

    assert (
        result
        == "The persisted task is blocked on write_text_file: "
        "Permission denied: /tmp/note.txt"
    )


def test_structural_finalization_fallback_stays_generic_when_no_tools_ran() -> None:
    result = _build_structural_finalization_fallback(
        reply="I created the biography file.",
        tool_results=[],
        finalization_required=True,
    )

    assert (
        result
        == "I did not execute any tools in this turn, so I cannot confirm "
        "additional completed actions."
    )
