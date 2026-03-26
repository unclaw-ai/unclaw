from __future__ import annotations

import pytest

from unclaw.core.reply_discipline import (
    _apply_post_tool_reply_discipline,
    _build_all_failed_tool_reply,
    _build_fast_grounding_guarded_reply,
)
from unclaw.core.session_manager import SessionGoalState
from unclaw.tools.contracts import ToolResult

pytestmark = pytest.mark.unit


def _fast_web_result(
    *,
    query: str = "Marine Leleu",
    grounding_note: str = "Marine Leleu\n- Marine Leleu is a French endurance athlete.",
    result_count: int = 1,
    output_text: str | None = None,
) -> ToolResult:
    return ToolResult.ok(
        tool_name="fast_web_search",
        output_text=output_text or grounding_note,
        payload={
            "query": query,
            "result_count": result_count,
            "grounding_note": grounding_note,
        },
    )


def _search_web_result() -> ToolResult:
    return ToolResult.ok(
        tool_name="search_web",
        output_text="Grounded search result.",
        payload={
            "query": "Ada Lovelace",
            "evidence_count": 4,
            "finding_count": 2,
            "display_sources": [
                {"title": "Britannica", "url": "https://example.com/britannica"},
                {"title": "History", "url": "https://example.com/history"},
            ],
        },
    )


def test_reply_discipline_uses_failed_tool_fallback_when_all_tools_fail() -> None:
    result = _apply_post_tool_reply_discipline(
        reply="Here is the answer.",
        user_input="Who is Ada Lovelace?",
        tool_results=[
            ToolResult.failure(tool_name="search_web", error="network failure"),
        ],
    )

    assert (
        result
        == "The tool step failed, so I couldn't confirm the requested details "
        "from retrieved tool evidence."
    )


@pytest.mark.parametrize(
    "user_input",
    (
        "Who is Ada Lovelace?",
        "Qui est Ada Lovelace ?",
        "Quien es Ada Lovelace?",
    ),
)
def test_all_failed_tool_builder_is_language_neutral_across_user_inputs(
    user_input: str,
) -> None:
    result = _build_all_failed_tool_reply(
        user_input=user_input,
        tool_results=[
            ToolResult.failure(tool_name="search_web", error="network failure"),
        ],
    )

    assert (
        result
        == "The tool step failed, so I couldn't confirm the requested details "
        "from retrieved tool evidence."
    )


def test_reply_discipline_uses_language_neutral_timeout_fallback_when_all_tools_fail() -> None:
    result = _apply_post_tool_reply_discipline(
        reply="Je vais repondre quand meme.",
        user_input="Qui est Marine Leleu ?",
        tool_results=[
            ToolResult.failure(
                tool_name="search_web",
                error="search_web timed out after 5 seconds.",
            ),
        ],
    )

    assert (
        result
        == "The tool step timed out, so I couldn't confirm the requested details "
        "from retrieved tool evidence."
    )


def test_reply_discipline_clamps_substantive_reply_for_thin_fast_web_results() -> None:
    reply = (
        "Marine Leleu is a French endurance athlete, author, speaker, and "
        "podcast host with a much broader biography."
    )
    result = _apply_post_tool_reply_discipline(
        reply=reply,
        user_input="Tell me everything you know about Marine Leleu.",
        tool_results=[_fast_web_result()],
    )

    assert (
        result
        == "Marine Leleu is a French endurance athlete. "
        "I couldn't confirm a fuller biography from that quick grounding probe alone."
    )


@pytest.mark.parametrize(
    "user_input",
    (
        "Who is Marine Leleu?",
        "Qui est Marine Leleu ?",
        "Quien es Marine Leleu?",
    ),
)
def test_fast_grounding_guarded_reply_is_language_neutral_across_user_inputs(
    user_input: str,
) -> None:
    result = _build_fast_grounding_guarded_reply(
        candidate_reply=(
            "Marine Leleu is a French endurance athlete. "
            "She also has a much broader biography."
        ),
        user_input=user_input,
        fast_results=[_fast_web_result()],
    )

    assert (
        result
        == "Marine Leleu is a French endurance athlete. "
        "I couldn't confirm a fuller biography from that quick grounding probe alone."
    )


def test_reply_discipline_clamps_substantive_reply_for_fast_web_entity_mismatches() -> None:
    reply = "Marine Leleu is a politician with a long public career."
    result = _apply_post_tool_reply_discipline(
        reply=reply,
        user_input="Who is Marine Leleu?",
        tool_results=[
            _fast_web_result(
                grounding_note="",
                output_text="No exact top match; different entity found.",
            ),
        ],
    )

    assert (
        result
        == "The quick web grounding appeared to match a different entity, so I "
        "couldn't confirm the requested details from that result alone."
    )


def test_reply_discipline_preserves_normal_grounded_replies() -> None:
    reply = "Ada Lovelace was an English mathematician and writer."

    result = _apply_post_tool_reply_discipline(
        reply=reply,
        user_input="Who is Ada Lovelace?",
        tool_results=[_search_web_result()],
    )

    assert result == reply


def test_reply_discipline_preserves_existing_limitation_acknowledgement() -> None:
    reply = "I couldn't confirm a fuller biography from that quick result alone."

    result = _apply_post_tool_reply_discipline(
        reply=reply,
        user_input="Tell me everything you know about Marine Leleu.",
        tool_results=[_fast_web_result()],
    )

    assert result == reply


def test_reply_discipline_blocks_file_write_claim_without_successful_write() -> None:
    result = _apply_post_tool_reply_discipline(
        reply="I saved the note locally.",
        user_input="Write the biography file.",
        tool_results=[],
    )

    assert result == "I haven't created or saved the requested file yet."


def test_reply_discipline_blocks_research_completion_claim_without_grounded_search() -> None:
    result = _apply_post_tool_reply_discipline(
        reply="I completed the research and the biography is complete.",
        user_input="Research this person in detail.",
        tool_results=[],
    )

    assert result == "I haven't run grounded search for that yet."


def test_reply_discipline_rewrites_unexecuted_verify_promise() -> None:
    result = _apply_post_tool_reply_discipline(
        reply="I will now verify the file.",
        user_input="Check the file.",
        tool_results=[],
    )

    assert result == "I haven't verified that file yet."


def test_reply_discipline_grounds_status_reply_when_no_persisted_task_exists() -> None:
    result = _apply_post_tool_reply_discipline(
        reply="The task is still in progress.",
        user_input="Where does this task stand?",
        tool_results=[],
    )

    assert result == "There is no persisted task progress for this session."


def test_reply_discipline_grounds_status_reply_when_persisted_task_conflicts() -> None:
    result = _apply_post_tool_reply_discipline(
        reply="The task is completed.",
        user_input="Where does this task stand?",
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
