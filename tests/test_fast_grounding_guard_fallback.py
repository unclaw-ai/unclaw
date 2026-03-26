from __future__ import annotations

import pytest

from unclaw.constants import EMPTY_RESPONSE_REPLY
from unclaw.core.reply_discipline import _apply_post_tool_reply_discipline
from unclaw.tools.contracts import ToolResult

pytestmark = pytest.mark.unit


def _thin_fast_web_result() -> ToolResult:
    return ToolResult.ok(
        tool_name="fast_web_search",
        output_text="Marine Leleu\n- Marine Leleu is a French endurance athlete.",
        payload={
            "query": "Marine Leleu",
            "result_count": 1,
            "grounding_note": (
                "Marine Leleu\n- Marine Leleu is a French endurance athlete."
            ),
        },
    )


def _mismatch_fast_web_result() -> ToolResult:
    return ToolResult.ok(
        tool_name="fast_web_search",
        output_text="No exact top match; different entity found.",
        payload={
            "query": "Marine Leleu",
            "result_count": 1,
            "grounding_note": "",
        },
    )


def test_substantive_reply_is_clamped_for_thin_fast_grounding() -> None:
    reply = (
        "Marine Leleu is a French endurance athlete, author, speaker, and "
        "podcast host with a much broader biography."
    )

    result = _apply_post_tool_reply_discipline(
        reply=reply,
        user_input="Tell me everything you know about Marine Leleu.",
        tool_results=[_thin_fast_web_result()],
    )

    assert (
        result
        == "Marine Leleu is a French endurance athlete. "
        "I couldn't confirm a fuller biography from that quick grounding probe alone."
    )


def test_substantive_reply_is_clamped_for_mismatch_fast_grounding() -> None:
    reply = "Marine Leleu is a politician with a long public career."

    result = _apply_post_tool_reply_discipline(
        reply=reply,
        user_input="Who is Marine Leleu?",
        tool_results=[_mismatch_fast_web_result()],
    )

    assert (
        result
        == "The quick web grounding appeared to match a different entity, so I "
        "couldn't confirm the requested details from that result alone."
    )


@pytest.mark.parametrize(
    ("reply", "tool_result", "expected"),
    (
        (
            "",
            _thin_fast_web_result(),
            "Marine Leleu is a French endurance athlete. "
            "I couldn't confirm a fuller biography from that quick grounding probe alone.",
        ),
        (
            EMPTY_RESPONSE_REPLY,
            _mismatch_fast_web_result(),
            "The quick web grounding appeared to match a different entity, so I "
            "couldn't confirm the requested details from that result alone.",
        ),
    ),
)
def test_effectively_empty_reply_still_uses_guarded_fast_grounding_fallback(
    reply: str,
    tool_result: ToolResult,
    expected: str,
) -> None:
    result = _apply_post_tool_reply_discipline(
        reply=reply,
        user_input="Who is Marine Leleu?",
        tool_results=[tool_result],
    )

    assert result == expected


@pytest.mark.parametrize(
    ("reply", "tool_result", "expected"),
    (
        (
            "I could not confirm a full biography from the quick result. "
            "The quick result stayed limited. "
            "I do not want to overstate it.",
            _thin_fast_web_result(),
            "Marine Leleu is a French endurance athlete. "
            "I couldn't confirm a fuller biography from that quick grounding probe alone.",
        ),
        (
            "I could not confirm the identity from that probe alone. "
            "The result looked uncertain and possibly mismatched. "
            "I do not want to guess from a weak quick-grounding result. "
            "I cannot verify the requested details from it.",
            _mismatch_fast_web_result(),
            "The quick web grounding appeared to match a different entity, so I "
            "couldn't confirm the requested details from that result alone.",
        ),
    ),
)
def test_limitation_style_reply_can_still_use_guarded_fast_grounding_fallback(
    reply: str,
    tool_result: ToolResult,
    expected: str,
) -> None:
    result = _apply_post_tool_reply_discipline(
        reply=reply,
        user_input="Who is Marine Leleu?",
        tool_results=[tool_result],
    )

    assert result == expected


def test_all_failed_fallback_is_unchanged() -> None:
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
