from __future__ import annotations

import pytest

from unclaw.core.runtime_support import (
    _build_completion_blocking_web_detail,
    _find_latest_completion_blocking_web_tool_result,
)
from unclaw.tools.contracts import ToolResult

pytestmark = pytest.mark.unit


def _thin_fast_web_result() -> ToolResult:
    return ToolResult.ok(
        tool_name="fast_web_search",
        output_text="Marine Leleu\n- Marine Leleu is a French endurance athlete.",
        payload={
            "query": "Marine Leleu",
            "result_count": 1,
            "match_quality": "exact",
            "supported_point_count": 1,
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
            "match_quality": "mismatch",
            "supported_point_count": 0,
            "grounding_note": "",
        },
    )


def test_structural_fast_grounding_mismatch_blocks_completion_without_text_parsing() -> None:
    tool_result = _find_latest_completion_blocking_web_tool_result(
        [
            _mismatch_fast_web_result(),
            ToolResult.ok(
                tool_name="write_text_file",
                output_text="Wrote text file: /tmp/note.txt",
                payload={"path": "/tmp/note.txt", "size": 10},
            ),
        ]
    )

    assert tool_result is not None
    assert (
        _build_completion_blocking_web_detail(tool_result)
        == "Quick web grounding matched a different entity or found no exact "
        "usable match."
    )


def test_structural_fast_grounding_thin_result_blocks_completion() -> None:
    tool_result = _find_latest_completion_blocking_web_tool_result(
        [
            _thin_fast_web_result(),
            ToolResult.ok(
                tool_name="write_text_file",
                output_text="Wrote text file: /tmp/note.txt",
                payload={"path": "/tmp/note.txt", "size": 10},
            ),
        ]
    )

    assert tool_result is not None
    assert (
        _build_completion_blocking_web_detail(tool_result)
        == "Quick web grounding was too thin to confirm requested details."
    )


def test_non_blocking_fast_grounding_is_ignored_when_supported_points_are_rich() -> None:
    rich_result = ToolResult.ok(
        tool_name="fast_web_search",
        output_text=(
            "Marine Leleu\n"
            "- Marine Leleu is a French endurance athlete.\n"
            "- Marine Leleu completed several public endurance challenges.\n"
        ),
        payload={
            "query": "Marine Leleu",
            "result_count": 3,
            "match_quality": "exact",
            "supported_point_count": 2,
            "grounding_note": "rich note",
        },
    )

    assert (
        _find_latest_completion_blocking_web_tool_result(
            [
                rich_result,
                ToolResult.ok(
                    tool_name="write_text_file",
                    output_text="Wrote text file: /tmp/note.txt",
                    payload={"path": "/tmp/note.txt", "size": 10},
                ),
            ]
        )
        is None
    )
