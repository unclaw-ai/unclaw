"""Tests for the local_text_search skill bundle.

Covers:
- search_local_text finds matches and returns structured payload
- Correct file_path, line_number, line_text, and snippet in each match
- Returns no matches (success=True) when nothing is found
- Fails honestly on empty query
- Fails honestly on nonexistent root
- Fails honestly on a root that is a file, not a directory
- Skips binary files transparently
- Respects the extensions filter
- Bounds max_results correctly (default and clamped)
- truncated flag is set when total_matches_found > max_results
- extensions_filter is None when no filter was provided
- register_skill_tools registers search_local_text
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip this entire module when the local_text_search skill is not installed locally.
# Skills are no longer bundled; install via `unclaw onboard` to run these tests.
pytest.importorskip(
    "skills.local_text_search.tool",
    reason="local_text_search skill not installed locally",
)

from skills.local_text_search.tool import (
    SEARCH_LOCAL_TEXT_DEFINITION,
    _DEFAULT_MAX_RESULTS,
    _MAX_MAX_RESULTS,
    _MIN_MAX_RESULTS,
    _build_snippet,
    _is_binary,
    _normalize_extensions,
    register_skill_tools,
    search_local_text,
)
from unclaw.tools.contracts import ToolCall
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# register_skill_tools
# ---------------------------------------------------------------------------


def test_register_skill_tools_registers_search_local_text() -> None:
    registry = ToolRegistry()
    register_skill_tools(registry)

    tool_names = {t.name for t in registry.list_tools()}
    assert SEARCH_LOCAL_TEXT_DEFINITION.name in tool_names


# ---------------------------------------------------------------------------
# search_local_text — happy paths
# ---------------------------------------------------------------------------


def test_search_finds_match_and_returns_structured_payload(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("Hello, world!\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "world", "root": str(tmp_path)},
        )
    )

    assert result.success is True
    assert result.payload is not None
    payload = result.payload
    assert payload["query"] == "world"
    assert payload["root"] == str(tmp_path)
    assert payload["total_matches_found"] == 1
    assert payload["truncated"] is False
    assert len(payload["matches"]) == 1

    match = payload["matches"][0]
    assert match["line_number"] == 1
    assert "world" in match["line_text"].lower()
    assert "world" in match["snippet"].lower()


def test_search_returns_correct_line_number(tmp_path: Path) -> None:
    content = "line one\nline two\nfind me here\nline four\n"
    (tmp_path / "file.txt").write_text(content, encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "find me", "root": str(tmp_path)},
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["matches"][0]["line_number"] == 3


def test_search_returns_empty_matches_when_nothing_found(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("unrelated content\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "xyz_not_present", "root": str(tmp_path)},
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["total_matches_found"] == 0
    assert result.payload["matches"] == []
    assert "No matches" in result.output_text


def test_search_is_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "file.py").write_text("UPPER lower Mixed\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "upper", "root": str(tmp_path)},
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["total_matches_found"] == 1


def test_search_recurses_into_subdirectories(tmp_path: Path) -> None:
    subdir = tmp_path / "sub" / "nested"
    subdir.mkdir(parents=True)
    (subdir / "deep.txt").write_text("deep match\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "deep match", "root": str(tmp_path)},
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["total_matches_found"] == 1
    assert str(subdir / "deep.txt") == result.payload["matches"][0]["file_path"]


def test_search_counts_multiple_matches_across_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("target here\ntarget again\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("no match\ntarget once\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "target", "root": str(tmp_path)},
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["total_matches_found"] == 3


# ---------------------------------------------------------------------------
# extensions filter
# ---------------------------------------------------------------------------


def test_search_respects_extensions_filter(tmp_path: Path) -> None:
    (tmp_path / "match.py").write_text("find this\n", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("find this\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={
                "query": "find this",
                "root": str(tmp_path),
                "extensions": [".py"],
            },
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["total_matches_found"] == 1
    assert result.payload["matches"][0]["file_path"].endswith(".py")


def test_search_extensions_filter_normalizes_missing_dot(tmp_path: Path) -> None:
    (tmp_path / "file.md").write_text("needle\n", encoding="utf-8")
    (tmp_path / "file.py").write_text("needle\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={
                "query": "needle",
                "root": str(tmp_path),
                "extensions": ["md"],  # no leading dot
            },
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["total_matches_found"] == 1
    assert result.payload["matches"][0]["file_path"].endswith(".md")


def test_search_extensions_filter_is_none_when_not_provided(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("hello\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "hello", "root": str(tmp_path)},
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["extensions_filter"] is None


def test_search_extensions_filter_is_set_when_provided(tmp_path: Path) -> None:
    (tmp_path / "file.py").write_text("hello\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={
                "query": "hello",
                "root": str(tmp_path),
                "extensions": [".py"],
            },
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["extensions_filter"] == [".py"]


# ---------------------------------------------------------------------------
# max_results and truncation
# ---------------------------------------------------------------------------


def test_search_respects_max_results(tmp_path: Path) -> None:
    lines = "\n".join(f"match on line {i}" for i in range(30))
    (tmp_path / "big.txt").write_text(lines + "\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "match", "root": str(tmp_path), "max_results": 5},
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert len(result.payload["matches"]) == 5
    assert result.payload["total_matches_found"] == 30
    assert result.payload["truncated"] is True
    assert "truncated" in result.output_text


def test_search_clamps_max_results_to_max(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("line\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "line", "root": str(tmp_path), "max_results": 9999},
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["max_results"] == _MAX_MAX_RESULTS


def test_search_clamps_max_results_to_min(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("match\nmatch\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "match", "root": str(tmp_path), "max_results": 0},
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["max_results"] == _MIN_MAX_RESULTS
    assert len(result.payload["matches"]) <= _MIN_MAX_RESULTS


def test_search_default_max_results_is_twenty(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(tool_name="search_local_text", arguments={"query": "x", "root": str(tmp_path)})
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["max_results"] == _DEFAULT_MAX_RESULTS


# ---------------------------------------------------------------------------
# Binary file skipping
# ---------------------------------------------------------------------------


def test_search_skips_binary_files(tmp_path: Path) -> None:
    binary_file = tmp_path / "data.bin"
    binary_file.write_bytes(b"match\x00this\x00binary")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={
                "query": "match",
                "root": str(tmp_path),
                "extensions": [".bin"],
            },
        )
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["total_matches_found"] == 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_search_fails_on_empty_query(tmp_path: Path) -> None:
    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "", "root": str(tmp_path)},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert "query" in result.error.lower() or "empty" in result.error.lower()


def test_search_fails_on_whitespace_only_query(tmp_path: Path) -> None:
    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "   ", "root": str(tmp_path)},
        )
    )

    assert result.success is False
    assert result.error is not None


def test_search_fails_on_nonexistent_root() -> None:
    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "anything", "root": "/nonexistent/path/xyz999"},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert "exist" in result.error.lower() or "not" in result.error.lower()


def test_search_fails_when_root_is_a_file(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("content\n", encoding="utf-8")

    result = search_local_text(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "content", "root": str(f)},
        )
    )

    assert result.success is False
    assert result.error is not None


# ---------------------------------------------------------------------------
# Internal helpers (unit)
# ---------------------------------------------------------------------------


def test_is_binary_returns_true_for_null_byte_file(tmp_path: Path) -> None:
    f = tmp_path / "bin.dat"
    f.write_bytes(b"hello\x00world")
    assert _is_binary(f) is True


def test_is_binary_returns_false_for_text_file(tmp_path: Path) -> None:
    f = tmp_path / "text.txt"
    f.write_text("plain text content\n", encoding="utf-8")
    assert _is_binary(f) is False


def test_normalize_extensions_adds_leading_dot() -> None:
    result = _normalize_extensions(["py", ".md", "TXT"])
    assert result is not None
    assert ".py" in result
    assert ".md" in result
    assert ".txt" in result


def test_normalize_extensions_returns_none_for_none() -> None:
    assert _normalize_extensions(None) is None


def test_normalize_extensions_returns_none_for_empty_list() -> None:
    assert _normalize_extensions([]) is None


def test_build_snippet_centers_on_match() -> None:
    line = "the quick brown fox jumps over the lazy dog"
    snippet = _build_snippet(line, "fox", 10)
    assert "fox" in snippet


def test_build_snippet_adds_ellipsis_when_truncated() -> None:
    line = "a" * 50 + "TARGET" + "b" * 50
    snippet = _build_snippet(line, "TARGET", 20)
    assert "TARGET" in snippet
    # With short context, both sides should be truncated
    assert "\u2026" in snippet
