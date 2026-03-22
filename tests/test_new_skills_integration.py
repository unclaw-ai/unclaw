"""Integration tests for the git_repo and local_text_search skill bundles.

Covers:
- Both new skills are discoverable under ./skills/ alongside weather
- Both can be activated (tools registered) alongside weather in one registry
- All new tools are callable via ToolDispatcher and return ToolResults
- Tools return structured payloads
- Tools fail honestly on invalid inputs (via dispatcher)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.skills.bundle_tools import register_active_skill_tools
from unclaw.skills.file_loader import discover_skill_bundles, shipped_skill_bundle_root
from unclaw.tools.contracts import ToolCall
from unclaw.tools.dispatcher import ToolDispatcher
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Discovery: both new skills exist in ./skills/
# ---------------------------------------------------------------------------


def test_git_repo_skill_bundle_is_discoverable_in_shipped_skills_root() -> None:
    skills_root = shipped_skill_bundle_root()
    bundles = discover_skill_bundles(skills_root=skills_root)
    skill_ids = {b.skill_id for b in bundles}

    assert "git_repo" in skill_ids


def test_local_text_search_skill_bundle_is_discoverable_in_shipped_skills_root() -> None:
    skills_root = shipped_skill_bundle_root()
    bundles = discover_skill_bundles(skills_root=skills_root)
    skill_ids = {b.skill_id for b in bundles}

    assert "local_text_search" in skill_ids


def test_all_three_skills_discoverable_alongside_each_other() -> None:
    skills_root = shipped_skill_bundle_root()
    bundles = discover_skill_bundles(skills_root=skills_root)
    skill_ids = {b.skill_id for b in bundles}

    assert "weather" in skill_ids
    assert "git_repo" in skill_ids
    assert "local_text_search" in skill_ids


def test_skill_bundles_have_valid_skill_md_content() -> None:
    skills_root = shipped_skill_bundle_root()
    bundles = {b.skill_id: b for b in discover_skill_bundles(skills_root=skills_root)}

    for skill_id in ("git_repo", "local_text_search"):
        bundle = bundles[skill_id]
        assert bundle.display_name, f"{skill_id} has no display_name"
        assert bundle.summary, f"{skill_id} has no summary"
        assert bundle.tool_hints, f"{skill_id} has no tool hints"


# ---------------------------------------------------------------------------
# Activation: all three skills register cleanly together
# ---------------------------------------------------------------------------


def test_all_three_skills_activate_in_one_registry() -> None:
    registry = ToolRegistry()
    registered = register_active_skill_tools(
        registry,
        enabled_skill_ids=("weather", "git_repo", "local_text_search"),
    )

    assert "weather" in registered
    assert "git_repo" in registered
    assert "local_text_search" in registered


def test_git_repo_tools_present_after_activation() -> None:
    registry = ToolRegistry()
    register_active_skill_tools(registry, enabled_skill_ids=("git_repo",))

    tool_names = {t.name for t in registry.list_tools()}
    assert "git_status" in tool_names
    assert "git_recent_commits" in tool_names
    assert "git_diff_summary" in tool_names


def test_local_text_search_tool_present_after_activation() -> None:
    registry = ToolRegistry()
    register_active_skill_tools(registry, enabled_skill_ids=("local_text_search",))

    tool_names = {t.name for t in registry.list_tools()}
    assert "search_local_text" in tool_names


def test_skill_ownership_is_tracked_correctly() -> None:
    registry = ToolRegistry()
    register_active_skill_tools(
        registry,
        enabled_skill_ids=("git_repo", "local_text_search"),
    )

    assert registry.get_owner_skill_id("git_status") == "git_repo"
    assert registry.get_owner_skill_id("git_recent_commits") == "git_repo"
    assert registry.get_owner_skill_id("git_diff_summary") == "git_repo"
    assert registry.get_owner_skill_id("search_local_text") == "local_text_search"


# ---------------------------------------------------------------------------
# Callability: tools execute via ToolDispatcher
# ---------------------------------------------------------------------------


def test_git_status_callable_via_dispatcher() -> None:
    registry = ToolRegistry()
    register_active_skill_tools(registry, enabled_skill_ids=("git_repo",))
    dispatcher = ToolDispatcher(registry=registry)

    result = dispatcher.dispatch(
        ToolCall(
            tool_name="git_status",
            arguments={"repo_path": str(_repo_root())},
        )
    )

    assert result.tool_name == "git_status"
    assert result.success is True
    assert result.payload is not None


def test_git_recent_commits_callable_via_dispatcher() -> None:
    registry = ToolRegistry()
    register_active_skill_tools(registry, enabled_skill_ids=("git_repo",))
    dispatcher = ToolDispatcher(registry=registry)

    result = dispatcher.dispatch(
        ToolCall(
            tool_name="git_recent_commits",
            arguments={"repo_path": str(_repo_root()), "limit": 3},
        )
    )

    assert result.tool_name == "git_recent_commits"
    assert result.success is True
    assert result.payload is not None
    assert isinstance(result.payload["commits"], list)


def test_git_diff_summary_callable_via_dispatcher() -> None:
    registry = ToolRegistry()
    register_active_skill_tools(registry, enabled_skill_ids=("git_repo",))
    dispatcher = ToolDispatcher(registry=registry)

    result = dispatcher.dispatch(
        ToolCall(
            tool_name="git_diff_summary",
            arguments={"repo_path": str(_repo_root()), "target": "HEAD"},
        )
    )

    assert result.tool_name == "git_diff_summary"
    assert result.success is True
    assert result.payload is not None


def test_search_local_text_callable_via_dispatcher(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("integration test content\n", encoding="utf-8")
    registry = ToolRegistry()
    register_active_skill_tools(registry, enabled_skill_ids=("local_text_search",))
    dispatcher = ToolDispatcher(registry=registry)

    result = dispatcher.dispatch(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "integration test", "root": str(tmp_path)},
        )
    )

    assert result.tool_name == "search_local_text"
    assert result.success is True
    assert result.payload is not None
    assert result.payload["total_matches_found"] == 1


# ---------------------------------------------------------------------------
# Honest failure via dispatcher
# ---------------------------------------------------------------------------


def test_git_status_fails_honestly_on_invalid_path_via_dispatcher() -> None:
    registry = ToolRegistry()
    register_active_skill_tools(registry, enabled_skill_ids=("git_repo",))
    dispatcher = ToolDispatcher(registry=registry)

    result = dispatcher.dispatch(
        ToolCall(
            tool_name="git_status",
            arguments={"repo_path": "/absolutely/nonexistent/path/xyz"},
        )
    )

    assert result.tool_name == "git_status"
    assert result.success is False
    assert result.error is not None


def test_search_local_text_fails_honestly_on_empty_query_via_dispatcher(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    register_active_skill_tools(registry, enabled_skill_ids=("local_text_search",))
    dispatcher = ToolDispatcher(registry=registry)

    result = dispatcher.dispatch(
        ToolCall(
            tool_name="search_local_text",
            arguments={"query": "", "root": str(tmp_path)},
        )
    )

    assert result.tool_name == "search_local_text"
    assert result.success is False
    assert result.error is not None
