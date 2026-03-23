"""Integration tests for the catalog-only skill architecture.

Skills are no longer bundled with the main repo.  These tests verify the
runtime behaviour with synthetic skill bundles installed in a temp directory,
simulating what happens after catalog installation.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from unclaw.skills.bundle_tools import register_active_skill_tools
from unclaw.skills.file_loader import discover_skill_bundles, local_skill_install_root
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolPermissionLevel
from unclaw.tools.dispatcher import ToolDispatcher
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill_bundle(
    skills_root: Path,
    skill_id: str,
    skill_md: str,
    *,
    tool_py: str | None = None,
) -> Path:
    bundle_dir = skills_root / skill_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    if tool_py is not None:
        (bundle_dir / "tool.py").write_text(tool_py, encoding="utf-8")
    return bundle_dir


# ---------------------------------------------------------------------------
# Empty install root
# ---------------------------------------------------------------------------


def test_local_skill_install_root_exists_in_repo() -> None:
    """The skills/ directory must exist (even if empty after a fresh checkout)."""
    root = local_skill_install_root()
    assert root.is_dir()


def test_discover_skill_bundles_returns_empty_for_empty_root(tmp_path: Path) -> None:
    """Empty install directory → no bundles discovered."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    assert discover_skill_bundles(skills_root=skills_root) == ()


def test_discover_skill_bundles_returns_empty_when_root_missing(tmp_path: Path) -> None:
    assert discover_skill_bundles(skills_root=tmp_path / "nonexistent") == ()


# ---------------------------------------------------------------------------
# Synthetic skill bundles — simulates post-catalog-install state
# ---------------------------------------------------------------------------


def test_installed_skill_is_discovered(tmp_path: Path) -> None:
    """A skill installed into skills_root is discovered correctly."""
    skills_root = tmp_path / "skills"
    _write_skill_bundle(skills_root, "alpha", "# Alpha\n\nAlpha skill.\n")

    bundles = discover_skill_bundles(skills_root=skills_root)
    assert len(bundles) == 1
    assert bundles[0].skill_id == "alpha"


def test_multiple_installed_skills_all_discovered(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    for sid in ("bravo", "alpha", "charlie"):
        _write_skill_bundle(skills_root, sid, f"# {sid.title()}\n\nSummary.\n")

    bundles = discover_skill_bundles(skills_root=skills_root)
    # Sorted alphabetically
    assert tuple(b.skill_id for b in bundles) == ("alpha", "bravo", "charlie")


def test_skill_bundle_with_tool_registers_its_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synthetic skill with a valid register_skill_tools hook registers cleanly."""
    skills_root = tmp_path / "skills"
    _write_skill_bundle(
        skills_root,
        "synth",
        "# Synth\n\nSynthetic skill.\nTool hints: Use synth_action.\n",
        tool_py="# placeholder",
    )

    # Inject a fake module implementing the hook
    module_name = "skills.synth.tool"
    fake_mod = types.ModuleType(module_name)
    _def = ToolDefinition(
        name="synth_action",
        description="synthetic tool",
        permission_level=ToolPermissionLevel.NETWORK,
        arguments={},
    )

    def _register(registry) -> None:
        registry.register(_def, lambda call: None)

    fake_mod.register_skill_tools = _register  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake_mod)

    registry = ToolRegistry()
    registered = register_active_skill_tools(
        registry,
        enabled_skill_ids=("synth",),
        skills_root=skills_root,
    )

    assert "synth" in registered
    assert any(t.name == "synth_action" for t in registry.list_tools())


def test_skill_ownership_tracked_per_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_bundle(skills_root, "s1", "# S1\n\nSkill 1.\n", tool_py="# s1")
    _write_skill_bundle(skills_root, "s2", "# S2\n\nSkill 2.\n", tool_py="# s2")

    def _make_module(module_name: str, tool_name: str) -> types.ModuleType:
        mod = types.ModuleType(module_name)
        _d = ToolDefinition(
            name=tool_name,
            description="x",
            permission_level=ToolPermissionLevel.NETWORK,
            arguments={},
        )

        def _register(registry) -> None:
            registry.register(_d, lambda call: None)

        mod.register_skill_tools = _register  # type: ignore[attr-defined]
        return mod

    monkeypatch.setitem(sys.modules, "skills.s1.tool", _make_module("skills.s1.tool", "tool_s1"))
    monkeypatch.setitem(sys.modules, "skills.s2.tool", _make_module("skills.s2.tool", "tool_s2"))

    registry = ToolRegistry()
    register_active_skill_tools(
        registry,
        enabled_skill_ids=("s1", "s2"),
        skills_root=skills_root,
    )

    assert registry.get_owner_skill_id("tool_s1") == "s1"
    assert registry.get_owner_skill_id("tool_s2") == "s2"


def test_runtime_works_with_zero_installed_skills(tmp_path: Path) -> None:
    """Registry creation with empty skills dir and no enabled IDs raises nothing."""
    registry = ToolRegistry()
    registered = register_active_skill_tools(
        registry,
        enabled_skill_ids=(),
        skills_root=tmp_path / "empty_skills",
    )
    assert registered == ()
    assert registry.list_tools() == []
