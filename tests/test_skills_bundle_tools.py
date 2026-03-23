"""Tests for generic file-first skill bundle tool loading.

Covers:
- register_active_skill_tools with a bundle that has register_skill_tools
- register_active_skill_tools skips bundles without tool.py
- register_active_skill_tools skips bundles whose tool.py lacks the hook
- probe_skill_tool_loading returns None for ok bundles, error string on failure
- startup Skills check appears in build_startup_report when skills are enabled
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from unclaw.skills.bundle_tools import probe_skill_tool_loading, register_active_skill_tools
from unclaw.skills.file_loader import load_skill_bundle
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill_bundle(
    skills_root: Path,
    skill_id: str,
    skill_md_text: str,
    *,
    tool_py_text: str | None = None,
) -> Path:
    bundle_dir = skills_root / skill_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "SKILL.md").write_text(skill_md_text, encoding="utf-8")
    if tool_py_text is not None:
        (bundle_dir / "tool.py").write_text(tool_py_text, encoding="utf-8")
    return bundle_dir


# ---------------------------------------------------------------------------
# register_active_skill_tools
# ---------------------------------------------------------------------------


def test_register_active_skill_tools_with_injected_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bundle with a valid register_skill_tools hook registers its tool."""
    skills_root = tmp_path / "skills"
    _write_skill_bundle(
        skills_root,
        "alpha",
        "# Alpha\n\nAlpha skill.\n",
        tool_py_text=(
            "from unclaw.tools.contracts import ToolDefinition, ToolPermissionLevel\n"
            "_DEF = ToolDefinition(name='alpha_tool', description='x', "
            "permission_level=ToolPermissionLevel.NETWORK)\n"
            "def register_skill_tools(registry) -> None:\n"
            "    registry.register(_DEF, lambda call: None)\n"
        ),
    )

    # Inject a fake module so importlib finds it from sys.modules
    module_name = "skills.alpha.tool"
    fake_module = types.ModuleType(module_name)
    from unclaw.tools.contracts import ToolDefinition, ToolPermissionLevel
    _def = ToolDefinition(
        name="alpha_tool",
        description="x",
        permission_level=ToolPermissionLevel.NETWORK,
        arguments={},
    )

    def _register(registry) -> None:
        registry.register(_def, lambda call: None)

    fake_module.register_skill_tools = _register  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    from unclaw.skills.file_loader import discover_skill_bundles
    bundles = discover_skill_bundles(skills_root=skills_root)

    import unclaw.skills.bundle_tools as bt
    monkeypatch.setattr(bt, "load_active_skill_bundles", lambda **kwargs: tuple(bundles))

    registry = ToolRegistry()
    registered = register_active_skill_tools(
        registry,
        enabled_skill_ids=("alpha",),
        skills_root=skills_root,
    )

    assert "alpha" in registered
    tool_names = {tool.name for tool in registry.list_tools()}
    assert "alpha_tool" in tool_names


def test_register_active_skill_tools_skips_bundle_without_tool_py(
    tmp_path: Path,
) -> None:
    """A prompt-only bundle (no tool.py) is silently skipped."""
    skills_root = tmp_path / "skills"
    _write_skill_bundle(skills_root, "promptonly", "# Prompt Only\n\nJust prompt.\n")
    from unclaw.skills.file_loader import discover_skill_bundles

    bundles = discover_skill_bundles(skills_root=skills_root)
    assert len(bundles) == 1
    bundle = bundles[0]

    registry = ToolRegistry()
    from unclaw.skills.bundle_tools import _register_bundle_tools

    result = _register_bundle_tools(registry, bundle)

    assert result is False
    assert registry.list_tools() == []


def test_register_bundle_tools_skips_tool_py_without_register_skill_tools_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_bundle(
        skills_root,
        "nohook",
        "# No Hook\n\nSkill.\n",
        tool_py_text="# tool.py without the hook\nSOME_VAR = 42\n",
    )
    from unclaw.skills.file_loader import discover_skill_bundles
    from unclaw.skills.bundle_tools import _register_bundle_tools

    bundles = discover_skill_bundles(skills_root=skills_root)
    bundle = bundles[0]

    # Inject a fake module so importlib doesn't try to load from sys.path
    fake_module = types.ModuleType("skills.nohook.tool")
    fake_module.SOME_VAR = 42  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "skills.nohook.tool", fake_module)

    registry = ToolRegistry()
    result = _register_bundle_tools(registry, bundle)
    assert result is False
    assert registry.list_tools() == []


# ---------------------------------------------------------------------------
# probe_skill_tool_loading
# ---------------------------------------------------------------------------


def test_probe_skill_tool_loading_returns_none_for_prompt_only_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_bundle(skills_root, "myskill", "# My Skill\n\nA skill.\n")
    from unclaw.skills.file_loader import discover_skill_bundles
    import unclaw.skills.bundle_tools as bt

    bundles = discover_skill_bundles(skills_root=skills_root)
    bundle = bundles[0]

    monkeypatch.setattr(bt, "load_active_skill_bundles", lambda **kwargs: (bundle,))

    result = bt.probe_skill_tool_loading(enabled_skill_ids=("myskill",))

    assert result == {"myskill": None}


def test_probe_skill_tool_loading_returns_error_for_missing_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_bundle(
        skills_root,
        "badskill",
        "# Bad Skill\n\nA skill.\n",
        tool_py_text="X = 1\n",
    )
    from unclaw.skills.file_loader import discover_skill_bundles
    import unclaw.skills.bundle_tools as bt

    bundles = discover_skill_bundles(skills_root=skills_root)
    bundle = bundles[0]

    module_name = "skills.badskill.tool"
    fake_module = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    monkeypatch.setattr(bt, "load_active_skill_bundles", lambda **kwargs: (bundle,))

    result = bt.probe_skill_tool_loading(enabled_skill_ids=("badskill",))

    assert "badskill" in result
    assert result["badskill"] is not None
    assert "register_skill_tools" in result["badskill"]


# ---------------------------------------------------------------------------
# Startup Skills check — isolated from developer-local skills
# ---------------------------------------------------------------------------


def test_build_startup_report_has_no_skills_check_when_empty(
    make_temp_project,
    monkeypatch,
) -> None:
    """No skills installed, none enabled → no Skills check emitted."""
    from unclaw.settings import load_settings
    from unclaw.startup import OllamaStatus, build_startup_report

    project_root = make_temp_project()  # enabled_skill_ids=[] by default
    settings = load_settings(project_root=project_root)

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(settings.default_model.model_name,),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="terminal",
        channel_enabled=True,
        required_profile_names=(settings.app.default_model_profile,),
    )

    skills_check = next(
        (check for check in report.checks if check.label == "Skills"),
        None,
    )
    # No skills → no Skills check
    assert skills_check is None


def test_build_startup_report_skills_check_ok_with_installed_skill(
    tmp_path: Path,
    make_temp_project,
    monkeypatch,
) -> None:
    """An installed + enabled skill shows up as OK in the startup check."""
    from unclaw.settings import load_settings
    from unclaw.startup import CheckStatus, OllamaStatus, build_startup_report

    # Create a synthetic skill bundle in the temp project's skills/ dir.
    skill_md = "# Demo Skill\n\nA demo skill.\n"
    project_root = make_temp_project(
        enabled_skill_ids=["demo"],
        install_skill_bundles={"demo": skill_md},
    )
    settings = load_settings(project_root=project_root)

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(settings.default_model.model_name,),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="terminal",
        channel_enabled=True,
        required_profile_names=(settings.app.default_model_profile,),
    )

    skills_check = next(
        (check for check in report.checks if check.label == "Skills"),
        None,
    )
    assert skills_check is not None
    # demo skill has no tool.py so probe succeeds (prompt-only = ok)
    assert skills_check.status is CheckStatus.OK
    assert "demo" in skills_check.detail


def test_build_startup_report_skills_check_warns_installed_but_not_enabled(
    tmp_path: Path,
    make_temp_project,
    monkeypatch,
) -> None:
    """A skill installed but not enabled → WARN in the startup check."""
    from unclaw.settings import load_settings
    from unclaw.startup import CheckStatus, OllamaStatus, build_startup_report

    skill_md = "# Extra\n\nExtra skill.\n"
    # Skill installed but NOT in enabled_skill_ids
    project_root = make_temp_project(
        enabled_skill_ids=[],
        install_skill_bundles={"extra": skill_md},
    )
    settings = load_settings(project_root=project_root)

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(settings.default_model.model_name,),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="terminal",
        channel_enabled=True,
        required_profile_names=(settings.app.default_model_profile,),
    )

    skills_check = next(
        (check for check in report.checks if check.label == "Skills"),
        None,
    )
    assert skills_check is not None
    assert skills_check.status is CheckStatus.WARN
    assert "extra" in skills_check.detail
