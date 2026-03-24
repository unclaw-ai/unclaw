from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from unclaw.core.executor import ToolExecutor, create_default_tool_registry
from unclaw.core.session_manager import SessionManager
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolPermissionLevel, ToolResult

pytestmark = pytest.mark.unit


def _set_control_preset(project_root: Path, preset_name: str) -> None:
    app_config_path = project_root / "config" / "app.yaml"
    payload = yaml.safe_load(app_config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["security"]["tools"]["files"]["control_preset"] = preset_name
    app_config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _install_synthetic_skill_module(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = project_root / "skills" / "synth"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "SKILL.md").write_text("# Synth\n\nSynthetic skill.\n", encoding="utf-8")
    (bundle_dir / "tool.py").write_text("# placeholder\n", encoding="utf-8")

    module_name = "skills.synth.tool"
    fake_module = types.ModuleType(module_name)
    definition = ToolDefinition(
        name="synth_action",
        description="Synthetic skill tool.",
        permission_level=ToolPermissionLevel.NETWORK,
        arguments={},
    )

    def _register(registry) -> None:  # type: ignore[no-untyped-def]
        registry.register(
            definition,
            lambda call: ToolResult.ok(tool_name=call.tool_name, output_text="ok"),
        )

    fake_module.register_skill_tools = _register  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake_module)


@pytest.mark.parametrize("preset_name", ["safe", "workspace", "full"])
def test_control_presets_keep_core_agent_tools_available(
    make_temp_project,
    monkeypatch: pytest.MonkeyPatch,
    preset_name: str,
) -> None:
    project_root = make_temp_project(enabled_skill_ids=["synth"])
    _set_control_preset(project_root, preset_name)
    _install_synthetic_skill_module(project_root, monkeypatch)

    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        registry = create_default_tool_registry(
            settings,
            session_manager=session_manager,
        )
    finally:
        session_manager.close()

    for tool_name in (
        "system_info",
        "fetch_url_text",
        "search_web",
        "fast_web_search",
        "inspect_session_history",
        "search_long_term_memory",
        "list_long_term_memory",
        "remember_long_term_memory",
        "forget_long_term_memory",
        "synth_action",
    ):
        assert registry.get(tool_name) is not None, (
            f"Expected {tool_name!r} to remain available in preset {preset_name!r}."
        )

    system_info_result = registry.get("system_info").handler(  # type: ignore[union-attr]
        ToolCall(tool_name="system_info", arguments={})
    )
    assert system_info_result.success is True


def test_safe_preset_keeps_runtime_reads_but_blocks_project_writes(make_temp_project) -> None:
    project_root = make_temp_project()
    _set_control_preset(project_root, "safe")
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    config_note = project_root / "config" / "note.txt"
    config_note.write_text("runtime-readable", encoding="utf-8")

    read_result = executor.execute(
        ToolCall(
            tool_name="read_text_file",
            arguments={"path": str(config_note)},
        )
    )
    assert read_result.success is True

    write_result = executor.execute(
        ToolCall(
            tool_name="write_text_file",
            arguments={
                "path": str(project_root / "outside-data.txt"),
                "content": "blocked",
            },
        )
    )
    assert write_result.success is False
    assert write_result.error is not None
    assert "outside the allowed local roots" in write_result.error

    data_write_result = executor.execute(
        ToolCall(
            tool_name="write_text_file",
            arguments={
                "path": "safe-note.txt",
                "content": "stored in data",
            },
        )
    )
    assert data_write_result.success is True
    assert (settings.paths.files_dir / "safe-note.txt").read_text(encoding="utf-8") == (
        "stored in data"
    )


def test_workspace_preset_reads_project_workspace_but_writes_only_to_project_data(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    _set_control_preset(project_root, "workspace")
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    workspace_note = project_root / "workspace-note.txt"
    workspace_note.write_text("workspace-readable", encoding="utf-8")

    read_result = executor.execute(
        ToolCall(
            tool_name="read_text_file",
            arguments={"path": str(workspace_note)},
        )
    )
    assert read_result.success is True

    blocked_write_result = executor.execute(
        ToolCall(
            tool_name="write_text_file",
            arguments={
                "path": str(project_root / "workspace-write.txt"),
                "content": "blocked",
            },
        )
    )
    assert blocked_write_result.success is False
    assert blocked_write_result.error is not None
    assert "outside the allowed local roots" in blocked_write_result.error

    terminal_result = executor.execute(
        ToolCall(
            tool_name="run_terminal_command",
            arguments={
                "command": "pwd",
                "working_directory": str(project_root),
            },
        )
    )
    assert terminal_result.success is True


def test_full_preset_allows_project_writes(make_temp_project) -> None:
    project_root = make_temp_project()
    _set_control_preset(project_root, "full")
    settings = load_settings(project_root=project_root)
    executor = ToolExecutor.with_default_tools(settings)

    write_result = executor.execute(
        ToolCall(
            tool_name="write_text_file",
            arguments={
                "path": str(project_root / "full-write.txt"),
                "content": "allowed",
            },
        )
    )

    assert write_result.success is True
    assert (project_root / "full-write.txt").read_text(encoding="utf-8") == "allowed"
