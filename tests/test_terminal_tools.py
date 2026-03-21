from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.capabilities import (
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.executor import create_default_tool_registry
from unclaw.tools.contracts import ToolCall, ToolPermissionLevel
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.terminal_tools import (
    RUN_TERMINAL_COMMAND_DEFINITION,
    register_terminal_tools,
    run_terminal_command,
)

pytestmark = pytest.mark.unit


def test_run_terminal_command_definition_permission_level() -> None:
    assert (
        RUN_TERMINAL_COMMAND_DEFINITION.permission_level
        is ToolPermissionLevel.LOCAL_EXECUTE
    )


def test_register_terminal_tools_adds_run_terminal_command() -> None:
    registry = ToolRegistry()
    register_terminal_tools(registry)

    assert registry.get(RUN_TERMINAL_COMMAND_DEFINITION.name) is not None


def test_run_terminal_command_is_in_default_registry() -> None:
    registry = create_default_tool_registry()

    assert registry.get(RUN_TERMINAL_COMMAND_DEFINITION.name) is not None


def test_run_terminal_command_executes_successfully_in_valid_working_directory(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    call = ToolCall(
        tool_name=RUN_TERMINAL_COMMAND_DEFINITION.name,
        arguments={
            "command": "pwd",
            "working_directory": str(workspace),
        },
    )

    result = run_terminal_command(
        call,
        allowed_roots=(tmp_path,),
        default_working_directory=tmp_path,
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["exit_code"] == 0
    assert result.payload["stdout"].strip() == str(workspace.resolve())
    assert result.payload["stderr"] == ""
    assert result.payload["working_directory"] == str(workspace.resolve())
    assert result.payload["timeout_occurred"] is False


def test_run_terminal_command_returns_failure_for_non_zero_exit_code(
    tmp_path: Path,
) -> None:
    call = ToolCall(
        tool_name=RUN_TERMINAL_COMMAND_DEFINITION.name,
        arguments={"command": "printf 'boom' >&2; exit 7"},
    )

    result = run_terminal_command(
        call,
        allowed_roots=(tmp_path,),
        default_working_directory=tmp_path,
    )

    assert result.success is False
    assert result.error == "Command exited with code 7."
    assert result.payload is not None
    assert result.payload["exit_code"] == 7
    assert result.payload["stdout"] == ""
    assert result.payload["stderr"] == "boom"
    assert result.payload["timeout_occurred"] is False


def test_run_terminal_command_fails_clearly_for_invalid_working_directory(
    tmp_path: Path,
) -> None:
    missing_directory = tmp_path / "missing"
    call = ToolCall(
        tool_name=RUN_TERMINAL_COMMAND_DEFINITION.name,
        arguments={
            "command": "pwd",
            "working_directory": str(missing_directory),
        },
    )

    result = run_terminal_command(
        call,
        allowed_roots=(tmp_path,),
        default_working_directory=tmp_path,
    )

    assert result.success is False
    assert result.error == f"Working directory does not exist: {missing_directory}"


def test_run_terminal_command_times_out_cleanly(tmp_path: Path) -> None:
    call = ToolCall(
        tool_name=RUN_TERMINAL_COMMAND_DEFINITION.name,
        arguments={
            "command": "sleep 2",
            "timeout_seconds": 0.1,
        },
    )

    result = run_terminal_command(
        call,
        allowed_roots=(tmp_path,),
        default_working_directory=tmp_path,
    )

    assert result.success is False
    assert result.error == "Command timed out after 0.1 seconds."
    assert result.payload is not None
    assert result.payload["exit_code"] is None
    assert result.payload["timeout_occurred"] is True


def test_run_terminal_command_captures_stdout_and_stderr(tmp_path: Path) -> None:
    call = ToolCall(
        tool_name=RUN_TERMINAL_COMMAND_DEFINITION.name,
        arguments={"command": "printf 'alpha'; printf 'beta' >&2"},
    )

    result = run_terminal_command(
        call,
        allowed_roots=(tmp_path,),
        default_working_directory=tmp_path,
    )

    assert result.success is True
    assert result.payload is not None
    assert result.payload["stdout"] == "alpha"
    assert result.payload["stderr"] == "beta"
    assert "Stdout:" in result.output_text
    assert "Stderr:" in result.output_text


def test_terminal_capability_context_exposes_run_terminal_command_guidance() -> None:
    registry = ToolRegistry()
    register_terminal_tools(registry)
    summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=True,
    )

    context = build_runtime_capability_context(summary)

    assert "run_terminal_command <command>" in context
    assert "Never invent terminal output" in context
    assert (
        "Local shell or terminal command execution via run_terminal_command."
        not in context
    )
