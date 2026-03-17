"""Targeted tests for the system_info tool — P3-1."""

from __future__ import annotations

import pytest

from unclaw.core.executor import create_default_tool_registry
from unclaw.tools.contracts import ToolCall, ToolPermissionLevel
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.system_tools import (
    SYSTEM_INFO_DEFINITION,
    get_system_info,
    register_system_tools,
)

pytestmark = pytest.mark.unit


def test_system_info_definition_permission_level() -> None:
    """system_info must be classified as LOCAL_READ (not NETWORK)."""
    assert SYSTEM_INFO_DEFINITION.permission_level is ToolPermissionLevel.LOCAL_READ


def test_system_info_definition_has_no_required_arguments() -> None:
    """system_info takes no arguments — it is a zero-arg read-only probe."""
    assert len(SYSTEM_INFO_DEFINITION.arguments) == 0


def test_register_system_tools_adds_system_info() -> None:
    """register_system_tools must add system_info to the registry."""
    registry = ToolRegistry()
    register_system_tools(registry)
    registered = registry.get("system_info")
    assert registered is not None
    assert registered.definition.name == "system_info"


def test_system_info_is_in_default_registry() -> None:
    """create_default_tool_registry must include system_info."""
    registry = create_default_tool_registry()
    registered = registry.get("system_info")
    assert registered is not None


def test_get_system_info_returns_success() -> None:
    """get_system_info must succeed and return a well-formed result."""
    call = ToolCall(tool_name="system_info", arguments={})
    result = get_system_info(call)
    assert result.success is True
    assert result.tool_name == "system_info"
    assert result.error is None


def test_get_system_info_output_contains_expected_fields() -> None:
    """Output text must contain the core bounded fields."""
    call = ToolCall(tool_name="system_info", arguments={})
    result = get_system_info(call)
    assert "OS:" in result.output_text
    assert "Python:" in result.output_text
    assert "CPU logical cores:" in result.output_text
    assert "Hostname:" in result.output_text
    assert "Local datetime:" in result.output_text
    assert "Locale:" in result.output_text


def test_get_system_info_payload_keys() -> None:
    """Payload must carry structured fields for programmatic use."""
    call = ToolCall(tool_name="system_info", arguments={})
    result = get_system_info(call)
    assert result.payload is not None
    for key in ("os", "os_release", "architecture", "python_version",
                "cpu_logical_cores", "hostname", "local_datetime", "locale"):
        assert key in result.payload, f"Missing payload key: {key}"


def test_get_system_info_does_not_expose_env_vars() -> None:
    """Output must not contain environment variable dumps or token-like strings."""
    call = ToolCall(tool_name="system_info", arguments={})
    result = get_system_info(call)
    output_upper = result.output_text.upper()
    assert "PATH=" not in output_upper
    assert "TOKEN" not in output_upper
    assert "SECRET" not in output_upper
    assert "PASSWORD" not in output_upper


def test_system_info_dispatched_via_registry() -> None:
    """system_info must be callable through the tool registry handler."""
    registry = ToolRegistry()
    register_system_tools(registry)
    registered = registry.get("system_info")
    assert registered is not None
    call = ToolCall(tool_name="system_info", arguments={})
    result = registered.handler(call)
    assert result.success is True
    assert result.tool_name == "system_info"
