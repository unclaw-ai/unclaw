"""Bounded read-only local machine and runtime information tool."""

from __future__ import annotations

import locale
import os
import platform
import socket
import sys
from datetime import datetime
from typing import Any

from unclaw.tools.contracts import (
    ToolCall,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry

SYSTEM_INFO_DEFINITION = ToolDefinition(
    name="system_info",
    description=(
        "Return a bounded read-only summary of the local machine and runtime: "
        "OS, Python version, CPU core count, hostname, local date/time, and locale."
    ),
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={},
)


def get_system_info(call: ToolCall) -> ToolResult:
    """Return a compact, safe, read-only snapshot of the local machine and runtime."""
    tool_name = SYSTEM_INFO_DEFINITION.name
    try:
        os_name = platform.system()
        os_release = platform.release()
        machine = platform.machine()
        python_version = sys.version.split()[0]
        cpu_cores = os.cpu_count()
        hostname = socket.gethostname()
        now = datetime.now().astimezone()
        local_datetime = now.strftime("%Y-%m-%d %H:%M:%S %Z")

        try:
            locale_info = locale.getlocale()
            locale_str = (
                f"{locale_info[0] or 'unknown'}/{locale_info[1] or 'unknown'}"
            )
        except Exception:
            locale_str = "unavailable"

        output_text = (
            f"OS: {os_name} {os_release} ({machine})\n"
            f"Python: {python_version}\n"
            f"CPU logical cores: {cpu_cores}\n"
            f"Hostname: {hostname}\n"
            f"Local datetime: {local_datetime}\n"
            f"Locale: {locale_str}"
        )

        payload: dict[str, Any] = {
            "os": os_name,
            "os_release": os_release,
            "architecture": machine,
            "python_version": python_version,
            "cpu_logical_cores": cpu_cores,
            "hostname": hostname,
            "local_datetime": now.isoformat(),
            "locale": locale_str,
        }

        return ToolResult.ok(
            tool_name=tool_name,
            output_text=output_text,
            payload=payload,
        )

    except Exception as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"system_info failed: {exc}",
        )


def register_system_tools(registry: ToolRegistry) -> None:
    """Register the built-in system information tool."""
    registry.register(SYSTEM_INFO_DEFINITION, get_system_info)


__all__ = [
    "SYSTEM_INFO_DEFINITION",
    "get_system_info",
    "register_system_tools",
]
