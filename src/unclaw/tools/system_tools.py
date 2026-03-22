"""Bounded read-only local machine and runtime information tool."""

from __future__ import annotations

import ctypes
import ctypes.util
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
        "OS, Python version, CPU core count, total RAM, hostname, local date/time, and locale. "
        "Call this tool when asked about the current date, time, day of week, OS, "
        "hardware specs, hostname, or locale — do not guess these facts."
    ),
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={},
)


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable GiB string."""
    return f"{n / (1024 ** 3):.1f} GiB"


def _ram_linux() -> str:
    """Read total RAM from /proc/meminfo (Linux only)."""
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return _fmt_bytes(kb * 1024)
    return "unavailable"


def _ram_macos() -> str:
    """Read total RAM via sysctlbyname("hw.memsize") (macOS only)."""
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return "unavailable"
    libc = ctypes.CDLL(libc_name)
    size = ctypes.c_uint64(0)
    size_of = ctypes.c_size_t(ctypes.sizeof(size))
    ret = libc.sysctlbyname(
        b"hw.memsize",
        ctypes.byref(size),
        ctypes.byref(size_of),
        None,
        0,
    )
    if ret != 0:
        return "unavailable"
    return _fmt_bytes(size.value)


def _ram_windows() -> str:
    """Read total RAM via GlobalMemoryStatusEx (Windows only)."""

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
    return _fmt_bytes(stat.ullTotalPhys)


def _get_ram_summary() -> str:
    """Return total physical RAM as a human-readable string, or 'unavailable'.

    Uses only the Python standard library.  Falls back cleanly to
    'unavailable' on any platform where the method is unavailable or fails.
    """
    os_name = platform.system()
    try:
        if os_name == "Linux":
            return _ram_linux()
        if os_name == "Darwin":
            return _ram_macos()
        if os_name == "Windows":
            return _ram_windows()
        return "unavailable"
    except Exception:
        return "unavailable"


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
        ram_summary = _get_ram_summary()

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
            f"RAM total: {ram_summary}\n"
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
            "ram_total": ram_summary,
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
    "_get_ram_summary",
    "get_system_info",
    "register_system_tools",
]
