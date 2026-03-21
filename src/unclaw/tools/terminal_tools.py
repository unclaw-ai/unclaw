"""Bounded local terminal execution tool."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from unclaw.constants import DEFAULT_RUNTIME_TOOL_TIMEOUT_SECONDS
from unclaw.tools.contracts import (
    ToolCall,
    ToolArgumentSpec,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.file_tools import resolve_allowed_roots
from unclaw.tools.registry import ToolRegistry

_DEFAULT_COMMAND_TIMEOUT_SECONDS = 10.0
_MAX_CAPTURE_CHARS = 4_000

RUN_TERMINAL_COMMAND_DEFINITION = ToolDefinition(
    name="run_terminal_command",
    description=(
        "Execute one bounded local shell command in a validated working directory. "
        "Captures bounded stdout/stderr, enforces a timeout, and does not provide "
        "an interactive terminal."
    ),
    permission_level=ToolPermissionLevel.LOCAL_EXECUTE,
    arguments={
        "command": ToolArgumentSpec(
            description="Shell command string to execute locally."
        ),
        "working_directory": ToolArgumentSpec(
            description=(
                "Optional working directory. Relative paths resolve from the "
                "configured default local working directory."
            )
        ),
        "timeout_seconds": ToolArgumentSpec(
            description=(
                "Optional timeout in seconds. Must be greater than zero and no "
                "greater than the configured maximum."
            ),
            value_type="number",
        ),
    },
)


def register_terminal_tools(
    registry: ToolRegistry,
    *,
    project_root: Path | None = None,
    configured_roots: tuple[str, ...] = (),
    default_working_directory: Path | None = None,
    max_timeout_seconds: float = DEFAULT_RUNTIME_TOOL_TIMEOUT_SECONDS,
) -> None:
    """Register the built-in local terminal execution tool."""
    allowed_roots = resolve_allowed_roots(
        project_root=project_root,
        configured_roots=configured_roots,
    )
    resolved_default_working_directory = (
        default_working_directory.expanduser().resolve()
        if default_working_directory is not None
        else allowed_roots[0]
    )

    def run_handler(call: ToolCall) -> ToolResult:
        return run_terminal_command(
            call,
            allowed_roots=allowed_roots,
            default_working_directory=resolved_default_working_directory,
            max_timeout_seconds=max_timeout_seconds,
        )

    registry.register(RUN_TERMINAL_COMMAND_DEFINITION, run_handler)


def run_terminal_command(
    call: ToolCall,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
    default_working_directory: Path | None = None,
    max_timeout_seconds: float = DEFAULT_RUNTIME_TOOL_TIMEOUT_SECONDS,
) -> ToolResult:
    """Execute one local shell command with bounded timeout and output."""
    tool_name = RUN_TERMINAL_COMMAND_DEFINITION.name

    try:
        command = _read_required_string_argument(call.arguments, "command")
        timeout_seconds = _read_timeout_seconds(
            call.arguments,
            max_timeout_seconds=max_timeout_seconds,
        )
        working_directory = _resolve_working_directory(
            call.arguments.get("working_directory"),
            default_working_directory=default_working_directory,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    validated_working_directory = _validate_working_directory(
        tool_name=tool_name,
        working_directory=working_directory,
        allowed_roots=allowed_roots,
    )
    if isinstance(validated_working_directory, ToolResult):
        return validated_working_directory

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=validated_working_directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout, stdout_truncated = _truncate_output(_normalize_output(exc.stdout))
        stderr, stderr_truncated = _truncate_output(_normalize_output(exc.stderr))
        error = f"Command timed out after {timeout_seconds:g} seconds."
        payload = _build_terminal_payload(
            command=command,
            working_directory=validated_working_directory,
            exit_code=None,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timeout_occurred=True,
        )
        return ToolResult.failure(
            tool_name=tool_name,
            error=error,
            output_text=_format_terminal_result(
                command=command,
                working_directory=validated_working_directory,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                timeout_occurred=True,
                error=error,
            ),
            payload=payload,
        )
    except OSError as exc:
        error = f"Could not launch command: {exc}"
        payload = _build_terminal_payload(
            command=command,
            working_directory=validated_working_directory,
            exit_code=None,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            timeout_occurred=False,
        )
        return ToolResult.failure(
            tool_name=tool_name,
            error=error,
            output_text=_format_terminal_result(
                command=command,
                working_directory=validated_working_directory,
                exit_code=None,
                stdout="",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                timeout_occurred=False,
                error=error,
            ),
            payload=payload,
        )

    stdout, stdout_truncated = _truncate_output(_normalize_output(completed.stdout))
    stderr, stderr_truncated = _truncate_output(_normalize_output(completed.stderr))
    payload = _build_terminal_payload(
        command=command,
        working_directory=validated_working_directory,
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timeout_occurred=False,
    )
    output_text = _format_terminal_result(
        command=command,
        working_directory=validated_working_directory,
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timeout_occurred=False,
        error=None,
    )

    if completed.returncode == 0:
        return ToolResult.ok(
            tool_name=tool_name,
            output_text=output_text,
            payload=payload,
        )

    return ToolResult.failure(
        tool_name=tool_name,
        error=f"Command exited with code {completed.returncode}.",
        output_text=output_text,
        payload=payload,
    )


def _read_required_string_argument(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Argument '{key}' must be a non-empty string.")
    return value.strip()


def _read_timeout_seconds(
    arguments: dict[str, Any],
    *,
    max_timeout_seconds: float,
) -> float:
    default_timeout_seconds = min(
        _DEFAULT_COMMAND_TIMEOUT_SECONDS,
        max_timeout_seconds,
    )
    value = arguments.get("timeout_seconds", default_timeout_seconds)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Argument 'timeout_seconds' must be a number.")

    timeout_seconds = float(value)
    if timeout_seconds <= 0:
        raise ValueError("Argument 'timeout_seconds' must be greater than zero.")
    if timeout_seconds > max_timeout_seconds:
        raise ValueError(
            "Argument 'timeout_seconds' exceeds the configured maximum of "
            f"{max_timeout_seconds:g} seconds."
        )
    return timeout_seconds


def _resolve_working_directory(
    raw_working_directory: Any,
    *,
    default_working_directory: Path | None,
) -> Path:
    base_directory = (
        default_working_directory.expanduser().resolve()
        if default_working_directory is not None
        else Path.cwd().resolve()
    )
    if raw_working_directory is None:
        return base_directory
    if not isinstance(raw_working_directory, str) or not raw_working_directory.strip():
        raise ValueError(
            "Argument 'working_directory' must be a non-empty string when provided."
        )

    requested_path = Path(raw_working_directory.strip()).expanduser()
    if requested_path.is_absolute():
        return requested_path.resolve()
    return (base_directory / requested_path).resolve()


def _validate_working_directory(
    *,
    tool_name: str,
    working_directory: Path,
    allowed_roots: tuple[Path, ...] | None,
) -> Path | ToolResult:
    normalized_allowed_roots = _normalize_allowed_roots(allowed_roots)
    if not _is_path_allowed(working_directory, normalized_allowed_roots):
        allowed_roots_text = ", ".join(str(root) for root in normalized_allowed_roots)
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"Working directory '{working_directory}' is outside the allowed "
                f"local roots. Allowed roots: {allowed_roots_text}."
            ),
        )
    if not working_directory.exists():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Working directory does not exist: {working_directory}",
        )
    if not working_directory.is_dir():
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Working directory is not a directory: {working_directory}",
        )
    return working_directory


def _normalize_allowed_roots(allowed_roots: tuple[Path, ...] | None) -> tuple[Path, ...]:
    if not allowed_roots:
        return (Path.cwd().resolve(),)
    return tuple(dict.fromkeys(root.resolve() for root in allowed_roots))


def _is_path_allowed(path: Path, allowed_roots: tuple[Path, ...]) -> bool:
    for root in allowed_roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _normalize_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _truncate_output(text: str) -> tuple[str, bool]:
    if len(text) <= _MAX_CAPTURE_CHARS:
        return text, False
    return text[:_MAX_CAPTURE_CHARS], True


def _build_terminal_payload(
    *,
    command: str,
    working_directory: Path,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    stdout_truncated: bool,
    stderr_truncated: bool,
    timeout_occurred: bool,
) -> dict[str, Any]:
    return {
        "command": command,
        "working_directory": str(working_directory),
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "timeout_occurred": timeout_occurred,
    }


def _format_terminal_result(
    *,
    command: str,
    working_directory: Path,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    stdout_truncated: bool,
    stderr_truncated: bool,
    timeout_occurred: bool,
    error: str | None,
) -> str:
    lines = [
        f"Command: {command}",
        f"Working directory: {working_directory}",
        f"Exit code: {exit_code if exit_code is not None else 'n/a'}",
        f"Timed out: {'yes' if timeout_occurred else 'no'}",
    ]
    if error is not None:
        lines.append(f"Error: {error}")
    lines.append("Stdout:")
    lines.append(stdout if stdout else "[no stdout]")
    if stdout_truncated:
        lines.append(f"[stdout truncated to {_MAX_CAPTURE_CHARS} chars]")
    lines.append("Stderr:")
    lines.append(stderr if stderr else "[no stderr]")
    if stderr_truncated:
        lines.append(f"[stderr truncated to {_MAX_CAPTURE_CHARS} chars]")
    return "\n".join(lines)


__all__ = [
    "RUN_TERMINAL_COMMAND_DEFINITION",
    "register_terminal_tools",
    "run_terminal_command",
]
