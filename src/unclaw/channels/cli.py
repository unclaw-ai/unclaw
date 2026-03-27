"""Terminal CLI entrypoint for the current Unclaw runtime."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from unclaw.bootstrap import bootstrap
from unclaw.core.command_handler import CommandHandler, CommandResult, CommandStatus
from unclaw.core.executor import ToolExecutor
from unclaw.core.research_flow import (
    is_search_tool_call,
    persist_tool_result,
    run_search_command,
)
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.core.timing import elapsed_ms
from unclaw.errors import UnclawError
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.memory import MemoryManager
from unclaw.memory.protocols import SessionMemorySummaryRefresher
from unclaw.schemas.chat import MessageRole
from unclaw.settings import Settings
from unclaw.startup import (
    StartupReport,
    build_banner,
    build_startup_report,
    format_startup_report,
    _should_use_color,
    _style_text,
)
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolResult

_TOOL_CALL_ARGUMENT_CHAR_LIMIT = 200
_HIDDEN_TOOL_ARGUMENT_NAMES = frozenset(
    {"mission_id", "mission_task_id", "mission_deliverable_id"}
)


@dataclass(slots=True)
class _TerminalAssistantStream:
    """Render streamed assistant output without faking partial tokens."""

    chunks: list[str] = field(default_factory=list)
    started: bool = False
    _suppressed: bool = field(default=False, init=False)
    _resume_prefix: bool = field(default=False, init=False)

    def suppress_live_output(self) -> None:
        """Buffer chunks silently; finish() will render only the final answer.

        Called when the runtime detects a WEB_SEARCH route so that the streamed
        model draft is never shown. The final grounded reply is printed once by
        finish(), eliminating the duplicate-answer-body problem.
        """
        self._suppressed = True

    def write(self, chunk: str) -> None:
        if not chunk:
            return

        if self._suppressed:
            # Buffer silently — do not print the draft to the terminal.
            self.chunks.append(chunk)
            return

        if not self.started or self._resume_prefix:
            sys.stdout.write("Unclaw> ")
            self.started = True
            self._resume_prefix = False

        sys.stdout.write(chunk)
        sys.stdout.flush()
        self.chunks.append(chunk)

    def render_status(self, line: str) -> None:
        if self.started and not self._suppressed:
            if not self.chunks or not self.chunks[-1].endswith("\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
            self._resume_prefix = True

        print(line, flush=True)

    def finish(self, final_text: str) -> None:
        # Suppressed path: stream was buffered; render only the final answer.
        # The user never saw the buffered draft, so no refinement marker is
        # needed — just show the clean final answer once.
        if self._suppressed:
            print(f"Unclaw> {final_text}")
            return

        if not self.started:
            print(f"Unclaw> {final_text}")
            return

        streamed_text = "".join(self.chunks)
        if streamed_text.strip() == final_text.strip():
            # Streamed text matches final — ensure terminal newline.
            if not streamed_text.endswith("\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
            return

        if final_text.startswith(streamed_text):
            suffix = final_text[len(streamed_text):]
            if suffix:
                sys.stdout.write(suffix)
            if not final_text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
            return

        # Fallback: render the final answer without any refinement marker.
        # The CLI now suppresses assistant streaming by default, so this
        # branch should be rare and primarily serves direct unit usage.
        if not streamed_text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        print(final_text)


def main(project_root: Path | None = None) -> int:
    """Run the interactive Unclaw CLI."""
    try:
        settings = bootstrap(project_root=project_root)
        startup_report = build_startup_report(
            settings,
            channel_name="terminal",
            channel_enabled=settings.app.channels.terminal_enabled,
            required_profile_names=(settings.app.default_model_profile,),
            optional_profile_names=tuple(
                profile_name
                for profile_name in settings.models
                if profile_name != settings.app.default_model_profile
            ),
            warm_default_model=True,
        )
        if startup_report.has_errors:
            print(_build_preflight_banner(settings))
            print(format_startup_report(startup_report))
            return 1

        session_manager = SessionManager.from_settings(settings)
        memory_manager = MemoryManager(session_manager=session_manager)
        current_session = session_manager.ensure_current_session()
        event_bus = EventBus()
        tracer = Tracer(
            event_bus=event_bus,
            event_repository=session_manager.event_repository,
            include_reasoning_text=settings.app.logging.include_reasoning_text,
        )
        tracer.runtime_log_path = (
            settings.paths.log_file_path if settings.app.logging.file_enabled else None
        )
        tool_executor = ToolExecutor.with_default_tools(settings)
        command_handler = CommandHandler(
            settings=settings,
            session_manager=session_manager,
            memory_manager=memory_manager,
            tracer=tracer,
        )
        tracer.trace_channel_started(
            channel_name="terminal",
            session_id=current_session.id,
            model_profile_name=command_handler.current_model_profile.name,
            thinking_enabled=command_handler.thinking_enabled is True,
        )
        tracer.trace_session_selected(
            session_id=current_session.id,
            title=current_session.title,
            reason="startup",
        )
        tracer.trace_model_profile_selected(
            session_id=current_session.id,
            model_profile_name=command_handler.current_model_profile.name,
            provider=command_handler.current_model_profile.provider,
            model_name=command_handler.current_model_profile.model_name,
            reason="startup",
        )
    except UnclawError as exc:
        print(f"Failed to start Unclaw: {exc}", file=sys.stderr)
        return 1

    try:
        _print_banner(
            settings=settings,
            session_id=current_session.id,
            command_handler=command_handler,
            startup_report=startup_report,
        )
        return run_cli(
            session_manager=session_manager,
            command_handler=command_handler,
            memory_manager=memory_manager,
            tracer=tracer,
            tool_executor=tool_executor,
        )
    finally:
        session_manager.close()


def run_cli(
    *,
    session_manager: SessionManager,
    command_handler: CommandHandler,
    memory_manager: SessionMemorySummaryRefresher,
    tracer: Tracer,
    tool_executor: ToolExecutor,
) -> int:
    """Run the interactive read-eval-print loop."""
    while True:
        try:
            user_input = input(_build_prompt(command_handler))
        except EOFError:
            print()
            print("Exiting Unclaw.")
            return 0
        except KeyboardInterrupt:
            print()
            print("Exiting Unclaw.")
            return 0

        stripped_input = user_input.strip()
        if not stripped_input:
            continue

        if stripped_input.startswith("/"):
            result = command_handler.handle(stripped_input)
            if result.refresh_tool_executor:
                tool_executor = ToolExecutor.with_default_tools(
                    command_handler.session_manager.settings
                )
            if result.list_tools:
                _render_tool_list(tool_executor.list_tools())
                continue
            if result.tool_call is not None:
                if is_search_tool_call(result.tool_call):
                    assistant_stream = _TerminalAssistantStream()
                    assistant_stream.suppress_live_output()
                    assistant_reply = run_search_command(
                        session_manager=session_manager,
                        command_handler=command_handler,
                        tracer=tracer,
                        tool_call=result.tool_call,
                        stream_output_func=assistant_stream.write,
                        tool_registry=tool_executor.registry,
                        tool_call_callback=lambda tool_call: assistant_stream.render_status(
                            _build_tool_call_visibility_line(
                                tool_call,
                                skill_id=tool_executor.registry.get_owner_skill_id(
                                    tool_call.tool_name
                                ),
                            )
                        ),
                    ).assistant_reply
                    assistant_stream.finish(assistant_reply)
                    _refresh_session_summary(
                        memory_manager=memory_manager,
                        session_id=session_manager.ensure_current_session().id,
                    )
                    continue

                session = session_manager.ensure_current_session()
                _tool_skill_id = tool_executor.registry.get_owner_skill_id(
                    result.tool_call.tool_name
                )
                tracer.trace_tool_started(
                    session_id=session.id,
                    tool_name=result.tool_call.tool_name,
                    arguments=result.tool_call.arguments,
                    skill_id=_tool_skill_id,
                )
                tool_started_at = perf_counter()
                tool_result = tool_executor.execute(result.tool_call)
                tracer.trace_tool_finished(
                    session_id=session.id,
                    tool_name=result.tool_call.tool_name,
                    success=tool_result.success,
                    output_length=len(tool_result.output_text),
                    error=tool_result.error,
                    tool_duration_ms=elapsed_ms(tool_started_at),
                    skill_id=_tool_skill_id,
                )
                persist_tool_result(
                    session_manager=session_manager,
                    session_id=session.id,
                    result=tool_result,
                    tool_call=result.tool_call,
                )
                _render_tool_result(tool_result)
                continue
            _render_command_result(result)
            if result.should_exit:
                return 0
            continue

        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            stripped_input,
            session_id=session.id,
        )
        assistant_stream = _TerminalAssistantStream()
        assistant_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=stripped_input,
            tracer=tracer,
            stream_output_func=assistant_stream.write,
            tool_call_callback=lambda tool_call: assistant_stream.render_status(
                _build_tool_call_visibility_line(
                    tool_call,
                    skill_id=tool_executor.registry.get_owner_skill_id(
                        tool_call.tool_name
                    ),
                )
            ),
            mission_event_callback=assistant_stream.render_status,
        )
        assistant_stream.finish(assistant_reply)
        _refresh_session_summary(
            memory_manager=memory_manager,
            session_id=session.id,
        )


def _print_banner(
    *,
    settings: Settings,
    session_id: str,
    command_handler: CommandHandler,
    startup_report: StartupReport,
) -> None:
    print(
        build_banner(
            title="Unclaw terminal",
            subtitle="Local-first assistant runtime with direct local model execution.",
            rows=(
                ("mode", "terminal"),
                ("session", session_id),
                ("pack", settings.model_pack),
                (
                    "default",
                    (
                        f"{command_handler.current_model_profile_name} -> "
                        f"{command_handler.current_model_profile.model_name}"
                    ),
                ),
                ("control", settings.app.security.tools.files.control_preset),
                ("thinking", command_handler.thinking_label),
                ("logging", settings.app.logging.mode),
            ),
        )
    )
    print(format_startup_report(startup_report))
    print("Type /help for commands. Press Ctrl-D or use /exit to leave.")


def _build_preflight_banner(settings: Settings) -> str:
    return build_banner(
        title="Unclaw terminal",
        subtitle="Local-first assistant runtime with direct local model execution.",
        rows=(
            ("mode", "terminal"),
            ("pack", settings.model_pack),
            (
                "default",
                f"{settings.app.default_model_profile} -> {settings.default_model.model_name}",
            ),
            ("control", settings.app.security.tools.files.control_preset),
            ("logging", settings.app.logging.mode),
        ),
    )


def _build_prompt(command_handler: CommandHandler) -> str:
    session = command_handler.session_manager.ensure_current_session()
    raw_prompt = (
        f"unclaw[{session.id} | model={command_handler.current_model_profile_name} | "
        f"think={command_handler.thinking_label}]> "
    )
    if _should_use_color():
        return _style_text(raw_prompt, "shrimp", True, bold=True)
    return raw_prompt


def _render_command_result(result: CommandResult) -> None:
    if not result.lines:
        return

    if result.status is CommandStatus.ERROR:
        first_line, *other_lines = result.lines
        print(f"Error: {first_line}")
        for line in other_lines:
            print(line)
        return

    for line in result.lines:
        print(line)


def _render_tool_list(tools: list[ToolDefinition]) -> None:
    if not tools:
        print("No built-in tools available.")
        return

    name_width = max(len("Name"), *(len(tool.name) for tool in tools))
    permission_width = max(
        len("Permission"),
        *(len(tool.permission_level.value) for tool in tools),
    )

    print("Built-in tools:")
    print(
        f"{'Name'.ljust(name_width)}  "
        f"{'Permission'.ljust(permission_width)}  "
        "Description"
    )
    for tool in tools:
        print(
            f"{tool.name.ljust(name_width)}  "
            f"{tool.permission_level.value.ljust(permission_width)}  "
            f"{tool.description}"
        )


def _render_tool_result(result: ToolResult) -> None:
    if result.success:
        print(result.output_text)
        return

    if not result.output_text:
        print(f"Error: {result.error}")
        return

    lines = result.output_text.splitlines() or [result.output_text]
    first_line, *other_lines = lines
    print(f"Error: {first_line}")
    for line in other_lines:
        print(line)


def _render_assistant_reply(reply_text: str) -> None:
    print(f"Unclaw> {reply_text}")


def _build_tool_call_visibility_line(
    tool_call: ToolCall,
    *,
    skill_id: str | None = None,
) -> str:
    rendered_arguments = _format_tool_call_arguments(
        _sanitize_tool_call_arguments(tool_call.arguments)
    )
    prefix = f"skill:{skill_id}" if skill_id else "tool"
    raw_line = f"[{prefix}] {tool_call.tool_name} {rendered_arguments}"
    if _should_use_color():
        return _style_text(raw_line, "dim", True)
    return raw_line


def _sanitize_tool_call_arguments(arguments: object) -> object:
    if not isinstance(arguments, dict):
        return arguments
    return {
        key: value
        for key, value in arguments.items()
        if key not in _HIDDEN_TOOL_ARGUMENT_NAMES
    }


def _format_tool_call_arguments(arguments: object) -> str:
    if not isinstance(arguments, dict):
        return "{}"

    try:
        rendered = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(", ", ": "),
        )
    except TypeError:
        rendered = repr(arguments)

    single_line = rendered.replace("\r", "\\r").replace("\n", "\\n")
    if len(single_line) <= _TOOL_CALL_ARGUMENT_CHAR_LIMIT:
        return single_line

    return f"{single_line[:_TOOL_CALL_ARGUMENT_CHAR_LIMIT - 3]}..."


def _refresh_session_summary(
    *,
    memory_manager: SessionMemorySummaryRefresher,
    session_id: str,
) -> None:
    if not isinstance(memory_manager, SessionMemorySummaryRefresher):
        return

    try:
        memory_manager.build_or_refresh_session_summary(session_id)
    except UnclawError as exc:
        print(f"Warning: could not refresh session summary: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
