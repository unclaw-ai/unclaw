"""Slash command parsing and handling shared by interactive channels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from shlex import split as shlex_split
from typing import TYPE_CHECKING

from unclaw.control_surface import (
    CONTROL_PRESET_NAMES,
    MIN_PROFILE_NUM_CTX,
    build_control_surface_summary,
)
from unclaw.core.executor import resolve_builtin_tool_command
from unclaw.core.session_manager import SessionManager, SessionManagerError
from unclaw.errors import ConfigurationError
from unclaw.memory.diagnostics import collect_memory_diagnostics, render_memory_diagnostics
from unclaw.schemas.session import SessionSummary
from unclaw.startup import warm_load_model_profile
from unclaw.settings import (
    ModelProfile,
    Settings,
    persist_control_preset,
    persist_profile_num_ctx,
)
from unclaw.tools.contracts import ToolCall

if TYPE_CHECKING:
    from unclaw.logs.tracer import Tracer
    from unclaw.memory.protocols import SessionMemoryCommandInterface

_FREEFORM_TOOL_COMMANDS = frozenset({"fetch", "ls", "read", "search"})


class CommandStatus(StrEnum):
    """Render-friendly command outcome types."""

    OK = "ok"
    ERROR = "error"
    EXIT = "exit"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Structured result returned by the slash command handler."""

    status: CommandStatus
    lines: tuple[str, ...]
    list_tools: bool = False
    tool_call: ToolCall | None = None
    session_id: str | None = None
    updated_settings: Settings | None = None
    refresh_tool_executor: bool = False

    @property
    def should_exit(self) -> bool:
        return self.status is CommandStatus.EXIT


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """One parsed slash command with tokenized and raw arguments."""

    name: str
    arguments: tuple[str, ...]
    raw_arguments: str


@dataclass(slots=True)
class CommandHandler:
    """Handle slash commands without executing any model calls."""

    settings: Settings
    session_manager: SessionManager
    memory_manager: SessionMemoryCommandInterface | None = None
    tracer: Tracer | None = None
    current_model_profile_name: str | None = None
    thinking_enabled: bool | None = None
    allow_exit: bool = True

    def __post_init__(self) -> None:
        if self.current_model_profile_name is None:
            self.current_model_profile_name = self.settings.app.default_model_profile
        elif self.current_model_profile_name not in self.settings.models:
            raise ValueError(
                f"Unknown model profile '{self.current_model_profile_name}'."
            )

        requested_thinking_enabled = (
            self.settings.app.thinking.default_enabled
            if self.thinking_enabled is None
            else self.thinking_enabled
        )
        self.thinking_enabled = requested_thinking_enabled

        if requested_thinking_enabled and not self.current_model_profile.thinking_supported:
            self.thinking_enabled = False
            self._trace_thinking_changed(reason=self._thinking_disabled_reason())

    @property
    def current_model_profile(self) -> ModelProfile:
        return self.settings.models[self.current_model_profile_name]

    @property
    def thinking_label(self) -> str:
        return "on" if self.thinking_enabled else "off"

    def handle(self, raw_command: str) -> CommandResult:
        """Parse one slash command and return a structured result."""
        try:
            parsed_command = self._parse_command(raw_command)
        except ValueError as exc:
            return self._error(str(exc))

        try:
            match parsed_command.name:
                case "new":
                    return self._handle_new(parsed_command.arguments)
                case "sessions":
                    return self._handle_sessions(parsed_command.arguments)
                case "use":
                    return self._handle_use(parsed_command.arguments)
                case "summary":
                    return self._handle_summary(parsed_command.arguments)
                case "session":
                    return self._handle_session(parsed_command.arguments)
                case "model":
                    return self._handle_model(parsed_command.arguments)
                case "profiles":
                    return self._handle_profiles(parsed_command.arguments)
                case "ctx":
                    return self._handle_ctx(parsed_command.arguments)
                case "control":
                    return self._handle_control(parsed_command.arguments)
                case "think":
                    return self._handle_think(parsed_command.arguments)
                case "tools":
                    return self._handle_tools(parsed_command.arguments)
                case "read":
                    return self._handle_tool_command(
                        parsed_command,
                        usage_line="/read <path>",
                    )
                case "ls":
                    return self._handle_tool_command(
                        parsed_command,
                        usage_line="/ls [path]",
                        default_argument_value=".",
                    )
                case "fetch":
                    return self._handle_tool_command(
                        parsed_command,
                        usage_line="/fetch <url>",
                    )
                case "search":
                    return self._handle_tool_command(
                        parsed_command,
                        usage_line="/search <query>",
                    )
                case "skills":
                    return self._handle_skills(parsed_command.arguments)
                case "memory-status":
                    return self._handle_memory_status(parsed_command.arguments)
                case "help":
                    return self._handle_help(parsed_command.arguments)
                case "exit":
                    return self._handle_exit(parsed_command.arguments)
                case _:
                    return self._error(
                        f"Unknown command '/{parsed_command.name}'. "
                        "Use /help to list commands."
                    )
        except SessionManagerError as exc:
            return self._error(str(exc))

    def _handle_new(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/new")

        session = self.session_manager.create_session()
        self._trace_session_started(session.id, session.title, source="command")
        return self._ok(
            f"Created and switched to session {session.id}.",
            f"Title: {session.title}",
            session_id=session.id,
        )

    def _handle_sessions(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/sessions")

        sessions = self.session_manager.list_sessions()
        if not sessions:
            return self._ok("No sessions found.")

        lines = ["Recent sessions:"]
        for session in sessions:
            lines.append(self._format_session_line(session))
        return self._ok(*lines)

    def _handle_use(self, arguments: tuple[str, ...]) -> CommandResult:
        if len(arguments) != 1:
            return self._usage("/use <session_id>")

        session = self.session_manager.switch_session(arguments[0])
        self._trace_session_selected(session.id, session.title, reason="command")
        return self._ok(
            f"Switched to session {session.id}.",
            f"Title: {session.title}",
            session_id=session.id,
        )

    def _handle_summary(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/summary")
        if self.memory_manager is None:
            return self._error("Session memory is not available in this channel.")

        summary_text = self.memory_manager.get_session_summary()
        return self._ok("Session summary:", summary_text)

    def _handle_session(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/session")
        if self.memory_manager is None:
            return self._error("Session memory is not available in this channel.")

        state = self.memory_manager.get_session_state()
        lines = [
            f"Session: {state.session_id}",
            f"Title: {state.title}",
            f"Updated: {state.updated_at}",
            (
                "Messages: "
                f"{state.message_count} total | "
                f"{state.user_message_count} user | "
                f"{state.assistant_message_count} assistant"
            ),
            f"Summary: {state.summary_text}",
        ]
        if state.recent_snippets:
            lines.append("Recent snippets:")
            lines.extend(state.recent_snippets)

        return self._ok(*lines)

    def _handle_model(self, arguments: tuple[str, ...]) -> CommandResult:
        if not arguments:
            profile = self.current_model_profile
            return self._ok(
                f"Current model profile: {profile.name}",
                f"Provider: {profile.provider} | Model: {profile.model_name}",
            )

        if len(arguments) != 1:
            return self._usage("/model <profile_name>")

        profile_name = arguments[0]
        profile = self.settings.models.get(profile_name)
        if profile is None:
            available_profiles = ", ".join(sorted(self.settings.models))
            return self._error(
                f"Unknown model profile '{profile_name}'. "
                f"Available profiles: {available_profiles}."
            )

        previous_profile_name = self.current_model_profile_name
        self.current_model_profile_name = profile_name
        self._trace_model_profile_selected(profile, reason="command")
        lines = [
            f"Selected model profile: {profile.name}",
            f"Provider: {profile.provider} | Model: {profile.model_name}",
        ]
        if self.thinking_enabled and not profile.thinking_supported:
            self.thinking_enabled = False
            lines.append(self._thinking_disabled_reason())
            self._trace_thinking_changed(reason=self._thinking_disabled_reason())

        if previous_profile_name != profile_name:
            self._refresh_loaded_model_profile(profile_name)

        return self._ok(*lines)

    def _handle_profiles(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/profiles")

        lines = ["Model profiles:"]
        for profile_name, profile in self.settings.models.items():
            marker = "*" if profile_name == self.current_model_profile_name else " "
            current_suffix = " | current" if marker == "*" else ""
            lines.append(
                f"{marker} {profile.name} | model={profile.model_name} | "
                f"ctx={self._format_num_ctx(profile.num_ctx)} | tools={profile.tool_mode}"
                f"{current_suffix}"
            )
        lines.append("Use /model <profile_name> to switch profiles.")
        lines.append("Use /ctx <profile_name> <num_ctx> to change one context window.")
        return self._ok(*lines)

    def _handle_ctx(self, arguments: tuple[str, ...]) -> CommandResult:
        if not arguments:
            lines = ["Context windows:"]
            for profile_name, profile in self.settings.models.items():
                marker = "*" if profile_name == self.current_model_profile_name else " "
                current_suffix = " | current" if marker == "*" else ""
                lines.append(
                    f"{marker} {profile.name} | ctx={self._format_num_ctx(profile.num_ctx)} "
                    f"| model={profile.model_name}{current_suffix}"
                )
            lines.append("Use /ctx <profile_name> <num_ctx> to save a new value.")
            return self._ok(*lines)

        if len(arguments) != 2:
            return self._usage("/ctx <profile_name> <num_ctx>")

        profile_name = arguments[0].strip().lower()
        if profile_name not in self.settings.models:
            available_profiles = ", ".join(sorted(self.settings.models))
            return self._error(
                f"Unknown model profile '{profile_name}'. "
                f"Available profiles: {available_profiles}."
            )

        try:
            num_ctx = int(arguments[1])
        except ValueError:
            return self._error("Context window must be an integer.")
        if num_ctx < MIN_PROFILE_NUM_CTX:
            return self._error(
                f"Context window must be at least {MIN_PROFILE_NUM_CTX} tokens."
            )

        previous_settings = self.settings
        try:
            updated_settings = persist_profile_num_ctx(
                self.settings,
                profile_name=profile_name,
                num_ctx=num_ctx,
            )
        except (ConfigurationError, OSError) as exc:
            return self._error(f"Could not save context window: {exc}")

        if updated_settings is previous_settings:
            return self._ok(
                *self._build_ctx_change_feedback(
                    profile_name=profile_name,
                    num_ctx=num_ctx,
                    changed=False,
                )
            )

        self._apply_updated_settings(updated_settings)
        return self._ok(
            *self._build_ctx_change_feedback(
                profile_name=profile_name,
                num_ctx=num_ctx,
                changed=True,
            ),
            updated_settings=updated_settings,
            refresh_tool_executor=True,
        )

    def _handle_control(self, arguments: tuple[str, ...]) -> CommandResult:
        if not arguments:
            return self._ok(*self._build_control_summary_lines())

        if len(arguments) != 1:
            return self._usage("/control <safe|workspace|full>")

        preset_name = arguments[0].strip().lower()
        if preset_name not in CONTROL_PRESET_NAMES:
            available_presets = ", ".join(CONTROL_PRESET_NAMES)
            return self._error(
                f"Unknown control preset '{preset_name}'. "
                f"Available presets: {available_presets}."
            )

        previous_settings = self.settings
        try:
            updated_settings = persist_control_preset(self.settings, preset_name)
        except (ConfigurationError, OSError) as exc:
            return self._error(f"Could not save control preset: {exc}")

        if updated_settings is not previous_settings:
            self._apply_updated_settings(updated_settings)
            lines = [
                f"Saved control preset: {preset_name}.",
                "New file and terminal tool access rules apply immediately in this CLI.",
                *self._build_control_summary_lines(),
            ]
            return self._ok(
                *lines,
                updated_settings=updated_settings,
                refresh_tool_executor=True,
            )

        return self._ok(f"Control preset unchanged: {preset_name}.")

    def _handle_think(self, arguments: tuple[str, ...]) -> CommandResult:
        if not arguments:
            lines = [f"Thinking mode: {self.thinking_label}"]
            if not self.current_model_profile.thinking_supported:
                lines.append(self._thinking_unsupported_status_note())
            return self._ok(*lines)

        if len(arguments) != 1:
            return self._usage("/think <on|off>")

        value = arguments[0].lower()
        if value not in {"on", "off"}:
            return self._usage("/think <on|off>")

        if value == "on" and not self.current_model_profile.thinking_supported:
            return self._error(self._thinking_enable_blocked_message())

        previous_value = self.thinking_enabled
        self.thinking_enabled = value == "on"
        if previous_value != self.thinking_enabled:
            self._trace_thinking_changed(reason="command")
        return self._ok(f"Thinking mode: {self.thinking_label}")

    def _handle_tools(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/tools")
        return CommandResult(
            status=CommandStatus.OK,
            lines=(),
            list_tools=True,
        )

    def _handle_tool_command(
        self,
        parsed_command: ParsedCommand,
        *,
        usage_line: str,
        default_argument_value: str | None = None,
    ) -> CommandResult:
        argument_value = self._read_freeform_argument(parsed_command)
        if argument_value is None:
            argument_value = default_argument_value
        if argument_value is None:
            return self._usage(usage_line)

        tool_name = resolve_builtin_tool_command(parsed_command.name)
        if tool_name is None:
            return self._error(
                f"Unknown tool command '/{parsed_command.name}'. "
                "Use /help to list commands."
            )

        argument_key = {
            "fetch": "url",
            "search": "query",
        }.get(parsed_command.name, "path")
        return CommandResult(
            status=CommandStatus.OK,
            lines=(),
            tool_call=ToolCall(
                tool_name=tool_name,
                arguments={argument_key: argument_value},
            ),
        )

    def _handle_skills(self, arguments: tuple[str, ...]) -> CommandResult:
        from unclaw.skills.manager import run_skill_command

        if not arguments:
            action = "list"
            skill_id = None
        elif arguments[0] == "search" and len(arguments) >= 2:
            action = "search"
            skill_id = " ".join(arguments[1:])
        elif len(arguments) == 2 and arguments[0] in {
            "install",
            "enable",
            "disable",
            "remove",
            "update",
        }:
            action, skill_id = arguments
        elif arguments == ("update", "--all"):
            action = "update"
            skill_id = "--all"
        else:
            return self._usage(
                "/skills [search <query>|install|enable|disable|remove|update <skill_id>|update --all]"
            )

        outcome = run_skill_command(self.settings, action=action, skill_id=skill_id)
        if outcome.updated_settings is not None:
            self.settings = outcome.updated_settings
            self.session_manager.settings = outcome.updated_settings

        if outcome.ok:
            return self._ok(
                *outcome.lines,
                updated_settings=outcome.updated_settings,
                refresh_tool_executor=outcome.refresh_runtime,
            )
        return self._error(
            *outcome.lines,
            updated_settings=outcome.updated_settings,
            refresh_tool_executor=outcome.refresh_runtime,
        )

    def _handle_memory_status(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/memory-status")

        diagnostics = collect_memory_diagnostics(
            data_dir=self.settings.paths.data_dir,
            session_db_path=self.settings.paths.database_path,
            current_session_id=self._current_session_id(),
        )
        return self._ok(*render_memory_diagnostics(diagnostics))

    def _handle_help(self, arguments: tuple[str, ...]) -> CommandResult:
        if arguments:
            return self._usage("/help")

        lines = [
            "Available slash commands for this channel:",
            "",
            "Sessions:",
            "/new               Start a fresh session.",
            "/sessions          List recent sessions.",
            "/use <session_id>  Switch to an existing session.",
            "",
            "Models:",
            "/model                 Show the active model profile.",
            "/model <profile_name>  Switch model profiles.",
            "/profiles              Show model profiles, tools, and context windows.",
            "/ctx                   Show context windows for each profile.",
            "/ctx <profile_name> <num_ctx>  Save a context window override for one profile.",
            "/think                 Show thinking mode status.",
            "/think on              Turn thinking mode on.",
            "/think off             Turn thinking mode off.",
            "",
            "Control:",
            "/control              Show the current local access preset and roots.",
            "/control <preset>     Set local access to safe, workspace, or full.",
            "",
            "Tools:",
            "/tools            List built-in tools.",
            "/read <path>      Read one local file inside allowed roots.",
            "/ls [path]        List one local directory inside allowed roots. Defaults to the current directory.",
            "/fetch <url>      Fetch one public URL.",
            "/search <query>  Search the public web, ground the answer, and append compact sources.",
            "",
            "Skills:",
            "/skills  Show enabled, installed, update, available, and orphaned skills.",
            "/skills search <query>  Search skills by id, name, summary, or tags.",
            "/skills install <skill_id>  Install one skill from the catalog.",
            "/skills enable <skill_id>  Enable one installed skill.",
            "/skills disable <skill_id>  Disable one enabled skill.",
            "/skills remove <skill_id>  Remove one installed skill bundle.",
            "/skills update <skill_id>  Update one installed skill.",
            "/skills update --all  Update every installed skill that has a newer catalog version.",
            "",
            "Memory:",
            "/session  Show the current session state.",
            "/summary  Show the saved session summary.",
            "/memory-status  Show memory layer diagnostics.",
            "",
            "General:",
            "/help  Show this command list with examples.",
            "",
            "Examples:",
            "/control workspace",
            "/ctx main 4096",
            "/ls .",
            "/ls /home/user/project",
            "/read README.md",
            "/search local ai agents",
        ]
        if self.allow_exit:
            lines.append("/exit  Leave the terminal runtime.")
            lines.append("")
            lines.append(
                "Tip: use 'unclaw logs' or 'unclaw logs full' in another terminal."
            )
        return self._ok(*lines)

    def _handle_exit(self, arguments: tuple[str, ...]) -> CommandResult:
        if not self.allow_exit:
            return self._error("The /exit command is only available in the CLI.")
        if arguments:
            return self._usage("/exit")
        return CommandResult(status=CommandStatus.EXIT, lines=("Exiting Unclaw.",))

    def _format_session_line(self, session: SessionSummary) -> str:
        marker = "*" if session.id == self.session_manager.current_session_id else " "
        return (
            f"{marker} {session.id} | {session.title} | "
            f"updated {session.updated_at}"
        )

    def _parse_command(self, raw_command: str) -> ParsedCommand:
        normalized = raw_command.strip()
        if not normalized.startswith("/"):
            raise ValueError("Commands must start with '/'.")
        if normalized == "/":
            raise ValueError("Empty command. Use /help to list available commands.")

        command_text = normalized[1:]
        command_name, _, raw_arguments = command_text.partition(" ")
        if not command_name:
            raise ValueError("Empty command. Use /help to list available commands.")

        lowered_command_name = command_name.lower()
        if lowered_command_name in _FREEFORM_TOOL_COMMANDS:
            arguments = self._parse_freeform_tool_arguments(raw_arguments)
            return ParsedCommand(
                name=lowered_command_name,
                arguments=arguments,
                raw_arguments=raw_arguments,
            )

        try:
            arguments = tuple(shlex_split(raw_arguments)) if raw_arguments else ()
        except ValueError as exc:
            raise ValueError(f"Could not parse command arguments: {exc}") from exc

        return ParsedCommand(
            name=lowered_command_name,
            arguments=arguments,
            raw_arguments=raw_arguments,
        )

    def _parse_freeform_tool_arguments(self, raw_arguments: str) -> tuple[str, ...]:
        raw_value = raw_arguments.strip()
        if not raw_value:
            return ()

        try:
            parsed_arguments = tuple(shlex_split(raw_value))
        except ValueError:
            parsed_arguments = ()

        if len(parsed_arguments) == 1:
            value = parsed_arguments[0].strip()
            return (value,) if value else ()

        value = self._unwrap_outer_quotes(raw_value)
        return (value,) if value else ()

    def _unwrap_outer_quotes(self, value: str) -> str:
        stripped_value = value.strip()
        if len(stripped_value) >= 2 and stripped_value[0] == stripped_value[-1]:
            if stripped_value[0] in {'"', "'"}:
                return stripped_value[1:-1].strip()
        return stripped_value

    def _read_freeform_argument(self, parsed_command: ParsedCommand) -> str | None:
        raw_value = parsed_command.raw_arguments.strip()
        if not raw_value:
            return None

        if len(parsed_command.arguments) == 1:
            value = parsed_command.arguments[0].strip()
            return value or None

        return raw_value

    def _build_control_summary_lines(self) -> tuple[str, ...]:
        summary = build_control_surface_summary(
            preset_name=self.settings.app.security.tools.files.control_preset,
            project_root=self.settings.paths.project_root,
            allowed_roots=self.settings.app.security.tools.files.allowed_roots,
        )
        lines = [
            f"Control preset: {summary.preset_name}",
            f"Meaning: {summary.preset_description}",
            f"Tool access: {summary.access_scope}",
            "Allowed roots:",
        ]
        lines.extend(f"- {root}" for root in summary.allowed_roots)
        return tuple(lines)

    def _build_ctx_change_feedback(
        self,
        *,
        profile_name: str,
        num_ctx: int,
        changed: bool,
    ) -> tuple[str, ...]:
        leading_line = (
            f"Saved context window: {profile_name}={num_ctx}."
            if changed
            else f"Context window already saved: {profile_name}={num_ctx}."
        )

        if profile_name != self.current_model_profile_name:
            return (
                leading_line,
                (
                    f"{profile_name} is not the active profile. "
                    "The saved value will be used the next time that model is loaded."
                ),
            )

        if self._refresh_loaded_model_profile(profile_name):
            return (
                leading_line,
                f"Reloaded active model profile: {profile_name}.",
                "The new context window will be used on the next turn in this CLI.",
            )

        return (
            leading_line,
            f"Could not refresh the active model profile: {profile_name}.",
            "The new value is guaranteed on next model reload or CLI restart.",
        )

    def _apply_updated_settings(self, updated_settings: Settings) -> None:
        self.settings = updated_settings
        self.session_manager.settings = updated_settings

    def _format_num_ctx(self, num_ctx: int | None) -> str:
        return str(num_ctx) if num_ctx is not None else "default"

    def _ok(
        self,
        *lines: str,
        session_id: str | None = None,
        updated_settings: Settings | None = None,
        refresh_tool_executor: bool = False,
    ) -> CommandResult:
        return CommandResult(
            status=CommandStatus.OK,
            lines=tuple(lines),
            session_id=session_id,
            updated_settings=updated_settings,
            refresh_tool_executor=refresh_tool_executor,
        )

    def _error(
        self,
        *lines: str,
        updated_settings: Settings | None = None,
        refresh_tool_executor: bool = False,
    ) -> CommandResult:
        return CommandResult(
            status=CommandStatus.ERROR,
            lines=tuple(lines),
            updated_settings=updated_settings,
            refresh_tool_executor=refresh_tool_executor,
        )

    def _usage(self, usage_line: str) -> CommandResult:
        return self._error(f"Usage: {usage_line}")

    def _thinking_disabled_reason(self) -> str:
        profile = self.current_model_profile
        if profile.name == "fast":
            return "Thinking mode was turned off because fast mode does not support thinking."
        return (
            f"Thinking mode was turned off because model profile '{profile.name}' "
            "does not support it."
        )

    def _thinking_unsupported_status_note(self) -> str:
        profile = self.current_model_profile
        if profile.name == "fast":
            return "Fast mode does not support thinking."
        return f"Model profile '{profile.name}' does not support thinking mode."

    def _thinking_enable_blocked_message(self) -> str:
        profile = self.current_model_profile
        if profile.name == "fast":
            return (
                "Fast mode does not support thinking. "
                "Switch to another model profile to turn it on."
            )
        return f"Model profile '{profile.name}' does not support thinking mode."

    def _current_session_id(self) -> str | None:
        current_session_id = getattr(self.session_manager, "current_session_id", None)
        return current_session_id if isinstance(current_session_id, str) else None

    def _trace_session_started(self, session_id: str, title: str, *, source: str) -> None:
        if self.tracer is None:
            return
        self.tracer.trace_session_started(
            session_id=session_id,
            title=title,
            source=source,
        )

    def _trace_session_selected(self, session_id: str, title: str, *, reason: str) -> None:
        if self.tracer is None:
            return
        self.tracer.trace_session_selected(
            session_id=session_id,
            title=title,
            reason=reason,
        )

    def _trace_model_profile_selected(self, profile: ModelProfile, *, reason: str) -> None:
        if self.tracer is None:
            return
        self.tracer.trace_model_profile_selected(
            session_id=self._current_session_id(),
            model_profile_name=profile.name,
            provider=profile.provider,
            model_name=profile.model_name,
            reason=reason,
        )

    def _refresh_loaded_model_profile(self, profile_name: str) -> bool:
        try:
            warm_load_model_profile(self.settings, profile_name=profile_name)
        except Exception:
            return False
        return True

    def _trace_thinking_changed(self, *, reason: str) -> None:
        if self.tracer is None:
            return
        self.tracer.trace_thinking_changed(
            session_id=self._current_session_id(),
            model_profile_name=self.current_model_profile.name,
            thinking_enabled=self.thinking_enabled is True,
            reason=reason,
        )
