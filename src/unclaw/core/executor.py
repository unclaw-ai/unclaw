"""Minimal tool executor wiring for the current runtime phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from unclaw.memory.long_term_store import LongTermStore
from unclaw.settings import Settings
from unclaw.skills.bundle_tools import register_active_skill_tools
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolResult
from unclaw.tools.dispatcher import ToolDispatcher
from unclaw.tools.file_tools import (
    DELETE_FILE_DEFINITION,
    LIST_DIRECTORY_DEFINITION,
    READ_TEXT_FILE_DEFINITION,
    WRITE_TEXT_FILE_DEFINITION,
    register_file_tools,
)
from unclaw.tools.long_term_memory_tools import (
    FORGET_LONG_TERM_MEMORY_DEFINITION,
    LIST_LONG_TERM_MEMORY_DEFINITION,
    REMEMBER_LONG_TERM_MEMORY_DEFINITION,
    SEARCH_LONG_TERM_MEMORY_DEFINITION,
    register_long_term_memory_tools,
)
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.session_tools import INSPECT_SESSION_HISTORY_DEFINITION, register_session_tools
from unclaw.tools.system_tools import SYSTEM_INFO_DEFINITION, register_system_tools
from unclaw.tools.terminal_tools import (
    RUN_TERMINAL_COMMAND_DEFINITION,
    register_terminal_tools,
)
from unclaw.tools.web_tools import (
    FAST_WEB_SEARCH_DEFINITION,
    FETCH_URL_TEXT_DEFINITION,
    SEARCH_WEB_DEFINITION,
    register_web_tools,
)

if TYPE_CHECKING:
    from unclaw.core.session_manager import SessionManager

BUILTIN_TOOL_COMMANDS = MappingProxyType(
    {
        "read": READ_TEXT_FILE_DEFINITION.name,
        "ls": LIST_DIRECTORY_DEFINITION.name,
        "fetch": FETCH_URL_TEXT_DEFINITION.name,
        "search": SEARCH_WEB_DEFINITION.name,
    }
)


def register_default_tools(registry: ToolRegistry) -> ToolRegistry:
    """Register the built-in tools on the provided registry.

    Used when no Settings object is available (e.g. test helpers).
    Skills are not loaded here — use create_default_tool_registry(settings) for
    a fully-configured runtime with active skills enabled.
    """
    register_file_tools(registry)
    register_web_tools(registry)
    register_system_tools(registry)
    register_terminal_tools(registry)
    return registry


def create_default_tool_registry(
    settings: Settings | None = None,
    session_manager: SessionManager | None = None,
) -> ToolRegistry:
    """Create a registry populated with the initial built-in tools."""
    registry = ToolRegistry()
    if settings is None:
        register_default_tools(registry)
    else:
        register_file_tools(
            registry,
            project_root=settings.paths.project_root,
            configured_roots=settings.app.security.tools.files.allowed_roots,
            default_write_dir=settings.paths.files_dir,
            default_read_dir=settings.paths.files_dir,
            allow_destructive_file_overwrite=(
                settings.app.security.tools.files.allow_destructive_file_overwrite
            ),
        )
        from unclaw.tools.web_research import (
            MAIN_RESEARCH_BUDGET,
            ResearchConfig,
            resolve_research_budget,
        )

        # Resolve research budget from active model pack.
        _profile_ctx = 8192  # safe default
        _main_profile = settings.models.get("main")
        if _main_profile is not None and _main_profile.num_ctx:
            _profile_ctx = _main_profile.num_ctx

        _research_budget = resolve_research_budget(
            effective_context=_profile_ctx,
            profile_name="main",
        )
        _research_config = ResearchConfig(settings=settings)

        register_web_tools(
            registry,
            allow_private_networks=settings.app.security.tools.fetch.allow_private_networks,
            research_config=_research_config,
            research_budget=_research_budget,
            workspace_base_dir=settings.paths.data_dir / "web_search",
        )
        register_active_skill_tools(
            registry,
            enabled_skill_ids=settings.skills.enabled_skill_ids,
            skills_root=settings.paths.project_root / "skills",
        )
        register_system_tools(registry)
        register_terminal_tools(
            registry,
            project_root=settings.paths.project_root,
            configured_roots=settings.app.security.tools.files.allowed_roots,
            default_working_directory=settings.paths.project_root,
            max_timeout_seconds=settings.app.runtime.tool_timeout_seconds,
        )

    if session_manager is not None:
        register_session_tools(registry, session_manager=session_manager)

    if settings is not None:
        long_term_db_path = settings.paths.data_dir / "memory" / "long_term.db"
        register_long_term_memory_tools(
            registry,
            long_term_store=LongTermStore(long_term_db_path),
        )

    return registry


def resolve_builtin_tool_command(command_name: str) -> str | None:
    """Resolve one CLI tool command name to a registered built-in tool name."""
    return BUILTIN_TOOL_COMMANDS.get(command_name)


@dataclass(slots=True)
class ToolExecutor:
    """Thin runtime-facing wrapper around the tool dispatcher."""

    registry: ToolRegistry
    dispatcher: ToolDispatcher = field(init=False)

    def __post_init__(self) -> None:
        self.dispatcher = ToolDispatcher(self.registry)

    @classmethod
    def with_default_tools(cls, settings: Settings | None = None) -> ToolExecutor:
        return cls(registry=create_default_tool_registry(settings))

    def list_tools(self) -> list[ToolDefinition]:
        return self.registry.list_builtin_tools()

    def execute(self, call: ToolCall) -> ToolResult:
        return self.dispatcher.dispatch(call)


def execute_tool_call(
    call: ToolCall,
    *,
    registry: ToolRegistry | None = None,
    settings: Settings | None = None,
) -> ToolResult:
    """Execute one tool call using the provided or default registry."""
    active_registry = (
        registry if registry is not None else create_default_tool_registry(settings)
    )
    dispatcher = ToolDispatcher(active_registry)
    return dispatcher.dispatch(call)


__all__ = [
    "BUILTIN_TOOL_COMMANDS",
    "DELETE_FILE_DEFINITION",
    "FAST_WEB_SEARCH_DEFINITION",
    "FORGET_LONG_TERM_MEMORY_DEFINITION",
    "INSPECT_SESSION_HISTORY_DEFINITION",
    "LIST_LONG_TERM_MEMORY_DEFINITION",
    "REMEMBER_LONG_TERM_MEMORY_DEFINITION",
    "RUN_TERMINAL_COMMAND_DEFINITION",
    "SEARCH_LONG_TERM_MEMORY_DEFINITION",
    "SYSTEM_INFO_DEFINITION",
    "ToolExecutor",
    "WRITE_TEXT_FILE_DEFINITION",
    "create_default_tool_registry",
    "execute_tool_call",
    "register_default_tools",
    "resolve_builtin_tool_command",
]
