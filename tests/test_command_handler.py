from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.core.command_handler import CommandHandler, CommandStatus
from unclaw.settings import load_settings


def test_help_lists_enriched_commands_for_cli() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/help")

    assert result.status is CommandStatus.OK
    assert result.lines[0] == "Available slash commands for this channel:"
    assert "/new               Start a fresh session." in result.lines
    assert "/sessions          List recent sessions." in result.lines
    assert "/use <session_id>  Switch to an existing session." in result.lines
    assert "/model                 Show the active model profile." in result.lines
    assert "/model <profile_name>  Switch model profiles." in result.lines
    assert "/profiles              Show model profiles, tools, and context windows." in result.lines
    assert "/ctx                   Show context windows for each profile." in result.lines
    assert (
        "/ctx <profile_name> <num_ctx>  Save a context window override for one profile."
        in result.lines
    )
    assert "/think                 Show thinking mode status." in result.lines
    assert "/think on              Turn thinking mode on." in result.lines
    assert "/think off             Turn thinking mode off." in result.lines
    assert "/control              Show the current local access preset and roots." in result.lines
    assert "/control <preset>     Set local access to safe, workspace, or full." in result.lines
    assert "/tools            List built-in tools." in result.lines
    assert "/read <path>      Read one local file inside allowed roots." in result.lines
    assert (
        "/ls [path]        List one local directory inside allowed roots. Defaults to the current directory."
        in result.lines
    )
    assert "/fetch <url>      Fetch one public URL." in result.lines
    assert (
        "/search <query>  Search the public web, ground the answer, and append compact sources."
        in result.lines
    )
    assert "/session  Show the current session state." in result.lines
    assert "/summary  Show the saved session summary." in result.lines
    assert "/skills  Show enabled, installed, update, available, and orphaned skills." in result.lines
    assert "/skills search <query>  Search skills by id, name, summary, or tags." in result.lines
    assert "/skills install <skill_id>  Install one skill from the catalog." in result.lines
    assert "/skills enable <skill_id>  Enable one installed skill." in result.lines
    assert "/skills disable <skill_id>  Disable one enabled skill." in result.lines
    assert "/skills remove <skill_id>  Remove one installed skill bundle." in result.lines
    assert "/skills update <skill_id>  Update one installed skill." in result.lines
    assert "/skills update --all  Update every installed skill that has a newer catalog version." in result.lines
    assert "/help  Show this command list with examples." in result.lines
    assert "Examples:" in result.lines
    assert "/control workspace" in result.lines
    assert "/ctx main 4096" in result.lines
    assert "/ls ." in result.lines
    assert "/ls /home/user/project" in result.lines
    assert "/read README.md" in result.lines
    assert "/search local ai agents" in result.lines
    assert "/exit  Leave the terminal runtime." in result.lines
    assert (
        "Tip: use 'unclaw logs' or 'unclaw logs full' in another terminal."
        in result.lines
    )


def test_help_omits_exit_for_telegram() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        allow_exit=False,
    )

    result = handler.handle("/help")

    assert result.status is CommandStatus.OK
    assert "/exit  Leave the terminal runtime." not in result.lines
    assert (
        "Tip: use 'unclaw logs' or 'unclaw logs full' in another terminal."
        not in result.lines
    )


def test_fast_profile_forces_thinking_off_on_startup() -> None:
    settings = _with_default_fast_and_thinking_enabled(_load_repo_settings())

    handler = CommandHandler(
        settings=settings,
        session_manager=_build_session_manager(),
    )

    assert handler.current_model_profile_name == "fast"
    assert handler.thinking_enabled is False


def test_switching_to_fast_turns_thinking_off_cleanly() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        current_model_profile_name="main",
        thinking_enabled=True,
    )

    result = handler.handle("/model fast")

    assert result.status is CommandStatus.OK
    assert handler.current_model_profile_name == "fast"
    assert handler.thinking_enabled is False
    assert (
        "Thinking mode was turned off because fast mode does not support thinking."
        in result.lines
    )


def test_switching_model_profile_warm_loads_the_selected_profile(monkeypatch) -> None:
    warmed_profiles: list[str] = []

    monkeypatch.setattr(
        "unclaw.core.command_handler.warm_load_model_profile",
        lambda settings, profile_name: warmed_profiles.append(profile_name),
    )

    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        current_model_profile_name="main",
        thinking_enabled=False,
    )

    result = handler.handle("/model deep")

    assert result.status is CommandStatus.OK
    assert handler.current_model_profile_name == "deep"
    assert warmed_profiles == ["deep"]


def test_model_status_does_not_trigger_warm_load(monkeypatch) -> None:
    warmed_profiles: list[str] = []

    monkeypatch.setattr(
        "unclaw.core.command_handler.warm_load_model_profile",
        lambda settings, profile_name: warmed_profiles.append(profile_name),
    )

    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/model")

    assert result.status is CommandStatus.OK
    assert warmed_profiles == []


def test_unknown_model_profile_does_not_trigger_warm_load(monkeypatch) -> None:
    warmed_profiles: list[str] = []

    monkeypatch.setattr(
        "unclaw.core.command_handler.warm_load_model_profile",
        lambda settings, profile_name: warmed_profiles.append(profile_name),
    )

    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/model ghost")

    assert result.status is CommandStatus.ERROR
    assert warmed_profiles == []


def test_warm_load_failure_does_not_block_model_switch(monkeypatch) -> None:
    monkeypatch.setattr(
        "unclaw.core.command_handler.warm_load_model_profile",
        lambda settings, profile_name: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        current_model_profile_name="main",
        thinking_enabled=False,
    )

    result = handler.handle("/model deep")

    assert result.status is CommandStatus.OK
    assert handler.current_model_profile_name == "deep"


def test_think_status_on_fast_explains_that_thinking_is_unsupported() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        current_model_profile_name="fast",
        thinking_enabled=False,
    )

    result = handler.handle("/think")

    assert result.status is CommandStatus.OK
    assert result.lines == (
        "Thinking mode: off",
        "Fast mode does not support thinking.",
    )


def test_think_on_fails_clearly_on_fast() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        current_model_profile_name="fast",
        thinking_enabled=False,
    )

    result = handler.handle("/think on")

    assert result.status is CommandStatus.ERROR
    assert handler.thinking_enabled is False
    assert result.lines == (
        "Fast mode does not support thinking. "
        "Switch to another model profile to turn it on.",
    )


def test_switching_back_to_supported_model_keeps_thinking_off() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
        current_model_profile_name="main",
        thinking_enabled=True,
    )

    assert handler.handle("/model fast").status is CommandStatus.OK

    result = handler.handle("/model main")

    assert result.status is CommandStatus.OK
    assert handler.current_model_profile_name == "main"
    assert handler.thinking_enabled is False


def test_profiles_lists_models_and_context_windows() -> None:
    settings = _load_repo_settings()
    handler = CommandHandler(
        settings=settings,
        session_manager=_build_session_manager(),
        current_model_profile_name="main",
    )

    result = handler.handle("/profiles")

    assert result.status is CommandStatus.OK
    assert result.lines[0] == "Model profiles:"
    assert any(
        "main | model=" in line and f"ctx={settings.models['main'].num_ctx}" in line
        for line in result.lines
    )
    assert any("| current" in line for line in result.lines)


def test_ctx_without_arguments_lists_current_context_windows() -> None:
    settings = _load_repo_settings()
    handler = CommandHandler(
        settings=settings,
        session_manager=_build_session_manager(),
        current_model_profile_name="main",
    )

    result = handler.handle("/ctx")

    assert result.status is CommandStatus.OK
    assert result.lines[0] == "Context windows:"
    assert any(
        f"main | ctx={settings.models['main'].num_ctx}" in line
        for line in result.lines
    )
    assert result.lines[-1] == "Use /ctx <profile_name> <num_ctx> to save a new value."


def test_control_without_arguments_shows_current_preset_and_roots() -> None:
    settings = _load_repo_settings()
    handler = CommandHandler(
        settings=settings,
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/control")

    assert result.status is CommandStatus.OK
    assert (
        result.lines[0]
        == f"Control preset: {settings.app.security.tools.files.control_preset}"
    )
    assert "Meaning:" in result.lines[1]
    assert result.lines[2].startswith("Tool access: ")
    assert result.lines[3].startswith("Core tools:")
    assert result.lines[4] == "Read roots:"
    assert "Write roots:" in result.lines
    assert "Terminal working roots:" in result.lines


def test_control_command_persists_new_preset(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = _build_session_manager(settings=settings)
    handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
    )

    result = handler.handle("/control safe")

    assert result.status is CommandStatus.OK
    assert result.updated_settings is not None
    assert result.refresh_tool_executor is True
    assert handler.settings.app.security.tools.files.control_preset == "safe"
    assert handler.settings.app.security.tools.files.read_allowed_roots == (
        "config",
        "data",
        "data/files",
    )
    assert handler.settings.app.security.tools.files.write_allowed_roots == ("data",)
    assert result.lines[0] == "Saved control preset: safe."
    assert (
        result.lines[1]
        == "New file and terminal tool access rules apply immediately in this CLI."
    )


def test_control_command_persists_after_settings_reload(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    handler = CommandHandler(
        settings=settings,
        session_manager=_build_session_manager(settings=settings),
    )

    result = handler.handle("/control safe")

    assert result.status is CommandStatus.OK

    reloaded_settings = load_settings(project_root=project_root)
    reloaded_handler = CommandHandler(
        settings=reloaded_settings,
        session_manager=_build_session_manager(settings=reloaded_settings),
    )

    reloaded_result = reloaded_handler.handle("/control")

    assert reloaded_result.status is CommandStatus.OK
    assert reloaded_result.lines[0] == "Control preset: safe"


def test_control_command_rejects_invalid_preset() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/control ghost")

    assert result.status is CommandStatus.ERROR
    assert result.lines == (
        "Unknown control preset 'ghost'. Available presets: safe, workspace, full.",
    )


def test_control_command_reports_config_write_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        "unclaw.core.command_handler.persist_control_preset",
        lambda settings, preset_name: (_ for _ in ()).throw(OSError("disk full")),
    )
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/control full")

    assert result.status is CommandStatus.ERROR
    assert result.lines == ("Could not save control preset: disk full",)


def test_ctx_command_persists_new_context_window(
    make_temp_project,
    monkeypatch,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = _build_session_manager(settings=settings)
    handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
    )
    refreshed_profiles: list[str] = []

    monkeypatch.setattr(
        "unclaw.core.command_handler.warm_load_model_profile",
        lambda settings, profile_name: refreshed_profiles.append(profile_name),
    )

    result = handler.handle("/ctx main 4096")

    assert result.status is CommandStatus.OK
    assert result.updated_settings is not None
    assert result.refresh_tool_executor is True
    assert handler.settings.models["main"].num_ctx == 4096
    assert refreshed_profiles == ["main"]
    assert result.lines == (
        "Saved context window: main=4096.",
        "Reloaded active model profile: main.",
        "The new context window will be used on the next turn in this CLI.",
    )


def test_ctx_command_falls_back_cleanly_when_active_profile_refresh_fails(
    make_temp_project,
    monkeypatch,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    handler = CommandHandler(
        settings=settings,
        session_manager=_build_session_manager(settings=settings),
    )

    monkeypatch.setattr(
        "unclaw.core.command_handler.warm_load_model_profile",
        lambda settings, profile_name: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    result = handler.handle("/ctx main 4096")

    assert result.status is CommandStatus.OK
    assert result.lines == (
        "Saved context window: main=4096.",
        "Could not refresh the active model profile: main.",
        "The new value is guaranteed on next model reload or CLI restart.",
    )


def test_ctx_command_for_inactive_profile_skips_refresh_and_explains_next_load(
    make_temp_project,
    monkeypatch,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    handler = CommandHandler(
        settings=settings,
        session_manager=_build_session_manager(settings=settings),
        current_model_profile_name="main",
    )
    refreshed_profiles: list[str] = []

    monkeypatch.setattr(
        "unclaw.core.command_handler.warm_load_model_profile",
        lambda settings, profile_name: refreshed_profiles.append(profile_name),
    )

    result = handler.handle("/ctx deep 4096")

    assert result.status is CommandStatus.OK
    assert refreshed_profiles == []
    assert result.lines == (
        "Saved context window: deep=4096.",
        "deep is not the active profile. The saved value will be used the next time that model is loaded.",
    )


def test_ctx_command_can_retry_refresh_when_value_is_already_saved(
    make_temp_project,
    monkeypatch,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    handler = CommandHandler(
        settings=settings,
        session_manager=_build_session_manager(settings=settings),
    )
    refreshed_profiles: list[str] = []

    monkeypatch.setattr(
        "unclaw.core.command_handler.warm_load_model_profile",
        lambda settings, profile_name: refreshed_profiles.append(profile_name),
    )

    result = handler.handle(f"/ctx main {settings.models['main'].num_ctx}")

    assert result.status is CommandStatus.OK
    assert refreshed_profiles == ["main"]
    assert result.lines == (
        f"Context window already saved: main={settings.models['main'].num_ctx}.",
        "Reloaded active model profile: main.",
        "The new context window will be used on the next turn in this CLI.",
    )


def test_ctx_command_rejects_invalid_num_ctx() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/ctx main nope")

    assert result.status is CommandStatus.ERROR
    assert result.lines == ("Context window must be an integer.",)


def test_ctx_command_rejects_too_small_num_ctx() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/ctx main 512")

    assert result.status is CommandStatus.ERROR
    assert result.lines == ("Context window must be at least 1024 tokens.",)


@pytest.mark.parametrize(
    ("command", "expected_line"),
    (
        ("/profiles extra", "Usage: /profiles"),
        ("/ctx main", "Usage: /ctx <profile_name> <num_ctx>"),
        ("/control safe extra", "Usage: /control <safe|workspace|full>"),
    ),
)
def test_control_profile_and_ctx_commands_preserve_usage_contracts(
    command: str,
    expected_line: str,
) -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle(command)

    assert result.status is CommandStatus.ERROR
    assert result.lines == (expected_line,)


def test_refresh_loaded_model_profile_returns_true_on_success(monkeypatch) -> None:
    refreshed_profiles: list[str] = []
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    monkeypatch.setattr(
        "unclaw.core.command_handler.warm_load_model_profile",
        lambda settings, profile_name: refreshed_profiles.append(profile_name),
    )

    assert handler._refresh_loaded_model_profile("main") is True
    assert refreshed_profiles == ["main"]


def test_refresh_loaded_model_profile_returns_false_on_failure(monkeypatch) -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    monkeypatch.setattr(
        "unclaw.core.command_handler.warm_load_model_profile",
        lambda settings, profile_name: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    assert handler._refresh_loaded_model_profile("main") is False


def test_ls_defaults_to_current_directory_when_no_path_is_given() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/ls")

    assert result.status is CommandStatus.OK
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "list_directory"
    assert result.tool_call.arguments == {"path": "."}


def test_search_uses_freeform_query_and_wires_to_search_tool() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/search local-first agent runtime")

    assert result.status is CommandStatus.OK
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "search_web"
    assert result.tool_call.arguments == {
        "query": "local-first agent runtime",
    }


def test_search_accepts_apostrophes_in_natural_language_queries() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/search qu'est-ce qu'il s'est passe aujourd'hui ?")

    assert result.status is CommandStatus.OK
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "search_web"
    assert result.tool_call.arguments == {
        "query": "qu'est-ce qu'il s'est passe aujourd'hui ?",
    }


def test_read_keeps_useful_quoted_paths_working() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle('/read "docs/My Notes.txt"')

    assert result.status is CommandStatus.OK
    assert result.tool_call is not None
    assert result.tool_call.tool_name == "read_text_file"
    assert result.tool_call.arguments == {"path": "docs/My Notes.txt"}


def _load_repo_settings():
    return load_settings(
        project_root=Path(__file__).resolve().parents[1],
        include_local_overrides=False,
    )


def _build_session_manager(settings=None) -> SimpleNamespace:
    return SimpleNamespace(current_session_id="sess-current", settings=settings)


def _with_default_fast_and_thinking_enabled(settings):
    return replace(
        settings,
        app=replace(
            settings.app,
            default_model_profile="fast",
            thinking=replace(settings.app.thinking, default_enabled=True),
        ),
    )
