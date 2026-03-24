"""Interactive onboarding for first-time and advanced local setup."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from unclaw.bootstrap import bootstrap
from unclaw.errors import UnclawError
from unclaw.local_secrets import local_secrets_path
from unclaw.onboarding_files import (
    build_onboarding_file_payloads,
    recommended_model_profiles,
    write_onboarding_files,
)
from unclaw.onboarding_flow import (
    build_onboarding_banner as _build_onboarding_banner,
    detect_system_ram_gib as _detect_system_ram_gib,
    default_channel_preset as _default_channel_preset,
    enabled_channels_from_preset as _enabled_channels_from_preset,
    load_existing_local_secrets as _load_existing_local_secrets,
    load_existing_telegram_config as _load_existing_telegram_config,
    post_configure_ollama as _post_configure_ollama,
    print_plan_summary as _print_plan_summary,
    prompt_catalog_skill_selection as _prompt_catalog_skill_selection,
    prompt_channel_preset as _prompt_channel_preset,
    prompt_model_pack as _prompt_model_pack,
    prompt_model_profiles as _prompt_model_profiles,
    prompt_telegram_bot_token as _prompt_telegram_bot_token,
)
from unclaw.model_packs import DEV_MODEL_PACK_NAME, recommend_model_pack
from unclaw.onboarding_types import (
    InputFunc,
    MenuOption,
    ModelProfileDraft,
    OnboardingPlan,
    OutputFunc,
    PROFILE_ORDER,
    PromptUI,
)
from unclaw.onboarding_ui import (
    FallbackPromptUIBase,
    InteractivePromptUIBase,
    ask_select_question as _run_select_question,
)
from unclaw.settings import Settings
from unclaw.startup import (
    find_missing_model_profiles,
    inspect_ollama,
)
from unclaw.terminal_styles import onboarding_questionary_style_entries

try:
    import questionary
except ImportError:  # pragma: no cover - exercised in environments without the UX dependency.
    questionary = None

_PROFILE_ORDER = PROFILE_ORDER

_SETUP_MODE_OPTIONS = (
    MenuOption(
        value="recommended",
        label="Recommended setup",
        description="Start with the recommended local model set and guided defaults.",
    ),
    MenuOption(
        value="advanced",
        label="Advanced setup",
        description="Choose logging, channels, and every model profile manually.",
    ),
)

_LOGGING_OPTIONS = (
    MenuOption(
        value="simple",
        label="Simple logs",
        description="Cleaner day-to-day output with less terminal noise.",
    ),
    MenuOption(
        value="full",
        label="Full logs",
        description="More runtime detail for testing, tuning, and debugging.",
    ),
)


class FallbackPromptUI(FallbackPromptUIBase):
    """Compatibility wrapper for the plain-text onboarding prompt UI."""


class InteractivePromptUI(InteractivePromptUIBase):
    """Compatibility wrapper for the questionary-backed onboarding UI."""

    def _questionary(self) -> Any:
        if questionary is None:  # pragma: no cover - guarded by _build_prompt_ui.
            raise RuntimeError("questionary is not available")
        return questionary

    def _questionary_style(self) -> Any:
        return _QUESTIONARY_STYLE

    def _ask_select_question(
        self,
        prompt: str,
        *,
        choices: list[Any],
        initial_choice: str,
        instruction: str,
    ) -> str | None:
        return _ask_select_question(
            prompt,
            choices=choices,
            initial_choice=initial_choice,
            instruction=instruction,
        )


if questionary is not None:
    _QUESTIONARY_STYLE = questionary.Style(
        list(onboarding_questionary_style_entries())
    )
else:  # pragma: no cover - exercised when questionary is unavailable.
    _QUESTIONARY_STYLE = None


def _ask_select_question(
    prompt: str,
    *,
    choices: list[Any],
    initial_choice: str,
    instruction: str,
) -> str | None:
    return _run_select_question(
        prompt,
        choices=choices,
        initial_choice=initial_choice,
        instruction=instruction,
        style=_QUESTIONARY_STYLE,
    )


def main(project_root: Path | None = None) -> int:
    """Run interactive onboarding from the command line."""

    try:
        settings = bootstrap(project_root=project_root)
        return run_onboarding(settings)
    except (KeyboardInterrupt, EOFError):
        print("\nOnboarding cancelled.", file=sys.stderr)
        return 1
    except UnclawError as exc:
        print(f"Failed to run Unclaw onboarding: {exc}", file=sys.stderr)
        return 1


def run_onboarding(
    settings: Settings,
    *,
    input_func: InputFunc = input,
    output_func: OutputFunc = print,
) -> int:
    """Run the interactive onboarding flow with injectable I/O for tests."""

    prompt_ui = _build_prompt_ui(input_func=input_func, output_func=output_func)
    telegram_config, telegram_warning = _load_existing_telegram_config(settings)
    local_secrets, local_secrets_warning = _load_existing_local_secrets(settings)
    ollama_status = inspect_ollama()

    output_func(
        _build_onboarding_banner(settings=settings, ollama_status=ollama_status)
    )
    output_func("Welcome to Unclaw setup.")
    output_func("This guide only updates local files in this project.")
    output_func("Your models, chats, and secrets stay on your machine.")
    if prompt_ui.interactive:
        output_func("Use arrow keys to move, Enter to confirm, and Ctrl-C to cancel.")
    if telegram_warning is not None:
        output_func(telegram_warning)
    if local_secrets_warning is not None:
        output_func(local_secrets_warning)

    prompt_ui.section(
        "🛠 Setup style",
        "Choose how much Unclaw should decide for you during first-time setup.",
    )
    setup_mode = prompt_ui.select(
        "How do you want to configure Unclaw?",
        options=_SETUP_MODE_OPTIONS,
        default="recommended",
    )
    beginner_mode = setup_mode == "recommended"

    prompt_ui.section(
        "🪵 Logging",
        "Choose how much runtime detail you want during startup and normal use.",
    )
    logging_mode = prompt_ui.select(
        "Which log view should Unclaw use day to day?",
        options=_LOGGING_OPTIONS,
        default="simple" if beginner_mode else settings.app.logging.mode,
    )

    prompt_ui.section(
        "📡 Channels",
        "Choose how people should talk to Unclaw in this project.",
    )
    enabled_channels = _enabled_channels_from_preset(
        _prompt_channel_preset(
            prompt_ui=prompt_ui,
            default_preset=_default_channel_preset(settings),
        )
    )

    prompt_ui.section(
        "🔌 Optional skills",
        "Skills extend built-in tools with optional domain-specific capabilities. "
        "Selected skills are installed from the official catalog.",
    )
    _catalog_entries, _catalog_error = _fetch_onboarding_catalog(
        settings.catalog.url, output_func=output_func
    )
    selected_skill_ids = _prompt_catalog_skill_selection(
        settings,
        prompt_ui=prompt_ui,
        catalog_entries=_catalog_entries,
    )

    prompt_ui.section(
        "🧠 Model profiles",
        "Choose a pack for your machine, or switch to manual control with the dev pack.",
    )
    detected_ram_gib = _detect_system_ram_gib()
    recommended_pack = recommend_model_pack(detected_ram_gib)
    default_pack = recommended_pack if beginner_mode else settings.model_pack
    selected_pack = _prompt_model_pack(
        prompt_ui=prompt_ui,
        detected_ram_gib=detected_ram_gib,
        default_pack=default_pack,
    )
    manual_model_pack = selected_pack == DEV_MODEL_PACK_NAME
    automatic_configuration = beginner_mode and not manual_model_pack

    if manual_model_pack:
        model_profiles = _prompt_model_profiles(
            settings,
            prompt_ui=prompt_ui,
            ollama_status=ollama_status,
            recommended_pack=recommended_pack,
        )
    else:
        model_profiles = recommended_model_profiles(selected_pack)
        prompt_ui.info(f"Using the {selected_pack} pack model lineup:")
        for profile_name in _PROFILE_ORDER:
            prompt_ui.info(
                f"- {profile_name}: {model_profiles[profile_name].model_name}"
            )

    telegram_bot_token = local_secrets.telegram_bot_token
    telegram_bot_token_env_var = telegram_config.bot_token_env_var
    if "telegram" in enabled_channels:
        telegram_bot_token = _prompt_telegram_bot_token(
            existing_token=local_secrets.telegram_bot_token,
            prompt_ui=prompt_ui,
        )

    plan = OnboardingPlan(
        beginner_mode=beginner_mode,
        automatic_configuration=automatic_configuration,
        logging_mode=logging_mode,
        enabled_channels=enabled_channels,
        enabled_skill_ids=selected_skill_ids,
        default_profile="main",
        model_pack=selected_pack,
        model_profiles=model_profiles,
        telegram_bot_token=telegram_bot_token,
        telegram_bot_token_env_var=telegram_bot_token_env_var,
        telegram_allowed_chat_ids=tuple(sorted(telegram_config.allowed_chat_ids)),
        telegram_polling_timeout_seconds=telegram_config.polling_timeout_seconds,
    )

    prompt_ui.section(
        "📝 Review",
        "Check the plan before Unclaw writes local configuration files.",
    )
    _print_plan_summary(plan, output_func=output_func)
    output_func(
        "- Local access: "
        f"{settings.app.security.tools.files.control_preset} preset "
        "(change later with `/control`)."
    )
    should_write_config = prompt_ui.confirm(
        "Write this configuration now?",
        default=True,
    )
    if not should_write_config:
        output_func("No files were changed.")
        return 1

    # Install selected skills from the remote catalog before writing config.
    installed_skill_ids = _install_onboarding_skills(
        settings,
        plan=plan,
        catalog_entries=_catalog_entries,
        output_func=output_func,
    )
    # Only enable skills that were actually installed.
    plan = OnboardingPlan(
        beginner_mode=plan.beginner_mode,
        automatic_configuration=plan.automatic_configuration,
        logging_mode=plan.logging_mode,
        enabled_channels=plan.enabled_channels,
        enabled_skill_ids=installed_skill_ids,
        default_profile=plan.default_profile,
        model_pack=plan.model_pack,
        model_profiles=plan.model_profiles,
        telegram_bot_token=plan.telegram_bot_token,
        telegram_bot_token_env_var=plan.telegram_bot_token_env_var,
        telegram_allowed_chat_ids=plan.telegram_allowed_chat_ids,
        telegram_polling_timeout_seconds=plan.telegram_polling_timeout_seconds,
    )

    write_onboarding_files(settings, plan)
    configured_settings = bootstrap(project_root=settings.paths.project_root)
    output_func("")
    output_func("Saved local configuration:")
    output_func(f"- {configured_settings.paths.app_config_path}")
    output_func(f"- {configured_settings.paths.models_config_path}")
    output_func(f"- {configured_settings.paths.config_dir / 'telegram.yaml'}")
    if plan.telegram_bot_token is not None:
        output_func(f"- {local_secrets_path(configured_settings)}")

    ollama_status = _post_configure_ollama(
        configured_settings,
        plan,
        prompt_ui=prompt_ui,
        inspect_ollama_func=inspect_ollama,
    )

    output_func("")
    output_func("What to do next:")
    output_func("- Start the terminal experience with `unclaw start`.")
    output_func("- Review local access later with `/control` inside the CLI.")
    output_func("- Tune profile context later with `/ctx <profile> <num_ctx>`.")
    if "telegram" in enabled_channels:
        output_func("- Start your Telegram bot with `unclaw telegram`.")
        output_func(
            f"- Your bot token is stored locally in `{local_secrets_path(configured_settings)}`."
        )
        if plan.telegram_allowed_chat_ids:
            count = len(plan.telegram_allowed_chat_ids)
            label = "chat" if count == 1 else "chats"
            output_func(
                f"- Telegram access: {count} authorized {label} already configured. Review them with `unclaw telegram list`."
            )
        else:
            output_func(
                "- Telegram access: secure deny-by-default until you authorize a chat locally."
            )
            output_func(
                "- Tip: send one test message, then run `unclaw telegram allow-latest` on this machine."
            )
        output_func(
            f"- Advanced fallback: export {plan.telegram_bot_token_env_var}=<your bot token>"
        )
    else:
        output_func("- Enable Telegram later by rerunning `unclaw onboard`.")

    if ollama_status is not None:
        missing_profiles = find_missing_model_profiles(
            configured_settings,
            installed_model_names=ollama_status.model_names,
            profile_names=tuple(plan.model_profiles),
        )
        if missing_profiles:
            output_func("- Pull the remaining missing models before heavier use.")

    return 0


def _fetch_onboarding_catalog(
    catalog_url: str,
    *,
    output_func: OutputFunc,
) -> "tuple[list, str | None]":
    """Fetch the remote catalog for onboarding skill selection.

    Returns ``(entries, error_message)``.  On failure, entries is empty and
    error_message describes the problem.  The caller should surface the error
    to the user but continue onboarding without skill installation.
    """
    from unclaw.skills.remote_catalog import CatalogFetchError, fetch_remote_catalog

    try:
        entries = fetch_remote_catalog(catalog_url)
        return entries, None
    except CatalogFetchError as exc:
        msg = (
            f"Could not reach the skills catalog ({exc}). "
            "Skill installation will be skipped — you can run `unclaw onboard` "
            "again once the network is available."
        )
        output_func(msg)
        return [], msg


def _install_onboarding_skills(
    settings: "Settings",
    *,
    plan: "OnboardingPlan",
    catalog_entries: "list",
    output_func: OutputFunc,
) -> "tuple[str, ...]":
    """Install skills selected during onboarding from the remote catalog.

    Downloads each selected skill into ``./skills/`` and returns the IDs of
    skills that were successfully installed.  Skills that fail to install are
    reported to *output_func* and excluded from the returned tuple.
    """
    if not plan.enabled_skill_ids:
        return ()

    from unclaw.skills.installer import SkillInstallError, install_skill
    from unclaw.skills.remote_catalog import RemoteCatalogEntry

    catalog_by_id: dict[str, RemoteCatalogEntry] = {
        e.skill_id: e
        for e in catalog_entries
        if isinstance(e, RemoteCatalogEntry)
    }
    skills_root = settings.paths.project_root / "skills"
    installed: list[str] = []

    for skill_id in plan.enabled_skill_ids:
        entry = catalog_by_id.get(skill_id)
        if entry is None:
            # Skill was selected but not found in catalog — skip silently.
            continue
        output_func(f"Installing {entry.display_name}...")
        try:
            install_skill(
                entry,
                skills_root=skills_root,
                catalog_url=settings.catalog.url,
            )
            output_func(f"  ✓ {entry.display_name} installed.")
            installed.append(skill_id)
        except SkillInstallError as exc:
            output_func(f"  ✗ Could not install {entry.display_name}: {exc}")

    return tuple(installed)


def _build_prompt_ui(
    *,
    input_func: InputFunc,
    output_func: OutputFunc,
) -> PromptUI:
    if (
        input_func is input
        and output_func is print
        and questionary is not None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    ):
        return InteractivePromptUI(output_func=output_func)
    return FallbackPromptUI(input_func=input_func, output_func=output_func)


__all__ = [
    "FallbackPromptUI",
    "InteractivePromptUI",
    "MenuOption",
    "ModelProfileDraft",
    "OnboardingPlan",
    "PromptUI",
    "build_onboarding_file_payloads",
    "main",
    "recommended_model_profiles",
    "run_onboarding",
    "write_onboarding_files",
]
