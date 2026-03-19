"""Shared onboarding flow helpers outside of the prompt UI layer."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from unclaw.channels.telegram_bot import TelegramConfig, load_telegram_config
from unclaw.errors import ConfigurationError, UnclawError
from unclaw.local_secrets import (
    DEFAULT_TELEGRAM_TOKEN_ENV_VAR,
    LocalSecrets,
    load_local_secrets,
    validate_telegram_bot_token,
)
from unclaw.onboarding_files import recommended_model_profiles
from unclaw.onboarding_types import (
    MenuOption,
    ModelProfileDraft,
    OnboardingPlan,
    OutputFunc,
    PROFILE_DESCRIPTIONS,
    PROFILE_ORDER,
    PromptUI,
)
from unclaw.settings import ModelProfile, Settings
from unclaw.startup import (
    OllamaStatus,
    build_banner,
    find_missing_model_profiles,
    inspect_ollama,
    ollama_install_guidance,
    start_ollama_server,
    wait_for_ollama,
)

_CHANNEL_PRESET_OPTIONS = (
    MenuOption(
        value="terminal_only",
        label="Terminal only",
        description="Keep Unclaw in the terminal only.",
    ),
    MenuOption(
        value="terminal_and_telegram",
        label="Terminal + Telegram",
        description="Use both the terminal and your Telegram bot.",
    ),
    MenuOption(
        value="telegram_only",
        label="Telegram only",
        description="Run Unclaw only through Telegram.",
    ),
)


def load_existing_telegram_config(
    settings: Settings,
) -> tuple[TelegramConfig, str | None]:
    try:
        return load_telegram_config(settings), None
    except UnclawError as exc:
        warning = (
            "Telegram config note: "
            f"{exc} Using {DEFAULT_TELEGRAM_TOKEN_ENV_VAR} as the advanced fallback "
            "environment variable."
        )
        return (
            TelegramConfig(
                bot_token_env_var=DEFAULT_TELEGRAM_TOKEN_ENV_VAR,
                polling_timeout_seconds=30,
                allowed_chat_ids=frozenset(),
            ),
            warning,
        )


def load_existing_local_secrets(
    settings: Settings,
) -> tuple[LocalSecrets, str | None]:
    try:
        return load_local_secrets(settings), None
    except UnclawError as exc:
        warning = (
            "Local secrets note: "
            f"{exc} Unclaw will ask you to rewrite the Telegram token for this project."
        )
        return LocalSecrets(), warning


def default_channel_preset(settings: Settings) -> str:
    terminal_enabled = settings.app.channels.terminal_enabled
    telegram_enabled = settings.app.channels.telegram_enabled
    if terminal_enabled and telegram_enabled:
        return "terminal_and_telegram"
    if telegram_enabled:
        return "telegram_only"
    return "terminal_only"


def prompt_channel_preset(
    *,
    prompt_ui: PromptUI,
    default_preset: str,
) -> str:
    return prompt_ui.select(
        "Which channel setup should Unclaw enable?",
        options=_CHANNEL_PRESET_OPTIONS,
        default=default_preset,
        help_text=(
            "Pick one clear setup. If Telegram is enabled, Unclaw will ask for "
            "the bot token and store it locally for this project."
        ),
    )


def enabled_channels_from_preset(channel_preset: str) -> tuple[str, ...]:
    match channel_preset:
        case "terminal_only":
            return ("terminal",)
        case "terminal_and_telegram":
            return ("terminal", "telegram")
        case "telegram_only":
            return ("telegram",)
        case _:
            raise ValueError(f"Unsupported channel preset: {channel_preset}")


def prompt_model_profiles(
    settings: Settings,
    *,
    prompt_ui: PromptUI,
    ollama_status: OllamaStatus,
) -> dict[str, ModelProfileDraft]:
    profiles: dict[str, ModelProfileDraft] = {}
    recommended = recommended_model_profiles()
    installed_model_names = (
        ollama_status.model_names if ollama_status.is_running else ()
    )

    for profile_name in PROFILE_ORDER:
        current_profile = settings.models.get(profile_name)
        current_model_name = (
            current_profile.model_name
            if current_profile is not None
            else recommended[profile_name].model_name
        )
        recommended_model_name = recommended[profile_name].model_name

        options, default_choice = _build_profile_menu_options(
            profile_name=profile_name,
            has_current_profile=current_profile is not None,
            current_model_name=current_model_name,
            recommended_model_name=recommended_model_name,
            installed_model_names=installed_model_names,
        )
        selection = prompt_ui.select(
            f"{profile_name.title()} profile",
            options=options,
            default=default_choice,
            help_text=_build_profile_help_text(
                profile_name=profile_name,
                installed_model_names=installed_model_names,
            ),
        )

        if selection == "custom":
            template = (
                _draft_from_profile(current_profile)
                if current_profile is not None
                else recommended[profile_name]
            )
            model_name = prompt_ui.text(
                f"Custom model name for {profile_name}",
                default=current_model_name,
                help_text=(
                    "Enter the exact local model name you plan to pull or already "
                    "have in Ollama."
                ),
                validator=_validate_model_name,
            )
        elif selection == "recommended":
            template = recommended[profile_name]
            model_name = recommended_model_name
        elif selection == "installed":
            template = (
                _draft_from_profile(current_profile)
                if current_profile is not None
                else recommended[profile_name]
            )
            model_name = _prompt_installed_model_name(
                profile_name=profile_name,
                current_model_name=current_model_name,
                recommended_model_name=recommended_model_name,
                installed_model_names=installed_model_names,
                prompt_ui=prompt_ui,
            )
        else:
            template = (
                _draft_from_profile(current_profile)
                if current_profile is not None
                else recommended[profile_name]
            )
            model_name = current_model_name

        profiles[profile_name] = ModelProfileDraft(
            provider=template.provider,
            model_name=model_name,
            temperature=template.temperature,
            thinking_supported=template.thinking_supported,
            tool_mode=template.tool_mode,
            num_ctx=template.num_ctx,
            keep_alive=template.keep_alive,
            planner_profile=template.planner_profile,
        )

    return profiles


def prompt_telegram_bot_token(
    *,
    existing_token: str | None,
    prompt_ui: PromptUI,
) -> str:
    prompt_ui.section(
        "🤖 Telegram bot",
        (
            "Connect your Telegram bot with the token from BotFather. "
            "Unclaw stores it locally in `config/secrets.yaml` for this project."
        ),
    )
    prompt_ui.info("Get this token from BotFather after you create or open your bot.")
    prompt_ui.info("Example format: 123456789:AA...")
    prompt_ui.info("The token stays visible while you type so you can verify the paste.")
    prompt_ui.info(
        "Secure by default: `allowed_chat_ids: []` denies all Telegram chats until you authorize one explicitly."
    )
    prompt_ui.info(
        "Advanced fallback: the Telegram channel can still read "
        f"`{DEFAULT_TELEGRAM_TOKEN_ENV_VAR}` from the environment."
    )

    if existing_token is not None:
        keep_existing = prompt_ui.confirm(
            "Keep the Telegram bot token already stored in `config/secrets.yaml`?",
            default=True,
            help_text="Choose No if you want to replace it with a fresh token from BotFather.",
        )
        if keep_existing:
            return existing_token

    return prompt_ui.text(
        "Telegram bot token",
        default="",
        help_text=(
            "Paste the full token from BotFather. Unclaw will write it to "
            "`config/secrets.yaml` for this project."
        ),
        instruction="Visible while typing. Press Enter to save it locally.",
        validator=_validate_telegram_bot_token,
    )


def print_plan_summary(plan: OnboardingPlan, *, output_func: OutputFunc) -> None:
    output_func(
        "- Setup style: "
        f"{'recommended guided' if plan.beginner_mode else 'advanced custom'}"
    )
    output_func(f"- Logging: {plan.logging_mode}")
    output_func(
        "- Channels: "
        + ", ".join(
            channel_name
            for channel_name in ("terminal", "telegram")
            if channel_name in plan.enabled_channels
        )
    )
    output_func("- Model lineup:")
    for profile_name in PROFILE_ORDER:
        output_func(f"  {profile_name}: {plan.model_profiles[profile_name].model_name}")
    if "telegram" in plan.enabled_channels:
        output_func("- Telegram token: stored locally in config/secrets.yaml")
        output_func(f"- Telegram env fallback: {plan.telegram_bot_token_env_var}")
        if plan.telegram_allowed_chat_ids:
            count = len(plan.telegram_allowed_chat_ids)
            label = "chat" if count == 1 else "chats"
            output_func(f"- Telegram access: allowlist with {count} authorized {label}")
        else:
            output_func("- Telegram access: secure deny-by-default (no chats authorized yet)")


def post_configure_ollama(
    settings: Settings,
    plan: OnboardingPlan,
    *,
    prompt_ui: PromptUI,
    inspect_ollama_func: Callable[..., OllamaStatus] = inspect_ollama,
) -> OllamaStatus | None:
    prompt_ui.section(
        "🦙 Local model runtime",
        "Check Ollama now so startup is smoother the next time you launch Unclaw.",
    )
    ollama_status = inspect_ollama_func()
    if not ollama_status.is_installed:
        prompt_ui.info("Ollama is not installed yet.")
        prompt_ui.info(ollama_install_guidance())
        return None

    if not ollama_status.is_running:
        should_start_ollama = prompt_ui.confirm(
            "Ollama is not running. Start it now with `ollama serve`?",
            default=plan.automatic_configuration,
        )
        if should_start_ollama:
            try:
                log_path = settings.paths.logs_dir / "ollama-serve.log"
                start_ollama_server(log_path)
            except OSError as exc:
                prompt_ui.info(f"Could not start Ollama automatically: {exc}")
                prompt_ui.info("Start it manually with `ollama serve`.")
                return None

            ollama_status = wait_for_ollama()
            if ollama_status.is_running:
                prompt_ui.info("Ollama is now running.")
            else:
                prompt_ui.info("Ollama still does not look reachable.")
                prompt_ui.info(
                    "Start it manually with `ollama serve`, then rerun onboarding."
                )
                return None
        else:
            prompt_ui.info("Start Ollama later with `ollama serve`.")
            return None

    missing_profiles = find_missing_model_profiles(
        settings,
        installed_model_names=ollama_status.model_names,
        profile_names=tuple(plan.model_profiles),
    )
    if not missing_profiles:
        prompt_ui.info("All selected models are already available in Ollama.")
        return ollama_status

    missing_models = _unique_model_names(missing_profiles)
    prompt_ui.info(
        "Missing local models: "
        + ", ".join(
            f"{profile_name}={model_name}"
            for profile_name, model_name in missing_profiles
        )
    )
    should_pull_models = prompt_ui.confirm(
        "Pull the missing models now?",
        default=plan.automatic_configuration or plan.beginner_mode,
    )
    if not should_pull_models:
        prompt_ui.info("Pull them later with:")
        for model_name in missing_models:
            prompt_ui.info(f"- ollama pull {model_name}")
        return ollama_status

    for model_name in missing_models:
        prompt_ui.info(f"Pulling {model_name}...")
        completed_process = subprocess.run(["ollama", "pull", model_name], check=False)
        if completed_process.returncode == 0:
            prompt_ui.info(f"Finished pulling {model_name}.")
            continue
        prompt_ui.info(
            f"`ollama pull {model_name}` exited with code {completed_process.returncode}."
        )

    return inspect_ollama_func()


def build_onboarding_banner(*, settings: Settings, ollama_status: OllamaStatus) -> str:
    return build_banner(
        title="Unclaw setup",
        subtitle="Guided local setup for channels, models, and startup defaults.",
        rows=(
            ("project", str(settings.paths.project_root)),
            ("config", str(settings.paths.config_dir)),
            ("ollama", _describe_ollama_status(ollama_status)),
        ),
    )


def _build_profile_menu_options(
    *,
    profile_name: str,
    has_current_profile: bool,
    current_model_name: str,
    recommended_model_name: str,
    installed_model_names: tuple[str, ...],
) -> tuple[tuple[MenuOption, ...], str]:
    options: list[MenuOption] = [
        MenuOption(
            value="current",
            label=f"Keep current model: {current_model_name}",
            description=(
                "Leave this profile exactly as it is."
                if has_current_profile
                else "No current profile was found, so this uses the detected fallback."
            ),
        ),
        MenuOption(
            value="recommended",
            label=f"Use recommended model: {recommended_model_name}",
            description=(
                "This already matches the current config."
                if current_model_name == recommended_model_name
                else "Use the updated local-friendly default for this role."
            ),
        ),
    ]
    if installed_model_names:
        options.append(
            MenuOption(
                value="installed",
                label="Choose from installed Ollama models",
                description=f"{len(installed_model_names)} local model(s) detected in Ollama.",
            )
        )
    options.append(
        MenuOption(
            value="custom",
            label="Enter a custom model name",
            description="Type another local model name yourself.",
        )
    )
    return tuple(options), "current" if has_current_profile else "recommended"


def _build_profile_help_text(
    *,
    profile_name: str,
    installed_model_names: tuple[str, ...],
) -> str:
    base_text = PROFILE_DESCRIPTIONS[profile_name]
    if installed_model_names:
        return (
            f"{base_text} Ollama is running with "
            f"{len(installed_model_names)} installed model(s) available."
        )
    return (
        f"{base_text} Start Ollama if you want to browse installed local models during setup."
    )


def _prompt_installed_model_name(
    *,
    profile_name: str,
    current_model_name: str,
    recommended_model_name: str,
    installed_model_names: tuple[str, ...],
    prompt_ui: PromptUI,
) -> str:
    options = tuple(
        MenuOption(
            value=model_name,
            label=model_name,
            description=_describe_installed_model(
                model_name=model_name,
                current_model_name=current_model_name,
                recommended_model_name=recommended_model_name,
            ),
        )
        for model_name in installed_model_names
    )
    default_model_name = _resolve_installed_model_default(
        installed_model_names=installed_model_names,
        current_model_name=current_model_name,
        recommended_model_name=recommended_model_name,
    )
    return prompt_ui.select(
        f"Installed Ollama model for {profile_name}",
        options=options,
        default=default_model_name,
        help_text="Choose one of the local models already available in Ollama.",
    )


def _validate_model_name(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return "The model name cannot be empty."
    return None


def _validate_telegram_bot_token(value: str) -> str | None:
    try:
        validate_telegram_bot_token(value)
    except ConfigurationError as exc:
        return str(exc)
    return None


def _draft_from_profile(profile: ModelProfile) -> ModelProfileDraft:
    return ModelProfileDraft(
        provider=profile.provider,
        model_name=profile.model_name,
        temperature=profile.temperature,
        thinking_supported=profile.thinking_supported,
        tool_mode=profile.tool_mode,
        num_ctx=profile.num_ctx,
        keep_alive=profile.keep_alive,
        planner_profile=profile.planner_profile,
    )


def _describe_ollama_status(ollama_status: OllamaStatus) -> str:
    if not ollama_status.is_installed:
        return "not installed"
    if not ollama_status.is_running:
        return "installed, not running"
    return f"running with {len(ollama_status.model_names)} model(s)"


def _unique_model_names(missing_profiles: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(model_name for _profile_name, model_name in missing_profiles)
    )


def _resolve_installed_model_default(
    *,
    installed_model_names: tuple[str, ...],
    current_model_name: str,
    recommended_model_name: str,
) -> str:
    if current_model_name in installed_model_names:
        return current_model_name
    if recommended_model_name in installed_model_names:
        return recommended_model_name
    return installed_model_names[0]


def _describe_installed_model(
    *,
    model_name: str,
    current_model_name: str,
    recommended_model_name: str,
) -> str | None:
    tags: list[str] = []
    if model_name == current_model_name:
        tags.append("current")
    if model_name == recommended_model_name:
        tags.append("recommended")
    if not tags:
        return None
    return ", ".join(tags)


__all__ = [
    "build_onboarding_banner",
    "default_channel_preset",
    "enabled_channels_from_preset",
    "load_existing_local_secrets",
    "load_existing_telegram_config",
    "post_configure_ollama",
    "print_plan_summary",
    "prompt_channel_preset",
    "prompt_model_profiles",
    "prompt_telegram_bot_token",
]
