"""Interactive onboarding for first-time and advanced local setup."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
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
    default_channel_preset as _default_channel_preset,
    enabled_channels_from_preset as _enabled_channels_from_preset,
    load_existing_local_secrets as _load_existing_local_secrets,
    load_existing_telegram_config as _load_existing_telegram_config,
    post_configure_ollama as _post_configure_ollama,
    print_plan_summary as _print_plan_summary,
    prompt_channel_preset as _prompt_channel_preset,
    prompt_model_profiles as _prompt_model_profiles,
    prompt_telegram_bot_token as _prompt_telegram_bot_token,
)
from unclaw.onboarding_types import (
    InputFunc,
    MenuOption,
    ModelProfileDraft,
    OnboardingPlan,
    OutputFunc,
    PROFILE_ORDER,
    PromptUI,
)
from unclaw.settings import Settings
from unclaw.startup import find_missing_model_profiles, inspect_ollama

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


@dataclass(slots=True)
class FallbackPromptUI:
    """Readable plain-text onboarding prompts used in tests and non-TTY flows."""

    input_func: InputFunc
    output_func: OutputFunc
    interactive: bool = False

    def section(self, title: str, description: str | None = None) -> None:
        self.output_func("")
        self.output_func(title)
        self.output_func("-" * max(36, len(title)))
        if description:
            self.output_func(description)
        self.output_func("")

    def info(self, message: str = "") -> None:
        self.output_func(message)

    def confirm(
        self,
        prompt: str,
        *,
        default: bool,
        help_text: str | None = None,
    ) -> bool:
        if help_text:
            self.output_func(help_text)

        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            response = self.input_func(f"{prompt} {suffix} ").strip().lower()
            if not response:
                return default
            if response in {"y", "yes"}:
                return True
            if response in {"n", "no"}:
                return False
            self.output_func("Please answer yes or no.")

    def select(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default: str,
        help_text: str | None = None,
    ) -> str:
        default_value = _resolve_select_default(options=options, default=default)
        if help_text:
            self.output_func(help_text)
        self.output_func(prompt)

        default_index = 1
        for index, option in enumerate(options, start=1):
            if option.value == default_value:
                default_index = index
            suffix = f" - {option.description}" if option.description else ""
            self.output_func(f"  {index}. {option.label}{suffix}")

        while True:
            response = self.input_func(f"Select an option [{default_index}] ").strip()
            if not response:
                return default_value
            if response.isdigit():
                choice_index = int(response)
                if 1 <= choice_index <= len(options):
                    return options[choice_index - 1].value

            normalized = response.lower()
            for option in options:
                if normalized == option.value.lower():
                    return option.value
            self.output_func("Choose one of the listed options.")

    def checkbox(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default_values: tuple[str, ...],
        help_text: str | None = None,
    ) -> tuple[str, ...]:
        while True:
            if help_text:
                self.output_func(help_text)
            self.output_func(prompt)

            selected_values: list[str] = []
            default_lookup = set(default_values)
            for option in options:
                enabled = self.confirm(
                    f"Enable {option.label}?",
                    default=option.value in default_lookup,
                    help_text=option.description,
                )
                if enabled:
                    selected_values.append(option.value)

            ordered_values = tuple(
                option.value for option in options if option.value in selected_values
            )
            if ordered_values:
                return ordered_values
            self.output_func("Choose at least one option.")

    def text(
        self,
        prompt: str,
        *,
        default: str,
        help_text: str | None = None,
        instruction: str | None = None,
        validator: Callable[[str], str | None] | None = None,
    ) -> str:
        if help_text:
            self.output_func(help_text)
        if instruction:
            self.output_func(instruction)

        prompt_suffix = f" [{default}]" if default.strip() else ""
        while True:
            response = self.input_func(f"{prompt}{prompt_suffix} ").strip()
            value = response or default.strip()
            if not value:
                self.output_func("This value cannot be empty.")
                continue

            if validator is not None:
                error_message = validator(value)
                if error_message is not None:
                    self.output_func(error_message)
                    continue
            return value


@dataclass(slots=True)
class InteractivePromptUI:
    """Menu-driven onboarding prompts backed by questionary."""

    output_func: OutputFunc
    interactive: bool = True

    def section(self, title: str, description: str | None = None) -> None:
        self.output_func("")
        self.output_func(title)
        self.output_func("-" * max(36, len(title)))
        if description:
            self.output_func(description)
        self.output_func("")

    def info(self, message: str = "") -> None:
        self.output_func(message)

    def confirm(
        self,
        prompt: str,
        *,
        default: bool,
        help_text: str | None = None,
    ) -> bool:
        if help_text:
            self.output_func(help_text)
        answer = questionary.confirm(
            prompt,
            default=default,
            style=_QUESTIONARY_STYLE,
        ).ask()
        if answer is None:
            raise KeyboardInterrupt
        return answer

    def select(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default: str,
        help_text: str | None = None,
    ) -> str:
        if help_text:
            self.output_func(help_text)

        default_value = _resolve_select_default(options=options, default=default)
        answer = _ask_select_question(
            prompt,
            choices=[_build_questionary_choice(option) for option in options],
            initial_choice=default_value,
            instruction="Use arrow keys, then press Enter.",
        )
        if answer is None:
            raise KeyboardInterrupt
        return answer

    def checkbox(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default_values: tuple[str, ...],
        help_text: str | None = None,
    ) -> tuple[str, ...]:
        if help_text:
            self.output_func(help_text)

        default_lookup = set(default_values)
        while True:
            answer = questionary.checkbox(
                prompt,
                choices=[
                    questionary.Choice(
                        title=option.label,
                        value=option.value,
                        checked=option.value in default_lookup,
                        description=option.description,
                    )
                    for option in options
                ],
                initial_choice=_resolve_checkbox_initial_choice(
                    options=options,
                    default_values=default_values,
                ),
                instruction="Use arrow keys to move, Space to toggle, Enter to confirm.",
                style=_QUESTIONARY_STYLE,
            ).ask()
            if answer is None:
                raise KeyboardInterrupt

            ordered_values = tuple(
                option.value for option in options if option.value in set(answer)
            )
            if ordered_values:
                return ordered_values
            self.output_func("Choose at least one option.")

    def text(
        self,
        prompt: str,
        *,
        default: str,
        help_text: str | None = None,
        instruction: str | None = None,
        validator: Callable[[str], str | None] | None = None,
    ) -> str:
        while True:
            if help_text:
                self.output_func(help_text)
            answer = questionary.text(
                prompt,
                default=default,
                instruction=instruction or "Press Enter to confirm.",
                style=_QUESTIONARY_STYLE,
            ).ask()
            if answer is None:
                raise KeyboardInterrupt

            value = answer.strip() or default.strip()
            if not value:
                self.output_func("This value cannot be empty.")
                continue
            if validator is not None:
                error_message = validator(value)
                if error_message is not None:
                    self.output_func(error_message)
                    continue
            return value


if questionary is not None:
    _QUESTIONARY_STYLE = questionary.Style(
        [
            ("qmark", "fg:#f6c7b1 bold"),
            ("question", "fg:#fff4ec bold"),
            ("answer", "fg:#f6c7b1 bold"),
            ("pointer", "fg:#241511 bg:#f6c7b1 bold"),
            ("highlighted", "fg:#241511 bg:#f6c7b1 bold"),
            ("selected", "fg:#241511 bg:#f2b99b bold"),
            ("text", "fg:#dde3e8"),
            ("separator", "fg:#8f7b70"),
            ("instruction", "fg:#c79a85 italic"),
            ("disabled", "fg:#6d6159 italic"),
            ("bottom-toolbar", "noreverse"),
            ("validation-toolbar", "fg:#ffffff bg:#b42318 bold"),
        ]
    )
else:  # pragma: no cover - exercised when questionary is unavailable.
    _QUESTIONARY_STYLE = None


def _build_questionary_choice(option: MenuOption) -> Any:
    return questionary.Choice(
        title=option.label,
        value=option.value,
        description=option.description,
    )


def _resolve_checkbox_initial_choice(
    *,
    options: tuple[MenuOption, ...],
    default_values: tuple[str, ...],
) -> str:
    for value in default_values:
        for option in options:
            if option.value == value:
                return value
    return options[0].value


def _ask_select_question(
    prompt: str,
    *,
    choices: list[Any],
    initial_choice: str,
    instruction: str,
) -> str | None:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from questionary.constants import (
        DEFAULT_QUESTION_PREFIX,
        DEFAULT_SELECTED_POINTER,
    )
    from questionary.prompts import common as questionary_common
    from questionary.prompts.common import InquirerControl
    from questionary.question import Question
    from questionary.styles import merge_styles_default

    merged_style = merge_styles_default([_QUESTIONARY_STYLE])
    inquirer_control = InquirerControl(
        choices,
        default=None,
        pointer=DEFAULT_SELECTED_POINTER,
        use_indicator=False,
        use_shortcuts=False,
        show_selected=False,
        show_description=True,
        use_arrow_keys=True,
        initial_choice=initial_choice,
    )

    def get_prompt_tokens() -> list[tuple[str, str]]:
        return [
            ("class:qmark", DEFAULT_QUESTION_PREFIX),
            ("class:question", f" {prompt} "),
            ("class:instruction", instruction),
        ]

    layout = questionary_common.create_inquirer_layout(
        inquirer_control,
        get_prompt_tokens,
    )
    bindings = KeyBindings()

    @bindings.add(Keys.ControlQ, eager=True)
    @bindings.add(Keys.ControlC, eager=True)
    def _abort(event: Any) -> None:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    def _move_cursor_down(event: Any) -> None:
        inquirer_control.select_next()
        while not inquirer_control.is_selection_valid():
            inquirer_control.select_next()

    def _move_cursor_up(event: Any) -> None:
        inquirer_control.select_previous()
        while not inquirer_control.is_selection_valid():
            inquirer_control.select_previous()

    @bindings.add(Keys.Down, eager=True)
    def _down(event: Any) -> None:
        _move_cursor_down(event)

    @bindings.add(Keys.Up, eager=True)
    def _up(event: Any) -> None:
        _move_cursor_up(event)

    @bindings.add("j", eager=True)
    def _j(event: Any) -> None:
        _move_cursor_down(event)

    @bindings.add("k", eager=True)
    def _k(event: Any) -> None:
        _move_cursor_up(event)

    @bindings.add(Keys.ControlN, eager=True)
    def _ctrl_n(event: Any) -> None:
        _move_cursor_down(event)

    @bindings.add(Keys.ControlP, eager=True)
    def _ctrl_p(event: Any) -> None:
        _move_cursor_up(event)

    @bindings.add(Keys.ControlM, eager=True)
    def _submit(event: Any) -> None:
        inquirer_control.is_answered = True
        event.app.exit(result=inquirer_control.get_pointed_at().value)

    @bindings.add(Keys.Any)
    def _ignore_other_keys(event: Any) -> None:
        del event

    return Question(
        Application(
            layout=layout,
            key_bindings=bindings,
            style=merged_style,
        )
    ).ask()


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
    automatic_configuration = beginner_mode

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
        "🧠 Model profiles",
        "Choose the local models for quick replies, default chat, deeper work, and code tasks.",
    )
    if beginner_mode:
        model_profiles = recommended_model_profiles()
        prompt_ui.info("Using the recommended starter model lineup:")
        for profile_name in _PROFILE_ORDER:
            prompt_ui.info(
                f"- {profile_name}: {model_profiles[profile_name].model_name}"
            )
    else:
        model_profiles = _prompt_model_profiles(
            settings,
            prompt_ui=prompt_ui,
            ollama_status=ollama_status,
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
        default_profile="main",
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
    should_write_config = prompt_ui.confirm(
        "Write this configuration now?",
        default=True,
    )
    if not should_write_config:
        output_func("No files were changed.")
        return 1

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


def _resolve_select_default(*, options: tuple[MenuOption, ...], default: str) -> str:
    for option in options:
        if option.value == default:
            return default
    return options[0].value


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
