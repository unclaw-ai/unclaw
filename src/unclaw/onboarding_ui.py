"""Prompt UI implementations and questionary helpers for onboarding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from unclaw.onboarding_types import InputFunc, MenuOption, OutputFunc


@dataclass(slots=True)
class FallbackPromptUIBase:
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
        default_value = resolve_select_default(options=options, default=default)
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
class InteractivePromptUIBase:
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

        answer = self._questionary().confirm(
            prompt,
            default=default,
            style=self._questionary_style(),
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

        questionary_module = self._questionary()
        default_value = resolve_select_default(options=options, default=default)
        answer = self._ask_select_question(
            prompt,
            choices=[
                build_questionary_choice(questionary_module, option)
                for option in options
            ],
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

        questionary_module = self._questionary()
        default_lookup = set(default_values)
        while True:
            answer = questionary_module.checkbox(
                prompt,
                choices=[
                    questionary_module.Choice(
                        title=option.label,
                        value=option.value,
                        checked=option.value in default_lookup,
                        description=option.description,
                    )
                    for option in options
                ],
                initial_choice=resolve_checkbox_initial_choice(
                    options=options,
                    default_values=default_values,
                ),
                instruction="Use arrow keys to move, Space to toggle, Enter to confirm.",
                style=self._questionary_style(),
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

            answer = self._questionary().text(
                prompt,
                default=default,
                instruction=instruction or "Press Enter to confirm.",
                style=self._questionary_style(),
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

    def _questionary(self) -> Any:
        raise NotImplementedError

    def _questionary_style(self) -> Any:
        raise NotImplementedError

    def _ask_select_question(
        self,
        prompt: str,
        *,
        choices: list[Any],
        initial_choice: str,
        instruction: str,
    ) -> str | None:
        raise NotImplementedError


def build_questionary_choice(questionary_module: Any, option: MenuOption) -> Any:
    return questionary_module.Choice(
        title=option.label,
        value=option.value,
        description=option.description,
    )


def resolve_checkbox_initial_choice(
    *,
    options: tuple[MenuOption, ...],
    default_values: tuple[str, ...],
) -> str:
    for value in default_values:
        for option in options:
            if option.value == value:
                return value
    return options[0].value


def resolve_select_default(*, options: tuple[MenuOption, ...], default: str) -> str:
    for option in options:
        if option.value == default:
            return default
    return options[0].value


def ask_select_question(
    prompt: str,
    *,
    choices: list[Any],
    initial_choice: str,
    instruction: str,
    style: Any,
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

    merged_style = merge_styles_default([style])
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


__all__ = [
    "FallbackPromptUIBase",
    "InteractivePromptUIBase",
    "ask_select_question",
    "resolve_select_default",
]
