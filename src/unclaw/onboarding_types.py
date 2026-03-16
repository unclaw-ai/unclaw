"""Shared onboarding types and prompt contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

InputFunc: TypeAlias = Callable[[str], str]
OutputFunc: TypeAlias = Callable[[str], None]

PROFILE_ORDER = ("fast", "main", "deep", "codex")

PROFILE_DESCRIPTIONS = {
    "fast": "Quick replies and lighter local hardware usage.",
    "main": "Best default for most conversations and tool work.",
    "deep": "Stronger reasoning when extra latency is acceptable.",
    "codex": "Code-heavy work such as repositories, scripts, and fixes.",
}


@dataclass(frozen=True, slots=True)
class ModelProfileDraft:
    """Editable model profile settings used during onboarding."""

    provider: str
    model_name: str
    temperature: float
    thinking_supported: bool
    tool_mode: str


@dataclass(frozen=True, slots=True)
class OnboardingPlan:
    """One complete onboarding result ready to write to config files."""

    beginner_mode: bool
    automatic_configuration: bool
    logging_mode: str
    enabled_channels: tuple[str, ...]
    default_profile: str
    model_profiles: dict[str, ModelProfileDraft]
    telegram_bot_token: str | None
    telegram_bot_token_env_var: str
    telegram_allowed_chat_ids: tuple[int, ...]
    telegram_polling_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class MenuOption:
    """One selectable onboarding option."""

    value: str
    label: str
    description: str | None = None


class PromptUI(Protocol):
    """Minimal prompt surface shared by interactive and fallback onboarding UIs."""

    interactive: bool

    def section(self, title: str, description: str | None = None) -> None:
        ...

    def info(self, message: str = "") -> None:
        ...

    def confirm(
        self,
        prompt: str,
        *,
        default: bool,
        help_text: str | None = None,
    ) -> bool:
        ...

    def select(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default: str,
        help_text: str | None = None,
    ) -> str:
        ...

    def checkbox(
        self,
        prompt: str,
        *,
        options: tuple[MenuOption, ...],
        default_values: tuple[str, ...],
        help_text: str | None = None,
    ) -> tuple[str, ...]:
        ...

    def text(
        self,
        prompt: str,
        *,
        default: str,
        help_text: str | None = None,
        instruction: str | None = None,
        validator: Callable[[str], str | None] | None = None,
    ) -> str:
        ...


__all__ = [
    "InputFunc",
    "MenuOption",
    "ModelProfileDraft",
    "OnboardingPlan",
    "OutputFunc",
    "PROFILE_DESCRIPTIONS",
    "PROFILE_ORDER",
    "PromptUI",
]
