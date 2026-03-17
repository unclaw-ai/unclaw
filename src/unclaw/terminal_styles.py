"""Shared terminal presentation helpers for onboarding flows."""

from __future__ import annotations

from collections.abc import Callable

TerminalOutputFunc = Callable[[str], None]

_ONBOARDING_ACCENT_BOLD = "fg:#f6c7b1 bold"
_ONBOARDING_INVERTED_ACCENT_BOLD = "fg:#241511 bg:#f6c7b1 bold"


def render_onboarding_section(
    output_func: TerminalOutputFunc,
    title: str,
    description: str | None = None,
) -> None:
    """Render one onboarding section heading without changing existing layout."""

    output_func("")
    output_func(title)
    output_func("-" * max(36, len(title)))
    if description:
        output_func(description)
    output_func("")


def onboarding_questionary_style_entries() -> tuple[tuple[str, str], ...]:
    """Return the exact token styles used for interactive onboarding prompts."""

    return (
        ("qmark", _ONBOARDING_ACCENT_BOLD),
        ("question", "fg:#fff4ec bold"),
        ("answer", _ONBOARDING_ACCENT_BOLD),
        ("pointer", _ONBOARDING_INVERTED_ACCENT_BOLD),
        ("highlighted", _ONBOARDING_INVERTED_ACCENT_BOLD),
        ("selected", "fg:#241511 bg:#f2b99b bold"),
        ("text", "fg:#dde3e8"),
        ("separator", "fg:#8f7b70"),
        ("instruction", "fg:#c79a85 italic"),
        ("disabled", "fg:#6d6159 italic"),
        ("bottom-toolbar", "noreverse"),
        ("validation-toolbar", "fg:#ffffff bg:#b42318 bold"),
    )


__all__ = [
    "onboarding_questionary_style_entries",
    "render_onboarding_section",
]
