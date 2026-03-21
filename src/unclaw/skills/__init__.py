"""Optional internal skill manifests and registry primitives."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTED_MEMBERS = {
    "SkillAvailability": ("unclaw.skills.models", "SkillAvailability"),
    "SkillBudgetMetadata": ("unclaw.skills.models", "SkillBudgetMetadata"),
    "SkillInstallMode": ("unclaw.skills.models", "SkillInstallMode"),
    "SkillManifest": ("unclaw.skills.models", "SkillManifest"),
    "SkillPromptBudgetTier": ("unclaw.skills.models", "SkillPromptBudgetTier"),
    "SkillPromptFragment": ("unclaw.skills.models", "SkillPromptFragment"),
    "SkillPromptFragmentKind": ("unclaw.skills.models", "SkillPromptFragmentKind"),
    "SkillPromptSource": ("unclaw.skills.models", "SkillPromptSource"),
    "SkillPromptSourceKind": ("unclaw.skills.models", "SkillPromptSourceKind"),
    "SkillRegistry": ("unclaw.skills.registry", "SkillRegistry"),
    "SkillToolBinding": ("unclaw.skills.models", "SkillToolBinding"),
    "build_active_skill_context_notes": (
        "unclaw.skills.runtime",
        "build_active_skill_context_notes",
    ),
    "load_skill_registry": ("unclaw.skills.registry", "load_skill_registry"),
    "resolve_active_skill_manifests": (
        "unclaw.skills.runtime",
        "resolve_active_skill_manifests",
    ),
}

__all__ = list(_EXPORTED_MEMBERS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTED_MEMBERS[name]
    except KeyError as exc:  # pragma: no cover - standard attribute protocol
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTED_MEMBERS))
