"""Optional skills package with legacy manifests and file-first bundle helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTED_MEMBERS = {
    "SkillBundle": ("unclaw.skills.file_models", "SkillBundle"),
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
    "UnknownSkillIdError": ("unclaw.skills.file_models", "UnknownSkillIdError"),
    "build_active_skill_catalog": (
        "unclaw.skills.catalog",
        "build_active_skill_catalog",
    ),
    "build_active_skill_context_notes": (
        "unclaw.skills.runtime",
        "build_active_skill_context_notes",
    ),
    "discover_internal_skill_bundles": (
        "unclaw.skills.file_loader",
        "discover_internal_skill_bundles",
    ),
    "discover_skill_bundles": ("unclaw.skills.file_loader", "discover_skill_bundles"),
    "get_skill_bundle_roots": ("unclaw.skills.file_loader", "get_skill_bundle_roots"),
    "internal_skill_bundle_root": (
        "unclaw.skills.file_loader",
        "internal_skill_bundle_root",
    ),
    "list_known_skill_ids": ("unclaw.skills.file_loader", "list_known_skill_ids"),
    "load_active_skill_bundles": (
        "unclaw.skills.file_loader",
        "load_active_skill_bundles",
    ),
    "load_skill_bundle": ("unclaw.skills.file_loader", "load_skill_bundle"),
    "load_skill_registry": ("unclaw.skills.registry", "load_skill_registry"),
    "resolve_file_first_skill_id": (
        "unclaw.skills.file_loader",
        "resolve_file_first_skill_id",
    ),
    "resolve_active_skill_manifests": (
        "unclaw.skills.runtime",
        "resolve_active_skill_manifests",
    ),
    "resolve_legacy_manifest_skill_id": (
        "unclaw.skills.file_loader",
        "resolve_legacy_manifest_skill_id",
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
