"""Optional skills package — file-first bundle helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTED_MEMBERS = {
    "SkillBundle": ("unclaw.skills.file_models", "SkillBundle"),
    "UnknownSkillIdError": ("unclaw.skills.file_models", "UnknownSkillIdError"),
    "build_active_skill_catalog": (
        "unclaw.skills.catalog",
        "build_active_skill_catalog",
    ),
    "discover_skill_bundles": ("unclaw.skills.file_loader", "discover_skill_bundles"),
    "get_skill_bundle_roots": ("unclaw.skills.file_loader", "get_skill_bundle_roots"),
    "list_known_skill_ids": ("unclaw.skills.file_loader", "list_known_skill_ids"),
    "load_active_skill_bundles": (
        "unclaw.skills.file_loader",
        "load_active_skill_bundles",
    ),
    "load_skill_bundle": ("unclaw.skills.file_loader", "load_skill_bundle"),
    "local_skill_install_root": (
        "unclaw.skills.file_loader",
        "local_skill_install_root",
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
