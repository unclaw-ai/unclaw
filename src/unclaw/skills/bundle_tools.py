"""Generic Python tool registration from file-first skill bundles.

Convention
----------
A skill bundle may expose a ``register_skill_tools(registry: ToolRegistry) -> None``
function in its ``tool.py``.  If present and callable, the runtime calls it
when that skill is active to register the bundle's tools on the shared registry.

Bundles without a ``tool.py``, or without the hook, are silently skipped —
they are treated as prompt-only skills.

Usage
-----
Call :func:`register_active_skill_tools` once per registry construction, passing
the configured enabled skill IDs.  The function handles deduplication and import
errors internally.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from unclaw.skills.file_loader import load_active_skill_bundles
from unclaw.skills.file_models import SkillBundle, UnknownSkillIdError
from unclaw.tools.contracts import ToolDefinition, ToolHandler
from unclaw.tools.registry import ToolRegistry

# The name of the registration hook that a skill bundle's tool.py may expose.
_SKILL_TOOL_REGISTRAR = "register_skill_tools"


@dataclass(slots=True)
class _SkillOwnedRegistryProxy:
    """Thin proxy that injects skill_id ownership on every register() call.

    Passed to a skill bundle's ``register_skill_tools(registry)`` hook so that
    all tools registered by the bundle are automatically tagged with the
    bundle's skill_id.  The skill's tool.py needs no changes.
    """

    _registry: ToolRegistry
    _skill_id: str

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        self._registry.register(definition, handler, skill_id=self._skill_id)


def register_active_skill_tools(
    registry: ToolRegistry,
    *,
    enabled_skill_ids: Collection[str],
    skills_root: Path | None = None,
) -> tuple[str, ...]:
    """Register tools exposed by active skill bundle backends.

    For each active bundle that contains a ``tool.py`` with a
    ``register_skill_tools(registry)`` callable, that function is called to
    add the bundle's tools to *registry*.

    Pass *skills_root* to restrict discovery to a specific directory; when
    omitted the default local install root (``./skills/``) is used.

    Returns the ``skill_id``s whose tools were successfully registered.
    Bundles without the hook are silently skipped and not included in the
    return value.  Raises nothing on import failure — the bundle is skipped
    with a warning-level note (callers should surface this at startup).
    """
    try:
        active_bundles = load_active_skill_bundles(
            enabled_skill_ids=enabled_skill_ids,
            skills_root=skills_root,
        )
    except UnknownSkillIdError:
        return ()

    registered: list[str] = []
    for bundle in active_bundles:
        if _register_bundle_tools(registry, bundle):
            registered.append(bundle.skill_id)

    return tuple(registered)


def probe_skill_tool_loading(
    *,
    enabled_skill_ids: Collection[str],
    skills_root: Path | None = None,
    discovered_skill_bundles: Sequence[SkillBundle] | None = None,
) -> dict[str, str | None]:
    """Probe tool-loading for active skill bundles without side effects.

    Returns a mapping of ``skill_id`` → error string (or ``None`` if OK).
    A ``None`` value means the bundle registered successfully (or has no
    tool.py, which is also fine).  A string value is the failure reason.

    Pass ``discovered_skill_bundles`` when bundles have already been discovered
    by the caller (e.g. ``_build_skills_check``) to avoid a redundant directory
    scan.

    Intended for use in startup diagnostics only.
    """
    try:
        active_bundles = load_active_skill_bundles(
            enabled_skill_ids=enabled_skill_ids,
            skills_root=skills_root,
            discovered_skill_bundles=discovered_skill_bundles,
        )
    except UnknownSkillIdError as exc:
        return {sid: f"unknown skill id: {sid}" for sid in exc.unknown_skill_ids}

    result: dict[str, str | None] = {}
    for bundle in active_bundles:
        tool_py_path = bundle.bundle_dir / "tool.py"
        if not tool_py_path.is_file():
            result[bundle.skill_id] = None  # prompt-only, not an error
            continue

        module = _import_bundle_tool_module(bundle)
        if module is None:
            result[bundle.skill_id] = f"could not import skills.{bundle.skill_id}.tool"
            continue

        registrar = getattr(module, _SKILL_TOOL_REGISTRAR, None)
        if not callable(registrar):
            result[bundle.skill_id] = (
                f"skills.{bundle.skill_id}.tool has no callable {_SKILL_TOOL_REGISTRAR!r}"
            )
            continue

        result[bundle.skill_id] = None  # tools present and loadable

    return result


def _register_bundle_tools(registry: ToolRegistry, bundle: SkillBundle) -> bool:
    """Register tools from one bundle. Returns True if the hook was called."""
    tool_py_path = bundle.bundle_dir / "tool.py"
    if not tool_py_path.is_file():
        return False

    module = _import_bundle_tool_module(bundle)
    if module is None:
        return False

    registrar = getattr(module, _SKILL_TOOL_REGISTRAR, None)
    if not callable(registrar):
        return False

    proxy = _SkillOwnedRegistryProxy(_registry=registry, _skill_id=bundle.skill_id)
    registrar(proxy)
    return True


def _import_bundle_tool_module(bundle: SkillBundle) -> object | None:
    """Import ``skills.<skill_id>.tool`` from sys.modules or via importlib.

    Returns ``None`` if the import fails (e.g. the bundle dir is not on
    sys.path, or the module has a syntax error).
    """
    module_name = f"skills.{bundle.skill_id}.tool"

    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


__all__ = [
    "probe_skill_tool_loading",
    "register_active_skill_tools",
]
