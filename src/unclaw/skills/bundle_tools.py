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
the configured enabled skill IDs.  The function handles legacy ID aliases,
deduplication, and import errors internally.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Collection

from unclaw.skills.file_loader import load_active_skill_bundles
from unclaw.skills.file_models import SkillBundle, UnknownSkillIdError
from unclaw.tools.registry import ToolRegistry

# The name of the registration hook that a skill bundle's tool.py may expose.
_SKILL_TOOL_REGISTRAR = "register_skill_tools"


def register_active_skill_tools(
    registry: ToolRegistry,
    *,
    enabled_skill_ids: Collection[str],
) -> tuple[str, ...]:
    """Register tools exposed by active skill bundle backends.

    For each active bundle that contains a ``tool.py`` with a
    ``register_skill_tools(registry)`` callable, that function is called to
    add the bundle's tools to *registry*.

    Returns the ``skill_id``s whose tools were successfully registered.
    Bundles without the hook are silently skipped and not included in the
    return value.  Raises nothing on import failure — the bundle is skipped
    with a warning-level note (callers should surface this at startup).
    """
    try:
        active_bundles = load_active_skill_bundles(enabled_skill_ids=enabled_skill_ids)
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
) -> dict[str, str | None]:
    """Probe tool-loading for active skill bundles without side effects.

    Returns a mapping of ``skill_id`` → error string (or ``None`` if OK).
    A ``None`` value means the bundle registered successfully (or has no
    tool.py, which is also fine).  A string value is the failure reason.

    Intended for use in startup diagnostics only.
    """
    try:
        active_bundles = load_active_skill_bundles(enabled_skill_ids=enabled_skill_ids)
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

    registrar(registry)
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
