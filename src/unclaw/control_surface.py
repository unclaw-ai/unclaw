"""User-facing local control presets and summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONTROL_PRESET_SAFE = "safe"
CONTROL_PRESET_WORKSPACE = "workspace"
CONTROL_PRESET_FULL = "full"
CONTROL_PRESET_CUSTOM = "custom"

CONTROL_PRESET_NAMES = (
    CONTROL_PRESET_SAFE,
    CONTROL_PRESET_WORKSPACE,
    CONTROL_PRESET_FULL,
)
CONTROL_PRESET_NAME_SET = frozenset(CONTROL_PRESET_NAMES)
DEFAULT_CONTROL_PRESET = CONTROL_PRESET_WORKSPACE
MIN_PROFILE_NUM_CTX = 1024


@dataclass(frozen=True, slots=True)
class ControlSurfaceSummary:
    """Compact user-facing summary of local file and terminal access."""

    preset_name: str
    preset_description: str
    access_scope: str
    allowed_roots: tuple[Path, ...]


def normalize_control_preset_name(preset_name: str) -> str:
    """Normalize and validate one explicit control preset value."""

    normalized = preset_name.strip().lower()
    if normalized not in CONTROL_PRESET_NAME_SET:
        allowed = ", ".join(CONTROL_PRESET_NAMES)
        raise ValueError(f"Control preset must be one of: {allowed}.")
    return normalized


def resolve_control_surface(
    *,
    configured_preset_name: str | None,
    configured_roots: tuple[str, ...],
    project_root: Path,
    files_dir: Path,
) -> tuple[str, tuple[str, ...]]:
    """Resolve the effective preset name and config roots for runtime use."""

    if configured_preset_name is not None:
        normalized_preset = normalize_control_preset_name(configured_preset_name)
        return (
            normalized_preset,
            derive_control_root_strings(
                preset_name=normalized_preset,
                project_root=project_root,
                files_dir=files_dir,
            ),
        )

    inferred_preset = infer_control_preset_name(
        configured_roots=configured_roots,
        project_root=project_root,
        files_dir=files_dir,
    )
    return inferred_preset, configured_roots


def infer_control_preset_name(
    *,
    configured_roots: tuple[str, ...],
    project_root: Path,
    files_dir: Path,
) -> str:
    """Infer a preset label from configured roots for legacy compatibility."""

    normalized_configured_roots = tuple(
        dict.fromkeys(
            _resolve_config_root(root, project_root=project_root)
            for root in configured_roots
        )
    )

    for preset_name in CONTROL_PRESET_NAMES:
        if normalized_configured_roots == derive_control_root_paths(
            preset_name=preset_name,
            project_root=project_root,
            files_dir=files_dir,
        ):
            return preset_name

    return CONTROL_PRESET_CUSTOM


def derive_control_root_strings(
    *,
    preset_name: str,
    project_root: Path,
    files_dir: Path,
) -> tuple[str, ...]:
    """Return config-friendly root strings for one preset."""

    return tuple(
        _path_to_config_string(path, project_root=project_root)
        for path in derive_control_root_paths(
            preset_name=preset_name,
            project_root=project_root,
            files_dir=files_dir,
        )
    )


def derive_control_root_paths(
    *,
    preset_name: str,
    project_root: Path,
    files_dir: Path,
) -> tuple[Path, ...]:
    """Return the concrete allowed roots for one preset."""

    normalized_preset = normalize_control_preset_name(preset_name)
    resolved_project_root = project_root.expanduser().resolve()
    resolved_files_dir = files_dir.expanduser().resolve()

    if normalized_preset == CONTROL_PRESET_SAFE:
        return (resolved_files_dir,)
    if normalized_preset == CONTROL_PRESET_WORKSPACE:
        return (resolved_project_root,)

    home_dir = Path.home().expanduser().resolve()
    return tuple(dict.fromkeys((resolved_project_root, home_dir)))


def build_control_surface_summary(
    *,
    preset_name: str,
    project_root: Path,
    allowed_roots: tuple[str, ...],
) -> ControlSurfaceSummary:
    """Build one compact user-facing summary of local access."""

    resolved_roots = tuple(
        dict.fromkeys(
            _resolve_config_root(root, project_root=project_root)
            for root in allowed_roots
        )
    )
    normalized_preset = preset_name.strip().lower()
    return ControlSurfaceSummary(
        preset_name=normalized_preset,
        preset_description=_preset_description(normalized_preset),
        access_scope=_preset_scope_label(normalized_preset),
        allowed_roots=resolved_roots,
    )


def _resolve_config_root(raw_root: str, *, project_root: Path) -> Path:
    candidate = Path(raw_root).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _path_to_config_string(path: Path, *, project_root: Path) -> str:
    try:
        relative_path = path.relative_to(project_root)
    except ValueError:
        return str(path)
    if not relative_path.parts:
        return "."
    return relative_path.as_posix()


def _preset_description(preset_name: str) -> str:
    if preset_name == CONTROL_PRESET_SAFE:
        return "File and terminal tools stay inside the local files sandbox."
    if preset_name == CONTROL_PRESET_WORKSPACE:
        return "File and terminal tools stay inside this project workspace."
    if preset_name == CONTROL_PRESET_FULL:
        return "File and terminal tools can access your project and home directory."
    return "File and terminal tools follow the custom allowed roots in config."


def _preset_scope_label(preset_name: str) -> str:
    if preset_name == CONTROL_PRESET_FULL:
        return "broad"
    return "restricted"


__all__ = [
    "CONTROL_PRESET_CUSTOM",
    "CONTROL_PRESET_FULL",
    "CONTROL_PRESET_NAME_SET",
    "CONTROL_PRESET_NAMES",
    "CONTROL_PRESET_SAFE",
    "CONTROL_PRESET_WORKSPACE",
    "ControlSurfaceSummary",
    "DEFAULT_CONTROL_PRESET",
    "MIN_PROFILE_NUM_CTX",
    "build_control_surface_summary",
    "derive_control_root_paths",
    "derive_control_root_strings",
    "infer_control_preset_name",
    "normalize_control_preset_name",
    "resolve_control_surface",
]
