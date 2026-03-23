"""Shared lifecycle operations for optional catalog-backed skills."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from unclaw.settings import Settings, load_settings
from unclaw.skills.bundle_tools import clear_skill_tool_module_cache
from unclaw.skills.file_loader import discover_skill_bundles
from unclaw.skills.file_models import SkillBundle, clear_skill_bundle_cache
from unclaw.skills.installer import SkillInstallError, install_skill
from unclaw.skills.remote_catalog import CatalogFetchError, RemoteCatalogEntry, fetch_remote_catalog
from unclaw.skills.status import (
    SkillStatus,
    SkillStatusEntry,
    build_skill_status_report,
    read_local_skill_version,
    render_skill_search_results,
    render_skills_report,
    search_skill_status_entries,
)
from unclaw.skills.versioning import VersionComparison, compare_versions

_SKILL_ACTIONS = frozenset(
    {"list", "search", "install", "enable", "disable", "remove", "update"}
)
_BULK_UPDATE_SENTINEL = "--all"


@dataclass(frozen=True, slots=True)
class SkillCommandOutcome:
    """Result returned by one skill lifecycle command."""

    ok: bool
    lines: tuple[str, ...]
    updated_settings: Settings | None = None
    refresh_runtime: bool = False


@dataclass(frozen=True, slots=True)
class _SkillCatalogSnapshot:
    """Single-fetch view of catalog entries plus computed hub status."""

    catalog_entries: tuple[RemoteCatalogEntry, ...]
    status_entries: tuple[SkillStatusEntry, ...]


def run_skill_command(
    settings: Settings,
    *,
    action: str = "list",
    skill_id: str | None = None,
) -> SkillCommandOutcome:
    """Run one skills hub command using the shared catalog-only model."""

    normalized_action = action.strip().lower()
    if normalized_action not in _SKILL_ACTIONS:
        return SkillCommandOutcome(
            ok=False,
            lines=(f"Unknown skills command '{action}'.",),
        )

    if normalized_action == "list":
        return _list_skills(settings)

    normalized_argument = (skill_id or "").strip()
    if normalized_action == "search":
        if not normalized_argument:
            return SkillCommandOutcome(ok=False, lines=("A search query is required.",))
        return _search_skills(settings, normalized_argument)

    if normalized_action == "update" and normalized_argument == _BULK_UPDATE_SENTINEL:
        return _update_all_skills(settings)

    if not normalized_argument:
        return SkillCommandOutcome(ok=False, lines=("A skill id is required.",))

    match normalized_action:
        case "install":
            return _install_skill(settings, normalized_argument)
        case "enable":
            return _enable_skill(settings, normalized_argument)
        case "disable":
            return _disable_skill(settings, normalized_argument)
        case "remove":
            return _remove_skill(settings, normalized_argument)
        case "update":
            return _update_skill(settings, normalized_argument)
        case _:
            return SkillCommandOutcome(
                ok=False,
                lines=(f"Unknown skills command '{action}'.",),
            )


def _list_skills(settings: Settings) -> SkillCommandOutcome:
    snapshot = _load_catalog_snapshot(settings)
    if isinstance(snapshot, SkillCommandOutcome):
        return snapshot

    return SkillCommandOutcome(
        ok=True,
        lines=tuple(
            render_skills_report(
                snapshot.status_entries,
                catalog_url=settings.catalog.url,
            )
        ),
    )


def _search_skills(settings: Settings, query: str) -> SkillCommandOutcome:
    snapshot = _load_catalog_snapshot(settings)
    if isinstance(snapshot, SkillCommandOutcome):
        return snapshot

    matches = search_skill_status_entries(snapshot.status_entries, query)
    return SkillCommandOutcome(
        ok=True,
        lines=tuple(render_skill_search_results(matches, query=query)),
    )


def _install_skill(settings: Settings, skill_id: str) -> SkillCommandOutcome:
    entry = _find_catalog_entry(settings, skill_id)
    if isinstance(entry, SkillCommandOutcome):
        return entry

    local_bundle = _local_bundle_by_id(settings).get(skill_id)
    if local_bundle is not None:
        local_version = read_local_skill_version(local_bundle.bundle_dir)
        if local_version == entry.version:
            return SkillCommandOutcome(
                ok=True,
                lines=(_already_installed_message(skill_id, local_version),),
            )
        current_version = local_version or "unknown"
        return SkillCommandOutcome(
            ok=False,
            lines=(
                f"Skill '{skill_id}' is already installed at {current_version}. "
                f"Use `unclaw skills update {skill_id}`.",
            ),
        )

    try:
        _install_bundle_into_skills_root(settings, entry)
    except SkillInstallError as exc:
        return SkillCommandOutcome(
            ok=False,
            lines=(f"Could not install '{skill_id}'.", str(exc)),
        )

    version_suffix = f" (version {entry.version})" if entry.version else ""
    return SkillCommandOutcome(
        ok=True,
        lines=(f"Installed '{skill_id}' in ./skills/{skill_id}{version_suffix}.",),
    )


def _enable_skill(settings: Settings, skill_id: str) -> SkillCommandOutcome:
    if skill_id not in _local_bundle_by_id(settings):
        return SkillCommandOutcome(
            ok=False,
            lines=(
                f"Skill '{skill_id}' is not installed. "
                f"Run `unclaw skills install {skill_id}` first.",
            ),
        )

    if skill_id in settings.skills.enabled_skill_ids:
        return SkillCommandOutcome(
            ok=True,
            lines=(f"Skill '{skill_id}' is already enabled.",),
        )

    enabled_skill_ids = (*settings.skills.enabled_skill_ids, skill_id)
    try:
        updated_settings = _write_enabled_skill_ids(settings, enabled_skill_ids)
    except OSError as exc:
        return SkillCommandOutcome(
            ok=False,
            lines=("Could not update config/app.yaml.", str(exc)),
        )

    return SkillCommandOutcome(
        ok=True,
        lines=(f"Enabled '{skill_id}'.",),
        updated_settings=updated_settings,
        refresh_runtime=True,
    )


def _disable_skill(settings: Settings, skill_id: str) -> SkillCommandOutcome:
    if skill_id not in settings.skills.enabled_skill_ids:
        return SkillCommandOutcome(
            ok=True,
            lines=(f"Skill '{skill_id}' is already disabled.",),
        )

    enabled_skill_ids = tuple(
        configured_skill_id
        for configured_skill_id in settings.skills.enabled_skill_ids
        if configured_skill_id != skill_id
    )
    try:
        updated_settings = _write_enabled_skill_ids(settings, enabled_skill_ids)
    except OSError as exc:
        return SkillCommandOutcome(
            ok=False,
            lines=("Could not update config/app.yaml.", str(exc)),
        )

    return SkillCommandOutcome(
        ok=True,
        lines=(f"Disabled '{skill_id}'.",),
        updated_settings=updated_settings,
        refresh_runtime=True,
    )


def _remove_skill(settings: Settings, skill_id: str) -> SkillCommandOutcome:
    bundle = _local_bundle_by_id(settings).get(skill_id)
    if bundle is None:
        return SkillCommandOutcome(
            ok=False,
            lines=(f"Skill '{skill_id}' is not installed.",),
        )

    lines: list[str] = []
    updated_settings: Settings | None = None
    refresh_runtime = False

    if skill_id in settings.skills.enabled_skill_ids:
        disable_outcome = _disable_skill(settings, skill_id)
        if not disable_outcome.ok:
            return disable_outcome
        lines.extend(disable_outcome.lines)
        updated_settings = disable_outcome.updated_settings
        refresh_runtime = disable_outcome.refresh_runtime
        settings = updated_settings or settings

    try:
        _remove_bundle_dir(_skills_root(settings), skill_id)
        _invalidate_skill_runtime_cache(_skills_root(settings), skill_id)
    except OSError as exc:
        error_lines = [*lines, f"Could not remove './skills/{skill_id}'.", str(exc)]
        return SkillCommandOutcome(
            ok=False,
            lines=tuple(error_lines),
            updated_settings=updated_settings,
            refresh_runtime=refresh_runtime,
        )

    lines.append(f"Removed './skills/{skill_id}'.")
    return SkillCommandOutcome(
        ok=True,
        lines=tuple(lines),
        updated_settings=updated_settings,
        refresh_runtime=refresh_runtime,
    )


def _update_skill(settings: Settings, skill_id: str) -> SkillCommandOutcome:
    bundle = _local_bundle_by_id(settings).get(skill_id)
    if bundle is None:
        return SkillCommandOutcome(
            ok=False,
            lines=(f"Skill '{skill_id}' is not installed.",),
        )

    entry = _find_catalog_entry(settings, skill_id)
    if isinstance(entry, SkillCommandOutcome):
        return entry

    local_version = read_local_skill_version(bundle.bundle_dir)
    comparison = compare_versions(local_version, entry.version)
    if comparison in {VersionComparison.EQUAL, VersionComparison.LOCAL_NEWER}:
        return SkillCommandOutcome(
            ok=True,
            lines=(f"Skill '{skill_id}' is already current.",),
        )

    try:
        _replace_bundle_from_catalog(settings, entry)
        _invalidate_skill_runtime_cache(_skills_root(settings), skill_id)
    except (SkillInstallError, OSError) as exc:
        return SkillCommandOutcome(
            ok=False,
            lines=(f"Could not update '{skill_id}'.", str(exc)),
        )

    was_enabled = skill_id in settings.skills.enabled_skill_ids
    if comparison is VersionComparison.UNKNOWN:
        lines = [
            f"Refreshed '{skill_id}' from the catalog because version comparison was inconclusive.",
        ]
    else:
        lines = [f"Updated '{skill_id}' to {entry.version or 'the catalog version'}."]
    if was_enabled:
        lines.append(f"'{skill_id}' stayed enabled.")
    return SkillCommandOutcome(
        ok=True,
        lines=tuple(lines),
        refresh_runtime=was_enabled,
    )


def _update_all_skills(settings: Settings) -> SkillCommandOutcome:
    snapshot = _load_catalog_snapshot(settings)
    if isinstance(snapshot, SkillCommandOutcome):
        return snapshot

    installed_entries = [
        entry for entry in snapshot.status_entries if entry.installed_locally
    ]
    if not installed_entries:
        return SkillCommandOutcome(
            ok=True,
            lines=("No installed skills found in ./skills/.",),
        )

    catalog_by_id = {entry.skill_id: entry for entry in snapshot.catalog_entries}
    updated: list[str] = []
    already_current: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    refresh_runtime = False

    for entry in installed_entries:
        if entry.status is SkillStatus.ORPHANED:
            skipped.append(f"{entry.skill_id:<22}  not in current catalog")
            continue

        catalog_entry = catalog_by_id.get(entry.skill_id)
        if catalog_entry is None:
            skipped.append(f"{entry.skill_id:<22}  not in current catalog")
            continue

        if entry.version_comparison is VersionComparison.CATALOG_NEWER:
            try:
                _replace_bundle_from_catalog(settings, catalog_entry)
                _invalidate_skill_runtime_cache(_skills_root(settings), entry.skill_id)
            except (SkillInstallError, OSError) as exc:
                failed.append(f"{entry.skill_id:<22}  {exc}")
                continue

            enabled_suffix = " [enabled]" if entry.enabled_locally else ""
            updated.append(
                f"{entry.skill_id:<22}  {entry.local_version or '?'} -> {catalog_entry.version or '?'}{enabled_suffix}"
            )
            refresh_runtime = refresh_runtime or entry.enabled_locally
            continue

        if entry.version_comparison in {
            VersionComparison.EQUAL,
            VersionComparison.LOCAL_NEWER,
        }:
            version_text = entry.local_version or entry.catalog_version or "?"
            already_current.append(f"{entry.skill_id:<22}  {version_text}")
            continue

        skipped.append(
            f"{entry.skill_id:<22}  could not compare local {entry.local_version or '?'} "
            f"to catalog {catalog_entry.version or '?'}"
        )

    lines = _render_bulk_update_summary(
        updated=updated,
        already_current=already_current,
        skipped=skipped,
        failed=failed,
    )
    return SkillCommandOutcome(
        ok=not failed,
        lines=lines,
        refresh_runtime=refresh_runtime,
    )


def _load_catalog_snapshot(
    settings: Settings,
) -> _SkillCatalogSnapshot | SkillCommandOutcome:
    catalog_entries = _load_catalog_entries(settings)
    if isinstance(catalog_entries, SkillCommandOutcome):
        return catalog_entries

    status_entries = build_skill_status_report(
        local_bundles=discover_skill_bundles(skills_root=_skills_root(settings)),
        enabled_skill_ids=settings.skills.enabled_skill_ids,
        catalog_entries=catalog_entries,
    )
    return _SkillCatalogSnapshot(
        catalog_entries=tuple(catalog_entries),
        status_entries=tuple(status_entries),
    )


def _load_catalog_entries(
    settings: Settings,
) -> tuple[RemoteCatalogEntry, ...] | SkillCommandOutcome:
    try:
        return tuple(fetch_remote_catalog(settings.catalog.url))
    except CatalogFetchError as exc:
        return SkillCommandOutcome(ok=False, lines=_catalog_fetch_error_lines(exc))


def _find_catalog_entry(
    settings: Settings,
    skill_id: str,
) -> RemoteCatalogEntry | SkillCommandOutcome:
    catalog_entries = _load_catalog_entries(settings)
    if isinstance(catalog_entries, SkillCommandOutcome):
        return catalog_entries

    for entry in catalog_entries:
        if entry.skill_id == skill_id:
            return entry

    return SkillCommandOutcome(
        ok=False,
        lines=(f"Skill '{skill_id}' was not found in the catalog.",),
    )


def _local_bundle_by_id(settings: Settings) -> dict[str, SkillBundle]:
    return {
        bundle.skill_id: bundle
        for bundle in discover_skill_bundles(skills_root=_skills_root(settings))
    }


def _write_enabled_skill_ids(
    settings: Settings,
    enabled_skill_ids: tuple[str, ...],
) -> Settings:
    app_config_path = settings.paths.app_config_path
    try:
        payload = yaml.safe_load(app_config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OSError(f"Missing configuration file: {app_config_path}") from exc
    except yaml.YAMLError as exc:
        raise OSError(f"Invalid YAML in {app_config_path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Could not read {app_config_path}: {exc}") from exc

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise OSError(f"Configuration file must contain a mapping: {app_config_path}")

    skills_section = payload.get("skills")
    if not isinstance(skills_section, dict):
        skills_section = {}
        payload["skills"] = skills_section
    skills_section["enabled_skill_ids"] = list(dict.fromkeys(enabled_skill_ids))

    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    temp_path = app_config_path.with_name(f".{app_config_path.name}.tmp")
    try:
        temp_path.write_text(rendered, encoding="utf-8")
        temp_path.replace(app_config_path)
    except OSError as exc:
        raise OSError(f"Could not write {app_config_path}: {exc}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return load_settings(project_root=settings.paths.project_root)


def _install_bundle_into_skills_root(settings: Settings, entry: RemoteCatalogEntry) -> None:
    skills_root = _skills_root(settings)
    skills_root.mkdir(parents=True, exist_ok=True)
    _ensure_skills_package(skills_root)

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".unclaw-install-{entry.skill_id}-", dir=skills_root)
    )
    try:
        install_skill(entry, skills_root=staging_root, catalog_url=settings.catalog.url)
        staged_bundle = staging_root / entry.skill_id
        destination = skills_root / entry.skill_id
        staged_bundle.replace(destination)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    else:
        shutil.rmtree(staging_root, ignore_errors=True)


def _replace_bundle_from_catalog(settings: Settings, entry: RemoteCatalogEntry) -> None:
    skills_root = _skills_root(settings)
    destination = skills_root / entry.skill_id
    if not destination.is_dir():
        raise OSError(f"Skill '{entry.skill_id}' is not installed.")

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".unclaw-update-{entry.skill_id}-", dir=skills_root)
    )
    backup_root = Path(
        tempfile.mkdtemp(prefix=f".unclaw-backup-{entry.skill_id}-", dir=skills_root)
    )
    backup_bundle = backup_root / entry.skill_id

    try:
        install_skill(entry, skills_root=staging_root, catalog_url=settings.catalog.url)
        staged_bundle = staging_root / entry.skill_id
        destination.replace(backup_bundle)
        try:
            staged_bundle.replace(destination)
        except OSError:
            backup_bundle.replace(destination)
            raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)


def _remove_bundle_dir(skills_root: Path, skill_id: str) -> None:
    bundle_dir = (skills_root / skill_id).resolve()
    resolved_skills_root = skills_root.resolve()
    if bundle_dir.parent != resolved_skills_root:
        raise OSError(f"Refusing to remove unsafe skill path: {bundle_dir}")
    if not bundle_dir.exists():
        raise OSError(f"Skill '{skill_id}' is not installed.")
    if bundle_dir.is_symlink():
        raise OSError(f"Refusing to remove symlinked skill path: {bundle_dir}")
    shutil.rmtree(bundle_dir)


def _invalidate_skill_runtime_cache(skills_root: Path, skill_id: str) -> None:
    clear_skill_tool_module_cache(skill_id)
    clear_skill_bundle_cache(skills_root / skill_id / "SKILL.md")


def _ensure_skills_package(skills_root: Path) -> None:
    init_path = skills_root / "__init__.py"
    if init_path.exists():
        return
    init_path.write_text("", encoding="utf-8")


def _catalog_fetch_error_lines(exc: CatalogFetchError) -> tuple[str, ...]:
    return (
        "Could not fetch the skills catalog.",
        str(exc),
        "Local skill runtime is not affected.",
    )


def _already_installed_message(skill_id: str, version: str | None) -> str:
    if version:
        return f"Skill '{skill_id}' is already installed at {version}."
    return f"Skill '{skill_id}' is already installed."


def _render_bulk_update_summary(
    *,
    updated: list[str],
    already_current: list[str],
    skipped: list[str],
    failed: list[str],
) -> tuple[str, ...]:
    lines = ["Bulk update summary", ""]
    sections = (
        ("Updated", updated),
        ("Already current", already_current),
        ("Skipped", skipped),
        ("Failed", failed),
    )
    for index, (label, items) in enumerate(sections):
        lines.append(f"{label} ({len(items)})")
        if items:
            lines.extend(f"  {item}" for item in items)
        else:
            lines.append("  (none)")
        if index != len(sections) - 1:
            lines.append("")
    return tuple(lines)


def _skills_root(settings: Settings) -> Path:
    return settings.paths.project_root / "skills"


__all__ = ["SkillCommandOutcome", "run_skill_command"]
