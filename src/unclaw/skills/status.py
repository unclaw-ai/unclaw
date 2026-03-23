"""Status computation, search, and terminal rendering for the skills hub."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from unclaw.skills.file_models import SkillBundle
from unclaw.skills.versioning import VersionComparison, compare_versions

if TYPE_CHECKING:
    from unclaw.skills.remote_catalog import RemoteCatalogEntry


class SkillStatus(StrEnum):
    """Primary user-facing state of a skill in the hub."""

    ENABLED = "enabled"
    INSTALLED = "installed"
    UPDATE = "update"
    AVAILABLE = "available"
    ORPHANED = "orphaned"
    LOCAL_UNTRACKED = "orphaned"


@dataclass(frozen=True, slots=True)
class SkillStatusEntry:
    """Normalised per-skill status combining local and remote catalog data."""

    skill_id: str
    display_name: str
    installed_locally: bool
    enabled_locally: bool
    available_in_catalog: bool
    local_version: str | None
    catalog_version: str | None
    status: SkillStatus
    repo_relative_path: str | None
    summary: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    version_comparison: VersionComparison = VersionComparison.UNKNOWN


_REPORT_SECTION_ORDER = (
    SkillStatus.UPDATE,
    SkillStatus.ENABLED,
    SkillStatus.INSTALLED,
    SkillStatus.AVAILABLE,
    SkillStatus.ORPHANED,
)
_REPORT_SECTION_LABELS = {
    SkillStatus.UPDATE: "Update",
    SkillStatus.ENABLED: "Enabled",
    SkillStatus.INSTALLED: "Installed",
    SkillStatus.AVAILABLE: "Available",
    SkillStatus.ORPHANED: "Orphaned",
}
_SEARCH_RESULT_PRIORITY = {
    SkillStatus.UPDATE: 0,
    SkillStatus.ENABLED: 1,
    SkillStatus.INSTALLED: 2,
    SkillStatus.AVAILABLE: 3,
    SkillStatus.ORPHANED: 4,
}


def read_local_skill_version(bundle_dir: Path) -> str | None:
    """Return the ``version`` field from *bundle_dir*/_meta.json, or ``None``."""

    meta_path = bundle_dir / "_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    version = meta.get("version")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return None


def build_skill_status_report(
    *,
    local_bundles: Sequence[SkillBundle],
    enabled_skill_ids: Collection[str],
    catalog_entries: Sequence[RemoteCatalogEntry],
) -> list[SkillStatusEntry]:
    """Build a normalised skills hub report from local and catalog data."""

    enabled_ids = frozenset(enabled_skill_ids)
    catalog_by_id: dict[str, RemoteCatalogEntry] = {
        entry.skill_id: entry for entry in catalog_entries
    }
    local_by_id: dict[str, SkillBundle] = {bundle.skill_id: bundle for bundle in local_bundles}

    all_skill_ids = sorted(frozenset(catalog_by_id) | frozenset(local_by_id))

    results: list[SkillStatusEntry] = []
    for skill_id in all_skill_ids:
        local_bundle = local_by_id.get(skill_id)
        catalog_entry = catalog_by_id.get(skill_id)

        installed_locally = local_bundle is not None
        available_in_catalog = catalog_entry is not None
        enabled_locally = installed_locally and skill_id in enabled_ids

        local_version = (
            read_local_skill_version(local_bundle.bundle_dir)
            if local_bundle is not None
            else None
        )
        catalog_version = catalog_entry.version if catalog_entry is not None else None
        version_comparison = compare_versions(local_version, catalog_version)

        display_name = _pick_display_name(skill_id, local_bundle=local_bundle, catalog_entry=catalog_entry)
        summary = _pick_summary(local_bundle=local_bundle, catalog_entry=catalog_entry)
        tags = catalog_entry.tags if catalog_entry is not None else ()
        repo_relative_path = (
            catalog_entry.repo_relative_path if catalog_entry is not None else None
        )

        status = _compute_status(
            installed_locally=installed_locally,
            enabled_locally=enabled_locally,
            available_in_catalog=available_in_catalog,
            version_comparison=version_comparison,
        )

        results.append(
            SkillStatusEntry(
                skill_id=skill_id,
                display_name=display_name,
                installed_locally=installed_locally,
                enabled_locally=enabled_locally,
                available_in_catalog=available_in_catalog,
                local_version=local_version,
                catalog_version=catalog_version,
                status=status,
                repo_relative_path=repo_relative_path,
                summary=summary,
                tags=tags,
                version_comparison=version_comparison,
            )
        )

    return results


def search_skill_status_entries(
    entries: Sequence[SkillStatusEntry],
    query: str,
) -> list[SkillStatusEntry]:
    """Return deterministic case-insensitive matches for one query."""

    terms = tuple(term for term in query.casefold().split() if term)
    if not terms:
        return []

    matches = [
        entry
        for entry in entries
        if all(term in _searchable_text(entry) for term in terms)
    ]
    return sorted(
        matches,
        key=lambda entry: (_SEARCH_RESULT_PRIORITY[entry.status], entry.skill_id),
    )


def render_skills_report(
    entries: Sequence[SkillStatusEntry],
    *,
    catalog_url: str,
) -> list[str]:
    """Render the main grouped skills hub listing."""

    grouped_entries = _group_entries(entries)
    lines = [
        "Skills Summary: "
        + ", ".join(
            f"{len(grouped_entries[status])} {status.value}"
            for status in _REPORT_SECTION_ORDER
        ),
        "",
    ]

    for status in _REPORT_SECTION_ORDER:
        section_entries = grouped_entries[status]
        lines.append(f"{_REPORT_SECTION_LABELS[status]} ({len(section_entries)})")
        if section_entries:
            for entry in section_entries:
                lines.append(_format_report_line(entry))
        else:
            lines.append("  (none)")
        lines.append("")

    lines.append(f"Catalog: {catalog_url}")
    return lines


def render_skill_search_results(
    entries: Sequence[SkillStatusEntry],
    *,
    query: str,
) -> list[str]:
    """Render compact search results with current skill state."""

    if not entries:
        return [f"No skills matched '{query}'."]

    lines = [f"Search results for '{query}' ({len(entries)})"]
    for entry in entries:
        lines.append(_format_search_line(entry))
    return lines


def _compute_status(
    *,
    installed_locally: bool,
    available_in_catalog: bool,
    enabled_locally: bool = False,
    version_comparison: VersionComparison | None = None,
    local_version: str | None = None,
    catalog_version: str | None = None,
) -> SkillStatus:
    if version_comparison is None:
        version_comparison = compare_versions(local_version, catalog_version)
    if installed_locally and not available_in_catalog:
        return SkillStatus.ORPHANED
    if not installed_locally and available_in_catalog:
        return SkillStatus.AVAILABLE
    if installed_locally and available_in_catalog:
        if version_comparison is VersionComparison.CATALOG_NEWER:
            return SkillStatus.UPDATE
        if enabled_locally:
            return SkillStatus.ENABLED
        return SkillStatus.INSTALLED
    return SkillStatus.ORPHANED


def _pick_display_name(
    skill_id: str,
    *,
    local_bundle: SkillBundle | None,
    catalog_entry: RemoteCatalogEntry | None,
) -> str:
    if catalog_entry is not None:
        return catalog_entry.display_name
    if local_bundle is not None:
        return local_bundle.display_name
    return skill_id


def _pick_summary(
    *,
    local_bundle: SkillBundle | None,
    catalog_entry: RemoteCatalogEntry | None,
) -> str | None:
    if catalog_entry is not None and catalog_entry.summary:
        return catalog_entry.summary
    if local_bundle is not None:
        return local_bundle.summary
    return None


def _group_entries(
    entries: Sequence[SkillStatusEntry],
) -> dict[SkillStatus, list[SkillStatusEntry]]:
    grouped_entries = {status: [] for status in _REPORT_SECTION_ORDER}
    for entry in entries:
        grouped_entries.setdefault(entry.status, []).append(entry)
    return grouped_entries


def _format_report_line(entry: SkillStatusEntry) -> str:
    version_text = _version_text(entry)
    name_text = _display_name_text(entry)
    return f"  {entry.skill_id:<22}  {version_text:<18}  {name_text}"


def _format_search_line(entry: SkillStatusEntry) -> str:
    version_text = _version_text(entry)
    summary_suffix = f" - {entry.summary}" if entry.summary else ""
    enabled_suffix = " [enabled]" if entry.enabled_locally and entry.status is not SkillStatus.ENABLED else ""
    return (
        f"  {entry.skill_id:<22}  [{entry.status.value:<9}]  "
        f"{version_text:<18}  {entry.display_name}{summary_suffix}{enabled_suffix}"
    )


def _display_name_text(entry: SkillStatusEntry) -> str:
    suffix_parts: list[str] = []
    if entry.enabled_locally and entry.status is not SkillStatus.ENABLED:
        suffix_parts.append("[enabled]")
    if suffix_parts:
        return f"{entry.display_name} {' '.join(suffix_parts)}"
    return entry.display_name


def _version_text(entry: SkillStatusEntry) -> str:
    if entry.status is SkillStatus.UPDATE:
        return f"{entry.local_version or '?'} -> {entry.catalog_version or '?'}"
    if entry.installed_locally:
        return entry.local_version or "?"
    return entry.catalog_version or "?"


def _searchable_text(entry: SkillStatusEntry) -> str:
    return " ".join(
        part.casefold()
        for part in (
            entry.skill_id,
            entry.display_name,
            entry.summary or "",
            *entry.tags,
        )
        if part
    )


__all__ = [
    "SkillStatus",
    "SkillStatusEntry",
    "_compute_status",
    "build_skill_status_report",
    "read_local_skill_version",
    "render_skill_search_results",
    "render_skills_report",
    "search_skill_status_entries",
]
