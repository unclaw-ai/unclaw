"""External skills catalog reader, status model, and comparison layer.

Architecture
------------
This module is the **only** place that reaches out to the remote catalog.
It never falls back to a local filesystem path.  The catalog URL is always
passed in by the caller (ultimately sourced from ``settings.catalog.url``).

Public surface
--------------
- :class:`RemoteCatalogEntry`   — one entry from catalog.json
- :class:`SkillStatusEntry`     — normalised per-skill status
- :class:`SkillStatus`          — status enum
- :class:`CatalogFetchError`    — raised on any fetch/parse failure
- :func:`fetch_remote_catalog`  — fetch + parse catalog.json
- :func:`read_local_skill_version` — read version from local _meta.json
- :func:`build_skill_status_report` — compute status for all skills
- :func:`render_skills_report`  — terminal-friendly output lines
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from unclaw.skills.file_models import SkillBundle


class SkillStatus(StrEnum):
    """Computed status of a skill relative to local installation and catalog."""

    INSTALLED = "installed"
    """Locally installed and matches the catalog (or no version to compare)."""

    AVAILABLE = "available"
    """In the catalog but not installed locally."""

    UPDATE = "update"
    """Installed locally and in the catalog with a differing version."""

    LOCAL_UNTRACKED = "local_untracked"
    """Installed locally but absent from the catalog (custom/dev skill)."""


class CatalogFetchError(Exception):
    """Raised when the remote catalog cannot be fetched or parsed.

    The caller must handle this and must NOT silently fall back to any
    local filesystem path.
    """


@dataclass(frozen=True, slots=True)
class RemoteCatalogEntry:
    """One entry parsed from the remote catalog.json."""

    skill_id: str
    display_name: str
    version: str | None
    summary: str | None
    repo_relative_path: str | None
    public_entry_files: tuple[str, ...] = field(default_factory=tuple)
    """File names within *repo_relative_path* that must be downloaded to install this skill."""
    repository_owner: str | None = None
    """GitHub organisation / user that owns the skills repository."""
    repository_name: str | None = None
    """GitHub repository name (e.g. ``"skills"``)."""


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


# ── Fetch ─────────────────────────────────────────────────────────────────────


def fetch_remote_catalog(
    url: str,
    *,
    timeout: float = 10.0,
) -> list[RemoteCatalogEntry]:
    """Fetch and parse catalog.json from *url*.

    Raises :class:`CatalogFetchError` if the URL cannot be reached, the
    response is not valid JSON, or the payload is not a recognised catalog
    format.

    This function **never** falls back to a local path.  Passing a ``file://``
    URL is supported for tests but is otherwise unsupported.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise CatalogFetchError(
            f"Could not reach catalog at {url!r}: {exc}"
        ) from exc
    except OSError as exc:
        raise CatalogFetchError(
            f"Network error while fetching catalog from {url!r}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise CatalogFetchError(
            f"Catalog response from {url!r} is not valid JSON: {exc}"
        ) from exc

    return _parse_catalog_payload(payload, source_url=url)


# ── Local version ─────────────────────────────────────────────────────────────


def read_local_skill_version(bundle_dir: Path) -> str | None:
    """Return the ``version`` field from *bundle_dir*/_meta.json, or None.

    Returns None silently when _meta.json is absent, unreadable, or the
    version field is missing or empty.  This is the expected state for skill
    bundles that were installed before the _meta.json convention was adopted.
    """
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


# ── Comparison ────────────────────────────────────────────────────────────────


def build_skill_status_report(
    *,
    local_bundles: Sequence[SkillBundle],
    enabled_skill_ids: Collection[str],
    catalog_entries: Sequence[RemoteCatalogEntry],
) -> list[SkillStatusEntry]:
    """Build a normalised per-skill status list from local and remote data.

    The full union of skill IDs (local ∪ catalog) is enumerated.  Each entry
    carries installation state, enabled state, version strings, and a computed
    :class:`SkillStatus` label.

    Results are sorted alphabetically by ``skill_id``.
    """
    enabled_ids = frozenset(enabled_skill_ids)
    catalog_by_id: dict[str, RemoteCatalogEntry] = {
        e.skill_id: e for e in catalog_entries
    }
    local_by_id: dict[str, SkillBundle] = {b.skill_id: b for b in local_bundles}

    all_skill_ids = sorted(frozenset(catalog_by_id) | frozenset(local_by_id))

    results: list[SkillStatusEntry] = []
    for skill_id in all_skill_ids:
        local_bundle = local_by_id.get(skill_id)
        catalog_entry = catalog_by_id.get(skill_id)

        installed_locally = local_bundle is not None
        enabled_locally = skill_id in enabled_ids
        available_in_catalog = catalog_entry is not None

        local_version = (
            read_local_skill_version(local_bundle.bundle_dir)
            if local_bundle is not None
            else None
        )
        catalog_version = catalog_entry.version if catalog_entry is not None else None

        if catalog_entry is not None:
            display_name = catalog_entry.display_name
        elif local_bundle is not None:
            display_name = local_bundle.display_name
        else:
            display_name = skill_id

        repo_relative_path = (
            catalog_entry.repo_relative_path if catalog_entry is not None else None
        )

        status = _compute_status(
            installed_locally=installed_locally,
            available_in_catalog=available_in_catalog,
            local_version=local_version,
            catalog_version=catalog_version,
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
            )
        )

    return results


# ── Rendering ─────────────────────────────────────────────────────────────────


def render_skills_report(
    entries: list[SkillStatusEntry],
    *,
    catalog_url: str,
) -> list[str]:
    """Render a terminal-friendly skills status report as a list of lines.

    Three primary sections are emitted in order: Installed, Update, Available.
    Skills installed locally but absent from the catalog appear at the bottom
    of the Installed section labelled ``[untracked]``.
    """
    installed = [e for e in entries if e.status is SkillStatus.INSTALLED]
    local_untracked = [e for e in entries if e.status is SkillStatus.LOCAL_UNTRACKED]
    updates = [e for e in entries if e.status is SkillStatus.UPDATE]
    available = [e for e in entries if e.status is SkillStatus.AVAILABLE]

    lines: list[str] = []

    # ── Installed ──────────────────────────────────────────────────────────
    installed_count = len(installed) + len(local_untracked)
    lines.append(f"Installed ({installed_count})")
    if installed or local_untracked:
        for entry in installed:
            enabled_tag = " [enabled]" if entry.enabled_locally else ""
            version_str = entry.local_version or "?"
            lines.append(
                f"  {entry.skill_id:<22}  {version_str:<8}  "
                f"{entry.display_name}{enabled_tag}"
            )
        for entry in local_untracked:
            enabled_tag = " [enabled]" if entry.enabled_locally else ""
            lines.append(
                f"  {entry.skill_id:<22}  {'?':<8}  "
                f"{entry.display_name}{enabled_tag}  [untracked]"
            )
    else:
        lines.append("  (none)")

    lines.append("")

    # ── Update ─────────────────────────────────────────────────────────────
    lines.append(f"Update ({len(updates)})")
    if updates:
        for entry in updates:
            enabled_tag = " [enabled]" if entry.enabled_locally else ""
            lines.append(
                f"  {entry.skill_id:<22}  "
                f"local: {entry.local_version or '?':<8}  "
                f"catalog: {entry.catalog_version or '?':<8}  "
                f"{entry.display_name}{enabled_tag}"
            )
    else:
        lines.append("  (none)")

    lines.append("")

    # ── Available ──────────────────────────────────────────────────────────
    lines.append(f"Available ({len(available)})")
    if available:
        for entry in available:
            version_str = entry.catalog_version or "?"
            lines.append(
                f"  {entry.skill_id:<22}  {version_str:<8}  {entry.display_name}"
            )
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"Catalog: {catalog_url}")

    return lines


# ── Internal helpers ──────────────────────────────────────────────────────────


def _parse_catalog_payload(
    payload: object,
    *,
    source_url: str,
) -> list[RemoteCatalogEntry]:
    """Extract a list of catalog entries from a parsed JSON payload."""
    if isinstance(payload, list):
        skills_data = payload
        repo_owner_default: str | None = None
        repo_name_default: str | None = None
    elif isinstance(payload, dict):
        if "skills" not in payload:
            raise CatalogFetchError(
                f"Unrecognised catalog format from {source_url!r}: "
                "expected a JSON list or an object with a 'skills' key."
            )
        skills_data = payload["skills"]
        if not isinstance(skills_data, list):
            raise CatalogFetchError(
                f"Catalog from {source_url!r}: 'skills' field must be a list."
            )
        # Top-level repository metadata (optional)
        repo_section = payload.get("repository", {})
        repo_owner_default = _nullable_str(
            repo_section.get("owner") if isinstance(repo_section, dict) else None
        )
        repo_name_default = _nullable_str(
            repo_section.get("name") if isinstance(repo_section, dict) else None
        )
    else:
        raise CatalogFetchError(
            f"Unrecognised catalog format from {source_url!r}: "
            "expected a JSON list or an object with a 'skills' key."
        )

    entries: list[RemoteCatalogEntry] = []
    for item in skills_data:
        if not isinstance(item, dict):
            continue
        skill_id = item.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id.strip():
            continue

        # Per-entry repository metadata overrides the top-level default.
        item_repo = item.get("repository", {})
        if isinstance(item_repo, dict):
            repo_owner = _nullable_str(item_repo.get("owner")) or repo_owner_default
            repo_name = _nullable_str(item_repo.get("name")) or repo_name_default
        else:
            repo_owner = repo_owner_default
            repo_name = repo_name_default

        raw_files = item.get("public_entry_files")
        if isinstance(raw_files, list):
            entry_files = tuple(
                f for f in raw_files if isinstance(f, str) and f.strip()
            )
        else:
            entry_files = ()

        entries.append(
            RemoteCatalogEntry(
                skill_id=skill_id.strip(),
                display_name=_coerce_str(item.get("display_name"), fallback=skill_id),
                version=_nullable_str(item.get("version")),
                summary=_nullable_str(item.get("summary")),
                repo_relative_path=_nullable_str(item.get("repo_relative_path")),
                public_entry_files=entry_files,
                repository_owner=repo_owner,
                repository_name=repo_name,
            )
        )

    return entries


def _compute_status(
    *,
    installed_locally: bool,
    available_in_catalog: bool,
    local_version: str | None,
    catalog_version: str | None,
) -> SkillStatus:
    """Compute the :class:`SkillStatus` label for one skill.

    Version comparison is a simple exact string equality check.  If either
    version is None the comparison is skipped and the skill is considered
    up-to-date (INSTALLED).
    """
    if installed_locally and not available_in_catalog:
        return SkillStatus.LOCAL_UNTRACKED
    if not installed_locally and available_in_catalog:
        return SkillStatus.AVAILABLE
    if installed_locally and available_in_catalog:
        if (
            local_version is not None
            and catalog_version is not None
            and local_version != catalog_version
        ):
            return SkillStatus.UPDATE
        return SkillStatus.INSTALLED
    # Defensive: both False (should not occur in practice).
    return SkillStatus.LOCAL_UNTRACKED


def _coerce_str(value: object, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback.strip()


def _nullable_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


__all__ = [
    "CatalogFetchError",
    "RemoteCatalogEntry",
    "SkillStatus",
    "SkillStatusEntry",
    "build_skill_status_report",
    "fetch_remote_catalog",
    "read_local_skill_version",
    "render_skills_report",
]
