"""External catalog fetch and parse helpers for skills hub metadata."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from unclaw.skills.status import (
    SkillStatus,
    SkillStatusEntry,
    _compute_status,
    build_skill_status_report,
    read_local_skill_version,
    render_skill_search_results,
    render_skills_report,
    search_skill_status_entries,
)


class CatalogFetchError(Exception):
    """Raised when the remote catalog cannot be fetched or parsed."""


@dataclass(frozen=True, slots=True)
class RemoteCatalogEntry:
    """One entry parsed from the remote catalog.json."""

    skill_id: str
    display_name: str
    version: str | None
    summary: str | None
    repo_relative_path: str | None
    public_entry_files: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    repository_owner: str | None = None
    repository_name: str | None = None


def fetch_remote_catalog(
    url: str,
    *,
    timeout: float = 10.0,
) -> list[RemoteCatalogEntry]:
    """Fetch and parse ``catalog.json`` from one URL."""

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


def _parse_catalog_payload(
    payload: object,
    *,
    source_url: str,
) -> list[RemoteCatalogEntry]:
    """Extract one list of catalog entries from a parsed JSON payload."""

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

        item_repo = item.get("repository", {})
        if isinstance(item_repo, dict):
            repo_owner = _nullable_str(item_repo.get("owner")) or repo_owner_default
            repo_name = _nullable_str(item_repo.get("name")) or repo_name_default
        else:
            repo_owner = repo_owner_default
            repo_name = repo_name_default

        entries.append(
            RemoteCatalogEntry(
                skill_id=skill_id.strip(),
                display_name=_coerce_str(item.get("display_name"), fallback=skill_id),
                version=_nullable_str(item.get("version")),
                summary=_nullable_str(item.get("summary")),
                repo_relative_path=_nullable_str(item.get("repo_relative_path")),
                public_entry_files=_coerce_str_tuple(item.get("public_entry_files")),
                tags=_coerce_tags(item.get("tags")),
                repository_owner=repo_owner,
                repository_name=repo_name,
            )
        )

    return entries


def _coerce_str(value: object, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback.strip()


def _nullable_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )


def _coerce_tags(value: object) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return tuple(
            part.strip()
            for part in value.split(",")
            if part.strip()
        )
    return _coerce_str_tuple(value)


__all__ = [
    "CatalogFetchError",
    "RemoteCatalogEntry",
    "SkillStatus",
    "SkillStatusEntry",
    "_compute_status",
    "build_skill_status_report",
    "fetch_remote_catalog",
    "read_local_skill_version",
    "render_skill_search_results",
    "render_skills_report",
    "search_skill_status_entries",
]
