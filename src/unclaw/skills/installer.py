"""Skill bundle installation from the remote catalog.

Architecture
------------
The installer downloads every file listed in
``RemoteCatalogEntry.public_entry_files`` from the raw GitHub URL derived
from the catalog base URL and ``repo_relative_path``, then writes them into
``skills_root/<skill_id>/``.

A ``_meta.json`` sidecar is also written with the skill version so that
:func:`~unclaw.skills.remote_catalog.read_local_skill_version` can track
whether the local copy is up-to-date.

URL derivation
--------------
Given the catalog URL (e.g.
``https://raw.githubusercontent.com/unclaw-ai/skills/main/catalog.json``),
the base URL is the catalog URL with ``catalog.json`` stripped:
``https://raw.githubusercontent.com/unclaw-ai/skills/main/``.

Each file URL is then::

    <base_url><repo_relative_path>/<filename>

This approach is fully determined by the catalog URL — no separate
owner / repo / branch values are hardcoded.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from unclaw.skills.remote_catalog import RemoteCatalogEntry


class SkillInstallError(Exception):
    """Raised when a skill bundle cannot be installed."""


def catalog_base_url(catalog_url: str) -> str:
    """Return the base URL for raw file downloads derived from *catalog_url*.

    Strips the trailing filename (typically ``catalog.json``) from the URL so
    that skill file paths can be appended directly.

    >>> catalog_base_url("https://raw.githubusercontent.com/unclaw-ai/skills/main/catalog.json")
    'https://raw.githubusercontent.com/unclaw-ai/skills/main/'
    """
    parsed = urllib.parse.urlparse(catalog_url)
    # Strip the last path component (catalog.json) and keep the trailing slash.
    base_path = parsed.path.rsplit("/", 1)[0] + "/"
    return urllib.parse.urlunparse(parsed._replace(path=base_path))


def build_file_url(base_url: str, repo_relative_path: str, filename: str) -> str:
    """Build the raw download URL for one skill file.

    Args:
        base_url: Result of :func:`catalog_base_url`.
        repo_relative_path: ``repo_relative_path`` field from the catalog entry
            (e.g. ``"weather"``).
        filename: File name within the skill directory (e.g. ``"SKILL.md"``).
    """
    # Normalise: remove leading/trailing slashes to avoid double-slash URLs.
    rel = repo_relative_path.strip("/")
    return f"{base_url}{rel}/{filename}"


def install_skill(
    entry: RemoteCatalogEntry,
    *,
    skills_root: Path,
    catalog_url: str,
    timeout: float = 15.0,
) -> None:
    """Download and install one skill bundle from the remote catalog.

    Downloads every file in ``entry.public_entry_files`` into
    ``skills_root/<skill_id>/`` and writes ``_meta.json`` with the version.

    Raises :class:`SkillInstallError` when:

    - ``entry.repo_relative_path`` is missing
    - ``entry.public_entry_files`` is empty
    - any file download fails
    - writing to the filesystem fails
    """
    if not entry.repo_relative_path:
        raise SkillInstallError(
            f"Catalog entry for {entry.skill_id!r} has no repo_relative_path — "
            "cannot determine download location."
        )
    if not entry.public_entry_files:
        raise SkillInstallError(
            f"Catalog entry for {entry.skill_id!r} lists no public_entry_files — "
            "nothing to install."
        )

    base = catalog_base_url(catalog_url)
    bundle_dir = skills_root / entry.skill_id

    try:
        bundle_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SkillInstallError(
            f"Could not create skill directory {bundle_dir}: {exc}"
        ) from exc

    for filename in entry.public_entry_files:
        url = build_file_url(base, entry.repo_relative_path, filename)
        try:
            content = _download_bytes(url, timeout=timeout)
        except SkillInstallError:
            raise
        dest = bundle_dir / filename
        try:
            dest.write_bytes(content)
        except OSError as exc:
            raise SkillInstallError(
                f"Could not write {dest}: {exc}"
            ) from exc

    _write_meta(bundle_dir, version=entry.version)


def _download_bytes(url: str, *, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise SkillInstallError(
            f"HTTP {exc.code} when downloading {url!r}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SkillInstallError(
            f"Could not download {url!r}: {exc}"
        ) from exc
    except OSError as exc:
        raise SkillInstallError(
            f"Network error downloading {url!r}: {exc}"
        ) from exc


def _write_meta(bundle_dir: Path, *, version: str | None) -> None:
    meta: dict[str, object] = {}
    if version is not None:
        meta["version"] = version
    try:
        (bundle_dir / "_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise SkillInstallError(
            f"Could not write _meta.json in {bundle_dir}: {exc}"
        ) from exc


__all__ = [
    "SkillInstallError",
    "build_file_url",
    "catalog_base_url",
    "install_skill",
]
