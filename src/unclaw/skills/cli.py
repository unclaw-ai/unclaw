"""Terminal-facing entry point for the ``unclaw skills`` subcommand.

Shows the status of all known skills — installed locally, available in the
official catalog, and any pending updates — without modifying any local files.

Usage
-----
    unclaw skills
"""

from __future__ import annotations

from pathlib import Path

from unclaw.settings import load_settings
from unclaw.skills.file_loader import discover_skill_bundles
from unclaw.skills.remote_catalog import (
    CatalogFetchError,
    build_skill_status_report,
    fetch_remote_catalog,
    render_skills_report,
)


def main(
    *,
    project_root: Path | None = None,
    output_func=print,
) -> int:
    """Fetch the remote catalog and print the skill status report.

    Returns 0 on success, 1 on any error.  ``output_func`` is injectable for
    testing; defaults to ``print``.
    """
    try:
        settings = load_settings(project_root=project_root)
    except Exception as exc:
        output_func(f"Error loading settings: {exc}")
        return 1

    catalog_url = settings.catalog.url

    try:
        catalog_entries = fetch_remote_catalog(catalog_url)
    except CatalogFetchError as exc:
        output_func("Error: could not fetch the skills catalog.")
        output_func(str(exc))
        output_func("Local skill runtime is not affected.")
        return 1

    local_bundles = discover_skill_bundles(
        skills_root=settings.paths.project_root / "skills"
    )

    entries = build_skill_status_report(
        local_bundles=local_bundles,
        enabled_skill_ids=settings.skills.enabled_skill_ids,
        catalog_entries=catalog_entries,
    )

    for line in render_skills_report(entries, catalog_url=catalog_url):
        output_func(line)

    return 0
