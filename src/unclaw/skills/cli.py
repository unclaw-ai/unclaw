"""Terminal-facing entry point for catalog-backed skill lifecycle commands."""

from __future__ import annotations

from pathlib import Path

from unclaw.settings import load_settings
from unclaw.skills.manager import run_skill_command


def main(
    *,
    project_root: Path | None = None,
    action: str = "list",
    skill_id: str | None = None,
    output_func=print,
) -> int:
    """Run one ``unclaw skills`` command and write the user-facing output."""
    try:
        settings = load_settings(project_root=project_root)
    except Exception as exc:
        output_func(f"Error loading settings: {exc}")
        return 1

    result = run_skill_command(settings, action=action, skill_id=skill_id)
    for line in result.lines:
        output_func(line)

    return 0 if result.ok else 1
