"""Compact prompt catalog rendering for active file-first skill bundles."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

from unclaw.skills.file_loader import load_active_skill_bundles
from unclaw.skills.file_models import SkillBundle


def build_active_skill_catalog(
    *,
    enabled_skill_ids: Collection[str],
    skills_root: Path | None = None,
    discovered_skill_bundles: Sequence[SkillBundle] | None = None,
) -> str:
    active_skill_bundles = load_active_skill_bundles(
        enabled_skill_ids=enabled_skill_ids,
        skills_root=skills_root,
        discovered_skill_bundles=discovered_skill_bundles,
    )
    if not active_skill_bundles:
        return ""

    lines = ["Active optional skills:"]
    for skill_bundle in active_skill_bundles:
        description = " ".join(
            part
            for part in (skill_bundle.summary, *skill_bundle.tool_hints)
            if part
        )
        lines.append(f"- {skill_bundle.skill_id}: {description}")

    return "\n".join(lines)


__all__ = ["build_active_skill_catalog"]
