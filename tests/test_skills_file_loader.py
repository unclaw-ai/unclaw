from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.skills.catalog import build_active_skill_catalog
from unclaw.skills.file_loader import (
    discover_skill_bundles,
    load_active_skill_bundles,
    load_skill_bundle,
    shipped_skill_bundle_root,
)
from unclaw.skills.file_models import UnknownSkillIdError

pytestmark = pytest.mark.unit


def test_shipped_skill_bundle_root_points_at_top_level_skills_directory() -> None:
    skills_root = shipped_skill_bundle_root()
    project_root = skills_root.parent

    assert skills_root.name == "skills"
    assert (skills_root / "weather" / "SKILL.md").is_file()
    assert not (project_root / "src" / "unclaw" / "skills" / "weather" / "SKILL.md").exists()


def test_discover_skill_bundles_only_includes_directories_with_skill_md_in_stable_order(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    _write_skill_bundle(
        skills_root,
        "weather",
        "# Weather\n\nLive weather and short forecasts.\n",
    )
    _write_skill_bundle(
        skills_root,
        "alpha",
        "# Alpha\n\nAlpha summary.\n",
    )
    (skills_root / "__pycache__").mkdir()
    (skills_root / "__pycache__" / "SKILL.md").write_text(
        "# Cached bundle\n",
        encoding="utf-8",
    )
    (skills_root / "sticky_notes").mkdir()
    (skills_root / "sticky_notes" / "notes.txt").write_text(
        "not a skill\n",
        encoding="utf-8",
    )
    (skills_root / "runtime.py").write_text("# ignore me\n", encoding="utf-8")

    discovered_bundles = discover_skill_bundles(skills_root=skills_root)

    assert tuple(bundle.skill_id for bundle in discovered_bundles) == ("alpha", "weather")


def test_load_skill_bundle_reads_display_name_summary_tool_hints_and_raw_content(
    tmp_path: Path,
) -> None:
    skill_md_content = (
        "# Weather\n\n"
        "Live weather and short forecasts.\n\n"
        "Tool hints: Prefer `get_weather`; use `search_web` only as fallback.\n\n"
        "This bundle stays lightweight in Phase 1.\n"
    )
    bundle_dir = _write_skill_bundle(tmp_path, "weather", skill_md_content)

    bundle = load_skill_bundle(bundle_dir)

    assert bundle.skill_id == "weather"
    assert bundle.display_name == "Weather"
    assert bundle.summary == "Live weather and short forecasts."
    assert bundle.tool_hints == (
        "Prefer get_weather; use search_web only as fallback.",
    )
    assert bundle.load_raw_content() == skill_md_content


def test_load_active_skill_bundles_deduplicates_repeated_skill_ids(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    _write_skill_bundle(
        skills_root,
        "weather",
        "# Weather\n\nLive weather and short forecasts.\n",
    )

    active_bundles = load_active_skill_bundles(
        enabled_skill_ids=("weather", "weather"),
        skills_root=skills_root,
    )

    assert tuple(bundle.skill_id for bundle in active_bundles) == ("weather",)


def test_load_active_skill_bundles_fail_clearly_on_unknown_enabled_skill_ids(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    _write_skill_bundle(
        skills_root,
        "weather",
        "# Weather\n\nLive weather and short forecasts.\n",
    )

    with pytest.raises(UnknownSkillIdError, match="ghost.skill"):
        load_active_skill_bundles(
            enabled_skill_ids=("ghost.skill",),
            skills_root=skills_root,
        )


def test_build_active_skill_catalog_renders_compact_catalog_for_active_skills(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    _write_skill_bundle(
        skills_root,
        "weather",
        (
            "# Weather\n\n"
            "Live weather and short forecasts.\n\n"
            "Tool hints: Prefer `get_weather`; use `search_web` only as fallback for official alerts or missing details.\n"
        ),
    )

    catalog = build_active_skill_catalog(
        enabled_skill_ids=("weather",),
        skills_root=skills_root,
    )

    assert catalog == "\n".join(
        (
            "Active optional skills:",
            (
                "- weather: Live weather and short forecasts. "
                "Prefer get_weather; use search_web only as fallback for official alerts or missing details."
            ),
        )
    )


def _write_skill_bundle(skills_root: Path, skill_id: str, skill_md_content: str) -> Path:
    bundle_dir = skills_root / skill_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")
    return bundle_dir
