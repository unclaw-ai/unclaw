"""Tests for the catalog-only skill status model.

Covers:
- SkillStatus enum values (no "bundled" status)
- build_skill_status_report correctly computes status for all combinations
- render_skills_report output matches the new model
- /skills command output via command_handler
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.skills.remote_catalog import (
    RemoteCatalogEntry,
    SkillStatus,
    SkillStatusEntry,
    build_skill_status_report,
    render_skills_report,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# SkillStatus enum — no "bundled" status
# ---------------------------------------------------------------------------


def test_skill_status_enum_has_no_bundled_value() -> None:
    names = {s.name for s in SkillStatus}
    assert "BUNDLED" not in names
    assert "LOCAL_ONLY" not in names  # renamed to LOCAL_UNTRACKED


def test_skill_status_enum_has_required_values() -> None:
    values = {s.value for s in SkillStatus}
    assert "installed" in values
    assert "available" in values
    assert "update" in values
    assert "local_untracked" in values


# ---------------------------------------------------------------------------
# build_skill_status_report — status computation
# ---------------------------------------------------------------------------

def _make_bundle(skill_id: str, *, bundle_dir: Path) -> object:
    """Create a minimal SkillBundle-like stub via file_loader for testing."""
    from unclaw.skills.file_loader import load_skill_bundle
    skill_md = bundle_dir / skill_id / "SKILL.md"
    skill_md.parent.mkdir(parents=True, exist_ok=True)
    skill_md.write_text(f"# {skill_id.title()}\n\nSkill.\n", encoding="utf-8")
    return load_skill_bundle(bundle_dir / skill_id)


def test_skill_in_catalog_but_not_installed_is_available(tmp_path: Path) -> None:
    catalog = [
        RemoteCatalogEntry(
            skill_id="weather",
            display_name="Weather",
            version="1.0",
            summary="Weather skill.",
            repo_relative_path="weather",
        )
    ]
    report = build_skill_status_report(
        local_bundles=[],
        enabled_skill_ids=[],
        catalog_entries=catalog,
    )
    assert len(report) == 1
    assert report[0].status is SkillStatus.AVAILABLE
    assert not report[0].installed_locally
    assert not report[0].enabled_locally


def test_skill_installed_and_in_catalog_is_installed(tmp_path: Path) -> None:
    bundle = _make_bundle("weather", bundle_dir=tmp_path)
    catalog = [
        RemoteCatalogEntry(
            skill_id="weather",
            display_name="Weather",
            version="1.0",
            summary="Weather skill.",
            repo_relative_path="weather",
        )
    ]
    report = build_skill_status_report(
        local_bundles=[bundle],  # type: ignore[list-item]
        enabled_skill_ids=["weather"],
        catalog_entries=catalog,
    )
    assert report[0].status is SkillStatus.INSTALLED
    assert report[0].installed_locally
    assert report[0].enabled_locally


def test_skill_installed_with_different_version_is_update(tmp_path: Path) -> None:
    # Write _meta.json with local version
    bundle = _make_bundle("weather", bundle_dir=tmp_path)
    (tmp_path / "weather" / "_meta.json").write_text(
        '{"version": "0.9.0"}', encoding="utf-8"
    )
    catalog = [
        RemoteCatalogEntry(
            skill_id="weather",
            display_name="Weather",
            version="1.0.0",
            summary="Weather skill.",
            repo_relative_path="weather",
        )
    ]
    report = build_skill_status_report(
        local_bundles=[bundle],  # type: ignore[list-item]
        enabled_skill_ids=[],
        catalog_entries=catalog,
    )
    assert report[0].status is SkillStatus.UPDATE
    assert report[0].local_version == "0.9.0"
    assert report[0].catalog_version == "1.0.0"


def test_skill_installed_but_not_in_catalog_is_local_untracked(tmp_path: Path) -> None:
    bundle = _make_bundle("custom_skill", bundle_dir=tmp_path)
    report = build_skill_status_report(
        local_bundles=[bundle],  # type: ignore[list-item]
        enabled_skill_ids=[],
        catalog_entries=[],
    )
    assert report[0].status is SkillStatus.LOCAL_UNTRACKED
    assert report[0].installed_locally


def test_report_is_sorted_alphabetically(tmp_path: Path) -> None:
    catalog = [
        RemoteCatalogEntry(
            skill_id="zebra", display_name="Zebra", version=None,
            summary=None, repo_relative_path="zebra",
        ),
        RemoteCatalogEntry(
            skill_id="alpha", display_name="Alpha", version=None,
            summary=None, repo_relative_path="alpha",
        ),
    ]
    report = build_skill_status_report(
        local_bundles=[], enabled_skill_ids=[], catalog_entries=catalog,
    )
    assert [e.skill_id for e in report] == ["alpha", "zebra"]


# ---------------------------------------------------------------------------
# render_skills_report
# ---------------------------------------------------------------------------


def _make_status_entry(
    skill_id: str,
    *,
    status: SkillStatus,
    enabled: bool = False,
    local_version: str | None = None,
    catalog_version: str | None = None,
) -> SkillStatusEntry:
    return SkillStatusEntry(
        skill_id=skill_id,
        display_name=skill_id.title(),
        installed_locally=status in (SkillStatus.INSTALLED, SkillStatus.UPDATE, SkillStatus.LOCAL_UNTRACKED),
        enabled_locally=enabled,
        available_in_catalog=status in (SkillStatus.INSTALLED, SkillStatus.AVAILABLE, SkillStatus.UPDATE),
        local_version=local_version,
        catalog_version=catalog_version,
        status=status,
        repo_relative_path=None,
    )


def test_render_report_installed_section() -> None:
    entries = [_make_status_entry("weather", status=SkillStatus.INSTALLED, enabled=True)]
    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")

    installed_lines = [l for l in lines if "weather" in l.lower() and "installed" not in l.lower()]
    assert any("weather" in l for l in lines)
    assert any("[enabled]" in l for l in lines)
    assert any("Installed (1)" in l for l in lines)


def test_render_report_available_section() -> None:
    entries = [_make_status_entry("weather", status=SkillStatus.AVAILABLE)]
    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")

    assert any("Available (1)" in l for l in lines)
    assert any("weather" in l for l in lines)


def test_render_report_update_section() -> None:
    entries = [
        _make_status_entry(
            "weather",
            status=SkillStatus.UPDATE,
            local_version="0.9",
            catalog_version="1.0",
        )
    ]
    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")

    assert any("Update (1)" in l for l in lines)
    assert any("0.9" in l for l in lines)
    assert any("1.0" in l for l in lines)


def test_render_report_local_untracked_shows_untracked_label() -> None:
    entries = [_make_status_entry("custom", status=SkillStatus.LOCAL_UNTRACKED)]
    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")

    assert any("[untracked]" in l for l in lines)
    # No "bundled" label anywhere
    assert not any("bundled" in l.lower() for l in lines)


def test_render_report_no_bundled_wording_anywhere() -> None:
    """The rendered report must not contain the word 'bundled'."""
    entries = [
        _make_status_entry("a", status=SkillStatus.INSTALLED, enabled=True),
        _make_status_entry("b", status=SkillStatus.AVAILABLE),
        _make_status_entry("c", status=SkillStatus.UPDATE, local_version="1", catalog_version="2"),
        _make_status_entry("d", status=SkillStatus.LOCAL_UNTRACKED),
    ]
    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")
    full_text = "\n".join(lines).lower()
    assert "bundled" not in full_text


def test_render_report_shows_catalog_url() -> None:
    lines = render_skills_report([], catalog_url="https://example.com/catalog.json")
    assert any("https://example.com/catalog.json" in l for l in lines)


def test_render_report_empty_sections_show_none() -> None:
    lines = render_skills_report([], catalog_url="https://example.com/catalog.json")
    assert any("(none)" in l for l in lines)
    assert any("Installed (0)" in l for l in lines)
    assert any("Available (0)" in l for l in lines)
    assert any("Update (0)" in l for l in lines)
