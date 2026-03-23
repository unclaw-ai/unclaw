"""Tests for the catalog-backed skills hub status model."""

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


def test_skill_status_enum_has_no_bundled_value() -> None:
    names = {status.name for status in SkillStatus}
    assert "BUNDLED" not in names
    assert "LOCAL_ONLY" not in names


def test_skill_status_enum_has_required_values() -> None:
    values = {status.value for status in SkillStatus}
    assert "enabled" in values
    assert "installed" in values
    assert "available" in values
    assert "update" in values
    assert "orphaned" in values


def _make_bundle(skill_id: str, *, bundle_dir: Path) -> object:
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


def test_enabled_skill_in_catalog_is_enabled(tmp_path: Path) -> None:
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

    assert report[0].status is SkillStatus.ENABLED
    assert report[0].installed_locally
    assert report[0].enabled_locally


def test_skill_installed_with_different_version_is_update(tmp_path: Path) -> None:
    bundle = _make_bundle("weather", bundle_dir=tmp_path)
    (tmp_path / "weather" / "_meta.json").write_text(
        '{"version": "0.9.0"}',
        encoding="utf-8",
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


def test_skill_installed_but_not_in_catalog_is_orphaned(tmp_path: Path) -> None:
    bundle = _make_bundle("custom_skill", bundle_dir=tmp_path)

    report = build_skill_status_report(
        local_bundles=[bundle],  # type: ignore[list-item]
        enabled_skill_ids=[],
        catalog_entries=[],
    )

    assert report[0].status is SkillStatus.ORPHANED
    assert report[0].installed_locally


def test_report_is_sorted_alphabetically(tmp_path: Path) -> None:
    catalog = [
        RemoteCatalogEntry(
            skill_id="zebra",
            display_name="Zebra",
            version=None,
            summary=None,
            repo_relative_path="zebra",
        ),
        RemoteCatalogEntry(
            skill_id="alpha",
            display_name="Alpha",
            version=None,
            summary=None,
            repo_relative_path="alpha",
        ),
    ]

    report = build_skill_status_report(
        local_bundles=[],
        enabled_skill_ids=[],
        catalog_entries=catalog,
    )

    assert [entry.skill_id for entry in report] == ["alpha", "zebra"]


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
        installed_locally=status is not SkillStatus.AVAILABLE,
        enabled_locally=enabled,
        available_in_catalog=status is not SkillStatus.ORPHANED,
        local_version=local_version,
        catalog_version=catalog_version,
        status=status,
        repo_relative_path=None,
    )


def test_render_report_sections_cover_all_user_facing_groups() -> None:
    entries = [
        _make_status_entry("weather", status=SkillStatus.ENABLED, enabled=True),
        _make_status_entry("notes", status=SkillStatus.INSTALLED),
        _make_status_entry("calendar", status=SkillStatus.UPDATE, local_version="1", catalog_version="2"),
        _make_status_entry("maps", status=SkillStatus.AVAILABLE, catalog_version="1"),
        _make_status_entry("custom", status=SkillStatus.ORPHANED),
    ]

    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")
    text = "\n".join(lines)

    assert "Enabled (1)" in text
    assert "Installed (1)" in text
    assert "Update (1)" in text
    assert "Available (1)" in text
    assert "Orphaned (1)" in text


def test_render_report_no_bundled_wording_anywhere() -> None:
    lines = render_skills_report([], catalog_url="https://example.com/catalog.json")
    full_text = "\n".join(lines).lower()
    assert "bundled" not in full_text


def test_render_report_shows_catalog_url() -> None:
    lines = render_skills_report([], catalog_url="https://example.com/catalog.json")
    assert any("https://example.com/catalog.json" in line for line in lines)


def test_render_report_empty_sections_show_none() -> None:
    lines = render_skills_report([], catalog_url="https://example.com/catalog.json")
    assert sum(1 for line in lines if "(none)" in line) == 5
    assert any("Enabled (0)" in line for line in lines)
    assert any("Installed (0)" in line for line in lines)
    assert any("Available (0)" in line for line in lines)
    assert any("Update (0)" in line for line in lines)
    assert any("Orphaned (0)" in line for line in lines)
