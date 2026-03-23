"""Tests for the remote skills catalog reader and status computation.

Coverage
--------
- Catalog parsing from the remote-style source (object + bare list formats)
- Graceful failure when the catalog is unavailable or malformed
- ``read_local_skill_version`` with and without _meta.json
- Status computation via ``_compute_status`` (all five paths)
- Full ``build_skill_status_report`` integration
- ``render_skills_report`` output structure
- ``unclaw skills`` CLI entry point (success + failure + catalog unavailable)
- ``/skills`` slash command (success + catalog fetch failure)
"""

from __future__ import annotations

import json
import urllib.error
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.core.command_handler import CommandHandler, CommandStatus
from unclaw.settings import CatalogSettings, load_settings
from unclaw.skills.file_models import SkillBundle
from unclaw.skills.manager import SkillCommandOutcome
from unclaw.skills.remote_catalog import (
    CatalogFetchError,
    RemoteCatalogEntry,
    SkillStatus,
    SkillStatusEntry,
    _compute_status,
    _parse_catalog_payload,
    build_skill_status_report,
    fetch_remote_catalog,
    read_local_skill_version,
    render_skills_report,
)

pytestmark = pytest.mark.unit


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_bundle(tmp_path: Path, skill_id: str, display_name: str = "") -> SkillBundle:
    bundle_dir = tmp_path / skill_id
    bundle_dir.mkdir(exist_ok=True)
    name = display_name or skill_id.replace("_", " ").title()
    (bundle_dir / "SKILL.md").write_text(
        f"# {name}\n\nA test skill.\n",
        encoding="utf-8",
    )
    return SkillBundle(
        skill_id=skill_id,
        bundle_dir=bundle_dir,
        skill_md_path=bundle_dir / "SKILL.md",
        display_name=name,
        summary=f"{name} skill.",
    )


def _make_catalog_entry(
    skill_id: str,
    version: str = "1.0.0",
    display_name: str = "",
) -> RemoteCatalogEntry:
    return RemoteCatalogEntry(
        skill_id=skill_id,
        display_name=display_name or skill_id.replace("_", " ").title(),
        version=version,
        summary=None,
        repo_relative_path=f"skills/{skill_id}",
    )


def _make_status_entry(
    skill_id: str,
    *,
    status: SkillStatus,
    installed_locally: bool = True,
    enabled_locally: bool = False,
    available_in_catalog: bool = True,
    local_version: str | None = None,
    catalog_version: str | None = "1.0.0",
) -> SkillStatusEntry:
    return SkillStatusEntry(
        skill_id=skill_id,
        display_name=skill_id.replace("_", " ").title(),
        installed_locally=installed_locally,
        enabled_locally=enabled_locally,
        available_in_catalog=available_in_catalog,
        local_version=local_version,
        catalog_version=catalog_version,
        status=status,
        repo_relative_path=f"skills/{skill_id}",
    )


def _fake_urlopen(payload: object):
    """Return a urlopen-compatible context manager that yields the given payload."""
    raw = json.dumps(payload).encode("utf-8")

    class _Response:
        def read(self):
            return raw

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    return lambda url, timeout: _Response()


def _load_repo_settings():
    return load_settings(project_root=Path(__file__).resolve().parents[1])


def _build_session_manager() -> SimpleNamespace:
    return SimpleNamespace(current_session_id="sess-test")


# ── fetch_remote_catalog ──────────────────────────────────────────────────────


def test_fetch_parses_skills_object_format(monkeypatch) -> None:
    payload = {
        "skills": [
            {
                "skill_id": "weather",
                "display_name": "Weather",
                "version": "0.1.0",
                "summary": "Live weather.",
                "repo_relative_path": "skills/weather",
            }
        ]
    }
    monkeypatch.setattr(
        "unclaw.skills.remote_catalog.urllib.request.urlopen",
        _fake_urlopen(payload),
    )

    entries = fetch_remote_catalog("https://example.com/catalog.json")

    assert len(entries) == 1
    e = entries[0]
    assert e.skill_id == "weather"
    assert e.version == "0.1.0"
    assert e.display_name == "Weather"
    assert e.summary == "Live weather."
    assert e.repo_relative_path == "skills/weather"


def test_fetch_parses_bare_list_format(monkeypatch) -> None:
    payload = [{"skill_id": "git_repo", "display_name": "Git Repo", "version": "0.2.0"}]
    monkeypatch.setattr(
        "unclaw.skills.remote_catalog.urllib.request.urlopen",
        _fake_urlopen(payload),
    )

    entries = fetch_remote_catalog("https://example.com/catalog.json")

    assert len(entries) == 1
    assert entries[0].skill_id == "git_repo"
    assert entries[0].version == "0.2.0"


def test_fetch_raises_on_network_error(monkeypatch) -> None:
    def _raise(url, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(
        "unclaw.skills.remote_catalog.urllib.request.urlopen",
        _raise,
    )

    with pytest.raises(CatalogFetchError, match="connection refused"):
        fetch_remote_catalog("https://example.com/catalog.json")


def test_fetch_raises_on_invalid_json(monkeypatch) -> None:
    class _Bad:
        def read(self):
            return b"not json{"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(
        "unclaw.skills.remote_catalog.urllib.request.urlopen",
        lambda url, timeout: _Bad(),
    )

    with pytest.raises(CatalogFetchError, match="not valid JSON"):
        fetch_remote_catalog("https://example.com/catalog.json")


def test_fetch_raises_on_unrecognised_format(monkeypatch) -> None:
    monkeypatch.setattr(
        "unclaw.skills.remote_catalog.urllib.request.urlopen",
        _fake_urlopen({"wrong_key": "value"}),
    )

    with pytest.raises(CatalogFetchError, match="Unrecognised catalog format"):
        fetch_remote_catalog("https://example.com/catalog.json")


def test_fetch_skips_entries_without_skill_id(monkeypatch) -> None:
    payload = {"skills": [{"display_name": "No Id Skill", "version": "1.0.0"}]}
    monkeypatch.setattr(
        "unclaw.skills.remote_catalog.urllib.request.urlopen",
        _fake_urlopen(payload),
    )

    entries = fetch_remote_catalog("https://example.com/catalog.json")
    assert entries == []


def test_fetch_fills_display_name_fallback_from_skill_id(monkeypatch) -> None:
    payload = {"skills": [{"skill_id": "my_skill"}]}
    monkeypatch.setattr(
        "unclaw.skills.remote_catalog.urllib.request.urlopen",
        _fake_urlopen(payload),
    )

    entries = fetch_remote_catalog("https://example.com/catalog.json")
    assert entries[0].display_name == "my_skill"


# ── read_local_skill_version ──────────────────────────────────────────────────


def test_read_version_returns_none_when_no_meta(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "my_skill"
    bundle_dir.mkdir()
    assert read_local_skill_version(bundle_dir) is None


def test_read_version_returns_version_string(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "my_skill"
    bundle_dir.mkdir()
    (bundle_dir / "_meta.json").write_text(
        json.dumps({"skill_id": "my_skill", "version": "2.3.4"}),
        encoding="utf-8",
    )
    assert read_local_skill_version(bundle_dir) == "2.3.4"


def test_read_version_returns_none_on_malformed_json(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "my_skill"
    bundle_dir.mkdir()
    (bundle_dir / "_meta.json").write_text("not json{", encoding="utf-8")
    assert read_local_skill_version(bundle_dir) is None


def test_read_version_returns_none_when_version_key_missing(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "my_skill"
    bundle_dir.mkdir()
    (bundle_dir / "_meta.json").write_text(
        json.dumps({"skill_id": "my_skill"}),
        encoding="utf-8",
    )
    assert read_local_skill_version(bundle_dir) is None


def test_read_version_strips_whitespace(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "my_skill"
    bundle_dir.mkdir()
    (bundle_dir / "_meta.json").write_text(
        json.dumps({"version": "  1.0.0  "}),
        encoding="utf-8",
    )
    assert read_local_skill_version(bundle_dir) == "1.0.0"


# ── _compute_status ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "installed,in_catalog,local_ver,catalog_ver,expected",
    [
        (True, True, "1.0.0", "1.0.0", SkillStatus.INSTALLED),
        (True, True, "1.0.0", "1.0.0", SkillStatus.INSTALLED),
        (True, True, None, "1.0.0", SkillStatus.INSTALLED),   # no local ver → installed
        (True, True, "1.0.0", None, SkillStatus.INSTALLED),   # no catalog ver → installed
        (True, True, None, None, SkillStatus.INSTALLED),       # neither ver → installed
        (True, True, "1.0.0", "2.0.0", SkillStatus.UPDATE),   # exact mismatch → update
        (True, False, None, None, SkillStatus.LOCAL_UNTRACKED),
        (True, False, "1.0.0", None, SkillStatus.LOCAL_UNTRACKED),
        (False, True, None, "1.0.0", SkillStatus.AVAILABLE),
        (False, True, None, None, SkillStatus.AVAILABLE),
    ],
)
def test_compute_status_all_paths(
    installed, in_catalog, local_ver, catalog_ver, expected
) -> None:
    result = _compute_status(
        installed_locally=installed,
        available_in_catalog=in_catalog,
        local_version=local_ver,
        catalog_version=catalog_ver,
    )
    assert result is expected


# ── build_skill_status_report ─────────────────────────────────────────────────


def test_build_report_installed_and_enabled(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "weather")
    entry = _make_catalog_entry("weather", version="0.1.0")

    report = build_skill_status_report(
        local_bundles=[bundle],
        enabled_skill_ids=["weather"],
        catalog_entries=[entry],
    )

    assert len(report) == 1
    r = report[0]
    assert r.skill_id == "weather"
    assert r.installed_locally is True
    assert r.enabled_locally is True
    assert r.available_in_catalog is True
    assert r.status is SkillStatus.INSTALLED
    assert r.catalog_version == "0.1.0"
    assert r.local_version is None   # no _meta.json


def test_build_report_installed_not_enabled(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "weather")
    entry = _make_catalog_entry("weather")

    report = build_skill_status_report(
        local_bundles=[bundle],
        enabled_skill_ids=[],
        catalog_entries=[entry],
    )

    assert report[0].enabled_locally is False
    assert report[0].installed_locally is True


def test_build_report_available_for_catalog_only_skill(tmp_path: Path) -> None:
    entry = _make_catalog_entry("new_skill", version="1.0.0")

    report = build_skill_status_report(
        local_bundles=[],
        enabled_skill_ids=[],
        catalog_entries=[entry],
    )

    assert len(report) == 1
    r = report[0]
    assert r.installed_locally is False
    assert r.enabled_locally is False
    assert r.status is SkillStatus.AVAILABLE


def test_build_report_local_only_for_uncatalogued_skill(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "custom_skill")

    report = build_skill_status_report(
        local_bundles=[bundle],
        enabled_skill_ids=[],
        catalog_entries=[],
    )

    assert len(report) == 1
    r = report[0]
    assert r.installed_locally is True
    assert r.available_in_catalog is False
    assert r.status is SkillStatus.LOCAL_UNTRACKED


def test_build_report_update_when_versions_differ(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "weather")
    (tmp_path / "weather" / "_meta.json").write_text(
        json.dumps({"skill_id": "weather", "version": "0.1.0"}),
        encoding="utf-8",
    )
    entry = _make_catalog_entry("weather", version="0.2.0")

    report = build_skill_status_report(
        local_bundles=[bundle],
        enabled_skill_ids=[],
        catalog_entries=[entry],
    )

    assert len(report) == 1
    r = report[0]
    assert r.status is SkillStatus.UPDATE
    assert r.local_version == "0.1.0"
    assert r.catalog_version == "0.2.0"


def test_build_report_no_update_when_versions_match(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "weather")
    (tmp_path / "weather" / "_meta.json").write_text(
        json.dumps({"version": "0.1.0"}),
        encoding="utf-8",
    )
    entry = _make_catalog_entry("weather", version="0.1.0")

    report = build_skill_status_report(
        local_bundles=[bundle],
        enabled_skill_ids=[],
        catalog_entries=[entry],
    )

    assert report[0].status is SkillStatus.INSTALLED


def test_build_report_union_covers_all_skill_ids(tmp_path: Path) -> None:
    bundle_a = _make_bundle(tmp_path, "skill_a")
    entry_b = _make_catalog_entry("skill_b")

    report = build_skill_status_report(
        local_bundles=[bundle_a],
        enabled_skill_ids=[],
        catalog_entries=[entry_b],
    )

    ids = {r.skill_id for r in report}
    assert ids == {"skill_a", "skill_b"}


def test_build_report_sorted_by_skill_id(tmp_path: Path) -> None:
    bundles = [_make_bundle(tmp_path, sid) for sid in ["zebra", "alpha", "middle"]]

    report = build_skill_status_report(
        local_bundles=bundles,
        enabled_skill_ids=[],
        catalog_entries=[],
    )

    assert [r.skill_id for r in report] == ["alpha", "middle", "zebra"]


def test_build_report_prefers_catalog_display_name(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "weather", display_name="Old Name")
    entry = _make_catalog_entry("weather", display_name="Official Name")

    report = build_skill_status_report(
        local_bundles=[bundle],
        enabled_skill_ids=[],
        catalog_entries=[entry],
    )

    assert report[0].display_name == "Official Name"


def test_build_report_uses_local_display_name_when_not_in_catalog(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, "custom", display_name="My Custom Skill")

    report = build_skill_status_report(
        local_bundles=[bundle],
        enabled_skill_ids=[],
        catalog_entries=[],
    )

    assert report[0].display_name == "My Custom Skill"


# ── render_skills_report ──────────────────────────────────────────────────────


def test_render_shows_three_labeled_sections() -> None:
    entries = [
        _make_status_entry("weather", status=SkillStatus.INSTALLED, enabled_locally=True),
        _make_status_entry(
            "new_skill",
            status=SkillStatus.AVAILABLE,
            installed_locally=False,
            available_in_catalog=True,
        ),
    ]
    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")
    text = "\n".join(lines)

    assert "Installed" in text
    assert "Update" in text
    assert "Available" in text
    assert "weather" in text
    assert "new_skill" in text
    assert "https://example.com/catalog.json" in text


def test_render_shows_none_in_empty_sections() -> None:
    lines = render_skills_report([], catalog_url="https://example.com/catalog.json")
    text = "\n".join(lines)
    assert text.count("(none)") == 3


def test_render_update_section_shows_both_versions() -> None:
    entries = [
        _make_status_entry(
            "weather",
            status=SkillStatus.UPDATE,
            local_version="0.1.0",
            catalog_version="0.2.0",
        )
    ]
    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")
    text = "\n".join(lines)

    assert "Update (1)" in text
    assert "0.1.0" in text
    assert "0.2.0" in text


def test_render_enabled_tag_present_when_enabled() -> None:
    entries = [
        _make_status_entry(
            "weather",
            status=SkillStatus.INSTALLED,
            enabled_locally=True,
            local_version="0.1.0",
        )
    ]
    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")
    assert any("[enabled]" in line for line in lines)


def test_render_no_enabled_tag_when_not_enabled() -> None:
    entries = [
        _make_status_entry(
            "weather",
            status=SkillStatus.INSTALLED,
            enabled_locally=False,
        )
    ]
    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")
    assert not any("[enabled]" in line for line in lines)


def test_render_local_only_labelled() -> None:
    entries = [
        _make_status_entry(
            "custom",
            status=SkillStatus.LOCAL_UNTRACKED,
            available_in_catalog=False,
            catalog_version=None,
        )
    ]
    lines = render_skills_report(entries, catalog_url="https://example.com/catalog.json")
    text = "\n".join(lines)

    assert "[untracked]" in text
    assert "Installed (1)" in text


# ── unclaw skills CLI ─────────────────────────────────────────────────────────


def test_skills_cli_calls_output_and_returns_zero(monkeypatch) -> None:
    from unclaw.skills import cli as skills_cli

    project_root = Path(__file__).resolve().parents[1]
    collected: list[str] = []

    monkeypatch.setattr(
        "unclaw.skills.cli.run_skill_command",
        lambda settings, action="list", skill_id=None: SkillCommandOutcome(
            ok=True,
            lines=("Installed (0)", "", "Update (0)", "", "Available (1)"),
        ),
    )

    rc = skills_cli.main(project_root=project_root, output_func=collected.append)

    assert rc == 0
    combined = "\n".join(collected)
    assert "Installed" in combined
    assert "Available" in combined


def test_skills_cli_returns_one_on_catalog_fetch_error(monkeypatch) -> None:
    from unclaw.skills import cli as skills_cli

    project_root = Path(__file__).resolve().parents[1]
    collected: list[str] = []

    monkeypatch.setattr(
        "unclaw.skills.cli.run_skill_command",
        lambda settings, action="list", skill_id=None: SkillCommandOutcome(
            ok=False,
            lines=(
                "Could not fetch the skills catalog.",
                "timeout",
                "Local skill runtime is not affected.",
            ),
        ),
    )

    rc = skills_cli.main(project_root=project_root, output_func=collected.append)

    assert rc == 1
    combined = "\n".join(collected)
    assert "catalog" in combined.lower()
    assert "timeout" in combined


def test_skills_cli_error_message_does_not_mention_local_path(monkeypatch) -> None:
    """The error must not suggest falling back to any local filesystem path."""
    from unclaw.skills import cli as skills_cli

    project_root = Path(__file__).resolve().parents[1]
    collected: list[str] = []

    monkeypatch.setattr(
        "unclaw.skills.cli.run_skill_command",
        lambda settings, action="list", skill_id=None: SkillCommandOutcome(
            ok=False,
            lines=(
                "Could not fetch the skills catalog.",
                "network error",
                "Local skill runtime is not affected.",
            ),
        ),
    )

    skills_cli.main(project_root=project_root, output_func=collected.append)

    combined = "\n".join(collected).lower()
    assert "/home/" not in combined
    assert "fallback" not in combined


# ── /skills slash command ─────────────────────────────────────────────────────


def test_slash_skills_returns_ok_and_three_sections(monkeypatch) -> None:
    monkeypatch.setattr(
        "unclaw.skills.manager.run_skill_command",
        lambda settings, action="list", skill_id=None: SkillCommandOutcome(
            ok=True,
            lines=("Installed (0)", "", "Update (0)", "", "Available (1)"),
        ),
    )

    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/skills")

    assert result.status is CommandStatus.OK
    combined = "\n".join(result.lines)
    assert "Installed" in combined
    assert "Update" in combined
    assert "Available" in combined


def test_slash_skills_returns_error_on_catalog_fetch_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "unclaw.skills.manager.run_skill_command",
        lambda settings, action="list", skill_id=None: SkillCommandOutcome(
            ok=False,
            lines=(
                "Could not fetch the skills catalog.",
                "timeout",
                "Local skill runtime is not affected.",
            ),
        ),
    )

    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/skills")

    assert result.status is CommandStatus.ERROR
    combined = "\n".join(result.lines)
    assert "catalog" in combined.lower()
    assert "not affected" in combined.lower()


def test_slash_skills_usage_error_with_arguments() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )

    result = handler.handle("/skills extra-arg")

    assert result.status is CommandStatus.ERROR
    assert "Usage" in result.lines[0]


# ── /help includes /skills ────────────────────────────────────────────────────


def test_help_includes_skills_command() -> None:
    handler = CommandHandler(
        settings=_load_repo_settings(),
        session_manager=_build_session_manager(),
    )
    result = handler.handle("/help")

    assert result.status is CommandStatus.OK
    assert "/skills  Show installed, available, and updatable skills." in result.lines


# ── settings catalog defaults ─────────────────────────────────────────────────


def test_settings_catalog_url_is_official_github_raw_url() -> None:
    settings = _load_repo_settings()
    assert settings.catalog.url == (
        "https://raw.githubusercontent.com/unclaw-ai/skills/main/catalog.json"
    )


def test_settings_catalog_url_is_configurable(make_temp_project) -> None:
    import yaml

    project_root = make_temp_project()
    app_config = project_root / "config" / "app.yaml"
    payload = yaml.safe_load(app_config.read_text(encoding="utf-8"))
    payload["catalog"] = {"url": "https://example.com/my_catalog.json"}
    app_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    settings = load_settings(project_root=project_root)
    assert settings.catalog.url == "https://example.com/my_catalog.json"


def test_settings_catalog_defaults_when_section_absent(make_temp_project) -> None:
    import yaml

    project_root = make_temp_project()
    app_config = project_root / "config" / "app.yaml"
    payload = yaml.safe_load(app_config.read_text(encoding="utf-8"))
    payload.pop("catalog", None)
    app_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    settings = load_settings(project_root=project_root)
    assert "raw.githubusercontent.com" in settings.catalog.url
