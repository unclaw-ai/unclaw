from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from unclaw.core.command_handler import CommandHandler, CommandStatus
from unclaw.settings import load_settings
from unclaw.skills.installer import catalog_base_url
from unclaw.skills.manager import SkillCommandOutcome, run_skill_command
from unclaw.skills.remote_catalog import RemoteCatalogEntry

pytestmark = pytest.mark.unit


def _catalog_entry(
    skill_id: str = "weather",
    *,
    version: str = "1.0.0",
    public_entry_files: tuple[str, ...] = ("SKILL.md",),
) -> RemoteCatalogEntry:
    return RemoteCatalogEntry(
        skill_id=skill_id,
        display_name=skill_id.title(),
        version=version,
        summary=f"{skill_id.title()} skill.",
        repo_relative_path=skill_id,
        public_entry_files=public_entry_files,
        repository_owner="unclaw-ai",
        repository_name="skills",
    )


def _mock_urlopen(url_to_content: dict[str, bytes]):
    class _FakeResponse:
        def __init__(self, content: bytes) -> None:
            self._content = content

        def read(self) -> bytes:
            return self._content

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def _open(url, timeout=None):
        del timeout
        content = url_to_content.get(url)
        if content is None:
            import urllib.error

            raise urllib.error.HTTPError(url, 404, "Not Found", MagicMock(), None)
        return _FakeResponse(content)

    return patch("unclaw.skills.installer.urllib.request.urlopen", side_effect=_open)


def _load_temp_settings(make_temp_project, **kwargs):
    project_root = make_temp_project(**kwargs)
    return project_root, load_settings(project_root=project_root)


def test_install_success_from_mocked_remote_catalog(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root, settings = _load_temp_settings(make_temp_project)
    entry = _catalog_entry(public_entry_files=("SKILL.md", "tool.py"))
    base = catalog_base_url(
        settings.catalog.url,
        repository_owner=entry.repository_owner or "",
        repository_name=entry.repository_name or "",
    )
    url_map = {
        f"{base}weather/SKILL.md": b"# Weather\n\nLive weather.\n",
        f"{base}weather/tool.py": b"VALUE = 1\n",
    }

    monkeypatch.setattr(
        "unclaw.skills.manager.fetch_remote_catalog",
        lambda url, **_: [entry],
    )

    with _mock_urlopen(url_map):
        outcome = run_skill_command(settings, action="install", skill_id="weather")

    assert outcome.ok is True
    assert "Installed 'weather'" in outcome.lines[0]
    bundle_dir = project_root / "skills" / "weather"
    assert (bundle_dir / "SKILL.md").read_text(encoding="utf-8").startswith("# Weather")
    assert (bundle_dir / "tool.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    meta = json.loads((bundle_dir / "_meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == "1.0.0"


def test_install_already_installed_same_version_is_noop(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root, settings = _load_temp_settings(make_temp_project)
    entry = _catalog_entry()
    base = catalog_base_url(
        settings.catalog.url,
        repository_owner=entry.repository_owner or "",
        repository_name=entry.repository_name or "",
    )
    url_map = {f"{base}weather/SKILL.md": b"# Weather\n\nLive weather.\n"}

    monkeypatch.setattr(
        "unclaw.skills.manager.fetch_remote_catalog",
        lambda url, **_: [entry],
    )

    with _mock_urlopen(url_map):
        first = run_skill_command(settings, action="install", skill_id="weather")
    assert first.ok is True

    marker_path = project_root / "skills" / "weather" / "local-note.txt"
    marker_path.write_text("keep me\n", encoding="utf-8")

    settings = load_settings(project_root=project_root)
    with _mock_urlopen(url_map):
        second = run_skill_command(settings, action="install", skill_id="weather")

    assert second.ok is True
    assert "already installed" in second.lines[0]
    assert marker_path.read_text(encoding="utf-8") == "keep me\n"


def test_enable_installed_skill(make_temp_project) -> None:
    project_root, settings = _load_temp_settings(
        make_temp_project,
        install_skill_bundles={"weather": "# Weather\n\nSkill.\n"},
    )

    outcome = run_skill_command(settings, action="enable", skill_id="weather")

    assert outcome.ok is True
    assert outcome.updated_settings is not None
    assert outcome.updated_settings.skills.enabled_skill_ids == ("weather",)
    assert load_settings(project_root=project_root).skills.enabled_skill_ids == ("weather",)


def test_enable_missing_skill_fails_clearly(make_temp_project) -> None:
    _, settings = _load_temp_settings(make_temp_project)

    outcome = run_skill_command(settings, action="enable", skill_id="weather")

    assert outcome.ok is False
    assert "not installed" in outcome.lines[0]


def test_disable_enabled_skill(make_temp_project) -> None:
    project_root, settings = _load_temp_settings(
        make_temp_project,
        enabled_skill_ids=["weather"],
        install_skill_bundles={"weather": "# Weather\n\nSkill.\n"},
    )

    outcome = run_skill_command(settings, action="disable", skill_id="weather")

    assert outcome.ok is True
    assert outcome.updated_settings is not None
    assert outcome.updated_settings.skills.enabled_skill_ids == ()
    assert load_settings(project_root=project_root).skills.enabled_skill_ids == ()


def test_remove_installed_skill(make_temp_project) -> None:
    project_root, settings = _load_temp_settings(
        make_temp_project,
        install_skill_bundles={"weather": "# Weather\n\nSkill.\n"},
    )

    outcome = run_skill_command(settings, action="remove", skill_id="weather")

    assert outcome.ok is True
    assert not (project_root / "skills" / "weather").exists()


def test_remove_enabled_skill_auto_disables_first(make_temp_project) -> None:
    project_root, settings = _load_temp_settings(
        make_temp_project,
        enabled_skill_ids=["weather"],
        install_skill_bundles={"weather": "# Weather\n\nSkill.\n"},
    )

    outcome = run_skill_command(settings, action="remove", skill_id="weather")

    assert outcome.ok is True
    assert outcome.updated_settings is not None
    assert outcome.refresh_runtime is True
    assert outcome.lines[0] == "Disabled 'weather'."
    assert outcome.lines[1] == "Removed './skills/weather'."
    assert not (project_root / "skills" / "weather").exists()
    assert load_settings(project_root=project_root).skills.enabled_skill_ids == ()


def test_update_when_newer_catalog_version_exists_preserves_enabled_state(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root, settings = _load_temp_settings(
        make_temp_project,
        enabled_skill_ids=["weather"],
        install_skill_bundles={"weather": "# Weather\n\nOld weather.\n"},
    )
    bundle_dir = project_root / "skills" / "weather"
    (bundle_dir / "_meta.json").write_text('{"version": "0.9.0"}\n', encoding="utf-8")

    entry = _catalog_entry(version="1.0.0")
    base = catalog_base_url(
        settings.catalog.url,
        repository_owner=entry.repository_owner or "",
        repository_name=entry.repository_name or "",
    )
    url_map = {f"{base}weather/SKILL.md": b"# Weather\n\nNew weather.\n"}

    monkeypatch.setattr(
        "unclaw.skills.manager.fetch_remote_catalog",
        lambda url, **_: [entry],
    )

    with _mock_urlopen(url_map):
        outcome = run_skill_command(settings, action="update", skill_id="weather")

    assert outcome.ok is True
    assert outcome.refresh_runtime is True
    assert (bundle_dir / "SKILL.md").read_text(encoding="utf-8") == "# Weather\n\nNew weather.\n"
    meta = json.loads((bundle_dir / "_meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == "1.0.0"
    assert load_settings(project_root=project_root).skills.enabled_skill_ids == ("weather",)


def test_update_when_already_current_is_noop(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root, settings = _load_temp_settings(
        make_temp_project,
        install_skill_bundles={"weather": "# Weather\n\nLive weather.\n"},
    )
    bundle_dir = project_root / "skills" / "weather"
    (bundle_dir / "_meta.json").write_text('{"version": "1.0.0"}\n', encoding="utf-8")
    marker_path = bundle_dir / "local-note.txt"
    marker_path.write_text("keep me\n", encoding="utf-8")

    monkeypatch.setattr(
        "unclaw.skills.manager.fetch_remote_catalog",
        lambda url, **_: [_catalog_entry(version="1.0.0")],
    )

    outcome = run_skill_command(settings, action="update", skill_id="weather")

    assert outcome.ok is True
    assert outcome.lines == ("Skill 'weather' is already up to date.",)
    assert marker_path.read_text(encoding="utf-8") == "keep me\n"


@pytest.mark.parametrize(
    ("raw_command", "expected_action", "expected_skill_id"),
    [
        ("/skills install weather", "install", "weather"),
        ("/skills enable weather", "enable", "weather"),
        ("/skills disable weather", "disable", "weather"),
        ("/skills remove weather", "remove", "weather"),
        ("/skills update weather", "update", "weather"),
    ],
)
def test_slash_skill_commands_delegate_to_shared_manager(
    monkeypatch,
    make_temp_project,
    raw_command: str,
    expected_action: str,
    expected_skill_id: str,
) -> None:
    _, settings = _load_temp_settings(make_temp_project)
    captured: list[tuple[str, str | None]] = []

    def _fake_run(settings_arg, *, action="list", skill_id=None):
        del settings_arg
        captured.append((action, skill_id))
        return SkillCommandOutcome(ok=True, lines=("ok",))

    monkeypatch.setattr("unclaw.skills.manager.run_skill_command", _fake_run)

    handler = CommandHandler(
        settings=settings,
        session_manager=SimpleNamespace(current_session_id="sess-test", settings=settings),
    )

    result = handler.handle(raw_command)

    assert result.status is CommandStatus.OK
    assert captured == [(expected_action, expected_skill_id)]

