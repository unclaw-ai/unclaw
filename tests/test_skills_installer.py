"""Tests for the skill bundle installer module.

Covers:
- catalog_base_url derivation
- build_file_url construction
- install_skill downloads files and writes _meta.json
- install_skill fails clearly on missing repo_relative_path
- install_skill fails clearly on empty public_entry_files
- install_skill fails clearly on HTTP errors
- installed skill is discoverable by discover_skill_bundles
- onboarding installs selected skills from mocked catalog responses
- onboarding enables installed skills in config
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from unclaw.skills.installer import (
    SkillInstallError,
    build_file_url,
    catalog_base_url,
    install_skill,
)
from unclaw.skills.remote_catalog import RemoteCatalogEntry

pytestmark = pytest.mark.unit

_CATALOG_URL = "https://raw.githubusercontent.com/unclaw-ai/skills/main/catalog.json"


# ---------------------------------------------------------------------------
# catalog_base_url
# ---------------------------------------------------------------------------


def test_catalog_base_url_strips_catalog_json() -> None:
    result = catalog_base_url(_CATALOG_URL)
    assert result == "https://raw.githubusercontent.com/unclaw-ai/skills/main/"


def test_catalog_base_url_with_trailing_path() -> None:
    url = "https://example.com/org/repo/main/subdir/catalog.json"
    result = catalog_base_url(url)
    assert result == "https://example.com/org/repo/main/subdir/"


# ---------------------------------------------------------------------------
# build_file_url
# ---------------------------------------------------------------------------


def test_build_file_url_constructs_correct_url() -> None:
    base = "https://raw.githubusercontent.com/unclaw-ai/skills/main/"
    url = build_file_url(base, "weather", "SKILL.md")
    assert url == "https://raw.githubusercontent.com/unclaw-ai/skills/main/weather/SKILL.md"


def test_build_file_url_strips_leading_slash_from_path() -> None:
    base = "https://raw.githubusercontent.com/unclaw-ai/skills/main/"
    url = build_file_url(base, "/weather/", "tool.py")
    assert url == "https://raw.githubusercontent.com/unclaw-ai/skills/main/weather/tool.py"


# ---------------------------------------------------------------------------
# install_skill — error cases
# ---------------------------------------------------------------------------


def test_install_skill_raises_when_repo_relative_path_missing(tmp_path: Path) -> None:
    entry = RemoteCatalogEntry(
        skill_id="weather",
        display_name="Weather",
        version="0.1.0",
        summary="Weather skill.",
        repo_relative_path=None,
        public_entry_files=("SKILL.md",),
    )
    with pytest.raises(SkillInstallError, match="repo_relative_path"):
        install_skill(entry, skills_root=tmp_path / "skills", catalog_url=_CATALOG_URL)


def test_install_skill_raises_when_public_entry_files_empty(tmp_path: Path) -> None:
    entry = RemoteCatalogEntry(
        skill_id="weather",
        display_name="Weather",
        version="0.1.0",
        summary="Weather skill.",
        repo_relative_path="weather",
        public_entry_files=(),
    )
    with pytest.raises(SkillInstallError, match="public_entry_files"):
        install_skill(entry, skills_root=tmp_path / "skills", catalog_url=_CATALOG_URL)


def test_install_skill_raises_on_http_error(tmp_path: Path) -> None:
    entry = RemoteCatalogEntry(
        skill_id="weather",
        display_name="Weather",
        version="0.1.0",
        summary="Weather skill.",
        repo_relative_path="weather",
        public_entry_files=("SKILL.md",),
    )

    import urllib.error

    http_error = urllib.error.HTTPError(
        url="https://example.com",
        code=404,
        msg="Not Found",
        hdrs=MagicMock(),
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(SkillInstallError, match="404"):
            install_skill(
                entry,
                skills_root=tmp_path / "skills",
                catalog_url=_CATALOG_URL,
            )


# ---------------------------------------------------------------------------
# install_skill — happy path with mocked HTTP
# ---------------------------------------------------------------------------


def _mock_urlopen(url_to_content: dict[str, bytes]):
    """Return a context-manager mock for urllib.request.urlopen."""

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
        content = url_to_content.get(url)
        if content is None:
            import urllib.error
            raise urllib.error.HTTPError(url, 404, "Not Found", MagicMock(), None)
        return _FakeResponse(content)

    return patch("urllib.request.urlopen", side_effect=_open)


def test_install_skill_downloads_files_and_writes_meta(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skill_md_content = b"# Weather\n\nLive weather.\n"
    base = catalog_base_url(_CATALOG_URL)
    url_map = {
        f"{base}weather/SKILL.md": skill_md_content,
        f"{base}weather/__init__.py": b"",
    }

    entry = RemoteCatalogEntry(
        skill_id="weather",
        display_name="Weather",
        version="0.1.0",
        summary="Live weather.",
        repo_relative_path="weather",
        public_entry_files=("SKILL.md", "__init__.py"),
    )

    with _mock_urlopen(url_map):
        install_skill(entry, skills_root=skills_root, catalog_url=_CATALOG_URL)

    bundle_dir = skills_root / "weather"
    assert (bundle_dir / "SKILL.md").read_bytes() == skill_md_content
    assert (bundle_dir / "__init__.py").exists()

    meta = json.loads((bundle_dir / "_meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == "0.1.0"


def test_install_skill_writes_meta_without_version_when_version_is_none(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    base = catalog_base_url(_CATALOG_URL)
    url_map = {f"{base}weather/SKILL.md": b"# Weather\n\nSkill.\n"}

    entry = RemoteCatalogEntry(
        skill_id="weather",
        display_name="Weather",
        version=None,
        summary="Skill.",
        repo_relative_path="weather",
        public_entry_files=("SKILL.md",),
    )

    with _mock_urlopen(url_map):
        install_skill(entry, skills_root=skills_root, catalog_url=_CATALOG_URL)

    meta = json.loads((bundle_dir := skills_root / "weather" / "_meta.json").read_text())
    assert "version" not in meta


def test_installed_skill_is_discovered_by_discover_skill_bundles(tmp_path: Path) -> None:
    """After install, the skill bundle is discoverable at runtime."""
    from unclaw.skills.file_loader import discover_skill_bundles

    skills_root = tmp_path / "skills"
    base = catalog_base_url(_CATALOG_URL)
    url_map = {f"{base}weather/SKILL.md": b"# Weather\n\nLive weather.\n"}

    entry = RemoteCatalogEntry(
        skill_id="weather",
        display_name="Weather",
        version="1.0.0",
        summary="Live weather.",
        repo_relative_path="weather",
        public_entry_files=("SKILL.md",),
    )

    with _mock_urlopen(url_map):
        install_skill(entry, skills_root=skills_root, catalog_url=_CATALOG_URL)

    bundles = discover_skill_bundles(skills_root=skills_root)
    assert len(bundles) == 1
    assert bundles[0].skill_id == "weather"
    assert bundles[0].display_name == "Weather"


# ---------------------------------------------------------------------------
# Onboarding skill install + enable — isolated from real network
# ---------------------------------------------------------------------------


def test_onboarding_installs_selected_skills_from_mocked_catalog(
    tmp_path: Path,
    make_temp_project,
) -> None:
    """_install_onboarding_skills downloads and installs catalog skills."""
    from unclaw.onboarding import _install_onboarding_skills
    from unclaw.onboarding_types import ModelProfileDraft, OnboardingPlan, PROFILE_ORDER
    from unclaw.onboarding_files import recommended_model_profiles
    from unclaw.settings import load_settings

    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    skill_md_bytes = b"# Weather\n\nLive weather.\n"
    base = catalog_base_url(settings.catalog.url)
    url_map = {f"{base}weather/SKILL.md": skill_md_bytes}

    catalog_entries = [
        RemoteCatalogEntry(
            skill_id="weather",
            display_name="Weather",
            version="0.1.0",
            summary="Live weather.",
            repo_relative_path="weather",
            public_entry_files=("SKILL.md",),
        )
    ]

    plan = OnboardingPlan(
        beginner_mode=True,
        automatic_configuration=True,
        logging_mode="simple",
        enabled_channels=("terminal",),
        enabled_skill_ids=("weather",),
        default_profile="main",
        model_pack="lite",
        model_profiles=recommended_model_profiles("lite"),
        telegram_bot_token=None,
        telegram_bot_token_env_var="TELEGRAM_BOT_TOKEN",
        telegram_allowed_chat_ids=(),
        telegram_polling_timeout_seconds=30,
    )

    output_lines: list[str] = []
    with _mock_urlopen(url_map):
        installed = _install_onboarding_skills(
            settings,
            plan=plan,
            catalog_entries=catalog_entries,
            output_func=output_lines.append,
        )

    assert "weather" in installed
    bundle_dir = settings.paths.project_root / "skills" / "weather"
    assert (bundle_dir / "SKILL.md").exists()


def test_onboarding_enables_installed_skills_in_config(
    tmp_path: Path,
    make_temp_project,
) -> None:
    """write_onboarding_files records the installed skill IDs in app.yaml."""
    import yaml
    from unclaw.onboarding_files import write_onboarding_files
    from unclaw.onboarding_types import OnboardingPlan
    from unclaw.onboarding_files import recommended_model_profiles
    from unclaw.settings import load_settings

    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    # Manually install a synthetic skill bundle so validation passes.
    skills_root = project_root / "skills"
    skills_root.mkdir(exist_ok=True)
    bundle_dir = skills_root / "weather"
    bundle_dir.mkdir()
    (bundle_dir / "SKILL.md").write_text("# Weather\n\nSkill.\n", encoding="utf-8")

    plan = OnboardingPlan(
        beginner_mode=True,
        automatic_configuration=True,
        logging_mode="simple",
        enabled_channels=("terminal",),
        enabled_skill_ids=("weather",),
        default_profile="main",
        model_pack="lite",
        model_profiles=recommended_model_profiles("lite"),
        telegram_bot_token=None,
        telegram_bot_token_env_var="TELEGRAM_BOT_TOKEN",
        telegram_allowed_chat_ids=(),
        telegram_polling_timeout_seconds=30,
    )

    write_onboarding_files(settings, plan)

    app_config_path = project_root / "config" / "app.yaml"
    payload = yaml.safe_load(app_config_path.read_text(encoding="utf-8"))
    assert "weather" in payload["skills"]["enabled_skill_ids"]


def test_onboarding_skips_install_when_catalog_unavailable(
    tmp_path: Path,
    make_temp_project,
) -> None:
    """When catalog is empty (fetch failed), _install_onboarding_skills returns ()."""
    from unclaw.onboarding import _install_onboarding_skills
    from unclaw.onboarding_types import OnboardingPlan
    from unclaw.onboarding_files import recommended_model_profiles
    from unclaw.settings import load_settings

    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    plan = OnboardingPlan(
        beginner_mode=True,
        automatic_configuration=True,
        logging_mode="simple",
        enabled_channels=("terminal",),
        enabled_skill_ids=("weather",),
        default_profile="main",
        model_pack="lite",
        model_profiles=recommended_model_profiles("lite"),
        telegram_bot_token=None,
        telegram_bot_token_env_var="TELEGRAM_BOT_TOKEN",
        telegram_allowed_chat_ids=(),
        telegram_polling_timeout_seconds=30,
    )

    installed = _install_onboarding_skills(
        settings,
        plan=plan,
        catalog_entries=[],  # catalog fetch failed → empty list
        output_func=lambda _: None,
    )

    assert installed == ()
