from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from unclaw.channels.telegram_config import load_telegram_config
from unclaw.errors import ConfigurationError
from unclaw.llm.model_profiles import resolve_model_profile
from unclaw.settings import load_settings

pytestmark = pytest.mark.unit


def test_load_settings_errors_when_app_config_file_is_missing(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    app_config_path = project_root / "config" / "app.yaml"
    app_config_path.unlink()

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert str(exc_info.value) == f"Missing configuration file: {app_config_path}"


def test_load_settings_errors_when_models_yaml_is_malformed(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    models_config_path = project_root / "config" / "models.yaml"
    models_config_path.write_text("profiles: [\n", encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert str(exc_info.value) == (
        f"Invalid YAML in configuration file: {models_config_path}"
    )


def test_load_settings_errors_when_default_profile_key_is_missing(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["models"].pop("default_profile")
    _write_yaml(app_config_path, app_payload)

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert (
        str(exc_info.value)
        == "Configuration key 'default_profile' must be a non-empty string."
    )


def test_load_settings_errors_when_channel_flag_has_invalid_type(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["channels"]["telegram_enabled"] = "yes"
    _write_yaml(app_config_path, app_payload)

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert str(exc_info.value) == "Configuration key 'telegram_enabled' must be a boolean."


def test_load_settings_errors_when_logging_mode_is_invalid(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["logging"]["mode"] = "verbose"
    _write_yaml(app_config_path, app_payload)

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert (
        str(exc_info.value)
        == "Configuration key 'mode' must be one of: full, simple."
    )


def test_load_settings_errors_when_logging_retention_is_negative(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["logging"]["retention_days"] = -1
    _write_yaml(app_config_path, app_payload)

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert (
        str(exc_info.value)
        == "Configuration key 'retention_days' must be a non-negative integer."
    )


def test_load_settings_errors_when_default_profile_is_not_defined_in_models(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["models"]["default_profile"] = "ghost"
    _write_yaml(app_config_path, app_payload)
    models_config_path = project_root / "config" / "models.yaml"

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert str(exc_info.value) == (
        "Default model profile 'ghost' is not defined in "
        f"{models_config_path}."
    )


def test_resolve_model_profile_marks_shipped_deep_profile_as_native_tool_capable(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)

    profile = resolve_model_profile(settings, "deep")

    assert profile.capabilities.tool_mode == "native"
    assert profile.capabilities.supports_native_tool_calling is True
    assert profile.keep_alive == "10m"


def test_load_settings_allows_profile_keep_alive_to_be_absent(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    models_config_path = project_root / "config" / "models.yaml"
    models_payload = _read_yaml(models_config_path)
    profiles = models_payload["profiles"]
    assert isinstance(profiles, dict)
    main_profile = profiles["main"]
    assert isinstance(main_profile, dict)
    main_profile.pop("keep_alive")
    _write_yaml(models_config_path, models_payload)

    settings = load_settings(project_root=project_root)
    profile = resolve_model_profile(settings, "main")

    assert settings.models["main"].keep_alive is None
    assert profile.keep_alive is None


def test_load_telegram_config_errors_when_yaml_is_malformed(tmp_path: Path) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    telegram_config_path = project_root / "config" / "telegram.yaml"
    telegram_config_path.write_text("allowed_chat_ids: [\n", encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        load_telegram_config(settings)

    assert str(exc_info.value) == (
        f"Invalid YAML in Telegram configuration file: {telegram_config_path}"
    )


def test_load_telegram_config_errors_when_chat_ids_include_boolean(
    tmp_path: Path,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    telegram_config_path = project_root / "config" / "telegram.yaml"
    telegram_payload = _read_yaml(telegram_config_path)
    telegram_payload["allowed_chat_ids"] = [True]
    _write_yaml(telegram_config_path, telegram_payload)

    with pytest.raises(ConfigurationError) as exc_info:
        load_telegram_config(settings)

    assert (
        str(exc_info.value)
        == "Telegram chat ids must be integers, not boolean values."
    )


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root


def _read_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
