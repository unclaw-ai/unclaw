from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from unclaw.channels.telegram_config import load_telegram_config
from unclaw.errors import ConfigurationError
from unclaw.llm.model_profiles import resolve_model_profile
from unclaw.model_packs import DEV_MODEL_PACK_NAME, recommend_model_pack
from unclaw.settings import load_settings

pytestmark = pytest.mark.unit


def test_load_settings_errors_when_app_config_file_is_missing(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    app_config_path = project_root / "config" / "app.yaml"
    app_config_path.unlink()

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert str(exc_info.value) == f"Missing configuration file: {app_config_path}"


def test_load_settings_errors_when_models_yaml_is_malformed(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    models_config_path = project_root / "config" / "models.yaml"
    models_config_path.write_text("profiles: [\n", encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert str(exc_info.value) == (
        f"Invalid YAML in configuration file: {models_config_path}"
    )


def test_load_settings_errors_when_default_profile_key_is_missing(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
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
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["channels"]["telegram_enabled"] = "yes"
    _write_yaml(app_config_path, app_payload)

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert str(exc_info.value) == "Configuration key 'telegram_enabled' must be a boolean."


def test_load_settings_errors_when_logging_mode_is_invalid(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
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
    make_temp_project,
) -> None:
    project_root = make_temp_project()
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


def test_load_settings_reads_public_facing_shipped_defaults(
    make_temp_project,
    write_models_config,
) -> None:
    project_root = make_temp_project()
    write_models_config(project_root, active_pack=DEV_MODEL_PACK_NAME)

    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)

    app_section = app_payload["app"]
    logging_section = app_payload["logging"]

    assert isinstance(app_section, dict)
    assert isinstance(logging_section, dict)

    settings = load_settings(project_root=project_root)

    assert settings.app.environment == app_section["environment"]
    assert settings.model_pack == DEV_MODEL_PACK_NAME
    assert settings.app.logging.level == logging_section["level"]
    assert settings.app.logging.mode == logging_section["mode"]


def test_load_settings_preserves_explicit_verbose_logging_overrides(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["app"]["environment"] = "development"
    app_payload["logging"]["level"] = "DEBUG"
    app_payload["logging"]["mode"] = "full"
    _write_yaml(app_config_path, app_payload)

    settings = load_settings(project_root=project_root)

    assert settings.app.environment == "development"
    assert settings.app.logging.level == "DEBUG"
    assert settings.app.logging.mode == "full"


def test_load_settings_reads_runtime_guardrail_config(
    make_temp_project,
) -> None:
    project_root = make_temp_project()

    settings = load_settings(project_root=project_root)

    assert settings.app.runtime.tool_timeout_seconds == 15.0
    assert settings.app.runtime.max_tool_calls_per_turn == 8


def test_load_settings_reads_enabled_skill_ids_from_config(
    make_temp_project,
) -> None:
    project_root = make_temp_project()

    settings = load_settings(project_root=project_root)

    assert settings.skills.enabled_skill_ids == ("weather",)


def test_load_settings_accepts_file_first_enabled_skill_ids_from_config(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["skills"]["enabled_skill_ids"] = ["weather"]
    _write_yaml(app_config_path, app_payload)

    settings = load_settings(project_root=project_root)

    assert settings.skills.enabled_skill_ids == ("weather",)


def test_load_settings_errors_when_enabled_skill_id_is_unknown(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["skills"]["enabled_skill_ids"] = ["ghost.skill"]
    _write_yaml(app_config_path, app_payload)

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert (
        str(exc_info.value)
        == "Configuration key 'skills.enabled_skill_ids' contains unknown skill id(s): ghost.skill."
    )


def test_load_settings_errors_when_runtime_tool_budget_is_negative(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["runtime"]["max_tool_calls_per_turn"] = -1
    _write_yaml(app_config_path, app_payload)

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert (
        str(exc_info.value)
        == "Configuration key 'max_tool_calls_per_turn' must be a non-negative integer."
    )


def test_load_settings_errors_when_default_profile_is_not_defined_in_models(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
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


def test_resolve_model_profile_marks_shipped_default_main_profile_as_native_tool_capable(
    make_temp_project,
    write_models_config,
) -> None:
    project_root = make_temp_project()
    write_models_config(project_root, active_pack=DEV_MODEL_PACK_NAME)
    settings = load_settings(project_root=project_root)

    assert settings.app.default_model_profile == "main"

    profile = resolve_model_profile(settings, settings.app.default_model_profile)

    assert settings.models["main"].tool_mode == "native"
    assert settings.models["main"].num_ctx == 8192
    assert profile.name == "main"
    assert profile.capabilities.tool_mode == "native"
    assert profile.capabilities.supports_native_tool_calling is True
    assert profile.num_ctx == 8192
    assert profile.keep_alive == "30m"


def test_load_settings_resolves_selected_sweet_pack_profiles(
    make_temp_project,
    pack_profiles,
    write_models_config,
) -> None:
    project_root = make_temp_project()
    manual_dev_profiles = _manual_dev_profiles_payload()
    write_models_config(
        project_root,
        active_pack="sweet",
        dev_profiles=manual_dev_profiles,
    )

    settings = load_settings(project_root=project_root)
    codex_profile = resolve_model_profile(settings, "codex")
    sweet_profiles = pack_profiles("sweet")

    assert settings.model_pack == "sweet"
    assert settings.dev_profiles["main"].model_name == "fixture-main:4b"
    assert settings.models["fast"].model_name == sweet_profiles["fast"].model_name
    assert settings.models["main"].model_name == sweet_profiles["main"].model_name
    assert settings.models["deep"].model_name == sweet_profiles["deep"].model_name
    assert settings.models["codex"].model_name == sweet_profiles["codex"].model_name
    assert codex_profile.capabilities.supports_native_tool_calling is False
    assert codex_profile.capabilities.tool_mode == "none"


def test_load_settings_supports_legacy_dev_pack_shape(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    models_config_path = project_root / "config" / "models.yaml"
    _write_yaml(
        models_config_path,
        {
            "pack": "dev",
            "profiles": {
                "fast": {
                    "provider": "ollama",
                    "model_name": "legacy-fast:1b",
                    "temperature": 0.2,
                    "thinking_supported": False,
                    "tool_mode": "none",
                    "num_ctx": 4096,
                    "keep_alive": "10m",
                },
                "main": {
                    "provider": "ollama",
                    "model_name": "legacy-main:4b",
                    "temperature": 0.3,
                    "thinking_supported": True,
                    "tool_mode": "native",
                    "num_ctx": 8192,
                    "keep_alive": "30m",
                },
                "deep": {
                    "provider": "ollama",
                    "model_name": "legacy-deep:9b",
                    "temperature": 0.2,
                    "thinking_supported": True,
                    "tool_mode": "native",
                    "num_ctx": 8192,
                    "keep_alive": "10m",
                },
                "codex": {
                    "provider": "ollama",
                    "model_name": "legacy-codex:7b",
                    "temperature": 0.1,
                    "thinking_supported": True,
                    "tool_mode": "none",
                    "num_ctx": 4096,
                    "keep_alive": "10m",
                },
            },
        },
    )

    settings = load_settings(project_root=project_root)

    assert settings.model_pack == DEV_MODEL_PACK_NAME
    assert settings.models["main"].model_name == "legacy-main:4b"
    assert settings.dev_profiles["codex"].model_name == "legacy-codex:7b"


def test_load_settings_preserves_legacy_profiles_as_dev_profiles_for_fixed_packs(
    make_temp_project,
    pack_profiles,
    write_models_config,
) -> None:
    project_root = make_temp_project()
    manual_dev_profiles = _manual_dev_profiles_payload()
    write_models_config(
        project_root,
        active_pack=DEV_MODEL_PACK_NAME,
        dev_profiles=manual_dev_profiles,
    )
    models_config_path = project_root / "config" / "models.yaml"
    legacy_payload = _read_yaml(models_config_path)
    dev_profiles = legacy_payload.pop("dev_profiles")
    assert isinstance(dev_profiles, dict)
    legacy_payload["pack"] = "sweet"
    legacy_payload["profiles"] = dev_profiles
    legacy_payload.pop("active_pack")
    _write_yaml(models_config_path, legacy_payload)

    settings = load_settings(project_root=project_root)
    sweet_profiles = pack_profiles("sweet")

    assert settings.model_pack == "sweet"
    assert settings.models["main"].model_name == sweet_profiles["main"].model_name
    assert settings.dev_profiles["main"].model_name == "fixture-main:4b"


def test_load_settings_errors_when_active_dev_pack_has_no_dev_profiles(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    models_config_path = project_root / "config" / "models.yaml"
    _write_yaml(
        models_config_path,
        {
            "active_pack": "dev",
            "dev_profiles": {},
        },
    )

    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(project_root=project_root)

    assert "Active model pack 'dev' requires at least one profile" in str(
        exc_info.value
    )


def test_resolve_model_profile_marks_shipped_deep_profile_as_native_tool_capable(
    make_temp_project,
    write_models_config,
) -> None:
    project_root = make_temp_project()
    write_models_config(project_root, active_pack=DEV_MODEL_PACK_NAME)
    settings = load_settings(project_root=project_root)

    profile = resolve_model_profile(settings, "deep")

    assert settings.models["deep"].num_ctx == 8192
    assert profile.capabilities.tool_mode == "native"
    assert profile.capabilities.supports_native_tool_calling is True
    assert profile.num_ctx == 8192
    assert profile.keep_alive == "10m"


def test_recommend_model_pack_uses_ram_thresholds() -> None:
    assert recommend_model_pack(8.0) == "lite"
    assert recommend_model_pack(16.0) == "lite"
    assert recommend_model_pack(16.1) == "sweet"
    assert recommend_model_pack(32.0) == "sweet"
    assert recommend_model_pack(48.0) == "power"
    assert recommend_model_pack(None) == "lite"


def test_load_settings_allows_profile_keep_alive_to_be_absent(
    make_temp_project,
    write_models_config,
) -> None:
    project_root = make_temp_project()
    write_models_config(project_root, active_pack=DEV_MODEL_PACK_NAME)
    models_config_path = project_root / "config" / "models.yaml"
    models_payload = _read_yaml(models_config_path)
    profiles = models_payload["dev_profiles"]
    assert isinstance(profiles, dict)
    main_profile = profiles["main"]
    assert isinstance(main_profile, dict)
    main_profile.pop("keep_alive")
    _write_yaml(models_config_path, models_payload)

    settings = load_settings(project_root=project_root)
    profile = resolve_model_profile(settings, "main")

    assert settings.models["main"].keep_alive is None
    assert profile.keep_alive is None


def test_load_telegram_config_errors_when_yaml_is_malformed(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    telegram_config_path = project_root / "config" / "telegram.yaml"
    telegram_config_path.write_text("allowed_chat_ids: [\n", encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        load_telegram_config(settings)

    assert str(exc_info.value) == (
        f"Invalid YAML in Telegram configuration file: {telegram_config_path}"
    )


def test_load_telegram_config_errors_when_chat_ids_include_boolean(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
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

def _read_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _manual_dev_profiles_payload() -> dict[str, dict[str, object]]:
    return {
        "fast": {
            "provider": "ollama",
            "model_name": "fixture-fast:1b",
            "temperature": 0.2,
            "thinking_supported": False,
            "tool_mode": "none",
            "num_ctx": 4096,
            "keep_alive": "10m",
        },
        "main": {
            "provider": "ollama",
            "model_name": "fixture-main:4b",
            "temperature": 0.3,
            "thinking_supported": True,
            "tool_mode": "native",
            "num_ctx": 8192,
            "keep_alive": "30m",
        },
        "deep": {
            "provider": "ollama",
            "model_name": "fixture-deep:9b",
            "temperature": 0.2,
            "thinking_supported": True,
            "tool_mode": "native",
            "num_ctx": 8192,
            "keep_alive": "10m",
        },
        "codex": {
            "provider": "ollama",
            "model_name": "fixture-codex:7b",
            "temperature": 0.1,
            "thinking_supported": True,
            "tool_mode": "none",
            "num_ctx": 4096,
            "keep_alive": "10m",
        },
    }
