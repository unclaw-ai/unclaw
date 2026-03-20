from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from unclaw.channels.telegram_config import load_telegram_config
from unclaw.errors import ConfigurationError
from unclaw.llm.model_profiles import resolve_model_profile, resolve_router_profile
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
) -> None:
    project_root = make_temp_project()

    settings = load_settings(project_root=project_root)

    assert settings.app.environment == "production"
    assert settings.model_pack == DEV_MODEL_PACK_NAME
    assert settings.app.logging.level == "INFO"
    assert settings.app.logging.mode == "simple"


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
) -> None:
    project_root = make_temp_project()
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
) -> None:
    project_root = make_temp_project()
    models_config_path = project_root / "config" / "models.yaml"
    _write_yaml(
        models_config_path,
        {
            "pack": "sweet",
            "profiles": {},
        },
    )

    settings = load_settings(project_root=project_root)
    codex_profile = resolve_model_profile(settings, "codex")

    assert settings.model_pack == "sweet"
    assert settings.models["fast"].model_name == "ministral-3:3b"
    assert settings.models["main"].model_name == "qwen3.5:9b"
    assert settings.models["deep"].model_name == "qwen3.5:14b"
    assert settings.models["codex"].model_name == "qwen2.5-codex:7b"
    assert codex_profile.capabilities.supports_native_tool_calling is False
    assert codex_profile.capabilities.tool_mode == "none"


def test_load_settings_defaults_to_dev_pack_when_pack_key_is_missing(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    models_config_path = project_root / "config" / "models.yaml"
    models_payload = _read_yaml(models_config_path)
    models_payload.pop("pack")
    _write_yaml(models_config_path, models_payload)

    settings = load_settings(project_root=project_root)

    assert settings.model_pack == DEV_MODEL_PACK_NAME
    assert settings.models["main"].model_name == "qwen3.5:4b"


def test_resolve_model_profile_marks_shipped_deep_profile_as_native_tool_capable(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    profile = resolve_model_profile(settings, "deep")

    assert settings.models["deep"].num_ctx == 8192
    assert profile.capabilities.tool_mode == "native"
    assert profile.capabilities.supports_native_tool_calling is True
    assert profile.num_ctx == 8192
    assert profile.keep_alive == "10m"


def test_load_settings_reads_dedicated_router_defaults(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    router_profile = resolve_router_profile(settings)

    assert settings.models["fast"].tool_mode == "none"
    assert settings.models["main"].planner_profile is None
    assert settings.models["deep"].planner_profile is None
    assert settings.models["codex"].planner_profile is None
    assert settings.models["codex"].tool_mode == "none"
    assert settings.router.enabled is True
    assert settings.router.model_name == "qwen3:1.7b"
    assert settings.router.timeout_seconds == 15.0
    assert router_profile.name == "router"
    assert router_profile.model_name == "qwen3:1.7b"


def test_recommend_model_pack_uses_ram_thresholds() -> None:
    assert recommend_model_pack(8.0) == "lite"
    assert recommend_model_pack(16.0) == "lite"
    assert recommend_model_pack(16.1) == "sweet"
    assert recommend_model_pack(32.0) == "sweet"
    assert recommend_model_pack(48.0) == "power"
    assert recommend_model_pack(None) == "lite"


def test_load_settings_allows_profile_keep_alive_to_be_absent(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
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
