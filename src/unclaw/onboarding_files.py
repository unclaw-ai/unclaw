"""Onboarding file payload construction and persistence."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from unclaw.errors import ConfigurationError
from unclaw.local_secrets import (
    LOCAL_SECRETS_FILE_NAME,
    LocalSecrets,
    ensure_local_secrets_permissions,
    local_secrets_path,
    write_local_secrets,
)
from unclaw.onboarding_types import ModelProfileDraft, OnboardingPlan, PROFILE_ORDER
from unclaw.settings import Settings

_RECOMMENDED_PROFILES: dict[str, ModelProfileDraft] = {
    "fast": ModelProfileDraft(
        provider="ollama",
        model_name="llama3.2:3b",
        temperature=0.2,
        thinking_supported=False,
        tool_mode="none",
        num_ctx=4096,
        keep_alive="10m",
    ),
    "main": ModelProfileDraft(
        provider="ollama",
        model_name="qwen3.5:4b",
        temperature=0.3,
        thinking_supported=True,
        tool_mode="native",
        num_ctx=8192,
        keep_alive="30m",
    ),
    "deep": ModelProfileDraft(
        provider="ollama",
        model_name="qwen3.5:9b",
        temperature=0.2,
        thinking_supported=True,
        tool_mode="native",
        num_ctx=8192,
        keep_alive="10m",
    ),
    "codex": ModelProfileDraft(
        provider="ollama",
        model_name="qwen2.5-coder:7b",
        temperature=0.1,
        thinking_supported=True,
        tool_mode="none",
        num_ctx=4096,
        keep_alive="10m",
    ),
}


def recommended_model_profiles() -> dict[str, ModelProfileDraft]:
    """Return a fresh copy of the recommended local model profiles."""

    return {
        name: ModelProfileDraft(
            provider=draft.provider,
            model_name=draft.model_name,
            temperature=draft.temperature,
            thinking_supported=draft.thinking_supported,
            tool_mode=draft.tool_mode,
            num_ctx=draft.num_ctx,
            keep_alive=draft.keep_alive,
        )
        for name, draft in _RECOMMENDED_PROFILES.items()
    }


def build_onboarding_file_payloads(
    settings: Settings,
    plan: OnboardingPlan,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build YAML payloads for app, model, and Telegram configuration."""

    app_payload: dict[str, object] = {
        "app": {
            "name": settings.app.name,
            "display_name": settings.app.display_name,
            "environment": settings.app.environment,
        },
        "paths": {
            "data_dir": settings.app.directories.data_dir,
            "logs_dir": settings.app.directories.logs_dir,
            "sessions_dir": settings.app.directories.sessions_dir,
            "cache_dir": settings.app.directories.cache_dir,
            "files_dir": settings.app.directories.files_dir,
            "database_file": settings.app.directories.database_file,
        },
        "logging": {
            "level": "DEBUG" if plan.logging_mode == "full" else "INFO",
            "mode": plan.logging_mode,
            "console_enabled": settings.app.logging.console_enabled,
            "file_enabled": settings.app.logging.file_enabled,
            "file_name": settings.app.logging.file_name,
            "include_reasoning_text": False,
        },
        "channels": {
            "terminal_enabled": "terminal" in plan.enabled_channels,
            "telegram_enabled": "telegram" in plan.enabled_channels,
        },
        "models": {
            "default_profile": plan.default_profile,
        },
        "thinking": {
            "default_enabled": settings.app.thinking.default_enabled,
        },
        "providers": {
            "ollama": {
                "timeout_seconds": settings.app.providers.ollama.timeout_seconds,
            },
        },
        "runtime": {
            "tool_timeout_seconds": settings.app.runtime.tool_timeout_seconds,
            "max_tool_calls_per_turn": settings.app.runtime.max_tool_calls_per_turn,
        },
        "security": {
            "tools": {
                "files": {
                    "allowed_roots": list(
                        settings.app.security.tools.files.allowed_roots
                    ),
                },
                "fetch": {
                    "allow_private_networks": (
                        settings.app.security.tools.fetch.allow_private_networks
                    ),
                },
            },
        },
    }

    models_payload: dict[str, object] = {"profiles": {}}
    profiles_section = models_payload["profiles"]
    assert isinstance(profiles_section, dict)
    for profile_name in PROFILE_ORDER:
        draft = plan.model_profiles[profile_name]
        profile_payload = {
            "provider": draft.provider,
            "model_name": draft.model_name,
            "temperature": draft.temperature,
            "thinking_supported": draft.thinking_supported,
            "tool_mode": draft.tool_mode,
        }
        if draft.num_ctx is not None:
            profile_payload["num_ctx"] = draft.num_ctx
        if draft.keep_alive is not None:
            profile_payload["keep_alive"] = draft.keep_alive
        profiles_section[profile_name] = profile_payload

    telegram_payload: dict[str, object] = {
        "bot_token_env_var": plan.telegram_bot_token_env_var,
        "polling_timeout_seconds": plan.telegram_polling_timeout_seconds,
        "allowed_chat_ids": list(plan.telegram_allowed_chat_ids),
    }
    return app_payload, models_payload, telegram_payload


def write_onboarding_files(settings: Settings, plan: OnboardingPlan) -> None:
    """Write the selected onboarding configuration to disk."""

    app_payload, models_payload, telegram_payload = build_onboarding_file_payloads(
        settings,
        plan,
    )
    paths_to_backup = [
        settings.paths.app_config_path,
        settings.paths.models_config_path,
        settings.paths.config_dir / "telegram.yaml",
    ]
    if plan.telegram_bot_token is not None:
        paths_to_backup.append(local_secrets_path(settings))

    _backup_existing_files(paths_to_backup)
    _write_yaml(settings.paths.app_config_path, app_payload)
    _write_yaml(settings.paths.models_config_path, models_payload)
    _write_yaml(settings.paths.config_dir / "telegram.yaml", telegram_payload)
    write_local_secrets(
        settings,
        LocalSecrets(telegram_bot_token=plan.telegram_bot_token),
    )


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    rendered = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=False,
    )
    path.write_text(rendered, encoding="utf-8")


def _backup_existing_files(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        backup_path = path.with_name(f"{path.name}.bak")
        try:
            shutil.copy2(path, backup_path)
            if path.name == LOCAL_SECRETS_FILE_NAME:
                ensure_local_secrets_permissions(backup_path)
        except OSError as exc:
            raise ConfigurationError(
                f"Could not create a backup before overwriting {path}: {exc}"
            ) from exc


__all__ = [
    "build_onboarding_file_payloads",
    "recommended_model_profiles",
    "write_onboarding_files",
]
