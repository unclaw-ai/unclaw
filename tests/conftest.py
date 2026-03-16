from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def make_temp_project(tmp_path: Path):
    def _make_temp_project(
        *,
        allowed_chat_ids: list[int] | None = None,
        remove_secrets: bool = False,
    ) -> Path:
        project_root = tmp_path / "project"
        shutil.copytree(_repo_root() / "config", project_root / "config")
        if allowed_chat_ids is not None:
            telegram_config_path = project_root / "config" / "telegram.yaml"
            payload = yaml.safe_load(telegram_config_path.read_text(encoding="utf-8"))
            assert isinstance(payload, dict)
            payload["allowed_chat_ids"] = allowed_chat_ids
            telegram_config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
        if remove_secrets:
            secrets_path = project_root / "config" / "secrets.yaml"
            if secrets_path.exists():
                secrets_path.unlink()
        return project_root

    return _make_temp_project


@pytest.fixture
def set_profile_tool_mode():
    def _set_profile_tool_mode(settings, profile_name: str, *, tool_mode: str) -> None:
        profile = settings.models[profile_name]
        settings.models[profile_name] = profile.__class__(
            name=profile.name,
            provider=profile.provider,
            model_name=profile.model_name,
            temperature=profile.temperature,
            thinking_supported=profile.thinking_supported,
            tool_mode=tool_mode,
            keep_alive=profile.keep_alive,
        )

    return _set_profile_tool_mode
