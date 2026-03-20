from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from unclaw.model_packs import DEV_MODEL_PACK_NAME, get_model_pack_profiles


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


_REAL_OLLAMA_TOOL_CAPABLE_MODEL_PREFIXES = (
    "qwen",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "mistral",
    "command-r",
    "firefunction",
    "hermes",
)


def _real_ollama_enabled() -> bool:
    return os.environ.get("REAL_OLLAMA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def pytest_collection_modifyitems(config, items) -> None:  # type: ignore[no-untyped-def]
    del config
    if _real_ollama_enabled():
        return

    skip_real_ollama = pytest.mark.skip(
        reason="Set REAL_OLLAMA=1 to run real_ollama tests.",
    )
    for item in items:
        if "real_ollama" in item.keywords:
            item.add_marker(skip_real_ollama)


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


@pytest.fixture(scope="session")
def real_ollama_model_name() -> str:
    if not _real_ollama_enabled():
        pytest.skip("Set REAL_OLLAMA=1 to run real_ollama tests.")

    from unclaw.llm.ollama_provider import OllamaProvider

    provider = OllamaProvider()
    if not provider.is_available(timeout_seconds=5):
        pytest.fail(
            "REAL_OLLAMA=1 but Ollama is not reachable at http://127.0.0.1:11434."
        )

    available_models = provider.list_models(timeout_seconds=5)
    configured_model = os.environ.get("REAL_OLLAMA_MODEL", "").strip()
    if configured_model:
        if configured_model not in available_models:
            pytest.fail(
                "REAL_OLLAMA_MODEL={!r} is not installed. Available models: {}.".format(
                    configured_model,
                    ", ".join(available_models) or "[none]",
                )
            )
        return configured_model

    for model_name in available_models:
        model_base = model_name.split(":")[0].lower()
        if any(
            model_base.startswith(prefix)
            for prefix in _REAL_OLLAMA_TOOL_CAPABLE_MODEL_PREFIXES
        ):
            return model_name

    pytest.fail(
        "REAL_OLLAMA=1 but no tool-capable Ollama model is installed. "
        "Set REAL_OLLAMA_MODEL to a pulled model such as qwen3.5:4b."
    )


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
            num_ctx=profile.num_ctx,
            keep_alive=profile.keep_alive,
            planner_profile=profile.planner_profile,
        )

    return _set_profile_tool_mode


def _serialize_profile_for_models_config(profile: object) -> dict[str, object]:
    if isinstance(profile, Mapping):
        payload = dict(profile)
    else:
        payload = {
            "provider": getattr(profile, "provider"),
            "model_name": getattr(profile, "model_name"),
            "temperature": getattr(profile, "temperature"),
            "thinking_supported": getattr(profile, "thinking_supported"),
            "tool_mode": getattr(profile, "tool_mode"),
        }
        num_ctx = getattr(profile, "num_ctx", None)
        keep_alive = getattr(profile, "keep_alive", None)
        planner_profile = getattr(profile, "planner_profile", None)
        if num_ctx is not None:
            payload["num_ctx"] = num_ctx
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if planner_profile is not None:
            payload["planner_profile"] = planner_profile

    return payload


@pytest.fixture
def pack_profiles():
    def _pack_profiles(pack_name: str):
        return get_model_pack_profiles(pack_name)

    return _pack_profiles


@pytest.fixture
def write_models_config():
    def _write_models_config(
        project_root: Path,
        *,
        active_pack: str = DEV_MODEL_PACK_NAME,
        dev_profiles: Mapping[str, object] | None = None,
        dev_profile_pack: str = "power",
    ) -> dict[str, object]:
        profiles_source = (
            dev_profiles
            if dev_profiles is not None
            else get_model_pack_profiles(dev_profile_pack)
        )
        payload = {
            "active_pack": active_pack,
            "dev_profiles": {
                profile_name: _serialize_profile_for_models_config(profile)
                for profile_name, profile in profiles_source.items()
            },
        }
        models_path = project_root / "config" / "models.yaml"
        models_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        return payload

    return _write_models_config
