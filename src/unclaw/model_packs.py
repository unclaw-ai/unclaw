"""Hardware-oriented default model packs for Unclaw."""

from __future__ import annotations

from dataclasses import dataclass

DEV_MODEL_PACK_NAME = "dev"
MODEL_PACK_ORDER = ("lite", "sweet", "power", DEV_MODEL_PACK_NAME)


@dataclass(frozen=True, slots=True)
class ModelPackProfile:
    provider: str
    model_name: str
    temperature: float
    thinking_supported: bool
    tool_mode: str
    num_ctx: int | None = None
    keep_alive: str | None = None
    planner_profile: str | None = None


@dataclass(frozen=True, slots=True)
class ModelPackDefinition:
    name: str
    title: str
    description: str
    ram_guidance: str
    profiles: dict[str, ModelPackProfile] | None = None


def _build_pack_profiles(
    *,
    fast_model_name: str,
    main_model_name: str,
    deep_model_name: str,
    codex_model_name: str,
) -> dict[str, ModelPackProfile]:
    return {
        "fast": ModelPackProfile(
            provider="ollama",
            model_name=fast_model_name,
            temperature=0.2,
            thinking_supported=False,
            tool_mode="none",
            num_ctx=4096,
            keep_alive="10m",
        ),
        "main": ModelPackProfile(
            provider="ollama",
            model_name=main_model_name,
            temperature=0.3,
            thinking_supported=True,
            tool_mode="native",
            num_ctx=8192,
            keep_alive="30m",
        ),
        "deep": ModelPackProfile(
            provider="ollama",
            model_name=deep_model_name,
            temperature=0.2,
            thinking_supported=True,
            tool_mode="native",
            num_ctx=8192,
            keep_alive="10m",
        ),
        "codex": ModelPackProfile(
            provider="ollama",
            model_name=codex_model_name,
            temperature=0.1,
            thinking_supported=True,
            tool_mode="none",
            num_ctx=4096,
            keep_alive="10m",
        ),
    }


_MODEL_PACKS = {
    "lite": ModelPackDefinition(
        name="lite",
        title="Lite",
        description="Lighter local models for smaller machines.",
        ram_guidance="Best for up to 16 GB of memory.",
        profiles=_build_pack_profiles(
            fast_model_name="ministral-3:3b",
            main_model_name="qwen3.5:4b",
            deep_model_name="qwen3.5:9b",
            codex_model_name="qwen2.5-coder:7b",
        ),
    ),
    "sweet": ModelPackDefinition(
        name="sweet",
        title="Sweet",
        description="Balanced local models for most laptops and desktops.",
        ram_guidance="Best for more than 16 GB and up to 32 GB of memory.",
        profiles=_build_pack_profiles(
            fast_model_name="ministral-3:3b",
            main_model_name="qwen3.5:9b",
            deep_model_name="ministral-3:14b",
            codex_model_name="qwen2.5-coder:7b",
        ),
    ),
    "power": ModelPackDefinition(
        name="power",
        title="Power",
        description="Stronger local models for larger-memory machines.",
        ram_guidance="Best for more than 32 GB of memory.",
        profiles=_build_pack_profiles(
            fast_model_name="ministral-3:8b",
            main_model_name="ministral-3:14b",
            deep_model_name="qwen3.5:27b",
            codex_model_name="deepcoder:14b",
        ),
    ),
    DEV_MODEL_PACK_NAME: ModelPackDefinition(
        name=DEV_MODEL_PACK_NAME,
        title="Dev",
        description="Manual and advanced model control.",
        ram_guidance="Manual / advanced / full control.",
        profiles=None,
    ),
}


def model_pack_names() -> tuple[str, ...]:
    return MODEL_PACK_ORDER


def get_model_pack_definition(pack_name: str) -> ModelPackDefinition:
    try:
        return _MODEL_PACKS[pack_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model pack: {pack_name}") from exc


def is_manual_model_pack(pack_name: str) -> bool:
    return pack_name == DEV_MODEL_PACK_NAME


def get_model_pack_profiles(pack_name: str) -> dict[str, ModelPackProfile]:
    definition = get_model_pack_definition(pack_name)
    if definition.profiles is None:
        raise ValueError(f"Model pack '{pack_name}' does not define fixed profiles.")
    return dict(definition.profiles)


def recommend_model_pack(total_ram_gib: float | None) -> str:
    if total_ram_gib is None or total_ram_gib <= 16:
        return "lite"
    if total_ram_gib <= 32:
        return "sweet"
    return "power"


__all__ = [
    "DEV_MODEL_PACK_NAME",
    "MODEL_PACK_ORDER",
    "ModelPackDefinition",
    "ModelPackProfile",
    "get_model_pack_definition",
    "get_model_pack_profiles",
    "is_manual_model_pack",
    "model_pack_names",
    "recommend_model_pack",
]
