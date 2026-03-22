from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from unclaw.core.capabilities import (
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.capability_fragments import load_builtin_capability_fragment_registry
from unclaw.settings import load_settings
from unclaw.skills.models import (
    SkillAvailability,
    SkillBudgetMetadata,
    SkillInstallMode,
    SkillManifest,
    SkillPromptBudgetTier,
    SkillPromptFragment,
    SkillPromptFragmentKind,
    SkillPromptSource,
    SkillPromptSourceKind,
    SkillToolBinding,
)
from unclaw.skills.registry import SkillRegistry, load_skill_registry
from unclaw.skills.runtime import (
    build_active_skill_context_notes,
    resolve_active_skill_manifests,
)
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.weather_tools import GET_WEATHER_DEFINITION

pytestmark = pytest.mark.unit


def _inline_source(reference: str) -> SkillPromptSource:
    return SkillPromptSource(
        kind=SkillPromptSourceKind.INLINE,
        reference=reference,
    )


def _fragment(
    *,
    skill_id: str,
    fragment_id: str,
    kind: SkillPromptFragmentKind = SkillPromptFragmentKind.CONTEXT,
    line: str = "Prompt line.",
) -> SkillPromptFragment:
    return SkillPromptFragment(
        fragment_id=fragment_id,
        skill_id=skill_id,
        name=f"{fragment_id} name",
        kind=kind,
        source=_inline_source(fragment_id),
        lines=(line,),
    )


def test_skill_manifest_validates_fragment_ownership_and_budget_metadata() -> None:
    with pytest.raises(ValueError, match="ownership mismatch"):
        SkillManifest(
            skill_id="alpha.skill",
            display_name="Alpha",
            version="0.1.0",
            description="Test skill.",
            prompt_fragments=(
                _fragment(
                    skill_id="beta.skill",
                    fragment_id="beta.skill.context.overview",
                ),
            ),
            budget=SkillBudgetMetadata(
                min_budget_tier=SkillPromptBudgetTier.COMPACT,
                prompt_priority=10,
                estimated_prompt_lines=1,
            ),
        )


def test_skill_availability_matches_future_runtime_constraints() -> None:
    availability = SkillAvailability(
        supported_model_profiles=("MAIN", "codex"),
        supported_model_packs=("Power",),
        required_builtin_tool_names=("read_text_file", "search_web"),
        requires_model_tool_support=True,
    )

    assert availability.matches(
        model_profile_name="main",
        model_pack="power",
        available_builtin_tool_names=("read_text_file", "search_web"),
        model_can_call_tools=True,
    )
    assert not availability.matches(
        model_profile_name="fast",
        model_pack="power",
        available_builtin_tool_names=("read_text_file", "search_web"),
        model_can_call_tools=True,
    )
    assert not availability.matches(
        model_profile_name="main",
        model_pack="lite",
        available_builtin_tool_names=("read_text_file", "search_web"),
        model_can_call_tools=True,
    )
    assert not availability.matches(
        model_profile_name="main",
        model_pack="power",
        available_builtin_tool_names=("read_text_file",),
        model_can_call_tools=True,
    )
    assert not availability.matches(
        model_profile_name="main",
        model_pack="power",
        available_builtin_tool_names=("read_text_file", "search_web"),
        model_can_call_tools=False,
    )


def test_skill_registry_has_stable_skill_and_prompt_fragment_order() -> None:
    registry = load_skill_registry()

    assert registry.list_skill_ids() == (
        "fabrication.three_d_printer",
        "messaging.telegram",
    )
    assert tuple(
        fragment.fragment_id for fragment in registry.list_prompt_fragments()
    ) == (
        "fabrication.three_d_printer.context.overview",
        "fabrication.three_d_printer.guidance.workflow",
        "fabrication.three_d_printer.safety.hardware_state",
        "messaging.telegram.context.overview",
        "messaging.telegram.guidance.formatting",
        "messaging.telegram.safety.delivery_state",
    )


def test_skill_registry_exposes_indexes_without_touching_builtin_capabilities() -> None:
    registry = load_skill_registry()
    builtins = load_builtin_capability_fragment_registry()

    telegram_skill = registry.get_skill("messaging.telegram")
    assert telegram_skill.display_name == "Telegram Messaging"
    assert telegram_skill.install_mode is SkillInstallMode.OPT_IN
    assert telegram_skill.enabled_by_default is False
    assert telegram_skill.budget.min_budget_tier is SkillPromptBudgetTier.COMPACT
    assert tuple(binding.tool_name for binding in telegram_skill.tool_bindings) == (
        "list_telegram_chats",
        "send_telegram_message",
    )

    assert tuple(
        skill.skill_id for skill in registry.get_skills_for_tool_name("printer_status")
    ) == ("fabrication.three_d_printer",)
    assert tuple(
        skill.skill_id for skill in registry.get_skills_for_tag("Telegram")
    ) == ("messaging.telegram",)
    assert builtins.list_capability_ids()[0] == "local_file_read"
    assert "information.weather" not in builtins.list_capability_ids()
    assert "messaging.telegram" not in builtins.list_capability_ids()
    assert "fabrication.three_d_printer" not in builtins.list_capability_ids()


def test_skill_registry_rejects_duplicate_skill_and_fragment_ids() -> None:
    alpha_skill = SkillManifest(
        skill_id="alpha.skill",
        display_name="Alpha",
        version="0.1.0",
        description="Alpha skill.",
        prompt_fragments=(
            _fragment(
                skill_id="alpha.skill",
                fragment_id="shared.fragment",
            ),
        ),
        tool_bindings=(
            SkillToolBinding(
                binding_id="alpha.skill.tool",
                tool_name="alpha_tool",
            ),
        ),
    )
    beta_skill = SkillManifest(
        skill_id="beta.skill",
        display_name="Beta",
        version="0.1.0",
        description="Beta skill.",
        prompt_fragments=(
            _fragment(
                skill_id="beta.skill",
                fragment_id="shared.fragment",
            ),
        ),
    )

    with pytest.raises(ValueError, match="Duplicate skill prompt fragment id"):
        SkillRegistry(skills=(alpha_skill, beta_skill))

    with pytest.raises(ValueError, match="Duplicate skill id"):
        SkillRegistry(skills=(alpha_skill, alpha_skill))


def test_load_skill_registry_is_cached_and_builtin_runtime_context_still_ignores_skills() -> None:
    registry = load_skill_registry()
    assert load_skill_registry() is registry

    summary = build_runtime_capability_summary(
        tool_registry=ToolRegistry(),
        memory_summary_available=False,
        model_can_call_tools=False,
    )
    context = build_runtime_capability_context(summary)

    assert "Telegram Messaging" not in context
    assert "3D Printer Operations" not in context
    assert "Weather" not in context


def test_legacy_manifest_registry_no_longer_owns_weather(
    make_temp_project,
) -> None:
    """Weather was removed from the legacy manifest registry.

    Prompt ownership for weather now lives exclusively in the file-first
    skills/weather/SKILL.md bundle (compact catalog + on-demand full load).
    The legacy runtime produces no manifests for weather regardless of profile.
    """
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    tool_registry = ToolRegistry()
    tool_registry.register(
        GET_WEATHER_DEFINITION,
        lambda call: None,  # pragma: no cover - tool execution is not used here
    )

    native_summary = build_runtime_capability_summary(
        tool_registry=tool_registry,
        memory_summary_available=False,
        model_can_call_tools=True,
    )
    non_native_summary = build_runtime_capability_summary(
        tool_registry=ToolRegistry(),
        memory_summary_available=False,
        model_can_call_tools=False,
    )

    # Legacy manifest registry must produce no weather manifests.
    assert resolve_active_skill_manifests(
        settings=settings,
        capability_summary=native_summary,
        model_profile_name="main",
    ) == ()
    assert resolve_active_skill_manifests(
        settings=settings,
        capability_summary=non_native_summary,
        model_profile_name="fast",
    ) == ()


def test_legacy_manifest_registry_produces_no_weather_when_file_first_id_is_configured(
    make_temp_project,
) -> None:
    """File-first skill ID 'weather' no longer maps to a legacy manifest entry."""
    project_root = make_temp_project()
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = _read_yaml(app_config_path)
    app_payload["skills"]["enabled_skill_ids"] = ["weather"]
    _write_yaml(app_config_path, app_payload)

    settings = load_settings(project_root=project_root)
    tool_registry = ToolRegistry()
    tool_registry.register(
        GET_WEATHER_DEFINITION,
        lambda call: None,  # pragma: no cover - tool execution is not used here
    )
    native_summary = build_runtime_capability_summary(
        tool_registry=tool_registry,
        memory_summary_available=False,
        model_can_call_tools=True,
    )

    # The legacy registry no longer owns weather; result must be empty.
    assert resolve_active_skill_manifests(
        settings=settings,
        capability_summary=native_summary,
        model_profile_name="main",
    ) == ()


def test_legacy_skill_context_notes_produce_no_weather_notes(
    make_temp_project,
) -> None:
    """Weather was removed from the legacy manifest registry.

    build_active_skill_context_notes must now return no weather notes for any
    profile. Weather guidance comes exclusively from the file-first SKILL.md
    (compact catalog + on-demand full load).
    """
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    tool_registry = ToolRegistry()
    tool_registry.register(
        GET_WEATHER_DEFINITION,
        lambda call: None,  # pragma: no cover - tool execution is not used here
    )

    main_summary = build_runtime_capability_summary(
        tool_registry=tool_registry,
        memory_summary_available=False,
        model_can_call_tools=True,
    )
    fast_summary = build_runtime_capability_summary(
        tool_registry=ToolRegistry(),
        memory_summary_available=False,
        model_can_call_tools=False,
    )

    assert build_active_skill_context_notes(
        settings=settings,
        capability_summary=main_summary,
        model_profile_name="main",
    ) == ()
    assert build_active_skill_context_notes(
        settings=settings,
        capability_summary=fast_summary,
        model_profile_name="fast",
    ) == ()


def _read_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
