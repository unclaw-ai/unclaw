from __future__ import annotations

import pytest

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
        "information.weather",
        "messaging.telegram",
    )
    assert tuple(
        fragment.fragment_id for fragment in registry.list_prompt_fragments()
    ) == (
        "fabrication.three_d_printer.context.overview",
        "fabrication.three_d_printer.guidance.workflow",
        "fabrication.three_d_printer.safety.hardware_state",
        "information.weather.context.overview",
        "information.weather.guidance.live_lookup",
        "information.weather.safety.grounded_claims",
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
    weather_skill = registry.get_skill("information.weather")
    assert weather_skill.display_name == "Weather"
    assert tuple(binding.tool_name for binding in weather_skill.tool_bindings) == (
        "get_weather",
        "search_web",
    )
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


def test_weather_skill_resolves_only_for_native_profiles_with_dedicated_weather_tool(
    make_temp_project,
) -> None:
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
    codex_summary = build_runtime_capability_summary(
        tool_registry=tool_registry,
        memory_summary_available=False,
        model_can_call_tools=settings.models["codex"].tool_mode == "native",
    )
    non_native_summary = build_runtime_capability_summary(
        tool_registry=ToolRegistry(),
        memory_summary_available=False,
        model_can_call_tools=False,
    )

    assert tuple(
        skill.skill_id
        for skill in resolve_active_skill_manifests(
            settings=settings,
            capability_summary=native_summary,
            model_profile_name="main",
        )
    ) == ("information.weather",)
    assert resolve_active_skill_manifests(
        settings=settings,
        capability_summary=codex_summary,
        model_profile_name="codex",
    ) == ()
    assert resolve_active_skill_manifests(
        settings=settings,
        capability_summary=non_native_summary,
        model_profile_name="fast",
    ) == ()


def test_weather_skill_context_notes_stay_compact_on_lite_and_absent_on_fast(
    make_temp_project,
) -> None:
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

    weather_notes = build_active_skill_context_notes(
        settings=settings,
        capability_summary=main_summary,
        model_profile_name="main",
    )

    assert weather_notes == (
        "\n".join(
            (
                "Active optional skill: Weather",
                "- For current weather or short-forecast questions, use get_weather before stating live weather details.",
                "- Use a precise place for the lookup. Answer from the returned current conditions and 7-day forecast, and state any assumption if the user was vague.",
                "- Use search_web only as a fallback for official alerts, longer-range outlooks, or when get_weather cannot resolve the requested place or detail.",
                "- Do not present temperature, precipitation, alerts, or forecast details as certain unless they are grounded by get_weather or search results from this conversation.",
            )
        ),
    )
    assert build_active_skill_context_notes(
        settings=settings,
        capability_summary=fast_summary,
        model_profile_name="fast",
    ) == ()
