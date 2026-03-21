from __future__ import annotations

import pytest

from unclaw.core.capabilities import (
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.capability_fragments import load_builtin_capability_fragment_registry
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
from unclaw.tools.registry import ToolRegistry

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


def test_load_skill_registry_is_cached_and_live_runtime_context_ignores_skills() -> None:
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
