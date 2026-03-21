"""Live optional skill activation and compact prompt-note rendering."""

from __future__ import annotations

from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.core.capability_budget import (
    CapabilityBudgetPolicy,
    resolve_capability_budget_policy,
)
from unclaw.settings import Settings
from unclaw.skills.models import (
    SkillManifest,
    SkillPromptBudgetTier,
    SkillPromptFragment,
    SkillPromptFragmentKind,
)
from unclaw.skills.registry import SkillRegistry, load_skill_registry

_SKILL_BUDGET_ORDER: dict[SkillPromptBudgetTier, int] = {
    SkillPromptBudgetTier.MINIMAL: 0,
    SkillPromptBudgetTier.COMPACT: 1,
    SkillPromptBudgetTier.STANDARD: 2,
}
_FRAGMENT_KINDS_BY_BUDGET: dict[
    SkillPromptBudgetTier,
    tuple[SkillPromptFragmentKind, ...],
] = {
    SkillPromptBudgetTier.MINIMAL: (
        SkillPromptFragmentKind.SAFETY,
    ),
    SkillPromptBudgetTier.COMPACT: (
        SkillPromptFragmentKind.GUIDANCE,
        SkillPromptFragmentKind.SAFETY,
    ),
    SkillPromptBudgetTier.STANDARD: (
        SkillPromptFragmentKind.CONTEXT,
        SkillPromptFragmentKind.GUIDANCE,
        SkillPromptFragmentKind.SAFETY,
    ),
}


def resolve_active_skill_manifests(
    *,
    settings: Settings,
    capability_summary: RuntimeCapabilitySummary,
    model_profile_name: str,
    budget_policy: CapabilityBudgetPolicy | None = None,
    skill_registry: SkillRegistry | None = None,
) -> tuple[SkillManifest, ...]:
    """Resolve explicitly enabled skills that are active for this runtime turn."""
    enabled_skill_ids = getattr(settings.skills, "enabled_skill_ids", ())
    if not enabled_skill_ids:
        return ()

    resolved_budget_policy = budget_policy or resolve_capability_budget_policy(
        model_pack=settings.model_pack,
        model_profile_name=model_profile_name,
    )
    active_budget_tier = SkillPromptBudgetTier(resolved_budget_policy.tier.value)
    registry = skill_registry or load_skill_registry()

    active_skills: list[SkillManifest] = []
    for skill_id in enabled_skill_ids:
        try:
            skill = registry.get_skill(skill_id)
        except KeyError:
            continue

        if not _skill_budget_matches(skill, active_budget_tier):
            continue
        if not skill.availability.matches(
            model_profile_name=model_profile_name,
            model_pack=settings.model_pack,
            available_builtin_tool_names=capability_summary.available_builtin_tool_names,
            model_can_call_tools=capability_summary.model_can_call_tools,
        ):
            continue
        active_skills.append(skill)

    return tuple(active_skills)


def build_active_skill_context_notes(
    *,
    settings: Settings,
    capability_summary: RuntimeCapabilitySummary,
    model_profile_name: str,
    budget_policy: CapabilityBudgetPolicy | None = None,
    skill_registry: SkillRegistry | None = None,
) -> tuple[str, ...]:
    """Render compact system-note strings for the active optional skills."""
    resolved_budget_policy = budget_policy or resolve_capability_budget_policy(
        model_pack=settings.model_pack,
        model_profile_name=model_profile_name,
    )
    active_budget_tier = SkillPromptBudgetTier(resolved_budget_policy.tier.value)
    active_skills = resolve_active_skill_manifests(
        settings=settings,
        capability_summary=capability_summary,
        model_profile_name=model_profile_name,
        budget_policy=resolved_budget_policy,
        skill_registry=skill_registry,
    )

    notes: list[str] = []
    for skill in active_skills:
        rendered_lines = tuple(
            line
            for fragment in _select_prompt_fragments(
                skill,
                active_budget_tier=active_budget_tier,
            )
            for line in fragment.lines
        )
        if not rendered_lines:
            continue
        notes.append(
            "\n".join(
                (
                    f"Active optional skill: {skill.display_name}",
                    *(f"- {line}" for line in rendered_lines),
                )
            )
        )

    return tuple(notes)


def _skill_budget_matches(
    skill: SkillManifest,
    active_budget_tier: SkillPromptBudgetTier,
) -> bool:
    return _SKILL_BUDGET_ORDER[active_budget_tier] >= _SKILL_BUDGET_ORDER[
        skill.budget.min_budget_tier
    ]


def _select_prompt_fragments(
    skill: SkillManifest,
    *,
    active_budget_tier: SkillPromptBudgetTier,
) -> tuple[SkillPromptFragment, ...]:
    allowed_kinds = frozenset(_FRAGMENT_KINDS_BY_BUDGET[active_budget_tier])
    return tuple(
        fragment
        for fragment in skill.prompt_fragments
        if fragment.kind in allowed_kinds
    )


__all__ = [
    "build_active_skill_context_notes",
    "resolve_active_skill_manifests",
]
