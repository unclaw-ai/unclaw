"""Typed capability-context budget policies for built-in prompt composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CapabilityBudgetTier(StrEnum):
    """High-level prompt budget tiers for built-in capability context."""

    MINIMAL = "minimal"
    COMPACT = "compact"
    STANDARD = "standard"


class CapabilitySectionDetail(StrEnum):
    """How much detail to render for a capability section."""

    NONE = "none"
    COMPACT = "compact"
    FULL = "full"


class CapabilityGuidanceDetail(StrEnum):
    """How much guidance detail to inject for active built-in capabilities."""

    MINIMAL = "minimal"
    COMPACT = "compact"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class CapabilityBudgetPolicy:
    """Resolved pack/profile prompt budget for capability-context composition."""

    tier: CapabilityBudgetTier
    available_tool_detail: CapabilitySectionDetail
    include_available_runtime_section: bool
    unavailable_capability_detail: CapabilitySectionDetail
    guidance_detail: CapabilityGuidanceDetail


MINIMAL_CAPABILITY_BUDGET_POLICY = CapabilityBudgetPolicy(
    tier=CapabilityBudgetTier.MINIMAL,
    available_tool_detail=CapabilitySectionDetail.NONE,
    include_available_runtime_section=False,
    unavailable_capability_detail=CapabilitySectionDetail.NONE,
    guidance_detail=CapabilityGuidanceDetail.MINIMAL,
)

COMPACT_CAPABILITY_BUDGET_POLICY = CapabilityBudgetPolicy(
    tier=CapabilityBudgetTier.COMPACT,
    available_tool_detail=CapabilitySectionDetail.COMPACT,
    include_available_runtime_section=True,
    unavailable_capability_detail=CapabilitySectionDetail.COMPACT,
    guidance_detail=CapabilityGuidanceDetail.COMPACT,
)

STANDARD_CAPABILITY_BUDGET_POLICY = CapabilityBudgetPolicy(
    tier=CapabilityBudgetTier.STANDARD,
    available_tool_detail=CapabilitySectionDetail.FULL,
    include_available_runtime_section=True,
    unavailable_capability_detail=CapabilitySectionDetail.FULL,
    guidance_detail=CapabilityGuidanceDetail.FULL,
)


def resolve_capability_budget_policy(
    *,
    model_pack: str,
    model_profile_name: str,
) -> CapabilityBudgetPolicy:
    """Resolve the live capability-context budget deterministically by pack/profile."""
    normalized_profile_name = model_profile_name.strip().lower()
    if normalized_profile_name == "fast":
        return MINIMAL_CAPABILITY_BUDGET_POLICY

    normalized_pack_name = model_pack.strip().lower()
    if normalized_pack_name == "lite":
        return COMPACT_CAPABILITY_BUDGET_POLICY

    return STANDARD_CAPABILITY_BUDGET_POLICY


__all__ = [
    "CapabilityBudgetPolicy",
    "CapabilityBudgetTier",
    "CapabilityGuidanceDetail",
    "CapabilitySectionDetail",
    "COMPACT_CAPABILITY_BUDGET_POLICY",
    "MINIMAL_CAPABILITY_BUDGET_POLICY",
    "STANDARD_CAPABILITY_BUDGET_POLICY",
    "resolve_capability_budget_policy",
]
