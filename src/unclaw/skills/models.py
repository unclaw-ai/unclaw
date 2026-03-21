"""Typed optional skill manifests and metadata for future runtime integration."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from enum import StrEnum


def _require_non_empty(value: str, *, field_name: str) -> str:
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized_value


def _normalize_string_tuple(
    values: tuple[str, ...],
    *,
    field_name: str,
    lowercase: bool = False,
) -> tuple[str, ...]:
    normalized_values: list[str] = []

    for raw_value in values:
        normalized_value = _require_non_empty(raw_value, field_name=field_name)
        if lowercase:
            normalized_value = normalized_value.lower()
        if normalized_value not in normalized_values:
            normalized_values.append(normalized_value)

    return tuple(normalized_values)


class SkillInstallMode(StrEnum):
    """How a skill should be treated when future runtime activation exists."""

    OPT_IN = "opt_in"
    ENABLED_BY_DEFAULT = "enabled_by_default"


class SkillPromptFragmentKind(StrEnum):
    """Semantic role of one skill-provided prompt fragment."""

    CONTEXT = "context"
    GUIDANCE = "guidance"
    SAFETY = "safety"


class SkillPromptSourceKind(StrEnum):
    """Where a skill prompt fragment's text is authored."""

    INLINE = "inline"


class SkillPromptBudgetTier(StrEnum):
    """Future prompt-budget buckets that a skill may target."""

    MINIMAL = "minimal"
    COMPACT = "compact"
    STANDARD = "standard"


@dataclass(frozen=True, slots=True)
class SkillPromptSource:
    """Source location for one skill prompt fragment."""

    kind: SkillPromptSourceKind
    reference: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference",
            _require_non_empty(self.reference, field_name="reference"),
        )


@dataclass(frozen=True, slots=True)
class SkillPromptFragment:
    """One prompt fragment that an optional skill may expose later."""

    fragment_id: str
    skill_id: str
    name: str
    kind: SkillPromptFragmentKind
    source: SkillPromptSource
    lines: tuple[str, ...]
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fragment_id",
            _require_non_empty(self.fragment_id, field_name="fragment_id"),
        )
        object.__setattr__(
            self,
            "skill_id",
            _require_non_empty(self.skill_id, field_name="skill_id"),
        )
        object.__setattr__(
            self,
            "name",
            _require_non_empty(self.name, field_name="name"),
        )
        if not self.lines:
            raise ValueError("lines must contain at least one prompt line.")
        object.__setattr__(
            self,
            "lines",
            _normalize_string_tuple(self.lines, field_name="lines"),
        )
        if self.description is not None:
            object.__setattr__(
                self,
                "description",
                _require_non_empty(self.description, field_name="description"),
            )


@dataclass(frozen=True, slots=True)
class SkillToolBinding:
    """Optional skill-to-tool declaration for future tool exposure."""

    binding_id: str
    tool_name: str
    description: str | None = None
    required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding_id",
            _require_non_empty(self.binding_id, field_name="binding_id"),
        )
        object.__setattr__(
            self,
            "tool_name",
            _require_non_empty(self.tool_name, field_name="tool_name"),
        )
        if self.description is not None:
            object.__setattr__(
                self,
                "description",
                _require_non_empty(self.description, field_name="description"),
            )


@dataclass(frozen=True, slots=True)
class SkillAvailability:
    """Typed future activation constraints for one skill."""

    supported_model_profiles: tuple[str, ...] = ()
    supported_model_packs: tuple[str, ...] = ()
    required_builtin_tool_names: tuple[str, ...] = ()
    requires_model_tool_support: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_model_profiles",
            _normalize_string_tuple(
                self.supported_model_profiles,
                field_name="supported_model_profiles",
                lowercase=True,
            ),
        )
        object.__setattr__(
            self,
            "supported_model_packs",
            _normalize_string_tuple(
                self.supported_model_packs,
                field_name="supported_model_packs",
                lowercase=True,
            ),
        )
        object.__setattr__(
            self,
            "required_builtin_tool_names",
            _normalize_string_tuple(
                self.required_builtin_tool_names,
                field_name="required_builtin_tool_names",
            ),
        )

    def matches(
        self,
        *,
        model_profile_name: str | None = None,
        model_pack: str | None = None,
        available_builtin_tool_names: Collection[str] = (),
        model_can_call_tools: bool = False,
    ) -> bool:
        """Evaluate whether a runtime surface satisfies this skill's constraints."""
        if self.supported_model_profiles:
            normalized_profile_name = (
                model_profile_name.strip().lower()
                if model_profile_name is not None
                else ""
            )
            if normalized_profile_name not in self.supported_model_profiles:
                return False

        if self.supported_model_packs:
            normalized_pack_name = (
                model_pack.strip().lower() if model_pack is not None else ""
            )
            if normalized_pack_name not in self.supported_model_packs:
                return False

        available_tool_names = frozenset(available_builtin_tool_names)
        if any(
            tool_name not in available_tool_names
            for tool_name in self.required_builtin_tool_names
        ):
            return False

        if self.requires_model_tool_support and model_can_call_tools is not True:
            return False

        return True


@dataclass(frozen=True, slots=True)
class SkillBudgetMetadata:
    """Prompt-budget hints for future optional skill composition."""

    min_budget_tier: SkillPromptBudgetTier = SkillPromptBudgetTier.COMPACT
    prompt_priority: int = 100
    estimated_prompt_lines: int = 0

    def __post_init__(self) -> None:
        if self.prompt_priority < 0:
            raise ValueError("prompt_priority must be >= 0.")
        if self.estimated_prompt_lines < 0:
            raise ValueError("estimated_prompt_lines must be >= 0.")


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """Canonical typed representation of one optional internal skill."""

    skill_id: str
    display_name: str
    version: str
    description: str
    install_mode: SkillInstallMode = SkillInstallMode.OPT_IN
    tags: tuple[str, ...] = ()
    prompt_fragments: tuple[SkillPromptFragment, ...] = ()
    tool_bindings: tuple[SkillToolBinding, ...] = ()
    availability: SkillAvailability = field(default_factory=SkillAvailability)
    budget: SkillBudgetMetadata = field(default_factory=SkillBudgetMetadata)

    def __post_init__(self) -> None:
        normalized_skill_id = _require_non_empty(self.skill_id, field_name="skill_id")
        object.__setattr__(self, "skill_id", normalized_skill_id)
        object.__setattr__(
            self,
            "display_name",
            _require_non_empty(self.display_name, field_name="display_name"),
        )
        object.__setattr__(
            self,
            "version",
            _require_non_empty(self.version, field_name="version"),
        )
        object.__setattr__(
            self,
            "description",
            _require_non_empty(self.description, field_name="description"),
        )
        object.__setattr__(
            self,
            "tags",
            _normalize_string_tuple(
                self.tags,
                field_name="tags",
                lowercase=True,
            ),
        )

        fragment_ids: set[str] = set()
        for fragment in self.prompt_fragments:
            if fragment.skill_id != normalized_skill_id:
                raise ValueError(
                    "Skill prompt fragment ownership mismatch: "
                    f"{fragment.fragment_id} belongs to {fragment.skill_id}, "
                    f"expected {normalized_skill_id}."
                )
            if fragment.fragment_id in fragment_ids:
                raise ValueError(
                    f"Duplicate skill prompt fragment id within manifest: {fragment.fragment_id}"
                )
            fragment_ids.add(fragment.fragment_id)

        binding_ids: set[str] = set()
        tool_names: set[str] = set()
        for binding in self.tool_bindings:
            if binding.binding_id in binding_ids:
                raise ValueError(
                    f"Duplicate skill tool binding id within manifest: {binding.binding_id}"
                )
            if binding.tool_name in tool_names:
                raise ValueError(
                    f"Duplicate skill tool name within manifest: {binding.tool_name}"
                )
            binding_ids.add(binding.binding_id)
            tool_names.add(binding.tool_name)

    @property
    def enabled_by_default(self) -> bool:
        return self.install_mode is SkillInstallMode.ENABLED_BY_DEFAULT


__all__ = [
    "SkillAvailability",
    "SkillBudgetMetadata",
    "SkillInstallMode",
    "SkillManifest",
    "SkillPromptBudgetTier",
    "SkillPromptFragment",
    "SkillPromptFragmentKind",
    "SkillPromptSource",
    "SkillPromptSourceKind",
    "SkillToolBinding",
]
