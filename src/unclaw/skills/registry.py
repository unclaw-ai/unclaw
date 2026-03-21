"""Deterministic registry for optional internal skill manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType

from unclaw.skills.manifests import load_internal_skill_manifests
from unclaw.skills.models import SkillManifest, SkillPromptFragment


@dataclass(frozen=True, slots=True)
class SkillRegistry:
    """Typed registry of optional skills kept separate from built-in capabilities."""

    skills: tuple[SkillManifest, ...]
    _skill_ids: tuple[str, ...] = field(init=False, repr=False)
    _prompt_fragments: tuple[SkillPromptFragment, ...] = field(init=False, repr=False)
    _skills_by_id: MappingProxyType = field(init=False, repr=False)
    _skills_by_tag: MappingProxyType = field(init=False, repr=False)
    _skills_by_tool_name: MappingProxyType = field(init=False, repr=False)
    _prompt_fragments_by_id: MappingProxyType = field(init=False, repr=False)
    _prompt_fragments_by_skill_id: MappingProxyType = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ordered_skills = tuple(sorted(self.skills, key=lambda skill: skill.skill_id))
        object.__setattr__(self, "skills", ordered_skills)

        skills_by_id: dict[str, SkillManifest] = {}
        skills_by_tag: dict[str, list[SkillManifest]] = {}
        skills_by_tool_name: dict[str, list[SkillManifest]] = {}
        prompt_fragments_by_id: dict[str, SkillPromptFragment] = {}
        prompt_fragments_by_skill_id: dict[str, tuple[SkillPromptFragment, ...]] = {}
        prompt_fragments: list[SkillPromptFragment] = []
        skill_ids: list[str] = []

        for skill in ordered_skills:
            if skill.skill_id in skills_by_id:
                raise ValueError(f"Duplicate skill id: {skill.skill_id}")

            skills_by_id[skill.skill_id] = skill
            skill_ids.append(skill.skill_id)

            for tag in skill.tags:
                skills_by_tag.setdefault(tag, []).append(skill)

            for tool_binding in skill.tool_bindings:
                skills_by_tool_name.setdefault(tool_binding.tool_name, []).append(skill)

            prompt_fragments_by_skill_id[skill.skill_id] = skill.prompt_fragments
            for prompt_fragment in skill.prompt_fragments:
                if prompt_fragment.fragment_id in prompt_fragments_by_id:
                    raise ValueError(
                        f"Duplicate skill prompt fragment id: {prompt_fragment.fragment_id}"
                    )
                prompt_fragments_by_id[prompt_fragment.fragment_id] = prompt_fragment
                prompt_fragments.append(prompt_fragment)

        object.__setattr__(self, "_skill_ids", tuple(skill_ids))
        object.__setattr__(self, "_prompt_fragments", tuple(prompt_fragments))
        object.__setattr__(self, "_skills_by_id", MappingProxyType(skills_by_id))
        object.__setattr__(
            self,
            "_skills_by_tag",
            MappingProxyType(
                {tag: tuple(tagged_skills) for tag, tagged_skills in skills_by_tag.items()}
            ),
        )
        object.__setattr__(
            self,
            "_skills_by_tool_name",
            MappingProxyType(
                {
                    tool_name: tuple(bound_skills)
                    for tool_name, bound_skills in skills_by_tool_name.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "_prompt_fragments_by_id",
            MappingProxyType(prompt_fragments_by_id),
        )
        object.__setattr__(
            self,
            "_prompt_fragments_by_skill_id",
            MappingProxyType(prompt_fragments_by_skill_id),
        )

    def list_skills(self) -> tuple[SkillManifest, ...]:
        return self.skills

    def list_skill_ids(self) -> tuple[str, ...]:
        return self._skill_ids

    def get_skill(self, skill_id: str) -> SkillManifest:
        try:
            return self._skills_by_id[skill_id]
        except KeyError as exc:
            raise KeyError(f"Unknown skill id: {skill_id}") from exc

    def list_prompt_fragments(self) -> tuple[SkillPromptFragment, ...]:
        return self._prompt_fragments

    def get_prompt_fragment(self, fragment_id: str) -> SkillPromptFragment:
        try:
            return self._prompt_fragments_by_id[fragment_id]
        except KeyError as exc:
            raise KeyError(f"Unknown skill prompt fragment id: {fragment_id}") from exc

    def get_prompt_fragments_for_skill(
        self,
        skill_id: str,
    ) -> tuple[SkillPromptFragment, ...]:
        return self._prompt_fragments_by_skill_id.get(skill_id, ())

    def get_skills_for_tag(self, tag: str) -> tuple[SkillManifest, ...]:
        return self._skills_by_tag.get(tag.strip().lower(), ())

    def get_skills_for_tool_name(self, tool_name: str) -> tuple[SkillManifest, ...]:
        return self._skills_by_tool_name.get(tool_name.strip(), ())


@lru_cache(maxsize=1)
def load_skill_registry() -> SkillRegistry:
    """Load the shipped internal skills registry without activating any skill."""
    return SkillRegistry(skills=load_internal_skill_manifests())


__all__ = ["SkillRegistry", "load_skill_registry"]
