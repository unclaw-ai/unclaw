"""On-demand skill selection for the current turn."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from unclaw.skills.file_models import SkillBundle


def select_skill_for_turn(
    user_message: str,
    active_bundles: Sequence[SkillBundle],
) -> SkillBundle | None:
    """Return the first active bundle whose key terms appear in user_message.

    Selection is lightweight, deterministic, and local — no network calls,
    no NLP, no model inference. Returns None when no bundle matches, keeping
    normal turns cheap.

    At most one bundle is returned. Load cost is zero until the caller reads
    the bundle's SKILL.md via ``bundle.load_raw_content()``.
    """
    if not active_bundles or not user_message.strip():
        return None

    normalised = user_message.lower()
    for bundle in active_bundles:
        if any(term in normalised for term in _key_terms(bundle)):
            return bundle
    return None


@lru_cache(maxsize=128)
def _key_terms(bundle: SkillBundle) -> tuple[str, ...]:
    """Return lowercase match terms derived from the bundle's id and display name.

    Cached per bundle instance: SkillBundle is frozen (immutable + hashable) so
    the result never changes for a given bundle. Avoids repeated string work on
    every turn when the same set of active bundles is used across requests.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for raw in (bundle.skill_id, bundle.display_name):
        for word in raw.lower().split():
            cleaned = word.strip(".,;:!?\"'()-")
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                terms.append(cleaned)
    return tuple(terms)


__all__ = ["select_skill_for_turn"]
