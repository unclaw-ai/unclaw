"""Tests for Phase 4: on-demand full skill loading.

Success criteria:
- select_skill_for_turn returns None with no bundles, empty message, or no match
- select_skill_for_turn returns a bundle when the user message matches key terms
- only the first matching bundle is returned
- _resolve_full_skill_content_for_turn returns "" when no match
- full SKILL.md is injected as a dedicated SYSTEM message when relevant
- compact catalog is still present alongside full-skill injection
- at most one full skill is injected per turn
- unknown skill id produces no injection and no crash
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.core.context_builder import build_context_messages
from unclaw.llm.base import LLMRole
from unclaw.skills.selector import select_skill_for_turn

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundle(
    skill_id: str,
    root: Path,
    *,
    summary: str = "A skill.",
) -> object:
    from unclaw.skills.file_models import SkillBundle

    bundle_dir = root / skill_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    skill_md = bundle_dir / "SKILL.md"
    skill_md.write_text(
        f"# {skill_id.title()}\n\n{summary}\n",
        encoding="utf-8",
    )
    return SkillBundle(
        skill_id=skill_id,
        bundle_dir=bundle_dir,
        skill_md_path=skill_md,
        display_name=skill_id.title(),
        summary=summary,
    )


def _make_session_manager(
    enabled_skill_ids: tuple[str, ...] = (),
    system_prompt: str = "Base system prompt.",
) -> SimpleNamespace:
    skills = SimpleNamespace(enabled_skill_ids=enabled_skill_ids)
    settings = SimpleNamespace(
        system_prompt=system_prompt,
        model_pack="dev",
        skills=skills,
    )
    return SimpleNamespace(
        settings=settings,
        list_messages=lambda session_id: [],
    )


def _minimal_capability_summary() -> RuntimeCapabilitySummary:
    return RuntimeCapabilitySummary(
        available_builtin_tool_names=("system_info",),
        local_file_read_available=True,
        local_directory_listing_available=False,
        url_fetch_available=False,
        web_search_available=False,
        system_info_available=False,
        memory_summary_available=False,
        model_can_call_tools=True,
    )


# ---------------------------------------------------------------------------
# Tests: select_skill_for_turn
# ---------------------------------------------------------------------------


class TestSelectSkillForTurn:
    def test_returns_none_when_no_active_bundles(self) -> None:
        assert select_skill_for_turn("what's the weather?", []) is None

    def test_returns_none_when_message_is_empty(self, tmp_path: Path) -> None:
        bundle = _make_bundle("weather", tmp_path, summary="Live weather.")
        assert select_skill_for_turn("", [bundle]) is None

    def test_returns_none_when_message_is_whitespace(self, tmp_path: Path) -> None:
        bundle = _make_bundle("weather", tmp_path, summary="Live weather.")
        assert select_skill_for_turn("   ", [bundle]) is None

    def test_returns_none_when_no_term_matches(self, tmp_path: Path) -> None:
        bundle = _make_bundle("weather", tmp_path, summary="Live weather.")
        assert select_skill_for_turn("tell me a joke", [bundle]) is None

    def test_returns_bundle_on_skill_id_match(self, tmp_path: Path) -> None:
        bundle = _make_bundle("weather", tmp_path, summary="Live weather.")
        result = select_skill_for_turn("what is the weather in Paris?", [bundle])
        assert result is bundle

    def test_match_is_case_insensitive(self, tmp_path: Path) -> None:
        bundle = _make_bundle("weather", tmp_path, summary="Live weather.")
        result = select_skill_for_turn("What is the WEATHER today?", [bundle])
        assert result is bundle

    def test_returns_bundle_on_display_name_word_match(self, tmp_path: Path) -> None:
        bundle = _make_bundle("cooking", tmp_path, summary="Recipe guidance.")
        result = select_skill_for_turn("cooking instructions for pasta", [bundle])
        assert result is bundle

    def test_returns_first_matching_bundle(self, tmp_path: Path) -> None:
        bundle_a = _make_bundle("weather", tmp_path / "a", summary="Live weather.")
        bundle_b = _make_bundle("cooking", tmp_path / "b", summary="Recipe guidance.")
        result = select_skill_for_turn("weather and cooking tips", [bundle_a, bundle_b])
        assert result is bundle_a

    def test_returns_second_bundle_when_first_does_not_match(self, tmp_path: Path) -> None:
        bundle_a = _make_bundle("weather", tmp_path / "a", summary="Live weather.")
        bundle_b = _make_bundle("cooking", tmp_path / "b", summary="Recipe guidance.")
        result = select_skill_for_turn("cooking pasta tonight", [bundle_a, bundle_b])
        assert result is bundle_b

    def test_unknown_skill_id_term_does_not_crash(self, tmp_path: Path) -> None:
        bundle = _make_bundle("weather", tmp_path, summary="Live weather.")
        result = select_skill_for_turn("completely unrelated message", [bundle])
        assert result is None


# ---------------------------------------------------------------------------
# Tests: context_builder integration
# ---------------------------------------------------------------------------


class TestFullSkillInjectionInContext:
    def _base_patches(self, monkeypatch, *, catalog: str = "") -> None:
        monkeypatch.setattr(
            "unclaw.core.context_builder.build_active_skill_catalog",
            lambda **kwargs: catalog,
        )

    def test_no_full_skill_injected_when_no_skills_enabled(self, monkeypatch) -> None:
        self._base_patches(monkeypatch)
        session_manager = _make_session_manager(enabled_skill_ids=())
        messages = build_context_messages(
            session_manager=session_manager,
            session_id="s1",
            user_message="Hello.",
            capability_summary=_minimal_capability_summary(),
            model_profile_name="native",
        )
        assert not any("# Weather" in m.content for m in messages)

    def test_no_full_skill_when_message_does_not_match(self, monkeypatch) -> None:
        self._base_patches(
            monkeypatch,
            catalog="Active optional skills:\n- weather: Live weather.",
        )
        monkeypatch.setattr(
            "unclaw.core.context_builder._resolve_full_skill_content_for_turn",
            lambda **kwargs: "",
        )
        session_manager = _make_session_manager(enabled_skill_ids=("weather",))
        messages = build_context_messages(
            session_manager=session_manager,
            session_id="s1",
            user_message="Tell me a joke.",
            capability_summary=_minimal_capability_summary(),
            model_profile_name="native",
        )
        assert not any("Full skill content" in m.content for m in messages)

    def test_full_skill_injected_as_system_message_when_relevant(self, monkeypatch) -> None:
        full_content = "# Weather\n\nFull skill content. Use get_weather.\n"
        self._base_patches(
            monkeypatch,
            catalog="Active optional skills:\n- weather: Live weather.",
        )
        monkeypatch.setattr(
            "unclaw.core.context_builder._resolve_full_skill_content_for_turn",
            lambda **kwargs: full_content,
        )
        session_manager = _make_session_manager(enabled_skill_ids=("weather",))
        messages = build_context_messages(
            session_manager=session_manager,
            session_id="s1",
            user_message="What is the weather in Paris?",
            capability_summary=_minimal_capability_summary(),
            model_profile_name="native",
        )
        system_msgs = [m for m in messages if m.role is LLMRole.SYSTEM]
        full_skill_msgs = [m for m in system_msgs if "Full skill content" in m.content]
        assert len(full_skill_msgs) == 1

    def test_compact_catalog_still_present_when_full_skill_injected(self, monkeypatch) -> None:
        full_content = "# Weather\n\nFull skill content.\n"
        catalog = "Active optional skills:\n- weather: Live weather."
        self._base_patches(monkeypatch, catalog=catalog)
        monkeypatch.setattr(
            "unclaw.core.context_builder._resolve_full_skill_content_for_turn",
            lambda **kwargs: full_content,
        )
        session_manager = _make_session_manager(enabled_skill_ids=("weather",))
        messages = build_context_messages(
            session_manager=session_manager,
            session_id="s1",
            user_message="What is the weather?",
            capability_summary=_minimal_capability_summary(),
            model_profile_name="native",
        )
        system_msgs = [m for m in messages if m.role is LLMRole.SYSTEM]
        catalog_msgs = [m for m in system_msgs if "Active optional skills" in m.content]
        full_skill_msgs = [m for m in system_msgs if "Full skill content" in m.content]
        assert len(catalog_msgs) == 1, "compact catalog must appear"
        assert len(full_skill_msgs) == 1, "full skill must appear"
        assert catalog_msgs[0] is not full_skill_msgs[0], "must be distinct messages"

    def test_full_skill_message_appears_after_catalog_message(self, monkeypatch) -> None:
        full_content = "# Weather\n\nFull skill content.\n"
        catalog = "Active optional skills:\n- weather: Live weather."
        self._base_patches(monkeypatch, catalog=catalog)
        monkeypatch.setattr(
            "unclaw.core.context_builder._resolve_full_skill_content_for_turn",
            lambda **kwargs: full_content,
        )
        session_manager = _make_session_manager(enabled_skill_ids=("weather",))
        messages = build_context_messages(
            session_manager=session_manager,
            session_id="s1",
            user_message="What is the weather?",
            capability_summary=_minimal_capability_summary(),
            model_profile_name="native",
        )
        system_msgs = [m for m in messages if m.role is LLMRole.SYSTEM]
        indexes = {m.content[:20]: i for i, m in enumerate(system_msgs)}
        catalog_idx = next(
            i for i, m in enumerate(system_msgs) if "Active optional skills" in m.content
        )
        full_skill_idx = next(
            i for i, m in enumerate(system_msgs) if "Full skill content" in m.content
        )
        assert full_skill_idx > catalog_idx, "full skill must follow catalog"

    def test_resolver_called_exactly_once_per_turn(self, monkeypatch) -> None:
        full_content = "# Weather\n\nFull skill content.\n"
        self._base_patches(
            monkeypatch,
            catalog="Active optional skills:\n- weather: Live weather.",
        )
        call_count: list[int] = []

        def _mock_resolver(**kwargs: object) -> str:
            call_count.append(1)
            return full_content

        monkeypatch.setattr(
            "unclaw.core.context_builder._resolve_full_skill_content_for_turn",
            _mock_resolver,
        )
        session_manager = _make_session_manager(enabled_skill_ids=("weather",))
        build_context_messages(
            session_manager=session_manager,
            session_id="s1",
            user_message="Weather in London?",
            capability_summary=_minimal_capability_summary(),
            model_profile_name="native",
        )
        assert len(call_count) == 1

    def test_stable_when_skill_id_unknown(self, monkeypatch) -> None:
        self._base_patches(monkeypatch)
        monkeypatch.setattr(
            "unclaw.core.context_builder._resolve_full_skill_content_for_turn",
            lambda **kwargs: "",
        )
        session_manager = _make_session_manager(enabled_skill_ids=("unknown.skill",))
        messages = build_context_messages(
            session_manager=session_manager,
            session_id="s1",
            user_message="Hello.",
            capability_summary=_minimal_capability_summary(),
            model_profile_name="native",
        )
        assert messages is not None
        assert not any("Full skill content" in m.content for m in messages)


# ---------------------------------------------------------------------------
# Integration tests: real shipped skill bundles
# ---------------------------------------------------------------------------


class TestOnDemandIntegration:
    def test_weather_bundle_selected_for_weather_query(self) -> None:
        from unclaw.skills.file_loader import load_active_skill_bundles

        active_bundles = load_active_skill_bundles(enabled_skill_ids=("weather",))
        result = select_skill_for_turn("What is the weather in Paris?", active_bundles)
        assert result is not None
        assert result.skill_id == "weather"

    def test_weather_bundle_not_selected_for_unrelated_query(self) -> None:
        from unclaw.skills.file_loader import load_active_skill_bundles

        active_bundles = load_active_skill_bundles(enabled_skill_ids=("weather",))
        result = select_skill_for_turn("Tell me a joke.", active_bundles)
        assert result is None

    def test_selected_weather_bundle_raw_content_loads(self) -> None:
        from unclaw.skills.file_loader import load_active_skill_bundles

        active_bundles = load_active_skill_bundles(enabled_skill_ids=("weather",))
        result = select_skill_for_turn("What is the weather in Paris?", active_bundles)
        assert result is not None
        content = result.load_raw_content()
        assert content.strip()
        assert "# Weather" in content
