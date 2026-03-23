"""Tests for the 3-layer research pipeline, budgets, and fast_web_search."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from unclaw.tools.contracts import ToolCall
from unclaw.tools.web_research import (
    DEEP_RESEARCH_BUDGET,
    FAST_RESEARCH_BUDGET,
    MAIN_RESEARCH_BUDGET,
    MergedResearchNote,
    ResearchBudget,
    ResearchWorkspace,
    SourceArtifact,
    SourceNote,
    build_fast_grounding_note,
    condense_source,
    format_research_output,
    merge_source_notes,
    resolve_research_budget,
    run_research_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(
    *,
    url: str = "https://example.com/page",
    title: str = "Example Page",
    text: str = "Some content about the topic.",
    fetch_success: bool = True,
) -> SourceArtifact:
    return SourceArtifact(
        url=url,
        title=title,
        cleaned_text=text,
        fetch_success=fetch_success,
    )


def _make_source_note(
    *,
    url: str = "https://example.com/page",
    title: str = "Example Page",
    text: str = "Condensed research note.",
    model_generated: bool = False,
) -> SourceNote:
    return SourceNote(
        url=url,
        title=title,
        condensed_text=text,
        model_generated=model_generated,
    )


def _make_fetched_pages(count: int) -> dict[str, tuple[str, str]]:
    return {
        f"https://example.com/{i}": (f"Source {i}", f"Content for source {i}. " * 20)
        for i in range(count)
    }


# ---------------------------------------------------------------------------
# Budget system
# ---------------------------------------------------------------------------


class TestResearchBudget:
    def test_pre_computed_tiers_have_sane_values(self) -> None:
        assert FAST_RESEARCH_BUDGET.max_sources == 3
        assert MAIN_RESEARCH_BUDGET.max_sources == 5
        assert DEEP_RESEARCH_BUDGET.max_sources == 8

        assert FAST_RESEARCH_BUDGET.max_source_chars < MAIN_RESEARCH_BUDGET.max_source_chars
        assert MAIN_RESEARCH_BUDGET.max_source_chars < DEEP_RESEARCH_BUDGET.max_source_chars

        assert FAST_RESEARCH_BUDGET.max_merged_note_chars < DEEP_RESEARCH_BUDGET.max_merged_note_chars

    def test_resolve_budget_fast_profile(self) -> None:
        budget = resolve_research_budget(effective_context=4096, profile_name="fast")
        assert budget.max_sources == 3
        assert budget.fast_grounding_max_results == 2

    def test_resolve_budget_main_profile(self) -> None:
        budget = resolve_research_budget(effective_context=8192, profile_name="main")
        assert budget.max_sources == 5
        assert budget.fast_grounding_max_results == 3

    def test_resolve_budget_deep_profile(self) -> None:
        budget = resolve_research_budget(effective_context=16384, profile_name="deep")
        assert budget.max_sources == 8
        assert budget.fast_grounding_max_results == 3

    def test_resolve_budget_codex_uses_fast_tier(self) -> None:
        budget = resolve_research_budget(effective_context=4096, profile_name="codex")
        assert budget.max_sources == 3

    def test_resolve_budget_unknown_profile_uses_main_defaults(self) -> None:
        budget = resolve_research_budget(effective_context=8192, profile_name="custom")
        assert budget.max_sources == 5

    def test_budget_adapts_to_context_window(self) -> None:
        small = resolve_research_budget(effective_context=4000, profile_name="main")
        large = resolve_research_budget(effective_context=32000, profile_name="main")
        assert small.max_source_chars <= large.max_source_chars
        assert small.max_merged_note_chars <= large.max_merged_note_chars

    def test_budget_clamps_with_tiny_context(self) -> None:
        budget = resolve_research_budget(effective_context=500, profile_name="main")
        # Should hit the minimum clamp values, not go below them
        assert budget.max_source_chars >= 1500
        assert budget.max_merged_note_chars >= 400

    def test_budget_clamps_with_huge_context(self) -> None:
        budget = resolve_research_budget(effective_context=200_000, profile_name="main")
        # Should hit the maximum clamp values, not exceed them
        assert budget.max_source_chars <= 5000
        assert budget.max_merged_note_chars <= 1000

    def test_profile_name_case_insensitive(self) -> None:
        budget_lower = resolve_research_budget(effective_context=8192, profile_name="deep")
        budget_upper = resolve_research_budget(effective_context=8192, profile_name="DEEP")
        assert budget_lower.max_sources == budget_upper.max_sources
        assert budget_lower.max_source_chars == budget_upper.max_source_chars


# ---------------------------------------------------------------------------
# Research workspace
# ---------------------------------------------------------------------------


class TestResearchWorkspace:
    def test_workspace_enforces_source_budget(self) -> None:
        budget = ResearchBudget(
            max_sources=2,
            max_source_chars=1000,
            max_source_note_chars=200,
            max_merged_note_chars=400,
            condensation_timeout_seconds=5.0,
        )
        workspace = ResearchWorkspace(query="test", budget=budget)

        assert workspace.add_source(_make_artifact(url="https://a.com"))
        assert workspace.add_source(_make_artifact(url="https://b.com"))
        assert not workspace.add_source(_make_artifact(url="https://c.com"))

        assert workspace.source_count == 2

    def test_workspace_properties(self) -> None:
        workspace = ResearchWorkspace(query="test", budget=MAIN_RESEARCH_BUDGET)
        assert workspace.source_count == 0
        assert not workspace.has_merged_note

        workspace.merged_note = MergedResearchNote(
            text="Note", source_count=0, model_generated=False
        )
        assert workspace.has_merged_note


# ---------------------------------------------------------------------------
# Layer B — per-source condensation (deterministic fallback)
# ---------------------------------------------------------------------------


class TestCondenseSource:
    def test_deterministic_fallback_produces_source_note(self) -> None:
        artifact = _make_artifact(text="This is important factual content about AI.")
        note = condense_source(
            artifact=artifact,
            query="AI facts",
            budget=MAIN_RESEARCH_BUDGET,
        )
        assert isinstance(note, SourceNote)
        assert note.url == artifact.url
        assert note.title == artifact.title
        assert not note.model_generated
        assert len(note.condensed_text) > 0

    def test_deterministic_fallback_respects_budget(self) -> None:
        long_text = "word " * 5000
        artifact = _make_artifact(text=long_text)
        budget = ResearchBudget(
            max_sources=5,
            max_source_chars=100,
            max_source_note_chars=50,
            max_merged_note_chars=200,
            condensation_timeout_seconds=5.0,
        )
        note = condense_source(artifact=artifact, query="test", budget=budget)
        assert len(note.condensed_text) <= 50

    def test_empty_text_produces_placeholder(self) -> None:
        artifact = _make_artifact(text="")
        note = condense_source(
            artifact=artifact,
            query="test",
            budget=MAIN_RESEARCH_BUDGET,
        )
        assert note.condensed_text  # Should not be empty
        assert not note.model_generated

    def test_deterministic_source_note_is_dense_and_header_free(self) -> None:
        artifact = _make_artifact(
            text=(
                "Marine Leleu is a French endurance athlete. "
                "She creates fitness content. "
                "She completes long-distance challenges."
            )
        )
        note = condense_source(
            artifact=artifact,
            query="Marine Leleu",
            budget=MAIN_RESEARCH_BUDGET,
        )

        assert "Source:" not in note.condensed_text
        assert "URL:" not in note.condensed_text
        assert "\n" not in note.condensed_text
        assert ";" in note.condensed_text or len(note.condensed_text.split()) <= 12

    @patch("unclaw.tools.web_research._call_condensation_model")
    def test_model_driven_condensation_when_provider_available(
        self, mock_call: MagicMock
    ) -> None:
        artifact = _make_artifact(text="Long factual content about renewable energy.")
        mock_call.return_value = "Renewable energy: key facts from source."

        note = condense_source(
            artifact=artifact,
            query="renewable energy",
            budget=MAIN_RESEARCH_BUDGET,
            provider=MagicMock(),
            profile=MagicMock(),
        )

        assert note.model_generated
        assert "Renewable energy" in note.condensed_text
        mock_call.assert_called_once()

    @patch("unclaw.tools.web_research._call_condensation_model")
    def test_model_failure_falls_back_to_deterministic(
        self, mock_call: MagicMock
    ) -> None:
        artifact = _make_artifact(text="Content about machine learning algorithms.")
        mock_call.return_value = None  # Simulate model failure

        note = condense_source(
            artifact=artifact,
            query="ML",
            budget=MAIN_RESEARCH_BUDGET,
            provider=MagicMock(),
            profile=MagicMock(),
        )

        assert not note.model_generated
        assert len(note.condensed_text) > 0


# ---------------------------------------------------------------------------
# Layer C — merged research note
# ---------------------------------------------------------------------------


class TestMergeSourceNotes:
    def test_empty_notes_produces_fallback(self) -> None:
        merged = merge_source_notes(
            source_notes=[],
            query="test",
            budget=MAIN_RESEARCH_BUDGET,
        )
        assert "No sources" in merged.text
        assert merged.source_count == 0
        assert not merged.model_generated

    def test_deterministic_merge_uses_compact_source_refs(self) -> None:
        notes = [
            _make_source_note(title="Source A", text="Fact A about quantum computing."),
            _make_source_note(title="Source B", text="Fact B about quantum computing."),
        ]
        merged = merge_source_notes(
            source_notes=notes,
            query="quantum computing",
            budget=MAIN_RESEARCH_BUDGET,
        )
        assert merged.source_count == 2
        assert not merged.model_generated
        assert "[1]" in merged.text
        assert "[2]" in merged.text
        assert "Fact A about quantum computing" in merged.text
        assert "Fact B about quantum computing" in merged.text

    def test_deterministic_merge_respects_budget(self) -> None:
        notes = [
            _make_source_note(title="S1", text="A" * 200),
            _make_source_note(title="S2", text="B" * 200),
            _make_source_note(title="S3", text="C" * 200),
        ]
        budget = ResearchBudget(
            max_sources=5,
            max_source_chars=1000,
            max_source_note_chars=200,
            max_merged_note_chars=100,
            condensation_timeout_seconds=5.0,
        )
        merged = merge_source_notes(
            source_notes=notes,
            query="test",
            budget=budget,
        )
        assert len(merged.text) <= 100

    def test_conflicting_sources_both_visible(self) -> None:
        notes = [
            _make_source_note(title="Pro Source", text="Solar energy is cost effective."),
            _make_source_note(title="Con Source", text="Solar energy is expensive to install."),
        ]
        merged = merge_source_notes(
            source_notes=notes,
            query="solar energy cost",
            budget=MAIN_RESEARCH_BUDGET,
        )
        assert "!=" in merged.text
        assert "cost effective" in merged.text
        assert "expensive to install" in merged.text
        assert "[1]" in merged.text
        assert "[2]" in merged.text


# ---------------------------------------------------------------------------
# Full pipeline (deterministic — no LLM)
# ---------------------------------------------------------------------------


class TestRunResearchPipeline:
    def test_pipeline_produces_complete_workspace(self) -> None:
        fetched_pages = _make_fetched_pages(3)
        workspace = run_research_pipeline(
            query="climate change",
            fetched_pages=fetched_pages,
            budget=MAIN_RESEARCH_BUDGET,
        )

        assert workspace.source_count == 3
        assert len(workspace.source_notes) == 3
        assert workspace.has_merged_note
        assert workspace.merged_note is not None
        assert workspace.merged_note.source_count == 3

    def test_pipeline_respects_max_sources(self) -> None:
        fetched_pages = _make_fetched_pages(10)
        budget = ResearchBudget(
            max_sources=3,
            max_source_chars=1000,
            max_source_note_chars=200,
            max_merged_note_chars=400,
            condensation_timeout_seconds=5.0,
        )
        workspace = run_research_pipeline(
            query="test",
            fetched_pages=fetched_pages,
            budget=budget,
        )
        assert workspace.source_count == 3
        assert len(workspace.source_notes) == 3

    def test_pipeline_with_empty_pages(self) -> None:
        workspace = run_research_pipeline(
            query="test",
            fetched_pages={},
            budget=MAIN_RESEARCH_BUDGET,
        )
        assert workspace.source_count == 0
        assert workspace.has_merged_note
        assert workspace.merged_note is not None
        assert "No sources" in workspace.merged_note.text

    def test_pipeline_with_weak_sources(self) -> None:
        fetched_pages = {
            "https://example.com/empty": ("Empty Page", ""),
            "https://example.com/real": ("Real Page", "Good content here."),
        }
        workspace = run_research_pipeline(
            query="test",
            fetched_pages=fetched_pages,
            budget=MAIN_RESEARCH_BUDGET,
        )
        assert workspace.source_count == 2
        assert workspace.has_merged_note
        # Should not crash — weak/empty sources produce deterministic fallback

    def test_pipeline_deterministic_without_config(self) -> None:
        fetched_pages = _make_fetched_pages(2)
        workspace = run_research_pipeline(
            query="test",
            fetched_pages=fetched_pages,
            budget=MAIN_RESEARCH_BUDGET,
            config=None,
        )
        # All condensation should be deterministic
        assert all(not note.model_generated for note in workspace.source_notes)
        assert workspace.merged_note is not None
        assert not workspace.merged_note.model_generated


# ---------------------------------------------------------------------------
# Fast grounding probe
# ---------------------------------------------------------------------------


class TestBuildFastGroundingNote:
    def test_empty_snippets_returns_no_results_message(self) -> None:
        note = build_fast_grounding_note(
            query="Marine Leleu",
            snippets=[],
            max_chars=300,
        )
        assert "Marine Leleu" in note
        assert "No web results" in note

    def test_produces_compact_grounding_note(self) -> None:
        snippets = [
            ("Marine Leleu - Athlete", "https://example.com/1", "French fitness athlete and content creator."),
            ("Marine Leleu Bio", "https://example.com/2", "Known for extreme sports challenges."),
        ]
        note = build_fast_grounding_note(
            query="Marine Leleu",
            snippets=snippets,
            max_chars=300,
        )
        assert "Marine Leleu" in note
        assert "Athlete" in note or "fitness" in note
        assert len(note) <= 300

    def test_respects_max_chars(self) -> None:
        snippets = [
            ("Title 1", "url1", "Very long snippet text. " * 50),
            ("Title 2", "url2", "Another long snippet. " * 50),
            ("Title 3", "url3", "Yet another snippet. " * 50),
        ]
        note = build_fast_grounding_note(
            query="test",
            snippets=snippets,
            max_chars=100,
        )
        assert len(note) <= 100

    def test_entity_mismatch_prevention_flow(self) -> None:
        """fast_web_search should help distinguish similar-sounding entities."""
        snippets_correct = [
            ("Marine Leleu", "url1", "French fitness athlete known for extreme sports."),
        ]
        snippets_wrong = [
            ("Marine Le Pen", "url1", "French politician and leader of Rassemblement National."),
        ]
        note_correct = build_fast_grounding_note(
            query="Marine Leleu", snippets=snippets_correct, max_chars=300
        )
        note_wrong = build_fast_grounding_note(
            query="Marine Leleu", snippets=snippets_wrong, max_chars=300
        )
        # The grounding note should reflect the actual snippet content,
        # letting the model decide whether the entity matches
        assert "fitness" in note_correct or "athlete" in note_correct or "sports" in note_correct
        assert "politician" in note_wrong or "Rassemblement" in note_wrong or "Le Pen" in note_wrong

    def test_marks_when_no_exact_entity_match_is_found(self) -> None:
        note = build_fast_grounding_note(
            query="Marine Leleu",
            snippets=[
                (
                    "Marine Le Pen",
                    "https://example.com/le-pen",
                    "French politician and leader of Rassemblement National.",
                ),
            ],
            max_chars=300,
        )

        assert "no exact top match" in note
        assert "!=" in note


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestFormatResearchOutput:
    def test_format_includes_query_and_stats(self) -> None:
        workspace = ResearchWorkspace(query="AI trends", budget=MAIN_RESEARCH_BUDGET)
        workspace.merged_note = MergedResearchNote(
            text="AI is advancing rapidly.", source_count=2, model_generated=False
        )
        workspace.source_notes = [
            _make_source_note(title="Source 1", url="https://a.com"),
            _make_source_note(title="Source 2", url="https://b.com"),
        ]

        output = format_research_output(
            workspace=workspace,
            query="AI trends",
            fetch_stats={"considered_candidate_count": 10, "fetch_attempt_count": 5, "fetch_success_count": 3},
        )

        assert "AI trends" in output
        assert "Source 1" in output
        assert "Source 2" in output
        assert "https://a.com" in output
        assert "Research note:" in output

    def test_format_without_merged_note(self) -> None:
        workspace = ResearchWorkspace(query="test", budget=MAIN_RESEARCH_BUDGET)
        output = format_research_output(workspace=workspace, query="test")
        assert "No research note" in output

    def test_format_indicates_condensation_mode(self) -> None:
        workspace = ResearchWorkspace(query="test", budget=MAIN_RESEARCH_BUDGET)
        workspace.merged_note = MergedResearchNote(
            text="Note", source_count=1, model_generated=False
        )
        output = format_research_output(workspace=workspace, query="test")
        assert "deterministic" in output

        workspace_model = ResearchWorkspace(query="test", budget=MAIN_RESEARCH_BUDGET)
        workspace_model.merged_note = MergedResearchNote(
            text="Note", source_count=1, model_generated=True
        )
        output_model = format_research_output(workspace=workspace_model, query="test")
        assert "model-driven" in output_model


# ---------------------------------------------------------------------------
# fast_web_search tool handler
# ---------------------------------------------------------------------------


class TestFastWebSearchTool:
    def test_missing_query_returns_failure(self) -> None:
        from unclaw.tools.web_tools import fast_web_search

        call = ToolCall(tool_name="fast_web_search", arguments={})
        result = fast_web_search(call)
        assert not result.success
        assert "query" in result.error.lower()

    def test_empty_query_returns_failure(self) -> None:
        from unclaw.tools.web_tools import fast_web_search

        call = ToolCall(tool_name="fast_web_search", arguments={"query": "  "})
        result = fast_web_search(call)
        assert not result.success

    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    def test_successful_grounding_probe(
        self, mock_parse: MagicMock, mock_search: MagicMock
    ) -> None:
        from unclaw.tools.web_tools import fast_web_search

        mock_search.return_value = "<html>results</html>"
        mock_parse.return_value = [
            {"title": "Marine Leleu", "url": "https://example.com/1", "snippet": "French fitness athlete."},
        ]

        call = ToolCall(tool_name="fast_web_search", arguments={"query": "Marine Leleu"})
        result = fast_web_search(call)

        assert result.success
        assert "Marine Leleu" in result.output_text
        assert result.payload is not None
        assert result.payload["query"] == "Marine Leleu"
        assert result.payload["result_count"] >= 1
        assert "grounding_note" in result.payload

    @patch("unclaw.tools.web_tools._search_public_web")
    def test_network_error_returns_failure(self, mock_search: MagicMock) -> None:
        from unclaw.tools.web_tools import fast_web_search

        mock_search.side_effect = OSError("Network down")

        call = ToolCall(tool_name="fast_web_search", arguments={"query": "test"})
        result = fast_web_search(call)

        assert not result.success
        assert "Could not search" in result.error


# ---------------------------------------------------------------------------
# Capability fragment registration
# ---------------------------------------------------------------------------


class TestCapabilityFragments:
    def test_fast_web_search_fragment_registered(self) -> None:
        from unclaw.core.capability_fragments import load_builtin_capability_fragment_registry

        registry = load_builtin_capability_fragment_registry()
        fragment = registry.get_fragment("available.fast_web_search")
        assert fragment is not None
        assert fragment.kind.value == "available_tool"

    def test_fast_web_grounding_guidance_registered(self) -> None:
        from unclaw.core.capability_fragments import load_builtin_capability_fragment_registry

        registry = load_builtin_capability_fragment_registry()
        fragment = registry.get_fragment("guidance.model_callable.fast_web_grounding")
        assert fragment is not None
        assert fragment.kind.value == "guidance"

    def test_fast_web_search_fragment_activates_when_available(self) -> None:
        from unclaw.core.capability_fragments import load_builtin_capability_fragment_registry

        registry = load_builtin_capability_fragment_registry()

        class FakeSummary:
            available_builtin_tool_names = ("fast_web_search",)
            local_file_read_available = False
            local_directory_listing_available = False
            url_fetch_available = False
            web_search_available = False
            system_info_available = False
            shell_command_execution_available = False
            memory_summary_available = False
            model_can_call_tools = False
            local_file_write_available = False
            session_history_recall_available = False
            long_term_memory_available = False
            fast_web_search_available = True

        fragment = registry.get_fragment("available.fast_web_search")
        assert fragment.matches(FakeSummary())

    def test_fast_web_search_fragment_inactive_when_unavailable(self) -> None:
        from unclaw.core.capability_fragments import load_builtin_capability_fragment_registry

        registry = load_builtin_capability_fragment_registry()

        class FakeSummary:
            available_builtin_tool_names = ()
            local_file_read_available = False
            local_directory_listing_available = False
            url_fetch_available = False
            web_search_available = False
            system_info_available = False
            shell_command_execution_available = False
            memory_summary_available = False
            model_can_call_tools = False
            local_file_write_available = False
            session_history_recall_available = False
            long_term_memory_available = False
            fast_web_search_available = False

        fragment = registry.get_fragment("available.fast_web_search")
        assert not fragment.matches(FakeSummary())


# ---------------------------------------------------------------------------
# RuntimeCapabilitySummary integration
# ---------------------------------------------------------------------------


class TestCapabilitySummaryIntegration:
    def test_fast_web_search_detected_in_summary(self) -> None:
        from unclaw.core.capabilities import build_runtime_capability_summary
        from unclaw.tools.registry import ToolRegistry
        from unclaw.tools.web_tools import FAST_WEB_SEARCH_DEFINITION

        registry = ToolRegistry()
        registry.register(FAST_WEB_SEARCH_DEFINITION, lambda call: None)

        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
        )
        assert summary.fast_web_search_available

    def test_fast_web_search_not_detected_when_missing(self) -> None:
        from unclaw.core.capabilities import build_runtime_capability_summary
        from unclaw.tools.registry import ToolRegistry

        registry = ToolRegistry()

        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
        )
        assert not summary.fast_web_search_available


# ---------------------------------------------------------------------------
# search_web remains usable in existing flows
# ---------------------------------------------------------------------------


class TestSearchWebBackwardCompatibility:
    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    @patch("unclaw.tools.web_tools._rank_search_results")
    @patch("unclaw.tools.web_tools._run_iterative_retrieval")
    @patch("unclaw.tools.web_tools._synthesize_search_knowledge")
    def test_search_web_works_without_research_config(
        self,
        mock_synth: MagicMock,
        mock_retrieval: MagicMock,
        mock_rank: MagicMock,
        mock_parse: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        from unclaw.tools.web_tools import search_web

        mock_search.return_value = "<html></html>"
        mock_parse.return_value = [
            {"title": "Result", "url": "https://example.com", "snippet": "Test."}
        ]
        mock_rank.return_value = [
            {"title": "Result", "url": "https://example.com", "snippet": "Test."}
        ]

        # Mock retrieval to return a minimal outcome
        mock_outcome = MagicMock()
        mock_outcome.sources = ()
        mock_outcome.evidence_items = ()
        mock_outcome.initial_result_count = 1
        mock_outcome.considered_candidate_count = 1
        mock_outcome.fetch_attempt_count = 1
        mock_outcome.fetch_success_count = 0
        mock_retrieval.return_value = mock_outcome

        # Mock synthesis
        mock_synth_result = MagicMock()
        mock_synth_result.fact_clusters = ()
        mock_synth_result.findings = ()
        mock_synth_result.statements = ()
        mock_synth.return_value = mock_synth_result

        call = ToolCall(tool_name="search_web", arguments={"query": "test query"})
        result = search_web(call)

        assert result.success
        assert result.tool_name == "search_web"


# ---------------------------------------------------------------------------
# Payload contract types
# ---------------------------------------------------------------------------


class TestPayloadContracts:
    def test_research_source_note_payload_shape(self) -> None:
        from unclaw.tools.contracts import ResearchSourceNotePayload

        payload: ResearchSourceNotePayload = {
            "url": "https://example.com",
            "title": "Test",
            "condensed_text": "Summary.",
            "model_generated": False,
        }
        assert payload["url"] == "https://example.com"
        assert not payload["model_generated"]

    def test_research_merged_note_payload_shape(self) -> None:
        from unclaw.tools.contracts import ResearchMergedNotePayload

        payload: ResearchMergedNotePayload = {
            "text": "Merged note.",
            "source_count": 3,
            "model_generated": True,
        }
        assert payload["source_count"] == 3

    def test_fast_search_payload_shape(self) -> None:
        from unclaw.tools.contracts import FastSearchWebPayload

        payload: FastSearchWebPayload = {
            "query": "test",
            "provider": "duckduckgo",
            "result_count": 2,
            "grounding_note": "Quick grounding for: test",
        }
        assert payload["result_count"] == 2


# ---------------------------------------------------------------------------
# Iterative refine merge (model-driven Layer C)
# ---------------------------------------------------------------------------


class TestMergeSourceNotesIterative:
    """Tests for the iterative refinement path in merge_source_notes."""

    @patch("unclaw.tools.web_research._call_condensation_model")
    def test_iterative_refine_called_for_multiple_sources(
        self, mock_call: MagicMock
    ) -> None:
        """With N sources, the model should be called N-1 times (one per integration step)."""
        mock_call.return_value = "Refined global note."
        notes = [
            _make_source_note(title="S1", text="Alice born 1990 in Paris."),
            _make_source_note(title="S2", text="Alice is a biologist."),
            _make_source_note(title="S3", text="Alice won an award in 2020."),
        ]
        merged = merge_source_notes(
            source_notes=notes,
            query="Alice",
            budget=MAIN_RESEARCH_BUDGET,
            provider=MagicMock(),
            profile=MagicMock(),
        )
        # 3 sources → 2 refine steps
        assert mock_call.call_count == 2
        assert merged.model_generated

    @patch("unclaw.tools.web_research._call_condensation_model")
    def test_iterative_refine_preserves_anchor_note_on_first_failure(
        self, mock_call: MagicMock
    ) -> None:
        """If the model call fails for a step, the current global note is kept unchanged."""
        # First refine call fails, second succeeds
        mock_call.side_effect = [None, "Refined note with S3."]
        notes = [
            _make_source_note(title="S1", text="Alice born 1990 in Paris."),
            _make_source_note(title="S2", text="Alice is a biologist."),
            _make_source_note(title="S3", text="Alice won an award in 2020."),
        ]
        merged = merge_source_notes(
            source_notes=notes,
            query="Alice",
            budget=MAIN_RESEARCH_BUDGET,
            provider=MagicMock(),
            profile=MagicMock(),
        )
        # Second call succeeded → model_generated is True
        assert merged.model_generated
        assert "S3" in merged.text or "award" in merged.text.lower() or "Refined" in merged.text

    @patch("unclaw.tools.web_research._call_condensation_model")
    def test_all_model_calls_fail_falls_back_to_deterministic(
        self, mock_call: MagicMock
    ) -> None:
        """When all iterative refine calls fail, deterministic fallback is used."""
        mock_call.return_value = None
        notes = [
            _make_source_note(title="S1", text="Alice born 1990 in Paris."),
            _make_source_note(title="S2", text="Alice is a biologist."),
        ]
        merged = merge_source_notes(
            source_notes=notes,
            query="Alice",
            budget=MAIN_RESEARCH_BUDGET,
            provider=MagicMock(),
            profile=MagicMock(),
        )
        assert not merged.model_generated
        assert merged.source_count == 2

    @patch("unclaw.tools.web_research._call_condensation_model")
    def test_iterative_refine_conflicts_visible(
        self, mock_call: MagicMock
    ) -> None:
        """The iterative refine prompt explicitly requests conflict marking."""
        # Track the prompt to verify it contains '!= ' instruction
        captured_prompts: list[str] = []

        def capture(*, provider, profile, system_prompt, user_prompt, **kwargs):  # type: ignore[override]
            captured_prompts.append(system_prompt + "\n" + user_prompt)
            return "!= claim_A / claim_B"

        mock_call.side_effect = capture

        notes = [
            _make_source_note(title="S1", text="Solar energy is cost effective."),
            _make_source_note(title="S2", text="Solar energy is expensive to install."),
        ]
        merged = merge_source_notes(
            source_notes=notes,
            query="solar energy cost",
            budget=MAIN_RESEARCH_BUDGET,
            provider=MagicMock(),
            profile=MagicMock(),
        )
        assert "!=" in merged.text
        # Ensure the system prompt instructed conflict marking
        assert any("!= " in p or "contradict" in p.lower() for p in captured_prompts)

    @patch("unclaw.tools.web_research._call_condensation_model")
    def test_single_source_uses_direct_condensation_not_iterative(
        self, mock_call: MagicMock
    ) -> None:
        """With exactly one source, no iterative step is needed."""
        mock_call.return_value = "Single source note."
        notes = [_make_source_note(title="S1", text="Alice born 1990 in Paris.")]
        merged = merge_source_notes(
            source_notes=notes,
            query="Alice",
            budget=MAIN_RESEARCH_BUDGET,
            provider=MagicMock(),
            profile=MagicMock(),
        )
        assert mock_call.call_count == 1
        assert merged.model_generated

    def test_iterative_refine_source_count_matches_input(self) -> None:
        """MergedResearchNote.source_count should equal the number of source notes."""
        notes = [
            _make_source_note(title="S1", text="Alice born 1990 in Paris."),
            _make_source_note(title="S2", text="Alice is a biologist."),
            _make_source_note(title="S3", text="Alice won an award in 2020."),
        ]
        merged = merge_source_notes(
            source_notes=notes,
            query="Alice",
            budget=MAIN_RESEARCH_BUDGET,
            # No provider → deterministic path, but count should still be right
        )
        assert merged.source_count == 3


# ---------------------------------------------------------------------------
# Structured extraction density tests (Layer B)
# ---------------------------------------------------------------------------


class TestStructuredExtractionDensity:
    """Verify the new extraction system prompt produces denser, header-free output."""

    def test_extraction_prompt_requests_no_headers(self) -> None:
        from unclaw.tools.web_research import _SOURCE_CONDENSATION_SYSTEM_PROMPT

        prompt = _SOURCE_CONDENSATION_SYSTEM_PROMPT.lower()
        assert "no headers" in prompt or "no header" in prompt

    def test_extraction_prompt_requests_semicolons(self) -> None:
        from unclaw.tools.web_research import _SOURCE_CONDENSATION_SYSTEM_PROMPT

        assert "semicolon" in _SOURCE_CONDENSATION_SYSTEM_PROMPT.lower()

    def test_extraction_prompt_lists_fact_categories(self) -> None:
        from unclaw.tools.web_research import _SOURCE_CONDENSATION_SYSTEM_PROMPT

        prompt = _SOURCE_CONDENSATION_SYSTEM_PROMPT.lower()
        # Should mention at least some specific fact categories
        category_hints = {"date", "location", "role", "name", "profession", "affiliation"}
        found = sum(1 for c in category_hints if c in prompt)
        assert found >= 3, f"Expected ≥3 fact category hints, found {found}"

    def test_deterministic_extraction_no_metadata_headers(self) -> None:
        """Deterministic source note should contain no 'Source:', 'URL:' headers."""
        artifact = _make_artifact(
            text=(
                "Marine Leleu is a French endurance athlete born in 1990. "
                "She has completed multiple Iron Man triathlons. "
                "She posts fitness content online."
            )
        )
        note = condense_source(
            artifact=artifact,
            query="Marine Leleu",
            budget=MAIN_RESEARCH_BUDGET,
        )
        assert "Source:" not in note.condensed_text
        assert "URL:" not in note.condensed_text
        assert "Title:" not in note.condensed_text

    @patch("unclaw.tools.web_research._call_condensation_model")
    def test_model_extraction_preserves_specific_facts(
        self, mock_call: MagicMock
    ) -> None:
        """Model extraction should be called with source text and return facts."""
        mock_call.return_value = (
            "Marine Leleu; born 1990; French endurance athlete; "
            "Iron Man finisher; fitness content creator"
        )
        artifact = _make_artifact(
            text=(
                "Marine Leleu, born in 1990, is a French endurance athlete who "
                "has finished multiple Iron Man triathlons and creates fitness content."
            )
        )
        note = condense_source(
            artifact=artifact,
            query="Marine Leleu",
            budget=MAIN_RESEARCH_BUDGET,
            provider=MagicMock(),
            profile=MagicMock(),
        )
        assert note.model_generated
        # Core facts should survive sanitization
        assert "Marine Leleu" in note.condensed_text or "1990" in note.condensed_text

    def test_fast_budget_source_note_chars_adequate_for_extraction(self) -> None:
        """FAST_RESEARCH_BUDGET max_source_note_chars should be ≥ 200."""
        from unclaw.tools.web_research import FAST_RESEARCH_BUDGET

        assert FAST_RESEARCH_BUDGET.max_source_note_chars >= 200

    def test_fast_budget_merged_note_chars_adequate(self) -> None:
        """FAST_RESEARCH_BUDGET max_merged_note_chars should be ≥ 250."""
        from unclaw.tools.web_research import FAST_RESEARCH_BUDGET

        assert FAST_RESEARCH_BUDGET.max_merged_note_chars >= 250

    def test_resolve_fast_budget_note_chars_adequate(self) -> None:
        """Dynamic fast budget for a 4096-ctx model should give ≥ 200 chars per note."""
        budget = resolve_research_budget(effective_context=4096, profile_name="fast")
        assert budget.max_source_note_chars >= 200

    def test_resolve_fast_budget_larger_than_old_floor(self) -> None:
        """New fast budget note chars floor (200) is higher than the old floor (150)."""
        budget = resolve_research_budget(effective_context=4096, profile_name="fast")
        assert budget.max_source_note_chars >= 200  # was 150 before


# ---------------------------------------------------------------------------
# Workspace: fast_web_search stays memory-only, search_web stays file-backed
# ---------------------------------------------------------------------------


class TestWorkspacePersistenceBoundary:
    """Ensure memory-only vs file-backed boundary is respected."""

    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    def test_fast_web_search_has_no_workspace_dir(
        self, mock_parse: MagicMock, mock_search: MagicMock
    ) -> None:
        """fast_web_search should never produce a workspace_dir in its payload."""
        from unclaw.tools.web_tools import fast_web_search

        mock_search.return_value = "<html></html>"
        mock_parse.return_value = [
            {"title": "T", "url": "https://example.com", "snippet": "s"}
        ]
        call = ToolCall(tool_name="fast_web_search", arguments={"query": "Marine Leleu"})
        result = fast_web_search(call)

        assert result.success
        assert result.payload is not None
        assert "workspace_dir" not in result.payload
        assert "workspace_id" not in result.payload

    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    @patch("unclaw.tools.web_tools._rank_search_results")
    @patch("unclaw.tools.web_tools._run_iterative_retrieval")
    @patch("unclaw.tools.web_tools._synthesize_search_knowledge")
    def test_search_web_without_workspace_dir_has_no_workspace(
        self,
        mock_synth: MagicMock,
        mock_retrieval: MagicMock,
        mock_rank: MagicMock,
        mock_parse: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """search_web without workspace_base_dir should have no workspace_id."""
        from unclaw.tools.web_tools import search_web

        mock_search.return_value = "<html></html>"
        mock_parse.return_value = []
        mock_rank.return_value = []

        mock_outcome = MagicMock()
        mock_outcome.sources = ()
        mock_outcome.evidence_items = ()
        mock_outcome.initial_result_count = 0
        mock_outcome.considered_candidate_count = 0
        mock_outcome.fetch_attempt_count = 0
        mock_outcome.fetch_success_count = 0
        mock_retrieval.return_value = mock_outcome

        mock_synth_result = MagicMock()
        mock_synth_result.fact_clusters = ()
        mock_synth_result.findings = ()
        mock_synth_result.statements = ()
        mock_synth.return_value = mock_synth_result

        call = ToolCall(tool_name="search_web", arguments={"query": "test"})
        result = search_web(call, workspace_base_dir=None)

        assert result.success
        assert result.payload is not None
        assert "workspace_id" not in result.payload
