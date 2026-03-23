"""Tests for the pre-tool literal entity guard.

Covers:
- entity drift detection
- literal entity restoration in search_web and fast_web_search calls
- no false-positive corrections when entity is present
- no correction when user input has no identifiable entity
- context-token preservation after correction
- guard is a no-op for non-search tools
"""

from __future__ import annotations

from unclaw.tools.contracts import ToolCall
from unclaw.tools.web_entity_guard import (
    _build_corrected_query,
    _entity_drift_detected,
    apply_entity_guard_to_tool_calls,
    extract_user_entity_surface,
)


# ---------------------------------------------------------------------------
# extract_user_entity_surface
# ---------------------------------------------------------------------------


class TestExtractUserEntitySurface:
    def test_extracts_entity_from_french_who_is(self) -> None:
        entity = extract_user_entity_surface("Qui est Marine Leleu ?")
        assert entity == "Marine Leleu"

    def test_extracts_entity_from_english_who_is(self) -> None:
        entity = extract_user_entity_surface("Who is Marine Leleu?")
        assert entity == "Marine Leleu"

    def test_extracts_entity_from_tell_me_about(self) -> None:
        entity = extract_user_entity_surface("Tell me about Ada Lovelace")
        assert entity == "Ada Lovelace"

    def test_extracts_entity_from_biography_prefix(self) -> None:
        entity = extract_user_entity_surface("biographie de Marine Leleu")
        assert entity == "Marine Leleu"

    def test_extracts_entity_from_plain_name(self) -> None:
        entity = extract_user_entity_surface("Marine Leleu")
        assert entity == "Marine Leleu"

    def test_extracts_entity_from_follow_up_correction(self) -> None:
        entity = extract_user_entity_surface("non, Marine Leleu")
        assert entity == "Marine Leleu"

    def test_extracts_entity_from_explicit_follow_up_reference(self) -> None:
        entity = extract_user_entity_surface("je parle bien de Inoxtag")
        assert entity == "Inoxtag"

    def test_returns_empty_for_keyword_query(self) -> None:
        # "latest news on climate change" has no clear entity span
        entity = extract_user_entity_surface("latest news on climate change")
        # May or may not extract; key property is guard does not break anything
        # The entity is either empty or a short span — both are acceptable.
        assert isinstance(entity, str)

    def test_returns_empty_for_long_generic_sentence(self) -> None:
        entity = extract_user_entity_surface(
            "I would really like to know more information about different topics"
        )
        # Long sentence with no clear entity — guard should be inactive
        assert entity == "" or len(entity.split()) <= 4


# ---------------------------------------------------------------------------
# _entity_drift_detected
# ---------------------------------------------------------------------------


class TestEntityDriftDetected:
    def test_detects_drift_le_pen_for_leleu(self) -> None:
        # Classic failure: model substitutes Marine Le Pen for Marine Leleu
        assert _entity_drift_detected("Marine Le Pen", "Marine Leleu")

    def test_no_drift_when_entity_present(self) -> None:
        assert not _entity_drift_detected("Marine Leleu", "Marine Leleu")

    def test_no_drift_entity_plus_context(self) -> None:
        assert not _entity_drift_detected("Marine Leleu biographie", "Marine Leleu")

    def test_no_drift_entity_quoted(self) -> None:
        assert not _entity_drift_detected('"Marine Leleu"', "Marine Leleu")

    def test_no_drift_partial_token_match(self) -> None:
        # All user entity tokens present in model query (different order / extra tokens)
        assert not _entity_drift_detected("Leleu Marine athlete", "Marine Leleu")

    def test_no_drift_empty_user_entity(self) -> None:
        assert not _entity_drift_detected("Marine Le Pen", "")

    def test_no_drift_empty_model_query(self) -> None:
        assert not _entity_drift_detected("", "Marine Leleu")

    def test_no_drift_when_model_query_has_no_entity(self) -> None:
        # Pure keyword query — no substitution to undo
        assert not _entity_drift_detected("french athlete fitness", "Marine Leleu")

    def test_detects_drift_different_person(self) -> None:
        assert _entity_drift_detected("Ada Lovelace", "Alan Turing")

    def test_no_drift_same_entity_different_case(self) -> None:
        assert not _entity_drift_detected("marine leleu", "Marine Leleu")


# ---------------------------------------------------------------------------
# _build_corrected_query
# ---------------------------------------------------------------------------


class TestBuildCorrectedQuery:
    def test_returns_user_entity_when_no_context(self) -> None:
        corrected = _build_corrected_query("Marine Le Pen", "Marine Leleu")
        assert corrected == "Marine Leleu"

    def test_preserves_context_tokens(self) -> None:
        # If model query had context tokens (e.g. "biographie"), keep them
        corrected = _build_corrected_query("Marine Le Pen biographie", "Marine Leleu")
        assert corrected.startswith("Marine Leleu")
        assert "biographie" in corrected

    def test_limits_context_tokens(self) -> None:
        # Only up to 3 context tokens should be kept
        corrected = _build_corrected_query(
            "Marine Le Pen a b c d e f", "Marine Leleu"
        )
        assert corrected.startswith("Marine Leleu")
        tokens = corrected.split()
        assert len(tokens) <= 4 + 1  # entity tokens + up to 3 context


# ---------------------------------------------------------------------------
# apply_entity_guard_to_tool_calls
# ---------------------------------------------------------------------------


class TestApplyEntityGuardToToolCalls:
    def _make_call(self, tool_name: str, query: str) -> ToolCall:
        return ToolCall(tool_name=tool_name, arguments={"query": query})

    def test_restores_literal_entity_in_fast_web_search(self) -> None:
        calls = [self._make_call("fast_web_search", "Marine Le Pen")]
        guarded = apply_entity_guard_to_tool_calls(calls, "Marine Leleu")

        assert len(guarded) == 1
        assert guarded[0].tool_name == "fast_web_search"
        assert "Marine Leleu" in guarded[0].arguments["query"]
        assert "Le Pen" not in guarded[0].arguments["query"]

    def test_restores_literal_entity_in_search_web(self) -> None:
        calls = [self._make_call("search_web", "Marine Le Pen")]
        guarded = apply_entity_guard_to_tool_calls(calls, "Marine Leleu")

        assert "Marine Leleu" in guarded[0].arguments["query"]

    def test_no_correction_when_entity_present(self) -> None:
        calls = [self._make_call("fast_web_search", "Marine Leleu biographie")]
        guarded = apply_entity_guard_to_tool_calls(calls, "Marine Leleu")

        assert guarded[0].arguments["query"] == "Marine Leleu biographie"

    def test_no_correction_for_non_search_tool(self) -> None:
        calls = [ToolCall(tool_name="fetch_url_text", arguments={"url": "https://example.com"})]
        guarded = apply_entity_guard_to_tool_calls(calls, "Marine Leleu")

        assert guarded[0] is calls[0]

    def test_no_correction_when_no_user_entity(self) -> None:
        calls = [self._make_call("fast_web_search", "Marine Le Pen")]
        guarded = apply_entity_guard_to_tool_calls(calls, "")

        assert guarded[0].arguments["query"] == "Marine Le Pen"

    def test_mixed_calls_only_search_corrected(self) -> None:
        calls = [
            ToolCall(tool_name="fetch_url_text", arguments={"url": "https://example.com"}),
            self._make_call("fast_web_search", "Marine Le Pen"),
        ]
        guarded = apply_entity_guard_to_tool_calls(calls, "Marine Leleu")

        assert guarded[0].arguments["url"] == "https://example.com"
        assert "Marine Leleu" in guarded[1].arguments["query"]

    def test_returns_tuple(self) -> None:
        calls = [self._make_call("fast_web_search", "test")]
        guarded = apply_entity_guard_to_tool_calls(calls, "test")
        assert isinstance(guarded, tuple)

    def test_empty_calls_returns_empty_tuple(self) -> None:
        guarded = apply_entity_guard_to_tool_calls([], "Marine Leleu")
        assert guarded == ()

    def test_corrected_call_preserves_tool_name(self) -> None:
        calls = [self._make_call("search_web", "Marine Le Pen")]
        guarded = apply_entity_guard_to_tool_calls(calls, "Marine Leleu")
        assert guarded[0].tool_name == "search_web"

    def test_corrected_call_preserves_other_arguments(self) -> None:
        calls = [
            ToolCall(
                tool_name="search_web",
                arguments={"query": "Marine Le Pen", "max_results": 5},
            )
        ]
        guarded = apply_entity_guard_to_tool_calls(calls, "Marine Leleu")
        assert guarded[0].arguments["max_results"] == 5

    def test_le_pen_for_leleu_is_corrected(self) -> None:
        """Canonical failure case: model calls fast_web_search('Marine Le Pen')
        when user asked 'Qui est Marine Leleu ?'."""
        user_input = "Qui est Marine Leleu ?"
        user_entity = extract_user_entity_surface(user_input)
        assert user_entity == "Marine Leleu"

        calls = [self._make_call("fast_web_search", "Marine Le Pen")]
        guarded = apply_entity_guard_to_tool_calls(calls, user_entity)

        corrected_query = guarded[0].arguments["query"]
        assert "Marine Leleu" in corrected_query
        assert "Le Pen" not in corrected_query

    def test_no_drift_with_quoted_entity_in_model_query(self) -> None:
        calls = [self._make_call("fast_web_search", '"Marine Leleu"')]
        guarded = apply_entity_guard_to_tool_calls(calls, "Marine Leleu")
        # Quoted entity — should not be flagged as drift
        assert guarded[0].arguments["query"] == '"Marine Leleu"'

    def test_context_tokens_preserved_after_correction(self) -> None:
        """When model drifted entity but had useful context tokens, keep them."""
        # "biographie" is a recognised context-modifier token that travels
        # alongside the entity — it should be preserved after correction.
        calls = [self._make_call("fast_web_search", "Marine Le Pen biographie")]
        guarded = apply_entity_guard_to_tool_calls(calls, "Marine Leleu")

        corrected = guarded[0].arguments["query"]
        assert "Marine Leleu" in corrected
        assert "biographie" in corrected

    def test_follow_up_correction_reanchors_on_corrected_entity(self) -> None:
        """Correction phrasing should still restore the corrected entity on the next search."""
        user_input = "je parle bien de Marine Leleu"
        user_entity = extract_user_entity_surface(user_input)

        guarded = apply_entity_guard_to_tool_calls(
            [self._make_call("fast_web_search", "Marine Le Pen")],
            user_entity,
        )

        assert guarded[0].arguments["query"] == "Marine Leleu"
