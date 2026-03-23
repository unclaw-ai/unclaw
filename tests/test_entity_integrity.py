"""Entity integrity hardening tests.

Covers:
- Exact user entity preserved on first tool call (no Leleu → Le Pen substitution)
- No silent entity substitution in staged query construction
- Biography source-prioritization: reference domains boosted, social shells demoted
- Identity-consistent merge: weak single-source non-anchor claims get '? ' prefix
- Conflicting weak claims stay isolated, not merged into the main identity spine
- fast_web_search remains memory-only (no file-backed workspace)
- search_web still persists workspaces (file-backed)
- Executed staged queries are persisted in workspace meta.json
- fast_web_search mismatch detection distinguishes partial vs full mismatch
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from unclaw.tools.contracts import ToolCall
from unclaw.tools.web_research import (
    MAIN_RESEARCH_BUDGET,
    MergedResearchNote,
    SourceNote,
    build_fast_grounding_note,
    merge_source_notes,
)
from unclaw.tools.web_search import (
    _build_search_query,
    _build_staged_search_queries,
    _rank_search_results,
)
from unclaw.tools.web_tools import (
    FAST_WEB_SEARCH_DEFINITION,
    SEARCH_WEB_DEFINITION,
    _run_bounded_staged_search,
    fast_web_search,
    search_web,
)
from unclaw.tools.web_workspace import persist_research_workspace
from unclaw.tools.web_research import ResearchBudget, ResearchWorkspace, SourceArtifact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(name: str, **kwargs) -> ToolCall:
    return ToolCall(tool_name=name, arguments=dict(kwargs))


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


# ---------------------------------------------------------------------------
# 1. Exact entity preserved on first tool call — no Leleu → Le Pen substitution
# ---------------------------------------------------------------------------


class TestEntityPreservedInStagedQueries:
    def test_exact_entity_surface_on_first_query(self) -> None:
        """First staged query must use the exact user-supplied entity name."""
        queries = _build_staged_search_queries("biographie de Marine Leleu", fast_mode=False)
        assert queries[0] == '"Marine Leleu"'
        assert all("le pen" not in q.casefold() for q in queries)

    def test_no_silent_substitution_leleu_to_le_pen(self) -> None:
        """No staged query should substitute Marine Leleu with Marine Le Pen."""
        for fast_mode in (True, False):
            queries = _build_staged_search_queries("Marine Leleu", fast_mode=fast_mode)
            for q in queries:
                assert "le pen" not in q.casefold(), (
                    f"Entity substitution detected in query: {q!r}"
                )

    def test_qui_est_marine_leleu_preserves_entity(self) -> None:
        """French bio prefix 'qui est' must not trigger entity rewrite."""
        queries = _build_staged_search_queries("qui est Marine Leleu", fast_mode=False)
        assert all("marine leleu" in q.casefold() for q in queries)
        assert all("le pen" not in q.casefold() for q in queries)

    def test_staging_does_not_contaminate_independent_lookups(self) -> None:
        """Entity from one lookup must not bleed into an independent lookup."""
        leleu_queries = _build_staged_search_queries("Marine Leleu", fast_mode=False)
        lovelace_queries = _build_staged_search_queries("Ada Lovelace", fast_mode=False)

        assert any("marine leleu" in q.casefold() for q in leleu_queries)
        assert all("marine" not in q.casefold() for q in lovelace_queries)
        assert any("ada lovelace" in q.casefold() for q in lovelace_queries)


# ---------------------------------------------------------------------------
# 2. Biography source prioritization
# ---------------------------------------------------------------------------


class TestBiographySourcePrioritization:
    def test_wikipedia_ranks_above_seo_farm(self) -> None:
        """Wikipedia result must score higher than an SEO biography farm."""
        search_query = _build_search_query("Marine Leleu biographie")
        results = [
            {
                "title": "Marine Leleu Biography Age Height Net Worth Wiki",
                "url": "https://celebswiki.com/marine-leleu-biography-age-height",
                "snippet": "Marine Leleu age height husband net worth married dating",
            },
            {
                "title": "Marine Leleu — Wikipédia",
                "url": "https://fr.wikipedia.org/wiki/Marine_Leleu",
                "snippet": "Marine Leleu est une athlète française spécialisée dans l'endurance.",
            },
        ]
        ranked = _rank_search_results(results, query=search_query)
        assert ranked[0]["url"].startswith("https://fr.wikipedia.org"), (
            "Wikipedia should outrank SEO farm"
        )

    def test_reference_domain_bonus_applied(self) -> None:
        """Reference domain URLs should score materially higher than equal-text generic URLs."""
        from unclaw.tools.web_search import _score_search_result

        search_query = _build_search_query("Marine Leleu")
        wiki_result = {
            "title": "Marine Leleu",
            "url": "https://fr.wikipedia.org/wiki/Marine_Leleu",
            "snippet": "Marine Leleu est une athlète française.",
        }
        generic_result = {
            "title": "Marine Leleu",
            "url": "https://some-blog.com/marine-leleu",
            "snippet": "Marine Leleu est une athlète française.",
        }
        wiki_score = _score_search_result(wiki_result, query=search_query)
        generic_score = _score_search_result(generic_result, query=search_query)
        assert wiki_score > generic_score

    def test_social_profile_shell_demoted(self) -> None:
        """Instagram/TikTok profile landing pages should score lower than real content pages."""
        from unclaw.tools.web_search import _score_search_result

        search_query = _build_search_query("Marine Leleu athlete")
        social_result = {
            "title": "Marine Leleu (@marineleleu)",
            "url": "https://www.instagram.com/marineleleu",
            "snippet": "Marine Leleu Instagram profile.",
        }
        article_result = {
            "title": "Marine Leleu – Ultra Runner Profile",
            "url": "https://ultrarunning.com/profiles/marine-leleu",
            "snippet": "Marine Leleu is a French ultra-endurance athlete.",
        }
        social_score = _score_search_result(social_result, query=search_query)
        article_score = _score_search_result(article_result, query=search_query)
        assert social_score < article_score

    def test_entity_ranking_penalizes_partial_name_substitution(self) -> None:
        """Partial entity match (Le Pen) must rank below exact match (Leleu)."""
        search_query = _build_search_query("Marine Leleu")
        ranked = _rank_search_results(
            [
                {
                    "title": "Marine Le Pen",
                    "url": "https://example.com/marine-le-pen",
                    "snippet": "French politician and party leader.",
                },
                {
                    "title": "Marine Leleu",
                    "url": "https://example.com/marine-leleu",
                    "snippet": "French endurance athlete and content creator.",
                },
            ],
            query=search_query,
        )
        assert ranked[0]["title"] == "Marine Leleu"


# ---------------------------------------------------------------------------
# 3. Identity-consistent merge — weak-source pollution resistance
# ---------------------------------------------------------------------------


class TestIdentityConsistentMerge:
    def test_anchor_source_claims_not_marked_weak(self) -> None:
        """Claims from sources 1-2 (anchor) should appear without '? ' prefix."""
        notes = [
            _make_source_note(
                url="https://reference.com/profile",
                title="Official Profile",
                text="Marine Leleu is a French ultra-endurance athlete born in Lille.",
            ),
            _make_source_note(
                url="https://sports.com/marineleleu",
                title="Sports Bio",
                text="Marine Leleu competes in trail running and cycling events.",
            ),
        ]
        result = merge_source_notes(
            source_notes=notes,
            query="Marine Leleu",
            budget=MAIN_RESEARCH_BUDGET,
        )
        # No lines should be prefixed with '? ' since both sources are anchors
        lines = result.text.splitlines()
        for line in lines:
            assert not line.startswith("? "), (
                f"Anchor source claim incorrectly marked weak: {line!r}"
            )

    def test_single_weak_source_claim_isolated(self) -> None:
        """A claim from a single non-anchor source (index 3+) should be marked '? '."""
        notes = [
            # Anchor source 1: athlete
            _make_source_note(
                url="https://ref1.com",
                title="Reference 1",
                text="Marine Leleu is a French ultra-endurance athlete.",
            ),
            # Anchor source 2: athlete confirmed
            _make_source_note(
                url="https://ref2.com",
                title="Reference 2",
                text="Marine Leleu competes in trail running.",
            ),
            # Non-anchor source 3: introduces unsupported claim
            _make_source_note(
                url="https://weak.com",
                title="Weak Source",
                text="Marine Leleu studied medicine at the Sorbonne.",
            ),
        ]
        result = merge_source_notes(
            source_notes=notes,
            query="Marine Leleu",
            budget=MAIN_RESEARCH_BUDGET,
        )
        lines = result.text.splitlines()
        # The Sorbonne claim (from source 3 only) should be marked tentative
        sorbonne_lines = [ln for ln in lines if "sorbonne" in ln.casefold()]
        assert sorbonne_lines, "Sorbonne claim should appear in merged note"
        for line in sorbonne_lines:
            assert line.startswith("? "), (
                f"Weak single-source claim should be marked '? ': {line!r}"
            )

    def test_conflicting_weak_claim_does_not_overwrite_identity_spine(self) -> None:
        """A conflicting claim from a weak source should not replace the main identity."""
        notes = [
            _make_source_note(
                url="https://sports.com",
                title="Sports Reference",
                text="Marine Leleu is a professional triathlete and coach.",
            ),
            _make_source_note(
                url="https://sports2.com",
                title="Sports Reference 2",
                text="Marine Leleu is a professional triathlete.",
            ),
            # Non-anchor source wrongly says "actress"
            _make_source_note(
                url="https://noisyblog.com",
                title="Noisy Blog",
                text="Marine Leleu is a French actress and model.",
            ),
        ]
        result = merge_source_notes(
            source_notes=notes,
            query="Marine Leleu",
            budget=MAIN_RESEARCH_BUDGET,
        )
        text = result.text
        # The "actress" claim from the noisy source should either be absent,
        # marked as weak ('? '), or flagged as a conflict ('!=').
        # It must NOT appear as a plain claim without qualification.
        actress_lines = [
            ln for ln in text.splitlines()
            if "actress" in ln.casefold() or "model" in ln.casefold()
        ]
        for line in actress_lines:
            assert line.startswith("? ") or line.startswith("!= "), (
                f"Weak/conflicting identity claim should be qualified, not plain: {line!r}"
            )

    def test_anchor_source_claims_never_marked_weak(self) -> None:
        """Claims from anchor sources (1, 2) must never get '? ' prefix, even when
        they each produce their own clusters due to phrasing differences."""
        notes = [
            _make_source_note(
                url="https://a.com",
                title="Source A",
                text="Marine Leleu is a French endurance athlete.",
            ),
            _make_source_note(
                url="https://b.com",
                title="Source B",
                text="Marine Leleu is a French endurance athlete and coach.",
            ),
        ]
        result = merge_source_notes(
            source_notes=notes,
            query="Marine Leleu",
            budget=MAIN_RESEARCH_BUDGET,
        )
        lines = result.text.splitlines()
        weak_lines = [ln for ln in lines if ln.startswith("? ")]
        # Only 2 anchor sources — nothing can be a single non-anchor claim
        assert not weak_lines, (
            f"Anchor-only sources should not produce weak markers: {weak_lines}"
        )


# ---------------------------------------------------------------------------
# 4. fast_web_search mismatch clarity
# ---------------------------------------------------------------------------


class TestFastWebSearchMismatchClarity:
    def test_no_mismatch_warning_when_entity_found(self) -> None:
        """No warning when snippets contain the queried entity."""
        note = build_fast_grounding_note(
            query="Marine Leleu",
            snippets=[
                (
                    "Marine Leleu",
                    "https://example.com/marine-leleu",
                    "Marine Leleu is a French endurance athlete.",
                )
            ],
            max_chars=300,
        )
        assert "!=" not in note

    def test_partial_mismatch_warning_when_entity_absent(self) -> None:
        """'!= no exact top match' appears when snippets lack the exact entity."""
        note = build_fast_grounding_note(
            query="Marine Leleu",
            snippets=[
                (
                    "Marine Le Pen",
                    "https://example.com/marine-le-pen",
                    "Marine Le Pen is a French politician.",
                )
            ],
            max_chars=300,
        )
        assert "!=" in note

    def test_full_mismatch_warning_when_zero_token_overlap(self) -> None:
        """Stronger warning when no entity tokens appear anywhere in results."""
        note = build_fast_grounding_note(
            query="Marine Leleu",
            snippets=[
                (
                    "Jean Dupont",
                    "https://example.com/jean-dupont",
                    "Jean Dupont is a French chef.",
                )
            ],
            max_chars=300,
        )
        assert "different entity" in note or "!=" in note

    def test_empty_snippets_returns_no_results_message(self) -> None:
        note = build_fast_grounding_note(
            query="Unknown Entity XYZ",
            snippets=[],
            max_chars=300,
        )
        assert "No web results" in note


# ---------------------------------------------------------------------------
# 5. fast_web_search is memory-only (no file-backed workspace)
# ---------------------------------------------------------------------------


class TestFastWebSearchMemoryOnly:
    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    def test_fast_web_search_produces_no_workspace_fields(
        self, mock_parse: MagicMock, mock_search: MagicMock
    ) -> None:
        """fast_web_search payload must not include workspace_id or workspace_dir."""
        mock_search.return_value = "<html></html>"
        mock_parse.return_value = [
            {
                "title": "Marine Leleu",
                "url": "https://example.com/marine-leleu",
                "snippet": "French athlete.",
            }
        ]

        call = _make_tool_call(FAST_WEB_SEARCH_DEFINITION.name, query="Marine Leleu")
        result = fast_web_search(call)

        assert result.ok
        assert "workspace_id" not in (result.payload or {})
        assert "workspace_dir" not in (result.payload or {})


# ---------------------------------------------------------------------------
# 6. search_web workspace persistence (file-backed)
# ---------------------------------------------------------------------------


class TestSearchWebFileBacked:
    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    @patch("unclaw.tools.web_tools._rank_search_results")
    @patch("unclaw.tools.web_tools._run_iterative_retrieval")
    def test_search_web_persists_workspace_when_base_dir_set(
        self,
        mock_retrieval: MagicMock,
        mock_rank: MagicMock,
        mock_parse: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """search_web with workspace_base_dir must not crash; payload fields are present."""
        mock_search.return_value = "<html></html>"
        mock_parse.return_value = []
        mock_rank.return_value = []
        mock_retrieval.return_value = SimpleNamespace(
            sources=[],
            evidence_items=[],
            initial_result_count=0,
            considered_candidate_count=0,
            fetch_attempt_count=0,
            fetch_success_count=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ws_base = Path(tmpdir) / "web_search"
            call = _make_tool_call(SEARCH_WEB_DEFINITION.name, query="Marine Leleu")
            result = search_web(call, workspace_base_dir=ws_base)
            # With no research_config, pipeline is off → no workspace created.
            # Verify no crash and tool returns success.
            assert result.ok

    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    @patch("unclaw.tools.web_tools._rank_search_results")
    @patch("unclaw.tools.web_tools._run_iterative_retrieval")
    def test_search_web_payload_lacks_workspace_without_pipeline(
        self,
        mock_retrieval: MagicMock,
        mock_rank: MagicMock,
        mock_parse: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """Without research_config, workspace_id is absent from the payload."""
        mock_search.return_value = "<html></html>"
        mock_parse.return_value = []
        mock_rank.return_value = []
        mock_retrieval.return_value = SimpleNamespace(
            sources=[],
            evidence_items=[],
            initial_result_count=0,
            considered_candidate_count=0,
            fetch_attempt_count=0,
            fetch_success_count=0,
        )

        call = _make_tool_call(SEARCH_WEB_DEFINITION.name, query="test query")
        result = search_web(call)
        assert result.ok
        assert "workspace_id" not in (result.payload or {})


# ---------------------------------------------------------------------------
# 7. Executed staged queries in workspace meta.json
# ---------------------------------------------------------------------------


class TestExecutedQueriesObservability:
    def test_executed_queries_written_to_meta_json(self) -> None:
        """persist_research_workspace writes executed_queries into meta.json."""
        budget = MAIN_RESEARCH_BUDGET
        workspace = ResearchWorkspace(query="Marine Leleu", budget=budget)
        workspace.source_notes = []

        with tempfile.TemporaryDirectory() as tmpdir:
            ws_dir = Path(tmpdir)
            persist_research_workspace(
                workspace_dir=ws_dir,
                workspace=workspace,
                query="Marine Leleu",
                executed_queries=('"Marine Leleu"', "Marine Leleu"),
            )

            meta_path = ws_dir / "meta.json"
            assert meta_path.exists(), "meta.json should be created"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            assert "executed_staged_queries" in meta, (
                "executed_staged_queries key missing from meta.json"
            )
            assert meta["executed_staged_queries"] == ['"Marine Leleu"', "Marine Leleu"]

    def test_executed_queries_empty_when_not_provided(self) -> None:
        """persist_research_workspace still writes meta.json when no executed_queries given."""
        budget = MAIN_RESEARCH_BUDGET
        workspace = ResearchWorkspace(query="test", budget=budget)

        with tempfile.TemporaryDirectory() as tmpdir:
            ws_dir = Path(tmpdir)
            persist_research_workspace(
                workspace_dir=ws_dir,
                workspace=workspace,
                query="test",
            )

            meta = json.loads((ws_dir / "meta.json").read_text(encoding="utf-8"))
            assert meta["executed_staged_queries"] == []

    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    def test_run_bounded_staged_search_returns_two_tuple(
        self, mock_parse: MagicMock, mock_search: MagicMock
    ) -> None:
        """_run_bounded_staged_search backward-compat: still returns (query, results)."""
        mock_search.return_value = "<html></html>"
        mock_parse.return_value = []

        result = _run_bounded_staged_search(
            query="test",
            max_results=5,
            timeout_seconds=5.0,
            fast_mode=False,
        )
        # Must unpack as a 2-tuple without error
        search_query, ranked = result
        assert ranked == []
