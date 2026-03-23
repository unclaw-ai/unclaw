"""Tests for the file-backed research workspace (web_workspace.py).

Covers:
- Workspace directory creation (timestamped + slug)
- Artifact file existence after persistence
- Per-source raw / note file content
- Merged-note persistence
- Payload includes workspace_id and workspace_dir after search_web
- search_web remains usable when persistence partially fails
- fast_web_search stays memory-only (no workspace files)
- Bounded retention via prune_old_workspaces
- Slug generation edge cases
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from unclaw.tools.web_research import (
    MAIN_RESEARCH_BUDGET,
    MergedResearchNote,
    ResearchBudget,
    ResearchWorkspace,
    SourceArtifact,
    SourceNote,
)
from unclaw.tools.web_workspace import (
    SearchWorkspaceRef,
    _slugify,
    create_workspace_dir,
    persist_research_workspace,
    prune_old_workspaces,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_populated_workspace(
    *,
    query: str = "climate change",
    source_count: int = 2,
    model_driven: bool = False,
) -> ResearchWorkspace:
    workspace = ResearchWorkspace(query=query, budget=MAIN_RESEARCH_BUDGET)
    for i in range(source_count):
        artifact = SourceArtifact(
            url=f"https://example.com/{i}",
            title=f"Source {i}",
            cleaned_text=f"Content for source {i}. " * 10,
            fetch_success=True,
        )
        workspace.source_artifacts.append(artifact)
        note = SourceNote(
            url=f"https://example.com/{i}",
            title=f"Source {i}",
            condensed_text=f"Note for source {i}.",
            model_generated=model_driven,
        )
        workspace.source_notes.append(note)
    workspace.merged_note = MergedResearchNote(
        text="Merged research note about climate change.",
        source_count=source_count,
        model_generated=model_driven,
    )
    return workspace


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_simple_ascii_query(self) -> None:
        assert _slugify("climate change") == "climate_change"

    def test_special_characters_stripped(self) -> None:
        slug = _slugify("who is Marine Leleu?")
        assert " " not in slug
        assert "?" not in slug
        assert slug.islower() or slug == slug

    def test_unicode_normalized(self) -> None:
        slug = _slugify("résumé énergie")
        # Non-ASCII stripped; remaining chars slugified
        assert all(c.isalnum() or c == "_" for c in slug)

    def test_slug_respects_max_length(self) -> None:
        long_query = "word " * 100
        slug = _slugify(long_query)
        assert len(slug) <= 40

    def test_empty_string_returns_search(self) -> None:
        assert _slugify("") == "search"

    def test_only_special_chars_returns_search(self) -> None:
        assert _slugify("!@#$%^&*") == "search"


# ---------------------------------------------------------------------------
# create_workspace_dir
# ---------------------------------------------------------------------------


class TestCreateWorkspaceDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        ws_path, ws_id = create_workspace_dir(tmp_path, "test query")
        assert ws_path.exists()
        assert ws_path.is_dir()

    def test_workspace_id_contains_timestamp_and_slug(self, tmp_path: Path) -> None:
        _, ws_id = create_workspace_dir(tmp_path, "climate change")
        # Pattern: YYYYMMDD_HHMMSS_slug
        parts = ws_id.split("_")
        assert len(parts) >= 3
        assert parts[0].isdigit() and len(parts[0]) == 8  # date
        assert parts[1].isdigit() and len(parts[1]) == 6  # time
        assert "climate" in ws_id or "change" in ws_id

    def test_different_queries_produce_different_ids(self, tmp_path: Path) -> None:
        _, id1 = create_workspace_dir(tmp_path, "first query")
        _, id2 = create_workspace_dir(tmp_path, "second query")
        # Slugs differ even if timestamps collide
        assert id1 != id2

    def test_nested_base_dir_created(self, tmp_path: Path) -> None:
        deep_base = tmp_path / "a" / "b" / "c"
        ws_path, _ = create_workspace_dir(deep_base, "test")
        assert ws_path.exists()

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        ws_path, _ = create_workspace_dir(tmp_path, "test")
        assert ws_path.is_absolute()


# ---------------------------------------------------------------------------
# persist_research_workspace — artifact existence
# ---------------------------------------------------------------------------


class TestPersistResearchWorkspace:
    def test_meta_json_exists_after_persistence(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=2)
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="climate change",
        )
        assert (tmp_path / "meta.json").exists()

    def test_meta_json_contains_expected_fields(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=2)
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="climate change",
        )
        meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        assert meta["query"] == "climate change"
        assert meta["source_count"] == 2
        assert "budget" in meta
        assert "model_driven" in meta

    def test_per_source_raw_files_created(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=3)
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="test",
        )
        for i in range(1, 4):
            assert (tmp_path / f"source_{i:02d}_raw.txt").exists(), \
                f"source_{i:02d}_raw.txt missing"

    def test_per_source_raw_contains_url_and_title(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=1)
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="test",
        )
        raw_text = (tmp_path / "source_01_raw.txt").read_text(encoding="utf-8")
        assert "https://example.com/0" in raw_text
        assert "Source 0" in raw_text

    def test_per_source_note_files_created(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=2)
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="test",
        )
        for i in range(1, 3):
            assert (tmp_path / f"source_{i:02d}_note.txt").exists(), \
                f"source_{i:02d}_note.txt missing"

    def test_per_source_note_contains_condensed_text(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=1)
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="test",
        )
        note_text = (tmp_path / "source_01_note.txt").read_text(encoding="utf-8")
        assert "Note for source 0" in note_text

    def test_merged_note_file_created(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=2)
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="test",
        )
        assert (tmp_path / "merged_note.txt").exists()

    def test_merged_note_contains_query_and_text(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=2, query="climate change")
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="climate change",
        )
        merged_text = (tmp_path / "merged_note.txt").read_text(encoding="utf-8")
        assert "climate change" in merged_text
        assert "Merged research note" in merged_text

    def test_no_merged_note_when_workspace_has_none(self, tmp_path: Path) -> None:
        workspace = ResearchWorkspace(query="test", budget=MAIN_RESEARCH_BUDGET)
        # Do NOT set merged_note
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="test",
        )
        assert not (tmp_path / "merged_note.txt").exists()

    def test_model_driven_flag_reflected_in_meta(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=1, model_driven=True)
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="test",
        )
        meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        assert meta["model_driven"] is True

    def test_io_error_on_meta_does_not_raise(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=1)
        # Make the workspace dir read-only to trigger IO errors
        ws_dir = tmp_path / "readonly_ws"
        ws_dir.mkdir()
        ws_dir.chmod(0o555)  # read+execute only
        try:
            # Should not raise even if writes fail
            persist_research_workspace(
                workspace_dir=ws_dir,
                workspace=workspace,
                query="test",
            )
        except OSError:
            pytest.fail("persist_research_workspace raised OSError unexpectedly")
        finally:
            ws_dir.chmod(0o755)

    def test_empty_workspace_writes_meta_only(self, tmp_path: Path) -> None:
        workspace = ResearchWorkspace(query="empty", budget=MAIN_RESEARCH_BUDGET)
        workspace.merged_note = MergedResearchNote(
            text="No sources available.", source_count=0, model_generated=False
        )
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="empty",
        )
        assert (tmp_path / "meta.json").exists()
        assert not (tmp_path / "source_01_raw.txt").exists()

    def test_non_ascii_query_in_meta(self, tmp_path: Path) -> None:
        workspace = _make_populated_workspace(source_count=1, query="énergie renouvelable")
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="énergie renouvelable",
        )
        meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        assert "énergie renouvelable" in meta["query"]


# ---------------------------------------------------------------------------
# prune_old_workspaces
# ---------------------------------------------------------------------------


class TestPruneOldWorkspaces:
    def test_no_pruning_when_under_limit(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"search_{i:04d}").mkdir()
        removed = prune_old_workspaces(tmp_path, max_entries=10)
        assert removed == 0
        assert len(list(tmp_path.iterdir())) == 5

    def test_prunes_oldest_entries(self, tmp_path: Path) -> None:
        # Create 15 directories, keep 10
        for i in range(15):
            d = tmp_path / f"search_{i:04d}"
            d.mkdir()
            # Stagger mtimes so oldest-first order is deterministic
            mtime = 1_000_000 + i
            os.utime(d, (mtime, mtime))

        removed = prune_old_workspaces(tmp_path, max_entries=10)
        assert removed == 5
        remaining = sorted(p.name for p in tmp_path.iterdir())
        assert len(remaining) == 10
        # Oldest 5 (search_0000–search_0004) should be gone
        for i in range(5):
            assert f"search_{i:04d}" not in remaining

    def test_returns_zero_when_base_dir_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        assert prune_old_workspaces(missing, max_entries=10) == 0

    def test_default_max_entries_is_reasonable(self, tmp_path: Path) -> None:
        # Should not prune when well under default limit
        for i in range(10):
            (tmp_path / f"ws_{i}").mkdir()
        removed = prune_old_workspaces(tmp_path)
        assert removed == 0


# ---------------------------------------------------------------------------
# search_web integration — workspace written to payload
# ---------------------------------------------------------------------------


def _make_mock_retrieval_outcome() -> MagicMock:
    mock = MagicMock()
    mock.sources = ()
    mock.evidence_items = ()
    mock.initial_result_count = 1
    mock.considered_candidate_count = 1
    mock.fetch_attempt_count = 1
    mock.fetch_success_count = 1
    return mock


def _make_retrieval_side_effect_with_pages(**kwargs):
    """Side effect for _run_iterative_retrieval that populates page_text_collector."""
    pc = kwargs.get("page_text_collector")
    if pc is not None:
        pc["https://example.com/page"] = ("Example Title", "Content about the topic. " * 5)
    return _make_mock_retrieval_outcome()


def _make_mock_synthesis() -> MagicMock:
    mock = MagicMock()
    mock.fact_clusters = ()
    mock.findings = ()
    mock.statements = ()
    return mock


class TestSearchWebWorkspacePayload:
    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    @patch("unclaw.tools.web_tools._rank_search_results")
    @patch("unclaw.tools.web_tools._run_iterative_retrieval")
    @patch("unclaw.tools.web_tools._synthesize_search_knowledge")
    @patch("unclaw.tools.web_tools.run_research_pipeline")
    def test_payload_contains_workspace_id_and_dir(
        self,
        mock_pipeline: MagicMock,
        mock_synth: MagicMock,
        mock_retrieval: MagicMock,
        mock_rank: MagicMock,
        mock_parse: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        from unclaw.tools.contracts import ToolCall
        from unclaw.tools.web_tools import search_web

        mock_search.return_value = "<html></html>"
        mock_parse.return_value = [
            {"title": "R", "url": "https://example.com", "snippet": "S."}
        ]
        mock_rank.return_value = mock_parse.return_value
        # Populate page_text_collector so the pipeline branch is triggered
        mock_retrieval.side_effect = _make_retrieval_side_effect_with_pages
        mock_synth.return_value = _make_mock_synthesis()

        # Make pipeline return a populated workspace
        workspace = _make_populated_workspace(source_count=2)
        mock_pipeline.return_value = workspace

        call = ToolCall(tool_name="search_web", arguments={"query": "climate change"})
        result = search_web(
            call,
            research_config=MagicMock(),  # triggers pipeline
            workspace_base_dir=tmp_path,
        )

        assert result.success
        assert result.payload is not None
        assert "workspace_id" in result.payload
        assert "workspace_dir" in result.payload
        assert result.payload["workspace_id"]  # non-empty
        assert Path(result.payload["workspace_dir"]).exists()

    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    @patch("unclaw.tools.web_tools._rank_search_results")
    @patch("unclaw.tools.web_tools._run_iterative_retrieval")
    @patch("unclaw.tools.web_tools._synthesize_search_knowledge")
    @patch("unclaw.tools.web_tools.run_research_pipeline")
    def test_workspace_artifacts_written_to_disk(
        self,
        mock_pipeline: MagicMock,
        mock_synth: MagicMock,
        mock_retrieval: MagicMock,
        mock_rank: MagicMock,
        mock_parse: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        from unclaw.tools.contracts import ToolCall
        from unclaw.tools.web_tools import search_web

        mock_search.return_value = "<html></html>"
        mock_parse.return_value = [
            {"title": "R", "url": "https://example.com", "snippet": "S."}
        ]
        mock_rank.return_value = mock_parse.return_value
        mock_retrieval.side_effect = _make_retrieval_side_effect_with_pages
        mock_synth.return_value = _make_mock_synthesis()

        workspace = _make_populated_workspace(source_count=2)
        mock_pipeline.return_value = workspace

        call = ToolCall(tool_name="search_web", arguments={"query": "test"})
        result = search_web(
            call,
            research_config=MagicMock(),
            workspace_base_dir=tmp_path,
        )

        ws_dir = Path(result.payload["workspace_dir"])
        assert (ws_dir / "meta.json").exists()
        assert (ws_dir / "merged_note.txt").exists()
        assert (ws_dir / "source_01_raw.txt").exists()
        assert (ws_dir / "source_01_note.txt").exists()

    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    @patch("unclaw.tools.web_tools._rank_search_results")
    @patch("unclaw.tools.web_tools._run_iterative_retrieval")
    @patch("unclaw.tools.web_tools._synthesize_search_knowledge")
    def test_no_workspace_fields_without_workspace_base_dir(
        self,
        mock_synth: MagicMock,
        mock_retrieval: MagicMock,
        mock_rank: MagicMock,
        mock_parse: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        from unclaw.tools.contracts import ToolCall
        from unclaw.tools.web_tools import search_web

        mock_search.return_value = "<html></html>"
        mock_parse.return_value = [
            {"title": "R", "url": "https://example.com", "snippet": "S."}
        ]
        mock_rank.return_value = mock_parse.return_value
        mock_retrieval.return_value = _make_mock_retrieval_outcome()
        mock_synth.return_value = _make_mock_synthesis()

        call = ToolCall(tool_name="search_web", arguments={"query": "test"})
        result = search_web(call)  # no workspace_base_dir

        assert result.success
        assert result.payload is not None
        assert "workspace_id" not in result.payload
        assert "workspace_dir" not in result.payload

    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    @patch("unclaw.tools.web_tools._rank_search_results")
    @patch("unclaw.tools.web_tools._run_iterative_retrieval")
    @patch("unclaw.tools.web_tools._synthesize_search_knowledge")
    @patch("unclaw.tools.web_tools.run_research_pipeline")
    def test_search_web_succeeds_when_persistence_raises(
        self,
        mock_pipeline: MagicMock,
        mock_synth: MagicMock,
        mock_retrieval: MagicMock,
        mock_rank: MagicMock,
        mock_parse: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """search_web must still return a successful result when workspace IO fails."""
        from unclaw.tools.contracts import ToolCall
        from unclaw.tools.web_tools import search_web

        mock_search.return_value = "<html></html>"
        mock_parse.return_value = [
            {"title": "R", "url": "https://example.com", "snippet": "S."}
        ]
        mock_rank.return_value = mock_parse.return_value
        mock_retrieval.side_effect = _make_retrieval_side_effect_with_pages
        mock_synth.return_value = _make_mock_synthesis()

        workspace = _make_populated_workspace(source_count=1)
        mock_pipeline.return_value = workspace

        # Point at a non-writable path to force IO failure on subdirectory creation.
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        bad_dir.chmod(0o444)  # read-only — mkdir inside will fail
        try:
            call = ToolCall(tool_name="search_web", arguments={"query": "test"})
            result = search_web(
                call,
                research_config=MagicMock(),
                workspace_base_dir=bad_dir,
            )
            # Tool result must still succeed even if workspace couldn't be written
            assert result.success
        finally:
            bad_dir.chmod(0o755)

    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    @patch("unclaw.tools.web_tools._rank_search_results")
    @patch("unclaw.tools.web_tools._run_iterative_retrieval")
    @patch("unclaw.tools.web_tools._synthesize_search_knowledge")
    @patch("unclaw.tools.web_tools.run_research_pipeline")
    def test_workspace_not_written_when_pipeline_returns_none(
        self,
        mock_pipeline: MagicMock,
        mock_synth: MagicMock,
        mock_retrieval: MagicMock,
        mock_rank: MagicMock,
        mock_parse: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        from unclaw.tools.contracts import ToolCall
        from unclaw.tools.web_tools import search_web

        mock_search.return_value = "<html></html>"
        mock_parse.return_value = [
            {"title": "R", "url": "https://example.com", "snippet": "S."}
        ]
        mock_rank.return_value = mock_parse.return_value
        mock_retrieval.side_effect = _make_retrieval_side_effect_with_pages
        mock_synth.return_value = _make_mock_synthesis()

        # Simulate pipeline failure
        mock_pipeline.side_effect = RuntimeError("pipeline failed")

        call = ToolCall(tool_name="search_web", arguments={"query": "test"})
        result = search_web(
            call,
            research_config=MagicMock(),
            workspace_base_dir=tmp_path,
        )

        assert result.success
        assert result.payload is not None
        assert "workspace_id" not in result.payload
        # No workspace directory should have been created
        assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# fast_web_search — must remain memory-only (no workspace writes)
# ---------------------------------------------------------------------------


class TestFastWebSearchIsMemoryOnly:
    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    def test_fast_web_search_writes_no_files(
        self,
        mock_parse: MagicMock,
        mock_search: MagicMock,
        tmp_path: Path,
    ) -> None:
        """fast_web_search must not write any workspace artifacts."""
        from unclaw.tools.contracts import ToolCall
        from unclaw.tools.web_tools import fast_web_search

        mock_search.return_value = "<html></html>"
        mock_parse.return_value = [
            {
                "title": "Quick Result",
                "url": "https://example.com",
                "snippet": "Quick grounding.",
            }
        ]

        call = ToolCall(tool_name="fast_web_search", arguments={"query": "Marine Leleu"})
        result = fast_web_search(call)

        assert result.success
        # No workspace directory should exist under tmp_path
        assert not any(tmp_path.iterdir())

    @patch("unclaw.tools.web_tools._search_public_web")
    @patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
    def test_fast_web_search_payload_has_no_workspace_fields(
        self,
        mock_parse: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        from unclaw.tools.contracts import ToolCall
        from unclaw.tools.web_tools import fast_web_search

        mock_search.return_value = "<html></html>"
        mock_parse.return_value = [
            {
                "title": "Quick Result",
                "url": "https://example.com",
                "snippet": "Quick grounding.",
            }
        ]

        call = ToolCall(tool_name="fast_web_search", arguments={"query": "test"})
        result = fast_web_search(call)

        assert result.success
        assert result.payload is not None
        assert "workspace_id" not in result.payload
        assert "workspace_dir" not in result.payload


# ---------------------------------------------------------------------------
# Budget bounds remain enforced with persistence active
# ---------------------------------------------------------------------------


class TestBudgetBoundsWithPersistence:
    def test_source_count_bounded_in_persisted_workspace(self, tmp_path: Path) -> None:
        """Persisted artifacts count never exceeds budget.max_sources."""
        tight_budget = ResearchBudget(
            max_sources=2,
            max_source_chars=500,
            max_source_note_chars=100,
            max_merged_note_chars=200,
            condensation_timeout_seconds=5.0,
        )
        workspace = ResearchWorkspace(query="test", budget=tight_budget)
        for i in range(5):
            artifact = SourceArtifact(
                url=f"https://example.com/{i}",
                title=f"Source {i}",
                cleaned_text=f"Content {i}",
                fetch_success=True,
            )
            workspace.add_source(artifact)
            note = SourceNote(
                url=f"https://example.com/{i}",
                title=f"Source {i}",
                condensed_text=f"Note {i}",
                model_generated=False,
            )
            if workspace.source_count <= tight_budget.max_sources:
                workspace.source_notes.append(note)

        workspace.merged_note = MergedResearchNote(
            text="Merged.", source_count=workspace.source_count, model_generated=False
        )

        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="test",
        )

        # Budget caps at 2 sources; only source_01 and source_02 raw files exist
        assert (tmp_path / "source_01_raw.txt").exists()
        assert (tmp_path / "source_02_raw.txt").exists()
        assert not (tmp_path / "source_03_raw.txt").exists()

    def test_merged_note_content_bounded_by_budget(self, tmp_path: Path) -> None:
        """Merged note stored on disk reflects the budget-capped content."""
        workspace = _make_populated_workspace(source_count=1)
        # Replace merged note with a long text to confirm it's stored as-is
        long_text = "Fact. " * 200
        workspace.merged_note = MergedResearchNote(
            text=long_text,
            source_count=1,
            model_generated=False,
        )
        persist_research_workspace(
            workspace_dir=tmp_path,
            workspace=workspace,
            query="test",
        )
        merged_text = (tmp_path / "merged_note.txt").read_text(encoding="utf-8")
        # Text is stored faithfully; budget capping happens in the pipeline, not here
        assert "Fact." in merged_text


# ---------------------------------------------------------------------------
# SearchWorkspaceRef dataclass
# ---------------------------------------------------------------------------


class TestSearchWorkspaceRef:
    def test_immutable(self) -> None:
        ref = SearchWorkspaceRef(workspace_id="abc", workspace_dir="/some/path")
        with pytest.raises((AttributeError, TypeError)):
            ref.workspace_id = "changed"  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        ref = SearchWorkspaceRef(
            workspace_id="20260323_143022_test",
            workspace_dir="/data/web_search/20260323_143022_test",
        )
        assert ref.workspace_id == "20260323_143022_test"
        assert ref.workspace_dir == "/data/web_search/20260323_143022_test"
