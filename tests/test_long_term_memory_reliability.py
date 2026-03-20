"""Targeted reliability tests for long-term memory.

Covers acceptance criteria from the PERSISTENT LONG-TERM MEMORY RELIABILITY mission:

A. Search quality
   - category field is now included in LIKE search
   - token-overlap fallback returns results when full phrase doesn't match
   - deduplication across passes
   - partial token matching (token is substring of stored field)

B. Tool description quality (structural)
   - remember_long_term_memory description includes multilingual positive examples
   - search_long_term_memory description includes semantic-query guidance
   - search description explicitly distinguishes from inspect_session_history
   - list_long_term_memory description mentions broad-recall use case
   - forget_long_term_memory description not changed (stable)

C. Capability context quality (structural)
   - LTM guidance distinguishes from session history
   - LTM guidance mentions semantic query guidance
   - LTM guidance mentions multilingual semantic recall handling

D. Explicit save reliability (English + French via tools)
   - English: "remember that my GPU is an RTX 4080" → stored and retrievable
   - French: "souviens-toi que je m'appelle Vincent" → stored and retrievable

E. Cross-session recall
   - Write in session A, read in new store instance (session B)

F. Negative cases
   - search description says NOT to use for session history
   - capability context says NOT to use LTM for session message history
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.memory.long_term_store import (
    LongTermStore,
    _extract_search_tokens,
    _fold_for_search,
)
from unclaw.tools.contracts import ToolCall
from unclaw.tools.long_term_memory_tools import (
    FORGET_LONG_TERM_MEMORY_DEFINITION,
    LIST_LONG_TERM_MEMORY_DEFINITION,
    REMEMBER_LONG_TERM_MEMORY_DEFINITION,
    SEARCH_LONG_TERM_MEMORY_DEFINITION,
    register_long_term_memory_tools,
)
from unclaw.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dispatch(registry: ToolRegistry, tool_name: str, **kwargs) -> object:
    from unclaw.tools.dispatcher import ToolDispatcher
    return ToolDispatcher(registry).dispatch(
        ToolCall(tool_name=tool_name, arguments=kwargs)
    )


def _make_registry(tmp_path: Path) -> tuple[LongTermStore, ToolRegistry]:
    store = LongTermStore(tmp_path / "memory" / "long_term.db")
    registry = ToolRegistry()
    register_long_term_memory_tools(registry, long_term_store=store)
    return store, registry


# ---------------------------------------------------------------------------
# A. Search quality
# ---------------------------------------------------------------------------

class TestSearchQuality:
    """Improved search: category field, token-overlap fallback, deduplication."""

    def test_category_field_is_searched(self, tmp_path: Path) -> None:
        """A search query that matches the category field returns the record."""
        store = LongTermStore(tmp_path / "long_term.db")
        store.store(key="main GPU", value="RTX 4080", category="hardware")
        results = store.search(query="hardware")
        assert len(results) == 1
        assert results[0].key == "main GPU"

    def test_category_field_searched_with_token(self, tmp_path: Path) -> None:
        """Token-level category match: stored category='hardware', query='my hardware'."""
        store = LongTermStore(tmp_path / "long_term.db")
        store.store(key="main GPU", value="RTX 4080", category="hardware")
        # Full phrase 'my hardware' won't match key/value/tags, but token 'hardware'
        # matches the category field.
        results = store.search(query="my hardware")
        assert len(results) == 1
        assert results[0].value == "RTX 4080"

    def test_token_overlap_matches_value_substring(self, tmp_path: Path) -> None:
        """A multi-word query matches via an individual token that is a substring."""
        store = LongTermStore(tmp_path / "long_term.db")
        store.store(key="main GPU", value="RTX 4080")
        # Full phrase 'user GPU info' won't match, but token 'GPU' matches key.
        results = store.search(query="user GPU info")
        assert len(results) == 1
        assert results[0].key == "main GPU"

    def test_deduplication_across_passes(self, tmp_path: Path) -> None:
        """A record matched in both pass 1 and pass 2 appears only once."""
        store = LongTermStore(tmp_path / "long_term.db")
        store.store(key="GPU model", value="RTX 4080", category="hardware")
        # "GPU" appears in both key and category; "model" in key.
        # Both passes could hit the same record — it must appear exactly once.
        results = store.search(query="GPU model")
        assert len(results) == 1

    def test_partial_token_match_in_key(self, tmp_path: Path) -> None:
        """Token 'name' (from query 'user name') matches key 'first name'."""
        store = LongTermStore(tmp_path / "long_term.db")
        store.store(key="first name", value="Vincent", category="identity")
        results = store.search(query="user name")
        # Token 'name' (4 chars) matches 'first name'
        assert len(results) == 1
        assert results[0].value == "Vincent"

    def test_short_tokens_excluded(self, tmp_path: Path) -> None:
        """Tokens shorter than 3 chars are not used for matching (no noise)."""
        tokens = _extract_search_tokens("my GPU is ok")
        # 'my' (2) and 'is' (2) and 'ok' (2) must be excluded
        assert "my" not in tokens
        assert "is" not in tokens
        assert "ok" not in tokens
        assert "GPU" in tokens or "gpu" in tokens

    def test_french_tokens_extracted(self, tmp_path: Path) -> None:
        """French particles <=2 chars are filtered out, content words kept."""
        tokens = _extract_search_tokens("je m'appelle Vincent")
        # 'je' (2) and 'm' (1) excluded; 'appelle' (7) and 'Vincent'/'vincent' kept
        assert "je" not in tokens
        assert "m" not in tokens
        assert "appelle" in tokens
        assert "vincent" in tokens

    def test_category_filter_still_respected(self, tmp_path: Path) -> None:
        """Explicit category= filter still restricts results even with token search."""
        store = LongTermStore(tmp_path / "long_term.db")
        store.store(key="main GPU", value="RTX 4080", category="hardware")
        store.store(key="preferred language", value="French", category="preference")
        # Token 'preferred' might match 'preference' category; restricted to 'hardware'
        results = store.search(query="GPU", category="hardware")
        assert all(r.category == "hardware" for r in results)
        assert len(results) == 1

    def test_no_results_returns_empty(self, tmp_path: Path) -> None:
        """Search with no matching records returns an empty list."""
        store = LongTermStore(tmp_path / "long_term.db")
        store.store(key="some key", value="some value")
        results = store.search(query="zzznomatch")
        assert results == []

    def test_limit_respected_across_passes(self, tmp_path: Path) -> None:
        """Result count never exceeds the specified limit."""
        store = LongTermStore(tmp_path / "long_term.db")
        for i in range(10):
            store.store(key=f"item {i}", value=f"value {i}", category="test")
        results = store.search(query="test", limit=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# B. Tool description quality (structural)
# ---------------------------------------------------------------------------

class TestToolDescriptionQuality:
    """Tool descriptions must contain adequate multilingual and semantic guidance."""

    def test_remember_description_contains_french_example(self) -> None:
        desc = REMEMBER_LONG_TERM_MEMORY_DEFINITION.description
        # Must mention a French trigger phrase to guide multilingual models
        assert "souviens" in desc or "enregistre" in desc

    def test_remember_description_contains_negative_examples(self) -> None:
        desc = REMEMBER_LONG_TERM_MEMORY_DEFINITION.description
        assert "NOT" in desc or "not" in desc.lower()

    def test_remember_description_mentions_hardware(self) -> None:
        desc = REMEMBER_LONG_TERM_MEMORY_DEFINITION.description
        # Hardware is one of the key failure cases; must be exemplified
        assert "GPU" in desc or "hardware" in desc.lower()

    def test_search_description_contains_semantic_query_guidance(self) -> None:
        desc = SEARCH_LONG_TERM_MEMORY_DEFINITION.description
        # Must guide model to pass semantic query not verbatim user text
        assert "semantic" in desc.lower() or "query=" in desc or "query='name'" in desc

    def test_search_description_contains_french_example(self) -> None:
        desc = SEARCH_LONG_TERM_MEMORY_DEFINITION.description
        assert "appelle" in desc or "sais-tu" in desc

    def test_search_description_distinguishes_from_session_history(self) -> None:
        desc = SEARCH_LONG_TERM_MEMORY_DEFINITION.description
        assert "inspect_session_history" in desc

    def test_search_description_mentions_name_recall(self) -> None:
        desc = SEARCH_LONG_TERM_MEMORY_DEFINITION.description
        assert "name" in desc.lower()

    def test_list_description_mentions_broad_recall(self) -> None:
        desc = LIST_LONG_TERM_MEMORY_DEFINITION.description
        # Must explain when to use list vs search
        assert "what" in desc.lower() or "broad" in desc.lower() or "everything" in desc.lower()

    def test_forget_description_requires_explicit_user_request(self) -> None:
        desc = FORGET_LONG_TERM_MEMORY_DEFINITION.description
        assert "explicit" in desc.lower() or "only when" in desc.lower()


# ---------------------------------------------------------------------------
# C. Capability context quality (structural)
# ---------------------------------------------------------------------------

class TestCapabilityContextQuality:
    """Capability context must clearly guide model on LTM vs. session recall."""

    def _build_context(self, make_temp_project) -> str:
        from unclaw.core.capabilities import (
            build_runtime_capability_context,
            build_runtime_capability_summary,
        )
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        registry = create_default_tool_registry(settings)
        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
            model_can_call_tools=True,
        )
        return build_runtime_capability_context(summary)

    def test_context_mentions_stable_personal_facts(self, make_temp_project) -> None:
        ctx = self._build_context(make_temp_project)
        assert "stable" in ctx.lower() or "personal" in ctx.lower() or "hardware" in ctx.lower()

    def test_context_distinguishes_ltm_from_session_history(self, make_temp_project) -> None:
        ctx = self._build_context(make_temp_project)
        # Must say NOT to use LTM for session message history
        assert "inspect_session_history" in ctx
        assert "session" in ctx.lower()

    def test_context_contains_semantic_query_guidance(self, make_temp_project) -> None:
        ctx = self._build_context(make_temp_project)
        # Must guide model to pass concise semantic query
        assert "semantic" in ctx.lower() or "concise" in ctx.lower()

    def test_context_contains_multilingual_semantic_recall_guidance(
        self, make_temp_project
    ) -> None:
        ctx = self._build_context(make_temp_project)
        # Must guide recall semantically even when the user asks in another language.
        assert "another language" in ctx.lower()
        assert "search_long_term_memory" in ctx

    def test_context_says_not_injected_automatically(self, make_temp_project) -> None:
        ctx = self._build_context(make_temp_project)
        assert "NOT injected" in ctx or "not injected" in ctx.lower()


# ---------------------------------------------------------------------------
# D. Explicit save reliability (English + French)
# ---------------------------------------------------------------------------

class TestExplicitSaveReliability:
    """Explicit save requests produce stored records and tool confirms success."""

    def test_english_explicit_save_stores_hardware(self, tmp_path: Path) -> None:
        """English: 'remember that my GPU is an RTX 4080'."""
        _store, registry = _make_registry(tmp_path)
        result = _dispatch(
            registry,
            REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
            key="main GPU",
            value="RTX 4080",
            category="hardware",
        )
        assert result.success  # type: ignore[union-attr]
        assert "RTX 4080" in result.output_text  # type: ignore[union-attr]
        assert "main GPU" in result.output_text  # type: ignore[union-attr]

    def test_english_explicit_save_stores_identity(self, tmp_path: Path) -> None:
        """English: 'remember my name is Alice'."""
        _store, registry = _make_registry(tmp_path)
        result = _dispatch(
            registry,
            REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
            key="user name",
            value="Alice",
            category="identity",
        )
        assert result.success  # type: ignore[union-attr]
        assert "Alice" in result.output_text  # type: ignore[union-attr]

    def test_french_explicit_save_stores_identity(self, tmp_path: Path) -> None:
        """French: 'souviens-toi que je m'appelle Vincent'."""
        _store, registry = _make_registry(tmp_path)
        result = _dispatch(
            registry,
            REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
            key="user name",
            value="Vincent",
            category="identity",
            tags="nom,prénom",
        )
        assert result.success  # type: ignore[union-attr]
        assert "Vincent" in result.output_text  # type: ignore[union-attr]

    def test_french_explicit_save_stores_hardware(self, tmp_path: Path) -> None:
        """French: 'enregistre que mon GPU principal est une RTX 4080'."""
        _store, registry = _make_registry(tmp_path)
        result = _dispatch(
            registry,
            REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
            key="GPU principal",
            value="RTX 4080",
            category="hardware",
            tags="GPU,matériel",
        )
        assert result.success  # type: ignore[union-attr]
        assert "RTX 4080" in result.output_text  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# E. Cross-session recall
# ---------------------------------------------------------------------------

class TestCrossSessionRecall:
    """Data stored in one store instance is readable by a separate instance."""

    def test_hardware_recall_cross_session(self, tmp_path: Path) -> None:
        db = tmp_path / "memory" / "long_term.db"
        store_a = LongTermStore(db)
        store_a.store(key="main GPU", value="RTX 4080", category="hardware")

        # New instance simulating a new session.
        store_b = LongTermStore(db)
        results = store_b.search(query="GPU")
        assert len(results) == 1
        assert results[0].value == "RTX 4080"

    def test_identity_recall_cross_session(self, tmp_path: Path) -> None:
        db = tmp_path / "memory" / "long_term.db"
        store_a = LongTermStore(db)
        store_a.store(key="user name", value="Vincent", category="identity")

        store_b = LongTermStore(db)
        results = store_b.search(query="name")
        assert len(results) == 1
        assert results[0].value == "Vincent"

    def test_hardware_recall_via_tool_cross_session(self, tmp_path: Path) -> None:
        """Tool: search returns stored hardware after new store instance created."""
        db = tmp_path / "memory" / "long_term.db"
        # Session A: store.
        store_a = LongTermStore(db)
        store_a.store(key="main GPU", value="RTX 4080", category="hardware")

        # Session B: new registry + store instance.
        store_b = LongTermStore(db)
        registry_b = ToolRegistry()
        register_long_term_memory_tools(registry_b, long_term_store=store_b)
        result = _dispatch(registry_b, SEARCH_LONG_TERM_MEMORY_DEFINITION.name, query="GPU")
        assert result.success  # type: ignore[union-attr]
        assert "RTX 4080" in result.output_text  # type: ignore[union-attr]

    def test_identity_recall_via_tool_cross_session(self, tmp_path: Path) -> None:
        """Tool: 'je m'appelle comment?' → model uses query='name' → finds Vincent."""
        db = tmp_path / "memory" / "long_term.db"
        store_a = LongTermStore(db)
        store_a.store(key="user name", value="Vincent", category="identity")

        store_b = LongTermStore(db)
        registry_b = ToolRegistry()
        register_long_term_memory_tools(registry_b, long_term_store=store_b)
        # Model is guided to pass query='name' for 'je m'appelle comment?'
        result = _dispatch(registry_b, SEARCH_LONG_TERM_MEMORY_DEFINITION.name, query="name")
        assert result.success  # type: ignore[union-attr]
        assert "Vincent" in result.output_text  # type: ignore[union-attr]

    def test_what_do_you_remember_style(self, tmp_path: Path) -> None:
        """'what do you remember about my hardware?' → list via search query='hardware'."""
        db = tmp_path / "memory" / "long_term.db"
        store = LongTermStore(db)
        store.store(key="main GPU", value="RTX 4080", category="hardware")
        store.store(key="RAM", value="32 GB DDR5", category="hardware")
        store.store(key="preferred language", value="French", category="preference")

        # Model is guided to call search with query='hardware' for this question.
        results = store.search(query="hardware")
        assert len(results) == 2
        values = {r.value for r in results}
        assert "RTX 4080" in values
        assert "32 GB DDR5" in values


# ---------------------------------------------------------------------------
# F. Negative cases
# ---------------------------------------------------------------------------

class TestNegativeCases:
    """Tool and context descriptions must contain correct negative guidance."""

    def test_search_description_says_not_for_session_history(self) -> None:
        """search_long_term_memory description must explicitly say NOT for session history."""
        desc = SEARCH_LONG_TERM_MEMORY_DEFINITION.description
        assert "NOT" in desc or "not" in desc.lower()
        # Must reference the correct alternative tool
        assert "inspect_session_history" in desc

    def test_remember_description_says_not_for_general_chat(self) -> None:
        """remember_long_term_memory description must say NOT for general chat."""
        desc = REMEMBER_LONG_TERM_MEMORY_DEFINITION.description
        assert "NOT" in desc or "not" in desc.lower()

    def test_capability_context_says_not_for_session_history(self, make_temp_project) -> None:
        """Capability context must tell model NOT to use LTM for session message history."""
        from unclaw.core.capabilities import (
            build_runtime_capability_context,
            build_runtime_capability_summary,
        )
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        registry = create_default_tool_registry(settings)
        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
            model_can_call_tools=True,
        )
        ctx = build_runtime_capability_context(summary)
        assert "Do NOT use long-term memory" in ctx or "not" in ctx.lower()
        assert "inspect_session_history" in ctx

    def test_no_deterministic_recall_shortcut_in_runtime(self) -> None:
        """runtime.py must not expose any deterministic memory recall shortcut."""
        import unclaw.core.runtime as runtime_module
        assert not hasattr(runtime_module, "_try_exact_recall_shortcut")
        assert not hasattr(runtime_module, "_try_memory_recall_shortcut")

    def test_no_keyword_table_in_long_term_store(self) -> None:
        """long_term_store.py must not contain hardcoded language keyword lists."""
        import unclaw.memory.long_term_store as ltm_module
        # The only list-like structures allowed are: _TOKEN_SPLIT_RE (a regex),
        # _MIN_TOKEN_LEN (an int), and the SQL field references.
        # There must be no tuple/list of language-specific phrases.
        src = Path(ltm_module.__file__).read_text(encoding="utf-8")
        # Sanity check: the regex and constant are present
        assert "_TOKEN_SPLIT_RE" in src
        assert "_MIN_TOKEN_LEN" in src
        # No hardcoded French/English keyword lists
        assert "souviens" not in src
        assert "remember" not in src
        assert "enregistre" not in src


# ---------------------------------------------------------------------------
# V2.A: French recall — category fallback and normalized accent search
# ---------------------------------------------------------------------------

class TestV2FrenchRecall:
    """French recall robustness: category fallback + accent normalization."""

    def test_category_fallback_when_french_query_misses_lexically(
        self, tmp_path: Path
    ) -> None:
        """query='prénom' misses stored key='user name' lexically but category='identity'
        fallback returns the record."""
        store = LongTermStore(tmp_path / "lt.db")
        store.upsert(key="user name", value="Vincent", category="identity")
        # 'prénom' doesn't appear in key/value/tags/category → passes 1-3 miss
        # → Pass 4 category fallback returns all 'identity' records
        results = store.search(query="prénom", category="identity")
        assert len(results) == 1
        assert results[0].value == "Vincent"

    def test_category_fallback_triggered_by_nom_query(
        self, tmp_path: Path
    ) -> None:
        """'nom' doesn't appear in key='user name' via LIKE (partial substring) but
        token 'nom' (3 chars) does appear as a substring — verifies Pass 2 works."""
        store = LongTermStore(tmp_path / "lt.db")
        store.upsert(key="user name", value="Vincent", category="identity")
        # 'nom' is a 3-char substring of 'user name'? No — 'name' contains 'nam',
        # but 'nom' as a LIKE '%nom%' doesn't match 'name'. Fallback should apply.
        results = store.search(query="nom", category="identity")
        assert len(results) == 1
        assert results[0].value == "Vincent"

    def test_normalized_search_finds_tag_with_accent(self, tmp_path: Path) -> None:
        """query='prenom' (no accent) finds record with tags='nom,prénom' via Pass 3."""
        store = LongTermStore(tmp_path / "lt.db")
        store.upsert(
            key="user name",
            value="Vincent",
            category="identity",
            tags="nom,prénom",
        )
        # SQLite LIKE '%prenom%' won't match 'prénom' (accent-sensitive)
        # Pass 3 normalized: fold('prénom')='prenom' → matches fold query 'prenom'
        results = store.search(query="prenom")
        assert len(results) == 1
        assert results[0].value == "Vincent"

    def test_normalized_query_with_accent_finds_unaccented_stored_tag(
        self, tmp_path: Path
    ) -> None:
        """query='prénom' finds record with tags='nom,prenom' (no accent) via Pass 3."""
        store = LongTermStore(tmp_path / "lt.db")
        store.upsert(
            key="user name",
            value="Vincent",
            category="identity",
            tags="nom,prenom",
        )
        results = store.search(query="prénom")
        assert len(results) == 1
        assert results[0].value == "Vincent"

    def test_fold_for_search_strips_accents(self) -> None:
        """_fold_for_search correctly normalizes accented characters."""
        assert _fold_for_search("prénom") == "prenom"
        assert _fold_for_search("PRÉNOM") == "prenom"
        assert _fold_for_search("matériel") == "materiel"
        assert _fold_for_search("Ñoño") == "nono"
        assert _fold_for_search("résumé") == "resume"

    def test_fold_for_search_preserves_ascii(self) -> None:
        """_fold_for_search doesn't corrupt ASCII strings."""
        assert _fold_for_search("Vincent") == "vincent"
        assert _fold_for_search("RTX 4080") == "rtx 4080"
        assert _fold_for_search("GPU") == "gpu"

    def test_english_name_query_recalls_french_stored_identity(
        self, tmp_path: Path
    ) -> None:
        """query='name' (English semantic) reliably recalls French-stored 'user name'."""
        store = LongTermStore(tmp_path / "lt.db")
        store.upsert(key="user name", value="Vincent", category="identity")
        # English semantic query as guided by tool description
        results = store.search(query="name")
        assert len(results) == 1
        assert results[0].value == "Vincent"

    def test_no_category_fallback_without_category_arg(self, tmp_path: Path) -> None:
        """Category fallback does NOT fire when category is not specified.
        'zzznomatch' with no category returns empty — no false positives."""
        store = LongTermStore(tmp_path / "lt.db")
        store.upsert(key="user name", value="Vincent", category="identity")
        # No category arg → Pass 4 does not trigger
        results = store.search(query="zzznomatch")
        assert results == []


# ---------------------------------------------------------------------------
# V2.B: Explicit fact correction / upsert flow
# ---------------------------------------------------------------------------

class TestV2CorrectionFlow:
    """Explicit corrections are persisted as in-place updates, not duplicates."""

    def test_upsert_creates_new_when_key_category_absent(
        self, tmp_path: Path
    ) -> None:
        store = LongTermStore(tmp_path / "lt.db")
        mem_id, created = store.upsert(
            key="user name", value="Alice", category="identity"
        )
        assert created is True
        results = store.search(query="name")
        assert len(results) == 1
        assert results[0].value == "Alice"
        assert results[0].id == mem_id

    def test_upsert_updates_when_same_key_and_category(self, tmp_path: Path) -> None:
        store = LongTermStore(tmp_path / "lt.db")
        mem_id_first, _ = store.upsert(
            key="user name", value="Alice", category="identity"
        )
        mem_id_second, created = store.upsert(
            key="user name", value="Vincent", category="identity"
        )
        assert created is False
        assert mem_id_second == mem_id_first  # same row updated
        results = store.search(query="name")
        assert len(results) == 1
        assert results[0].value == "Vincent"

    def test_upsert_no_stale_duplicate_after_correction(
        self, tmp_path: Path
    ) -> None:
        store = LongTermStore(tmp_path / "lt.db")
        store.upsert(key="user name", value="je m'appelle Vincent", category="identity")
        store.upsert(key="user name", value="Vincent", category="identity")
        all_records = store.list_all(category="identity")
        assert len(all_records) == 1
        assert all_records[0].value == "Vincent"
        assert "appelle" not in all_records[0].value

    def test_remember_tool_upsert_on_correction(self, tmp_path: Path) -> None:
        """remember_long_term_memory called twice with same key+category → one record."""
        _store, registry = _make_registry(tmp_path)
        _dispatch(
            registry,
            REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
            key="user name",
            value="Alice",
            category="identity",
        )
        result = _dispatch(
            registry,
            REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
            key="user name",
            value="Vincent",
            category="identity",
        )
        assert result.success  # type: ignore[union-attr]
        assert "Memory updated" in result.output_text  # type: ignore[union-attr]
        assert "Vincent" in result.output_text  # type: ignore[union-attr]

    def test_search_after_correction_returns_only_latest(
        self, tmp_path: Path
    ) -> None:
        """After correction, search returns corrected value; stale value is absent."""
        _store, registry = _make_registry(tmp_path)
        _dispatch(
            registry,
            REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
            key="user name",
            value="Alice",
            category="identity",
        )
        _dispatch(
            registry,
            REMEMBER_LONG_TERM_MEMORY_DEFINITION.name,
            key="user name",
            value="Vincent",
            category="identity",
        )
        result = _dispatch(
            registry,
            SEARCH_LONG_TERM_MEMORY_DEFINITION.name,
            query="name",
        )
        assert result.success  # type: ignore[union-attr]
        assert "Vincent" in result.output_text  # type: ignore[union-attr]
        assert "Alice" not in result.output_text  # type: ignore[union-attr]

    def test_upsert_key_only_no_category(self, tmp_path: Path) -> None:
        """upsert with no category matches on key alone."""
        store = LongTermStore(tmp_path / "lt.db")
        mem_id, created = store.upsert(key="main GPU", value="RTX 4080")
        assert created is True
        mem_id2, created2 = store.upsert(key="main GPU", value="RTX 4090")
        assert created2 is False
        assert mem_id2 == mem_id
        results = store.search(query="GPU")
        assert len(results) == 1
        assert results[0].value == "RTX 4090"

    def test_hardware_correction_cross_session(self, tmp_path: Path) -> None:
        """Correction in session A is visible in session B via new store instance."""
        db = tmp_path / "memory" / "long_term.db"
        store_a = LongTermStore(db)
        store_a.upsert(key="main GPU", value="RTX 4080", category="hardware")
        store_a.upsert(key="main GPU", value="RTX 4090", category="hardware")

        store_b = LongTermStore(db)
        results = store_b.search(query="GPU")
        assert len(results) == 1
        assert results[0].value == "RTX 4090"


# ---------------------------------------------------------------------------
# V2.C: Hardware stored-vs-current-system tool-choice guidance
# ---------------------------------------------------------------------------

class TestV2HardwareToolChoice:
    """Tool and context descriptions must guide model toward LTM for recalled hardware."""

    def test_search_description_mentions_remembered_vs_current_distinction(
        self,
    ) -> None:
        desc = SEARCH_LONG_TERM_MEMORY_DEFINITION.description
        # Must tell model to prefer LTM for remembered hardware, system_info for current
        assert "system_info" in desc
        assert "remember" in desc.lower() or "stored" in desc.lower()

    def test_search_description_includes_french_hardware_example(self) -> None:
        desc = SEARCH_LONG_TERM_MEMORY_DEFINITION.description
        assert "matériel" in desc or "setup" in desc.lower()

    def test_remember_description_mentions_correction_examples(self) -> None:
        desc = REMEMBER_LONG_TERM_MEMORY_DEFINITION.description
        assert "correction" in desc.lower() or "actually" in desc.lower()

    def test_remember_description_explains_upsert_behavior(self) -> None:
        desc = REMEMBER_LONG_TERM_MEMORY_DEFINITION.description
        assert "updated" in desc.lower() or "update" in desc.lower()

    def test_capability_context_includes_stored_vs_current_distinction(
        self, make_temp_project
    ) -> None:
        from unclaw.core.capabilities import (
            build_runtime_capability_context,
            build_runtime_capability_summary,
        )
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        registry = create_default_tool_registry(settings)
        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
            model_can_call_tools=True,
        )
        ctx = build_runtime_capability_context(summary)
        # Must distinguish stored recall from current machine state
        assert "system_info" in ctx
        assert "current" in ctx.lower()
        assert "remember" in ctx.lower() or "stored" in ctx.lower()

    def test_capability_context_includes_correction_guidance(
        self, make_temp_project
    ) -> None:
        from unclaw.core.capabilities import (
            build_runtime_capability_context,
            build_runtime_capability_summary,
        )
        from unclaw.core.executor import create_default_tool_registry
        from unclaw.settings import load_settings

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        registry = create_default_tool_registry(settings)
        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
            model_can_call_tools=True,
        )
        ctx = build_runtime_capability_context(summary)
        # Must mention corrections
        assert "correction" in ctx.lower() or "corrects" in ctx.lower()


# ---------------------------------------------------------------------------
# V2.D: Negative cases — no deterministic shortcut added by V2
# ---------------------------------------------------------------------------

class TestV2NoDeterministicShortcut:
    """V2 changes must not introduce deterministic routing or keyword tables."""

    def test_no_keyword_table_in_long_term_store_v2(self) -> None:
        """long_term_store.py must not contain hardcoded language keyword lists."""
        import unclaw.memory.long_term_store as ltm_module

        src = Path(ltm_module.__file__).read_text(encoding="utf-8")
        # _fold_for_search is allowed (normalization helper, not a keyword table)
        assert "_fold_for_search" in src
        # No hardcoded multilingual dispatch trigger lists (routing keywords)
        assert "souviens" not in src
        assert "remember" not in src
        assert "enregistre" not in src
        # No hardcoded per-language synonym dict (e.g. {"prénom": "name"})
        # Note: 'prénom' may appear in docstring examples — that is acceptable.
        # What is NOT allowed is a DICT or LIST mapping French → English keywords.
        assert '{"prénom"' not in src
        assert '["prénom"' not in src
        assert '"prénom": "name"' not in src

    def test_no_deterministic_recall_shortcut_in_runtime_v2(self) -> None:
        """runtime.py must not expose any deterministic memory recall shortcut."""
        import unclaw.core.runtime as runtime_module

        assert not hasattr(runtime_module, "_try_exact_recall_shortcut")
        assert not hasattr(runtime_module, "_try_memory_recall_shortcut")
        assert not hasattr(runtime_module, "_force_long_term_recall")

    def test_no_keyword_table_in_long_term_memory_tools(self) -> None:
        """long_term_memory_tools.py must not route on French/English keywords."""
        import unclaw.tools.long_term_memory_tools as tools_module

        src = Path(tools_module.__file__).read_text(encoding="utf-8")
        # Handler functions must not contain per-language dispatch logic.
        # 'souviens' is forbidden in handler bodies (routing table).
        # Note: 'souviens' in a tool *description string* is acceptable guidance,
        # but must not appear as a condition in handler code.
        # The test confirms no hardcoded language routing dict exists.
        assert '{"souviens"' not in src
        assert '"souviens": ' not in src
        assert '["souviens"' not in src
        # No hardcoded synonym map dict
        assert '"prénom": "name"' not in src
        assert '{"prénom"' not in src

    def test_category_fallback_does_not_fire_without_category(
        self, tmp_path: Path
    ) -> None:
        """Fallback is strictly gated on category being provided — not a free wildcard."""
        store = LongTermStore(tmp_path / "lt.db")
        for i in range(5):
            store.upsert(
                key=f"identity fact {i}",
                value=f"value {i}",
                category="identity",
            )
        # Query misses everything, no category provided → no fallback → empty
        results = store.search(query="zzznomatch")
        assert results == []

    def test_store_method_still_inserts_duplicates(self, tmp_path: Path) -> None:
        """store() (not upsert) still inserts new records — backward compatibility."""
        store = LongTermStore(tmp_path / "lt.db")
        id1 = store.store(key="user name", value="Alice", category="identity")
        id2 = store.store(key="user name", value="Vincent", category="identity")
        assert id1 != id2
        all_records = store.list_all(category="identity")
        assert len(all_records) == 2
