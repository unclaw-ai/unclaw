"""Model-driven 3-layer research pipeline for web search.

Layer A — discovery (handled by existing web_search / web_retrieval)
Layer B — per-source condensation (model-driven)
Layer C — merged research note (model-driven)

The pipeline produces a compact merged research note that small-context
local models can consume without raw page injection.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from unclaw.llm.base import LLMMessage, LLMProviderError, LLMRole
from unclaw.tools.web_search import _build_search_query
from unclaw.tools.web_text import (
    _build_signature_tokens,
    _clip_text,
    _content_tokens,
    _evidence_similarity,
    _extract_subject_tokens,
    _fold_for_match,
    _looks_promotional,
    _looks_site_descriptive,
    _normalize_text,
    _passage_has_noise_signals,
    _split_sentences,
    _strip_terminal_punctuation,
    _substring_similarity,
)


# ---------------------------------------------------------------------------
# Budget system — percentage-based with min/max clamps
# ---------------------------------------------------------------------------

_RESERVED_SYSTEM_OVERHEAD = 2000
_MIN_USABLE_CONTEXT = 1000
_MAX_SOURCE_NOTE_CLAUSES = 4
_MIN_NOTE_CONTENT_TOKENS = 2


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


@dataclass(frozen=True, slots=True)
class ResearchBudget:
    """Profile-aware budget for the 3-layer research pipeline.

    All character limits adapt to the effective context window of the
    active model profile, with safe min/max clamps.
    """

    max_sources: int
    max_source_chars: int
    max_source_note_chars: int
    max_merged_note_chars: int
    condensation_timeout_seconds: float
    fast_grounding_max_results: int = 3
    fast_grounding_max_chars: int = 300


# Pre-computed budget tiers for common profiles.

FAST_RESEARCH_BUDGET = ResearchBudget(
    max_sources=3,
    max_source_chars=1500,
    max_source_note_chars=250,
    max_merged_note_chars=400,
    condensation_timeout_seconds=6.0,
    fast_grounding_max_results=2,
    fast_grounding_max_chars=200,
)

MAIN_RESEARCH_BUDGET = ResearchBudget(
    max_sources=5,
    max_source_chars=3000,
    max_source_note_chars=400,
    max_merged_note_chars=800,
    condensation_timeout_seconds=8.0,
    fast_grounding_max_results=3,
    fast_grounding_max_chars=300,
)

DEEP_RESEARCH_BUDGET = ResearchBudget(
    max_sources=8,
    max_source_chars=5000,
    max_source_note_chars=600,
    max_merged_note_chars=1200,
    condensation_timeout_seconds=10.0,
    fast_grounding_max_results=3,
    fast_grounding_max_chars=400,
)


def resolve_research_budget(
    *,
    effective_context: int,
    profile_name: str,
) -> ResearchBudget:
    """Compute a research budget from effective context window and profile.

    The budget adapts automatically if context windows change.  Percentage-
    based stage allocations are clamped to safe min/max ranges.
    """
    available = max(effective_context - _RESERVED_SYSTEM_OVERHEAD, _MIN_USABLE_CONTEXT)
    normalized_profile = profile_name.strip().lower()

    if normalized_profile in {"fast", "codex"}:
        return ResearchBudget(
            max_sources=3,
            max_source_chars=_clamp(int(available * 0.30), 800, 2500),
            max_source_note_chars=_clamp(int(available * 0.08), 150, 400),
            max_merged_note_chars=_clamp(int(available * 0.12), 200, 600),
            condensation_timeout_seconds=6.0,
            fast_grounding_max_results=2,
            fast_grounding_max_chars=_clamp(int(available * 0.06), 120, 250),
        )

    if normalized_profile == "deep":
        return ResearchBudget(
            max_sources=8,
            max_source_chars=_clamp(int(available * 0.40), 2000, 8000),
            max_source_note_chars=_clamp(int(available * 0.12), 300, 800),
            max_merged_note_chars=_clamp(int(available * 0.20), 500, 1500),
            condensation_timeout_seconds=10.0,
            fast_grounding_max_results=3,
            fast_grounding_max_chars=_clamp(int(available * 0.08), 200, 400),
        )

    # main / default
    return ResearchBudget(
        max_sources=5,
        max_source_chars=_clamp(int(available * 0.35), 1500, 5000),
        max_source_note_chars=_clamp(int(available * 0.10), 200, 600),
        max_merged_note_chars=_clamp(int(available * 0.15), 400, 1000),
        condensation_timeout_seconds=8.0,
        fast_grounding_max_results=3,
        fast_grounding_max_chars=_clamp(int(available * 0.07), 150, 300),
    )


# ---------------------------------------------------------------------------
# Research workspace — lightweight in-memory artifacts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SourceArtifact:
    """One fetched source with cleaned text ready for condensation."""

    url: str
    title: str
    cleaned_text: str
    fetch_success: bool
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class SourceNote:
    """Per-source condensed research note (Layer B output)."""

    url: str
    title: str
    condensed_text: str
    model_generated: bool = True


@dataclass(frozen=True, slots=True)
class MergedResearchNote:
    """Merged research note from all source notes (Layer C output)."""

    text: str
    source_count: int
    model_generated: bool = True


@dataclass(slots=True)
class ResearchWorkspace:
    """Lightweight in-memory workspace for intermediate research artifacts.

    Can later support file-backed caching or local RAG evolution without
    changing the interface.
    """

    query: str
    budget: ResearchBudget
    source_artifacts: list[SourceArtifact] = field(default_factory=list)
    source_notes: list[SourceNote] = field(default_factory=list)
    merged_note: MergedResearchNote | None = None

    @property
    def source_count(self) -> int:
        return len(self.source_artifacts)

    @property
    def has_merged_note(self) -> bool:
        return self.merged_note is not None

    def add_source(self, artifact: SourceArtifact) -> bool:
        """Add a source artifact if within budget.  Returns False if full."""
        if len(self.source_artifacts) >= self.budget.max_sources:
            return False
        self.source_artifacts.append(artifact)
        return True


@dataclass(frozen=True, slots=True)
class _NoteClause:
    """One compact factual clause extracted from a source note."""

    text: str
    source_index: int
    signature_tokens: tuple[str, ...]
    content_tokens: tuple[str, ...]
    subject_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NoteCluster:
    """One clustered claim retained for the merged research note."""

    text: str
    source_indices: tuple[int, ...]
    signature_tokens: tuple[str, ...]
    content_tokens: tuple[str, ...]
    subject_tokens: tuple[str, ...]


# ---------------------------------------------------------------------------
# Condensation prompts
# ---------------------------------------------------------------------------

_SOURCE_CONDENSATION_SYSTEM_PROMPT = (
    "You are a factual research assistant. Read the source text and output "
    "only a dense note body. Keep only facts relevant to the query. Prefer "
    "short semicolon-separated clauses. Do not repeat the source title or "
    "URL. Use '? ' only when uncertainty materially matters."
)

_MERGE_SYSTEM_PROMPT = (
    "You are a factual research assistant. Merge source notes into a very "
    "compact final note. Output only note lines. Use '[1,2] claim' for "
    "supported claims and '!= variant [1] / other variant [2]' for clear "
    "disagreements. Keep uncertainty compact. No commentary or headers."
)


def _build_source_condensation_prompt(
    *,
    source_title: str,
    source_url: str,
    source_text: str,
    query: str,
    max_chars: int,
) -> str:
    return (
        f"Source: {source_title}\n"
        f"URL: {source_url}\n"
        f"Query context: {query}\n"
        f"Max note length: {max_chars} characters\n"
        "---\n"
        f"{source_text}\n"
        "---\n"
        "Write the note body only. Keep compact factual clauses relevant to "
        "the query. Do not repeat the source title or URL."
    )


def _build_merge_prompt(
    *,
    source_notes: Sequence[SourceNote],
    query: str,
    max_chars: int,
) -> str:
    notes_block = "\n\n".join(
        f"[{index}] {note.title}\n{note.condensed_text}"
        for index, note in enumerate(source_notes, start=1)
    )
    return (
        f"Query: {query}\n"
        f"Max merged note length: {max_chars} characters\n"
        f"Number of sources: {len(source_notes)}\n\n"
        f"Source notes:\n{notes_block}\n\n"
        "Write the merged note body only using compact claim lines and "
        "numeric source refs."
    )


# ---------------------------------------------------------------------------
# Model-driven condensation engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Configuration for the model-driven research pipeline.

    Captures everything needed to call the LLM for condensation without
    requiring the tool handler to know about settings internals.
    """

    settings: Any  # unclaw.settings.Settings — kept as Any to avoid circular import
    condensation_profile_name: str = "fast"


def _create_condensation_provider(config: ResearchConfig) -> Any | None:
    """Create a lightweight LLM provider for condensation calls."""
    try:
        from unclaw.core import orchestrator as orchestrator_module

        provider_class = getattr(orchestrator_module, "OllamaProvider", None)
        if provider_class is None:
            return None

        return provider_class(
            default_timeout_seconds=(
                config.settings.app.providers.ollama.timeout_seconds
            ),
        )
    except Exception:
        return None


def _resolve_condensation_profile(config: ResearchConfig) -> Any | None:
    """Resolve the model profile used for condensation LLM calls."""
    try:
        from unclaw.llm.model_profiles import resolve_model_profile

        return resolve_model_profile(
            config.settings,
            config.condensation_profile_name,
        )
    except Exception:
        return None


def _call_condensation_model(
    *,
    provider: Any,
    profile: Any,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float,
    max_output_chars: int,
) -> str | None:
    """Run one condensation LLM call.  Returns None on failure."""
    from dataclasses import replace as dataclass_replace

    from unclaw.llm.base import ResolvedModelProfile

    if not isinstance(profile, ResolvedModelProfile):
        return None

    condensation_profile = dataclass_replace(profile, temperature=0.1)
    messages = (
        LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
        LLMMessage(role=LLMRole.USER, content=user_prompt),
    )

    try:
        response = provider.chat(
            profile=condensation_profile,
            messages=messages,
            timeout_seconds=timeout_seconds,
            thinking_enabled=False,
        )
    except (LLMProviderError, Exception):
        return None

    content = response.content.strip()
    if not content:
        return None

    return _clip_text(content, limit=max_output_chars)


# ---------------------------------------------------------------------------
# Layer B — per-source condensation
# ---------------------------------------------------------------------------


def condense_source(
    *,
    artifact: SourceArtifact,
    query: str,
    budget: ResearchBudget,
    config: ResearchConfig | None = None,
    provider: Any | None = None,
    profile: Any | None = None,
) -> SourceNote:
    """Condense one source artifact into a compact research note.

    If a provider/profile are available, uses model-driven condensation.
    Otherwise falls back to deterministic text truncation.
    """
    source_text = _clip_text(
        artifact.cleaned_text,
        limit=budget.max_source_chars,
    )

    if provider is not None and profile is not None:
        user_prompt = _build_source_condensation_prompt(
            source_title=artifact.title,
            source_url=artifact.url,
            source_text=source_text,
            query=query,
            max_chars=budget.max_source_note_chars,
        )
        model_output = _call_condensation_model(
            provider=provider,
            profile=profile,
            system_prompt=_SOURCE_CONDENSATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            timeout_seconds=budget.condensation_timeout_seconds,
            max_output_chars=budget.max_source_note_chars,
        )
        if model_output:
            return SourceNote(
                url=artifact.url,
                title=artifact.title,
                condensed_text=_sanitize_model_note_output(
                    model_output,
                    max_chars=budget.max_source_note_chars,
                    multiline=False,
                ),
                model_generated=True,
            )

    # Deterministic fallback: truncate cleaned text
    fallback_text = _build_deterministic_source_note(
        source_text,
        max_chars=budget.max_source_note_chars,
    )
    return SourceNote(
        url=artifact.url,
        title=artifact.title,
        condensed_text=fallback_text,
        model_generated=False,
    )


def _build_deterministic_source_note(text: str, *, max_chars: int) -> str:
    """Build a dense deterministic source note with minimal formatting overhead."""
    clauses = _collect_compact_source_clauses(text)
    if not clauses:
        return "[No extractable content]"
    return _clip_text("; ".join(clauses), limit=max_chars)


# ---------------------------------------------------------------------------
# Layer C — merged research note
# ---------------------------------------------------------------------------


def merge_source_notes(
    *,
    source_notes: Sequence[SourceNote],
    query: str,
    budget: ResearchBudget,
    config: ResearchConfig | None = None,
    provider: Any | None = None,
    profile: Any | None = None,
) -> MergedResearchNote:
    """Merge per-source notes into one final compact research note.

    Uses model-driven synthesis when a provider is available, otherwise
    falls back to deterministic concatenation.
    """
    if not source_notes:
        return MergedResearchNote(
            text="No sources were available for research.",
            source_count=0,
            model_generated=False,
        )

    if provider is not None and profile is not None and len(source_notes) > 0:
        user_prompt = _build_merge_prompt(
            source_notes=source_notes,
            query=query,
            max_chars=budget.max_merged_note_chars,
        )
        model_output = _call_condensation_model(
            provider=provider,
            profile=profile,
            system_prompt=_MERGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            timeout_seconds=budget.condensation_timeout_seconds,
            max_output_chars=budget.max_merged_note_chars,
        )
        if model_output:
            return MergedResearchNote(
                text=_sanitize_model_note_output(
                    model_output,
                    max_chars=budget.max_merged_note_chars,
                    multiline=True,
                ),
                source_count=len(source_notes),
                model_generated=True,
            )

    # Deterministic fallback: concatenate source notes
    return _build_deterministic_merged_note(
        source_notes=source_notes,
        max_chars=budget.max_merged_note_chars,
    )


def _build_deterministic_merged_note(
    *,
    source_notes: Sequence[SourceNote],
    max_chars: int,
) -> MergedResearchNote:
    """Build a deterministic merged note with compact source refs."""
    clauses = _extract_note_clauses(source_notes)
    clusters = _cluster_note_clauses(clauses)
    lines = _build_merged_note_lines(clusters)
    return MergedResearchNote(
        text=_join_bounded_lines(lines, max_chars=max_chars)
        if lines
        else "No source notes available.",
        source_count=len(source_notes),
        model_generated=False,
    )


def _clean_note_unit(unit: str) -> str:
    text = _normalize_text(unit)
    text = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", text).strip()
    text = re.sub(
        r"^(?:source|title|url|query|note|notes|fact|facts|research note)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _iter_note_units(text: str) -> tuple[str, ...]:
    if not text:
        return ()

    normalized = _normalize_text(text)
    if not normalized:
        return ()

    units: list[str] = []
    for paragraph in normalized.splitlines():
        for chunk in re.split(r"\s*;\s*", paragraph):
            cleaned_chunk = _clean_note_unit(chunk)
            if not cleaned_chunk:
                continue
            sentence_units = _split_sentences(cleaned_chunk) or (cleaned_chunk,)
            for sentence in sentence_units:
                candidate = _clean_note_unit(sentence)
                if candidate:
                    units.append(candidate)
    return tuple(units)


def _note_unit_body(text: str) -> str:
    if text.startswith("!="):
        return text[2:].strip()
    if text.startswith("?"):
        return text[1:].strip()
    return text


def _collect_compact_source_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    seen_signatures: list[tuple[str, ...]] = []

    for candidate in _iter_note_units(text):
        body = _note_unit_body(candidate)
        content_tokens = _content_tokens(body)
        if len(content_tokens) < _MIN_NOTE_CONTENT_TOKENS:
            continue
        if _passage_has_noise_signals(body):
            continue
        if _looks_site_descriptive(body):
            continue
        if _looks_promotional(body):
            continue
        signature_tokens = _build_signature_tokens(body)
        if any(
            _evidence_similarity(signature_tokens, existing_signature) >= 0.8
            or _substring_similarity(body, " ".join(existing_signature)) >= 0.78
            for existing_signature in seen_signatures
        ):
            continue
        clauses.append(_strip_terminal_punctuation(candidate))
        seen_signatures.append(signature_tokens)
        if len(clauses) >= _MAX_SOURCE_NOTE_CLAUSES:
            break

    return clauses


def _sanitize_model_note_output(
    text: str,
    *,
    max_chars: int,
    multiline: bool,
) -> str:
    cleaned_units = [_clean_note_unit(unit) for unit in _normalize_text(text).splitlines()]
    unique_units: list[str] = []
    seen_units: set[str] = set()

    for unit in cleaned_units:
        if not unit:
            continue
        key = unit.casefold()
        if key in seen_units:
            continue
        unique_units.append(unit)
        seen_units.add(key)

    separator = "\n" if multiline else "; "
    compact_text = separator.join(unique_units).strip()
    if not compact_text:
        compact_text = _normalize_text(text)
    return _clip_text(compact_text, limit=max_chars)


def _extract_note_clauses(source_notes: Sequence[SourceNote]) -> tuple[_NoteClause, ...]:
    clauses: list[_NoteClause] = []

    for source_index, note in enumerate(source_notes, start=1):
        for candidate in _iter_note_units(note.condensed_text):
            body = _note_unit_body(candidate)
            content_tokens = _content_tokens(body)
            if len(content_tokens) < _MIN_NOTE_CONTENT_TOKENS:
                continue
            clauses.append(
                _NoteClause(
                    text=_strip_terminal_punctuation(candidate),
                    source_index=source_index,
                    signature_tokens=_build_signature_tokens(body),
                    content_tokens=content_tokens,
                    subject_tokens=_extract_subject_tokens(body),
                )
            )

    return tuple(clauses)


def _note_clause_similarity(left: _NoteClause, right: _NoteClause) -> float:
    return max(
        _evidence_similarity(left.signature_tokens, right.signature_tokens),
        _substring_similarity(left.text, right.text),
    )


def _cluster_note_clauses(clauses: Sequence[_NoteClause]) -> tuple[_NoteCluster, ...]:
    grouped_clauses: list[list[_NoteClause]] = []

    for clause in clauses:
        best_index: int | None = None
        best_similarity = 0.0
        for index, cluster_members in enumerate(grouped_clauses):
            similarity = max(
                _note_clause_similarity(clause, existing)
                for existing in cluster_members
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index

        if best_index is not None and best_similarity >= 0.74:
            grouped_clauses[best_index].append(clause)
            continue

        grouped_clauses.append([clause])

    clusters: list[_NoteCluster] = []
    for cluster_members in grouped_clauses:
        ordered_members = sorted(
            cluster_members,
            key=lambda item: (item.source_index, len(item.text), item.text.casefold()),
        )
        representative = ordered_members[0]
        source_indices = tuple(
            sorted({member.source_index for member in ordered_members})
        )
        subject_tokens = next(
            (
                member.subject_tokens
                for member in ordered_members
                if member.subject_tokens
            ),
            (),
        )
        clusters.append(
            _NoteCluster(
                text=representative.text,
                source_indices=source_indices,
                signature_tokens=representative.signature_tokens,
                content_tokens=representative.content_tokens,
                subject_tokens=subject_tokens,
            )
        )

    return tuple(
        sorted(
            clusters,
            key=lambda item: (-len(item.source_indices), item.source_indices, item.text.casefold()),
        )
    )


def _clusters_disagree(left: _NoteCluster, right: _NoteCluster) -> bool:
    if not left.subject_tokens or not right.subject_tokens:
        return False
    if left.subject_tokens != right.subject_tokens:
        return False
    similarity = max(
        _evidence_similarity(left.signature_tokens, right.signature_tokens),
        _substring_similarity(left.text, right.text),
    )
    return similarity < 0.55


def _format_source_refs(source_indices: tuple[int, ...]) -> str:
    return "[" + ",".join(str(index) for index in source_indices) + "]"


def _build_merged_note_lines(clusters: Sequence[_NoteCluster]) -> list[str]:
    lines: list[str] = []
    conflict_indexes: set[int] = set()

    for index, cluster in enumerate(clusters):
        if index in conflict_indexes or not cluster.subject_tokens:
            continue
        conflict_group: list[tuple[int, _NoteCluster]] = [(index, cluster)]
        for other_index in range(index + 1, len(clusters)):
            other_cluster = clusters[other_index]
            if other_index in conflict_indexes:
                continue
            if _clusters_disagree(cluster, other_cluster):
                conflict_group.append((other_index, other_cluster))
        if len(conflict_group) <= 1:
            continue
        lines.append(
            "!= "
            + " / ".join(
                f"{member.text} {_format_source_refs(member.source_indices)}"
                for _member_index, member in conflict_group
            )
        )
        conflict_indexes.update(member_index for member_index, _member in conflict_group)

    for index, cluster in enumerate(clusters):
        if index in conflict_indexes:
            continue
        lines.append(f"{_format_source_refs(cluster.source_indices)} {cluster.text}")

    return lines


def _join_bounded_lines(lines: Sequence[str], *, max_chars: int) -> str:
    kept_lines: list[str] = []
    remaining = max_chars

    for line in lines:
        normalized_line = _normalize_text(line)
        if not normalized_line:
            continue
        separator_cost = 1 if kept_lines else 0
        line_budget = remaining - separator_cost
        if line_budget <= 0:
            break
        if len(normalized_line) <= line_budget:
            kept_lines.append(normalized_line)
            remaining -= len(normalized_line) + separator_cost
            continue
        kept_lines.append(_clip_text(normalized_line, limit=line_budget))
        break

    return "\n".join(kept_lines)


# ---------------------------------------------------------------------------
# Full research pipeline
# ---------------------------------------------------------------------------


def run_research_pipeline(
    *,
    query: str,
    fetched_pages: Mapping[str, tuple[str, str]],
    budget: ResearchBudget,
    config: ResearchConfig | None = None,
) -> ResearchWorkspace:
    """Run the full 3-layer research pipeline.

    Args:
        query: The user's search query.
        fetched_pages: Mapping of url -> (title, cleaned_text) for each
            successfully fetched source.
        budget: Profile-aware research budget.
        config: Optional research config for model-driven condensation.

    Returns:
        A populated ResearchWorkspace with source artifacts, per-source
        notes, and a merged research note.
    """
    workspace = ResearchWorkspace(query=query, budget=budget)

    # Resolve LLM provider for condensation (may be None)
    provider = None
    profile = None
    if config is not None:
        provider = _create_condensation_provider(config)
        if provider is not None:
            profile = _resolve_condensation_profile(config)

    # Layer A output → Layer B input: build source artifacts
    source_entries = list(fetched_pages.items())[: budget.max_sources]
    for url, (title, cleaned_text) in source_entries:
        artifact = SourceArtifact(
            url=url,
            title=title,
            cleaned_text=cleaned_text,
            fetch_success=True,
            truncated=len(cleaned_text) > budget.max_source_chars,
        )
        workspace.add_source(artifact)

    # Layer B: per-source condensation
    for artifact in workspace.source_artifacts:
        note = condense_source(
            artifact=artifact,
            query=query,
            budget=budget,
            config=config,
            provider=provider,
            profile=profile,
        )
        workspace.source_notes.append(note)

    # Layer C: merge source notes into final research note
    workspace.merged_note = merge_source_notes(
        source_notes=workspace.source_notes,
        query=query,
        budget=budget,
        config=config,
        provider=provider,
        profile=profile,
    )

    return workspace


# ---------------------------------------------------------------------------
# Fast grounding probe
# ---------------------------------------------------------------------------


def build_fast_grounding_note(
    *,
    query: str,
    snippets: Sequence[tuple[str, str, str]],
    max_chars: int,
) -> str:
    """Build a tiny grounding note from search snippets.

    Args:
        query: The entity/topic query.
        snippets: Sequence of (title, url, snippet_text) tuples.
        max_chars: Maximum characters for the grounding note.

    Returns:
        A compact grounding note suitable for entity resolution.
    """
    if not snippets:
        return f"No web results found for: {query}"

    search_query = _build_search_query(query)
    parts: list[str] = [_clip_text(query, limit=max_chars)]
    remaining = max_chars - len(parts[0])

    exact_match_found = False
    for title, url, snippet_text in snippets:
        folded_metadata = _fold_for_match(f"{title} {snippet_text} {url}")
        if (
            search_query.normalized_entity
            and search_query.normalized_entity in folded_metadata
        ) or (
            search_query.entity_tokens
            and sum(
                1 for token in search_query.entity_tokens if token in folded_metadata.split()
            )
            >= len(search_query.entity_tokens)
        ):
            exact_match_found = True
            break

    if search_query.entity_tokens and not exact_match_found:
        warning_line = "!= no exact top match"
        if len(warning_line) + 1 <= remaining:
            parts.append(warning_line)
            remaining -= len(warning_line) + 1

    for title, url, snippet_text in snippets:
        normalized_snippet = _normalize_text(snippet_text)
        if not normalized_snippet:
            continue
        if remaining <= 4:
            break
        folded_metadata = _fold_for_match(f"{title} {snippet_text} {url}")
        entity_hit_count = sum(
            1 for token in search_query.entity_tokens if token in folded_metadata.split()
        )
        aligned = not search_query.entity_tokens or (
            entity_hit_count >= len(search_query.entity_tokens)
            or (
                search_query.normalized_entity
                and search_query.normalized_entity in folded_metadata
            )
        )
        entry_prefix = "- " if aligned else "!= "
        title_matches_query = (
            search_query.normalized_entity
            and _fold_for_match(title) == search_query.normalized_entity
        )
        snippet_limit = min(max(remaining - len(entry_prefix) - 3, 24), 110)
        if title_matches_query:
            entry = f"{entry_prefix}{_clip_text(normalized_snippet, limit=snippet_limit)}"
        else:
            entry = (
                f"{entry_prefix}{title} | "
                f"{_clip_text(normalized_snippet, limit=snippet_limit)}"
            )
        if len(entry) > remaining:
            break
        parts.append(entry)
        remaining -= len(entry) + 1

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------


def format_research_output(
    *,
    workspace: ResearchWorkspace,
    query: str,
    fetch_stats: Mapping[str, Any] | None = None,
) -> str:
    """Format the research workspace into compact tool output text."""
    lines: list[str] = []

    stats = fetch_stats or {}
    lines.append(f"Search query: {query}")
    if stats:
        lines.append(
            f"Sources considered: {stats.get('considered_candidate_count', '?')}"
        )
        lines.append(
            f"Sources fetched: {stats.get('fetch_success_count', '?')} "
            f"of {stats.get('fetch_attempt_count', '?')} attempted"
        )
    lines.append(f"Research sources: {workspace.source_count}")
    lines.append(
        f"Condensation: {'model-driven' if _any_model_generated(workspace) else 'deterministic'}"
    )
    lines.append("")

    if workspace.merged_note is not None:
        lines.append("Research note:")
        lines.append(workspace.merged_note.text)
    else:
        lines.append("No research note was produced.")

    if workspace.source_notes:
        lines.append("")
        lines.append("Sources:")
        for index, note in enumerate(workspace.source_notes, start=1):
            lines.append(f"[{index}] {note.title} | {note.url}")

    return "\n".join(lines)


def _any_model_generated(workspace: ResearchWorkspace) -> bool:
    if workspace.merged_note and workspace.merged_note.model_generated:
        return True
    return any(note.model_generated for note in workspace.source_notes)


__all__ = [
    "DEEP_RESEARCH_BUDGET",
    "FAST_RESEARCH_BUDGET",
    "MAIN_RESEARCH_BUDGET",
    "MergedResearchNote",
    "ResearchBudget",
    "ResearchConfig",
    "ResearchWorkspace",
    "SourceArtifact",
    "SourceNote",
    "build_fast_grounding_note",
    "condense_source",
    "format_research_output",
    "merge_source_notes",
    "resolve_research_budget",
    "run_research_pipeline",
]
