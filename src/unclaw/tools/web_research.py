"""Model-driven 3-layer research pipeline for web search.

Layer A — discovery (handled by existing web_search / web_retrieval)
Layer B — per-source condensation (model-driven)
Layer C — merged research note (model-driven)

The pipeline produces a compact merged research note that small-context
local models can consume without raw page injection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from unclaw.llm.base import LLMMessage, LLMProviderError, LLMRole
from unclaw.tools.web_text import _clip_text, _normalize_text


# ---------------------------------------------------------------------------
# Budget system — percentage-based with min/max clamps
# ---------------------------------------------------------------------------

_RESERVED_SYSTEM_OVERHEAD = 2000
_MIN_USABLE_CONTEXT = 1000


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


# ---------------------------------------------------------------------------
# Condensation prompts
# ---------------------------------------------------------------------------

_SOURCE_CONDENSATION_SYSTEM_PROMPT = (
    "You are a factual research assistant. Read the source text below and "
    "write a compact research note. Include only facts, findings, and "
    "specific details relevant to the query. Note any uncertainty or "
    "limitations. Be precise and concise — no commentary, no filler."
)

_MERGE_SYSTEM_PROMPT = (
    "You are a factual research assistant. Merge these source notes into "
    "one final research note. Highlight facts that appear in multiple "
    "sources. Note any disagreements between sources. Preserve uncertainty "
    "where sources are unclear. Track which sources support each claim. "
    "Stay factual and compact — no commentary, no filler."
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
        "Write a compact research note covering key facts relevant to the "
        "query. Include specific details worth keeping."
    )


def _build_merge_prompt(
    *,
    source_notes: Sequence[SourceNote],
    query: str,
    max_chars: int,
) -> str:
    notes_block = "\n\n".join(
        f"[Source: {note.title}]\n{note.condensed_text}"
        for note in source_notes
    )
    return (
        f"Query: {query}\n"
        f"Max merged note length: {max_chars} characters\n"
        f"Number of sources: {len(source_notes)}\n\n"
        f"Source notes:\n{notes_block}\n\n"
        "Write a merged research note. Highlight repeated facts, note "
        "disagreements, preserve uncertainty, and track source attribution."
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
                condensed_text=model_output,
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
    """Build a deterministic source note by extracting opening sentences."""
    normalized = _normalize_text(text)
    if not normalized:
        return "[No extractable content]"
    return _clip_text(normalized, limit=max_chars)


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
                text=model_output,
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
    """Build a deterministic merged note by concatenating source notes."""
    parts: list[str] = []
    remaining = max_chars

    for note in source_notes:
        header = f"[{note.title}]"
        entry_text = f"{header} {note.condensed_text}"
        if len(entry_text) > remaining:
            clipped = _clip_text(entry_text, limit=remaining)
            parts.append(clipped)
            break
        parts.append(entry_text)
        remaining -= len(entry_text) + 1  # +1 for separator

    return MergedResearchNote(
        text="\n".join(parts) if parts else "No source notes available.",
        source_count=len(source_notes),
        model_generated=False,
    )


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

    parts: list[str] = [f"Quick grounding for: {query}"]
    remaining = max_chars - len(parts[0]) - 1

    for title, url, snippet_text in snippets:
        normalized_snippet = _normalize_text(snippet_text)
        if not normalized_snippet:
            continue
        entry = f"- {title}: {_clip_text(normalized_snippet, limit=min(remaining - 3, 120))}"
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
            lines.append(f"{index}. {note.title}")
            lines.append(f"   URL: {note.url}")

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
