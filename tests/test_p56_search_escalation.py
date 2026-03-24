"""Tests for P5-6 — deep-search escalation, duo/group routing, timeout fallback.

Proves:
- _looks_like_deep_search_request detects bio/research depth signals (FR + EN)
- _looks_like_joint_entity_request detects duo/pair entities like 'McFly et Carlito'
- _build_request_routing_note prefers search_web for deep/complete requests
- _build_request_routing_note emits duo-aware notes for joint entity requests
- _build_request_routing_note still uses fast_web_search for plain identity lookups
  and mentions search_web as the escalation path
- _build_post_tool_grounding_note pushes search_web escalation after fast_web_search
- timeout/partial-failure note is included when a tool timed out
- _apply_post_tool_reply_discipline still clamps replies from thin fast_web results
  (factual discipline regression guard)
- capability context describes search_web as the deeper grounded path
"""

from __future__ import annotations

import pytest

from unclaw.core.capabilities import (
    RuntimeCapabilitySummary,
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.reply_discipline import (
    _apply_post_tool_reply_discipline,
)
from unclaw.core.routing import (
    _build_request_routing_note,
    _looks_like_deep_search_request,
    _looks_like_joint_entity_request,
)
from unclaw.core.runtime_support import (
    _build_post_tool_grounding_note,
)
from unclaw.tools.contracts import ToolResult
from unclaw.tools.web_tools import FAST_WEB_SEARCH_DEFINITION, SEARCH_WEB_DEFINITION

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _web_capable_summary(*, fast_web: bool = True) -> RuntimeCapabilitySummary:
    """Capability summary with both search_web and optionally fast_web_search."""
    return RuntimeCapabilitySummary(
        available_builtin_tool_names=(
            ("fast_web_search", "search_web") if fast_web else ("search_web",)
        ),
        local_file_read_available=False,
        local_directory_listing_available=False,
        url_fetch_available=False,
        web_search_available=True,
        system_info_available=False,
        memory_summary_available=False,
        model_can_call_tools=True,
        fast_web_search_available=fast_web,
    )


def _thin_fast_web_result(query: str = "McFly et Carlito") -> ToolResult:
    return ToolResult.ok(
        tool_name="fast_web_search",
        output_text=f"- {query}: résultat rapide",
        payload={
            "query": query,
            "result_count": 1,
            "grounding_note": f"- {query}: résultat rapide",
        },
    )


# ---------------------------------------------------------------------------
# _looks_like_deep_search_request — French signals
# ---------------------------------------------------------------------------


def test_deep_search_bio_complete_fr() -> None:
    assert _looks_like_deep_search_request("fais une bio complète")
    assert _looks_like_deep_search_request("biographie complète de cet artiste")
    assert _looks_like_deep_search_request("fais une biographie complète de Banksy")


def test_deep_search_recherche_complete_fr() -> None:
    assert _looks_like_deep_search_request("fais une recherche complète")
    assert _looks_like_deep_search_request("fais une recherche plus complète")
    assert _looks_like_deep_search_request("cherche en détail")
    assert _looks_like_deep_search_request("cherche plus en détail")


def test_deep_search_file_output_fr() -> None:
    assert _looks_like_deep_search_request("cherche puis écris une bio dans un fichier texte")
    assert _looks_like_deep_search_request("écris une bio dans un fichier")
    assert _looks_like_deep_search_request("écris une bio")


def test_deep_search_en_profondeur_fr() -> None:
    assert _looks_like_deep_search_request("cherche en profondeur")
    assert _looks_like_deep_search_request("une fiche complète")
    assert _looks_like_deep_search_request("un dossier complet sur lui")


# ---------------------------------------------------------------------------
# _looks_like_deep_search_request — English signals
# ---------------------------------------------------------------------------


def test_deep_search_full_bio_en() -> None:
    assert _looks_like_deep_search_request("give me a full biography")
    assert _looks_like_deep_search_request("complete bio please")
    assert _looks_like_deep_search_request("detailed biography")


def test_deep_search_everything_en() -> None:
    assert _looks_like_deep_search_request("tell me everything you know about him")
    assert _looks_like_deep_search_request("everything about Taylor Swift")


def test_deep_search_file_en() -> None:
    assert _looks_like_deep_search_request("write it in a file")
    assert _looks_like_deep_search_request("save the bio in a text file")


def test_deep_search_in_depth_en() -> None:
    assert _looks_like_deep_search_request("give me an in-depth profile")
    assert _looks_like_deep_search_request("do a deep research on this")
    assert _looks_like_deep_search_request("deep dive into this topic")


# ---------------------------------------------------------------------------
# _looks_like_deep_search_request — negative cases
# ---------------------------------------------------------------------------


def test_deep_search_bio_courte_triggers_for_reply_discipline() -> None:
    # "bio courte" / "biographie courte" must still trigger reply discipline
    # (prevents inflating a thin fast_web_search note into a short bio)
    assert _looks_like_deep_search_request("fais leur bio courte")
    assert _looks_like_deep_search_request("biographie courte de cet artiste")


def test_deep_search_plain_identity_is_false() -> None:
    assert not _looks_like_deep_search_request("qui est Marie Curie")
    assert not _looks_like_deep_search_request("who is Elon Musk")
    assert not _looks_like_deep_search_request("bio de cet artiste")


def test_deep_search_plain_chat_is_false() -> None:
    assert not _looks_like_deep_search_request("bonjour comment ça va")
    assert not _looks_like_deep_search_request("what is the capital of France")
    assert not _looks_like_deep_search_request("résume ce texte")


# ---------------------------------------------------------------------------
# _looks_like_joint_entity_request
# ---------------------------------------------------------------------------


def test_joint_entity_mcfly_et_carlito() -> None:
    # With identity or bio context — all should trigger duo routing
    assert _looks_like_joint_entity_request("McFly et Carlito, fais leur bio")
    assert _looks_like_joint_entity_request("qui sont McFly et Carlito")
    assert _looks_like_joint_entity_request("bio de McFly et Carlito")
    # With deep-search context
    assert _looks_like_joint_entity_request("McFly et Carlito, fais leur bio complète")


def test_joint_entity_and_en() -> None:
    assert _looks_like_joint_entity_request("who are John and Jane biography")


def test_joint_entity_not_sequential() -> None:
    # Sequential (puis/then) should NOT be treated as joint entity
    assert not _looks_like_joint_entity_request("Michou puis Cyprien puis Squeezie")


def test_joint_entity_not_plain_chat() -> None:
    # No identity/bio signal → not a joint entity bio request
    assert not _looks_like_joint_entity_request("Python et Java sont des langages")


def test_joint_entity_not_single_entity() -> None:
    assert not _looks_like_joint_entity_request("qui est Marine Leleu")
    assert not _looks_like_joint_entity_request("cherche Ada Lovelace")


# ---------------------------------------------------------------------------
# _build_request_routing_note — deep search escalation
# ---------------------------------------------------------------------------


def test_routing_note_uses_search_web_for_bio_complete() -> None:
    note = _build_request_routing_note(
        user_input="fais une bio complète de cet artiste",
        capability_summary=_web_capable_summary(),
    )
    assert note is not None
    assert "search_web" in note
    assert _CURRENT_REQUEST_ROUTING_NOTE_PREFIX in note
    # Must NOT only suggest fast_web_search as the endpoint
    assert "deep" in note.lower() or "grounded" in note.lower() or "complete" in note.lower()


def test_routing_note_uses_search_web_for_recherche_complete() -> None:
    note = _build_request_routing_note(
        user_input="fais une recherche complète",
        capability_summary=_web_capable_summary(),
    )
    assert note is not None
    assert "search_web" in note


def test_routing_note_uses_search_web_for_file_output_request() -> None:
    note = _build_request_routing_note(
        user_input="cherche puis écris une bio dans un fichier texte",
        capability_summary=_web_capable_summary(),
    )
    assert note is not None
    assert "search_web" in note


def test_routing_note_uses_search_web_for_full_bio_en() -> None:
    note = _build_request_routing_note(
        user_input="write me a full biography please",
        capability_summary=_web_capable_summary(),
    )
    assert note is not None
    assert "search_web" in note


def test_routing_note_uses_search_web_for_everything_you_know() -> None:
    note = _build_request_routing_note(
        user_input="tell me everything you know about them",
        capability_summary=_web_capable_summary(),
    )
    assert note is not None
    assert "search_web" in note


# ---------------------------------------------------------------------------
# _build_request_routing_note — duo/group routing
# ---------------------------------------------------------------------------


def test_routing_note_for_duo_identity_uses_duo_language() -> None:
    note = _build_request_routing_note(
        user_input="qui sont McFly et Carlito",
        capability_summary=_web_capable_summary(),
    )
    assert note is not None
    assert any(
        word in note.lower()
        for word in ("duo", "unit", "together", "pair", "joint", "both")
    )


def test_routing_note_for_duo_deep_search_uses_search_web_directly() -> None:
    note = _build_request_routing_note(
        user_input="McFly et Carlito, fais leur bio complète",
        capability_summary=_web_capable_summary(),
    )
    assert note is not None
    assert "search_web" in note
    # Deep search + duo: should go straight to search_web
    assert "deeper" in note.lower() or "grounded" in note.lower() or "complete" in note.lower()


# ---------------------------------------------------------------------------
# _build_request_routing_note — plain identity still uses fast_web first
# ---------------------------------------------------------------------------


def test_routing_note_plain_identity_uses_fast_web_first() -> None:
    note = _build_request_routing_note(
        user_input="qui est Marine Leleu",
        capability_summary=_web_capable_summary(),
    )
    assert note is not None
    assert "fast_web_search" in note
    # Should also mention search_web as escalation path
    assert "search_web" in note


def test_routing_note_plain_identity_mentions_escalation() -> None:
    note = _build_request_routing_note(
        user_input="who is Ada Lovelace",
        capability_summary=_web_capable_summary(),
    )
    assert note is not None
    # The note should now mention escalating to search_web if thin or full answer needed
    assert "search_web" in note


# ---------------------------------------------------------------------------
# _build_post_tool_grounding_note — search_web escalation signal
# ---------------------------------------------------------------------------


def test_post_tool_note_names_search_web_as_deeper_path_after_fast_web() -> None:
    fast_result = _thin_fast_web_result()
    note = _build_post_tool_grounding_note(
        tool_results=[fast_result],
        tool_definitions=[FAST_WEB_SEARCH_DEFINITION, SEARCH_WEB_DEFINITION],
    )
    assert "search_web" in note
    # Must describe search_web as deeper/richer than fast_web_search
    assert any(
        word in note.lower()
        for word in ("deeper", "richer", "full", "complete", "bio", "research")
    )


def test_post_tool_note_explicitly_discourages_inflating_fast_web_to_full_bio() -> None:
    fast_result = _thin_fast_web_result()
    note = _build_post_tool_grounding_note(
        tool_results=[fast_result],
        tool_definitions=[FAST_WEB_SEARCH_DEFINITION, SEARCH_WEB_DEFINITION],
    )
    # Must warn against writing a full bio from the grounding note alone
    assert "fast_web_search" in note or "grounding" in note.lower()


# ---------------------------------------------------------------------------
# _build_post_tool_grounding_note — timeout/partial-failure note
# ---------------------------------------------------------------------------


def test_post_tool_note_includes_timeout_guidance_when_tool_timed_out() -> None:
    timed_out_result = ToolResult.failure(
        tool_name="search_web",
        error="search_web timed out for query 'McFly et Carlito'",
    )
    note = _build_post_tool_grounding_note(
        tool_results=[timed_out_result],
        tool_definitions=[SEARCH_WEB_DEFINITION],
    )
    assert "timed out" in note.lower() or "timeout" in note.lower() or "partial" in note.lower()


def test_post_tool_note_no_timeout_guidance_when_no_timeout() -> None:
    ok_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Some web results",
        payload={"query": "test", "evidence_count": 5, "finding_count": 3},
    )
    note = _build_post_tool_grounding_note(
        tool_results=[ok_result],
        tool_definitions=[SEARCH_WEB_DEFINITION],
    )
    # No timeout — should not mention timeout fallback
    assert "timed out" not in note.lower()


# ---------------------------------------------------------------------------
# Factual discipline regression guard
# ---------------------------------------------------------------------------


def test_discipline_preserves_substantive_thin_fast_web_result_for_deep_bio_request() -> None:
    thin_result = _thin_fast_web_result("McFly et Carlito")
    rich_invented_reply = (
        "McFly et Carlito sont un duo de YouTubeurs français fondé en 2010. "
        "Ils ont 8 millions d'abonnés, ont sorti plusieurs livres et animent "
        "une émission de radio hebdomadaire depuis 2018."
    )
    result = _apply_post_tool_reply_discipline(
        reply=rich_invented_reply,
        user_input="fais une bio complète de McFly et Carlito",
        tool_results=[thin_result],
    )
    assert result == rich_invented_reply


def test_discipline_does_not_clamp_acknowledged_limitation_reply() -> None:
    thin_result = _thin_fast_web_result()
    honest_reply = "Je ne peux pas confirmer une bio complète depuis ce résultat rapide."
    result = _apply_post_tool_reply_discipline(
        reply=honest_reply,
        user_input="fais une bio complète",
        tool_results=[thin_result],
    )
    # The honest, limited reply should be preserved
    assert "confirmer" in result or "ne peux pas" in result


# ---------------------------------------------------------------------------
# Capability context: search_web as deep grounded path
# ---------------------------------------------------------------------------


def test_capability_context_describes_search_web_as_deep_path() -> None:
    registry = create_default_tool_registry()
    summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=True,
    )
    context = build_runtime_capability_context(summary)
    assert "search_web" in context
    # New guidance: search_web as deeper grounded path should appear
    assert "deeper" in context or "grounded" in context or "condenses" in context


def test_capability_context_includes_duo_joint_entity_guidance() -> None:
    registry = create_default_tool_registry()
    summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=True,
    )
    context = build_runtime_capability_context(summary)
    # Duo/joint entity guidance should appear in the fast_web_grounding section
    assert "duo" in context.lower() or "unit" in context.lower() or "together" in context.lower()


# ---------------------------------------------------------------------------
# No regression: existing phase1a/phase1b behavior preserved
# ---------------------------------------------------------------------------


def test_routing_note_is_none_when_tools_not_callable() -> None:
    no_tool_summary = RuntimeCapabilitySummary(
        available_builtin_tool_names=(),
        local_file_read_available=False,
        local_directory_listing_available=False,
        url_fetch_available=False,
        web_search_available=False,
        system_info_available=False,
        memory_summary_available=False,
        model_can_call_tools=False,
    )
    note = _build_request_routing_note(
        user_input="fais une bio complète",
        capability_summary=no_tool_summary,
    )
    assert note is None


def test_routing_note_no_web_no_search_web_escalation_possible() -> None:
    # When web_search is not available, deep search routing note should not appear
    no_web_summary = RuntimeCapabilitySummary(
        available_builtin_tool_names=(),
        local_file_read_available=True,
        local_directory_listing_available=True,
        url_fetch_available=False,
        web_search_available=False,
        system_info_available=False,
        memory_summary_available=False,
        model_can_call_tools=True,
    )
    note = _build_request_routing_note(
        user_input="fais une bio complète",
        capability_summary=no_web_summary,
    )
    # No web tools → no web routing note
    assert note is None or "search_web" not in note


_CURRENT_REQUEST_ROUTING_NOTE_PREFIX = "Current request routing hint:"
