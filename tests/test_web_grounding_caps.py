from __future__ import annotations

from datetime import date

from unclaw.core.search_grounding import shape_search_backed_reply
from unclaw.tools.web_synthesis import _FactCluster, _select_synthesized_findings


def _build_fact_cluster(*, prefix: str, word_count: int, score: float) -> _FactCluster:
    text = " ".join(f"{prefix}{index}" for index in range(word_count))
    return _FactCluster(
        merged_text=text,
        evidence=(),
        supporting_urls=(f"https://example.com/{prefix}",),
        source_titles=(f"{prefix.title()} Source",),
        score=score,
        query_relevance=5.0,
        evidence_quality=5.0,
        novelty=1.0,
        support_count=2,
    )


def test_select_synthesized_findings_uses_raised_output_caps() -> None:
    clusters = (
        _build_fact_cluster(prefix="alpha", word_count=80, score=10.0),
        _build_fact_cluster(prefix="bravo", word_count=14, score=9.0),
        _build_fact_cluster(prefix="charlie", word_count=14, score=8.0),
        _build_fact_cluster(prefix="delta", word_count=14, score=7.0),
        _build_fact_cluster(prefix="echo", word_count=14, score=6.0),
        _build_fact_cluster(prefix="foxtrot", word_count=14, score=5.0),
        _build_fact_cluster(prefix="golf", word_count=14, score=4.5),
        _build_fact_cluster(prefix="hotel", word_count=14, score=4.0),
        _build_fact_cluster(prefix="india", word_count=14, score=3.5),
    )

    findings = _select_synthesized_findings(clusters)

    assert len(findings) == 8
    assert findings[0].text.startswith("alpha0 alpha1 alpha2")
    assert 260 < len(findings[0].text) <= 400
    assert all(not finding.text.startswith("india0") for finding in findings)


def test_shape_search_backed_reply_uses_raised_grounding_fact_cap() -> None:
    payload = {
        "query": "tell me everything you know about Taylor Stone",
        "display_sources": [
            {
                "title": "Taylor Stone Profile",
                "url": "https://example.com/taylor-stone",
            }
        ],
        "synthesized_findings": [
            {
                "text": "Taylor Stone is a robotics researcher and startup founder.",
                "score": 9.0,
                "support_count": 2,
                "source_titles": ["Taylor Stone Profile"],
                "source_urls": ["https://example.com/taylor-stone"],
            },
            {
                "text": "She created the River Hand prosthetics project.",
                "score": 8.0,
                "support_count": 2,
                "source_titles": ["Taylor Stone Profile"],
                "source_urls": ["https://example.com/taylor-stone"],
            },
            {
                "text": "She leads the Applied Motion Lab.",
                "score": 7.0,
                "support_count": 2,
                "source_titles": ["Taylor Stone Profile"],
                "source_urls": ["https://example.com/taylor-stone"],
            },
            {
                "text": "She co-authored the GripKit open-source toolkit.",
                "score": 6.0,
                "support_count": 2,
                "source_titles": ["Taylor Stone Profile"],
                "source_urls": ["https://example.com/taylor-stone"],
            },
            {
                "text": "She mentors early-career biomedical engineers.",
                "score": 5.0,
                "support_count": 2,
                "source_titles": ["Taylor Stone Profile"],
                "source_urls": ["https://example.com/taylor-stone"],
            },
            {
                "text": "She speaks at neighborhood design meetups.",
                "score": 4.0,
                "support_count": 2,
                "source_titles": ["Taylor Stone Profile"],
                "source_urls": ["https://example.com/taylor-stone"],
            },
        ],
    }

    reply = shape_search_backed_reply(
        "Taylor Stone seems to be an inspiring creator who often shows up on podcasts and blogs.",
        payload=payload,
        query="tell me everything you know about Taylor Stone",
        current_date=date(2026, 3, 15),
    )

    assert "Taylor Stone is a robotics researcher and startup founder." in reply
    assert "She created the River Hand prosthetics project." in reply
    assert "She leads the Applied Motion Lab." in reply
    assert "She co-authored the GripKit open-source toolkit." in reply
    assert "She mentors early-career biomedical engineers." in reply
    assert "She speaks at neighborhood design meetups." not in reply


def test_shape_search_backed_reply_keeps_sparse_bio_grounding_evidence_bounded() -> None:
    payload = {
        "query": "who is Marine Leleu",
        "display_sources": [
            {
                "title": "Marine Leleu Profile",
                "url": "https://example.com/marine-leleu",
            }
        ],
        "synthesized_findings": [
            {
                "text": "Marine Leleu is a French endurance athlete.",
                "score": 7.0,
                "support_count": 1,
                "source_titles": ["Marine Leleu Profile"],
                "source_urls": ["https://example.com/marine-leleu"],
            }
        ],
    }

    reply = shape_search_backed_reply(
        (
            "Marine Leleu is a French endurance athlete, author, and podcast host. "
            "She is known for several public speaking tours."
        ),
        payload=payload,
        query="who is Marine Leleu",
        current_date=date(2026, 3, 15),
    )

    assert reply == "Marine Leleu is a French endurance athlete."
