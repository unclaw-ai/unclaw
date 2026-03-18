"""Tests for per-turn context budgeting — P4-2.

Covers:
1. Short conversations include full recent history.
2. Long conversations are clipped to the char budget.
3. Newest turns are preserved preferentially over old ones.
4. Required system/capability messages still appear.
5. Current user message is always preserved.
6. Both constraints (count cap and char budget) operate independently.
"""

from __future__ import annotations

from types import SimpleNamespace

from unclaw.core.context_builder import _budget_history, build_context_messages
from unclaw.llm.base import LLMRole
from unclaw.schemas.chat import ChatMessage, MessageRole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(role: MessageRole, content: str, index: int = 0) -> ChatMessage:
    return ChatMessage(
        id=f"msg_{index}",
        session_id="sess_test",
        role=role,
        content=content,
        created_at="2026-03-18T10:00:00Z",
    )


def _make_sm(messages: list[ChatMessage], system_prompt: str = "You are helpful.") -> object:
    """Minimal mock that satisfies what build_context_messages needs."""
    return SimpleNamespace(
        settings=SimpleNamespace(system_prompt=system_prompt),
        list_messages=lambda session_id: messages,
    )


# ---------------------------------------------------------------------------
# _budget_history — unit tests
# ---------------------------------------------------------------------------


def test_budget_history_short_conversation_includes_all_messages() -> None:
    """Short conversation fits within both caps — all messages returned."""
    messages = [_make_msg(MessageRole.USER, f"hi {i}", i) for i in range(5)]
    result = list(_budget_history(messages, max_history_size=20, max_history_chars=24_000))
    assert result == messages


def test_budget_history_count_cap_clips_long_history() -> None:
    """More messages than the count cap — trimmed to the most-recent N."""
    messages = [_make_msg(MessageRole.USER, f"msg {i}", i) for i in range(30)]
    result = list(_budget_history(messages, max_history_size=20, max_history_chars=24_000))
    assert result == messages[-20:]


def test_budget_history_char_budget_clips_before_count_cap() -> None:
    """Large messages hit the char budget before the count cap is reached."""
    large = "x" * 5_000  # 5000 chars each
    messages = [_make_msg(MessageRole.USER, large, i) for i in range(10)]
    # Budget 8000 chars → only 1 message fits (5000), 2nd would push to 10000
    result = list(_budget_history(messages, max_history_size=20, max_history_chars=8_000))
    assert len(result) == 1
    assert result[0].id == "msg_9"  # newest retained


def test_budget_history_newest_turns_preserved_when_budget_clips() -> None:
    """When the char budget clips, the most-recent messages survive."""
    messages = [_make_msg(MessageRole.USER, f"turn {i}", i) for i in range(10)]
    # Each "turn N" content is ~7 chars; budget 22 → fits 3 messages
    result = list(_budget_history(messages, max_history_size=20, max_history_chars=22))
    assert len(result) == 3
    assert result[-1].id == "msg_9"
    assert result[-2].id == "msg_8"
    assert result[-3].id == "msg_7"


def test_budget_history_result_is_chronological_order() -> None:
    """Returned messages are always oldest-to-newest regardless of budget."""
    messages = [_make_msg(MessageRole.USER, f"msg {i}", i) for i in range(6)]
    result = list(_budget_history(messages, max_history_size=20, max_history_chars=24_000))
    ids = [m.id for m in result]
    assert ids == sorted(ids)


def test_budget_history_none_char_budget_falls_back_to_count_only() -> None:
    """max_history_chars=None disables char budget; count cap still applies."""
    messages = [_make_msg(MessageRole.USER, f"msg {i}", i) for i in range(30)]
    result = list(_budget_history(messages, max_history_size=20, max_history_chars=None))
    assert result == messages[-20:]


def test_budget_history_none_count_cap_falls_back_to_char_budget_only() -> None:
    """max_history_size=None disables count cap; char budget still applies."""
    large = "x" * 1_000  # 1000 chars each
    messages = [_make_msg(MessageRole.USER, large, i) for i in range(50)]
    # Budget 3000 → 3 messages fit (3000 chars), newest 3
    result = list(_budget_history(messages, max_history_size=None, max_history_chars=3_000))
    assert len(result) == 3
    assert result[-1].id == "msg_49"


def test_budget_history_both_none_returns_full_history() -> None:
    """Both constraints None → full history returned unchanged."""
    messages = [_make_msg(MessageRole.USER, f"msg {i}", i) for i in range(50)]
    result = list(_budget_history(messages, max_history_size=None, max_history_chars=None))
    assert result == messages


def test_budget_history_zero_count_cap_returns_empty() -> None:
    """max_history_size=0 → empty history regardless of char budget."""
    messages = [_make_msg(MessageRole.USER, "hi", i) for i in range(5)]
    result = list(_budget_history(messages, max_history_size=0, max_history_chars=24_000))
    assert result == []


def test_budget_history_empty_input_returns_empty() -> None:
    """Empty history is handled gracefully."""
    result = list(_budget_history([], max_history_size=20, max_history_chars=24_000))
    assert result == []


# ---------------------------------------------------------------------------
# build_context_messages — integration tests
# ---------------------------------------------------------------------------


def test_build_context_messages_system_message_always_present() -> None:
    """System message is never dropped regardless of char budget."""
    sm = _make_sm([], system_prompt="Be concise.")
    context = build_context_messages(
        session_manager=sm,
        session_id="s",
        user_message="Hello",
        max_history_size=20,
        max_history_chars=24_000,
    )
    assert context[0].role == LLMRole.SYSTEM
    assert "Be concise." in context[0].content


def test_build_context_messages_user_message_always_present_under_tight_budget() -> None:
    """Current user message is never dropped even when history is fully clipped."""
    heavy = "x" * 6_000  # 6000 chars each
    messages = [_make_msg(MessageRole.USER, heavy, i) for i in range(5)]
    sm = _make_sm(messages)
    # Tiny budget → all history clipped, but user message must still appear
    context = build_context_messages(
        session_manager=sm,
        session_id="s",
        user_message="What is the answer?",
        max_history_size=20,
        max_history_chars=100,
    )
    user_msgs = [m for m in context if m.role == LLMRole.USER]
    assert any(m.content == "What is the answer?" for m in user_msgs)


def test_build_context_messages_history_within_budget_fully_included() -> None:
    """Short history within budget appears in context after system messages."""
    messages = [
        _make_msg(MessageRole.USER, "Earlier question", 0),
        _make_msg(MessageRole.ASSISTANT, "Earlier answer", 1),
    ]
    sm = _make_sm(messages)
    context = build_context_messages(
        session_manager=sm,
        session_id="s",
        user_message="Follow-up",
        max_history_size=20,
        max_history_chars=24_000,
    )
    contents = [m.content for m in context]
    assert "Earlier question" in contents
    assert "Earlier answer" in contents


def test_build_context_messages_oversized_history_clipped_by_budget() -> None:
    """Large tool output in history is dropped when budget is tight; current turn survives."""
    big_tool_output = "result data " * 500  # ~6000 chars
    messages = [
        _make_msg(MessageRole.USER, "Old question", 0),
        _make_msg(MessageRole.TOOL, big_tool_output, 1),
        _make_msg(MessageRole.ASSISTANT, "Old answer", 2),
    ]
    sm = _make_sm(messages)
    context = build_context_messages(
        session_manager=sm,
        session_id="s",
        user_message="New question",
        max_history_size=20,
        max_history_chars=500,  # smaller than the tool output alone
    )
    # Current user message must be present
    user_msgs = [m for m in context if m.role == LLMRole.USER]
    assert any(m.content == "New question" for m in user_msgs)
    # Big tool output must not appear (too large for budget)
    all_contents = " ".join(m.content for m in context)
    assert big_tool_output not in all_contents


def test_build_context_messages_capability_summary_preserved_under_tight_budget() -> None:
    """Capability summary system message is never dropped by the budget."""
    from unclaw.core.capabilities import RuntimeCapabilitySummary

    summary = RuntimeCapabilitySummary(
        available_builtin_tool_names=(),
        local_file_read_available=False,
        local_directory_listing_available=False,
        url_fetch_available=False,
        web_search_available=False,
        system_info_available=False,
        memory_summary_available=False,
    )
    big_content = "x" * 5_000
    messages = [_make_msg(MessageRole.USER, big_content, i) for i in range(5)]
    sm = _make_sm(messages)

    context = build_context_messages(
        session_manager=sm,
        session_id="s",
        user_message="Hello",
        max_history_size=20,
        max_history_chars=100,  # clips all history
        capability_summary=summary,
    )
    system_msgs = [m for m in context if m.role == LLMRole.SYSTEM]
    # At minimum: system prompt + capability summary
    assert len(system_msgs) >= 2
