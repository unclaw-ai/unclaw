from __future__ import annotations

from unclaw.core.command_handler import CommandHandler
from unclaw.core.research_flow import build_tool_history_content
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMMessage, LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.memory import MemoryManager
from unclaw.memory.summarizer import (
    parse_persisted_session_memory,
    summarize_session_messages,
)
from unclaw.schemas.chat import ChatMessage, MessageRole
from unclaw.settings import load_settings
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.contracts import ToolCall, ToolResult


def test_summarize_session_messages_retains_grounded_search_facts() -> None:
    tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: who is Jordan Lee\n",
        payload={
            "query": "who is Jordan Lee",
            "summary_points": [
                "Jordan Lee is a product designer and engineer.",
                "One profile lists the handle @jordancode.",
            ],
            "display_sources": [
                {
                    "title": "Company Bio",
                    "url": "https://company.example.com/jordan-lee",
                },
            ],
            "synthesized_findings": [
                {
                    "text": "Jordan Lee is a product designer and engineer.",
                    "score": 8.1,
                    "support_count": 2,
                    "source_titles": ["Company Bio"],
                    "source_urls": ["https://company.example.com/jordan-lee"],
                },
                {
                    "text": "One profile lists the handle @jordancode.",
                    "score": 4.2,
                    "support_count": 1,
                    "source_titles": ["Community Q&A"],
                    "source_urls": ["https://community.example.com/jordan-qa"],
                },
            ],
        },
    )
    tool_message = ChatMessage(
        id="msg_tool",
        session_id="sess_test",
        role=MessageRole.TOOL,
        content=build_tool_history_content(
            tool_result,
            tool_call=ToolCall(
                tool_name="search_web",
                arguments={"query": "who is Jordan Lee"},
            ),
        ),
        created_at="2026-03-16T10:00:00Z",
    )

    summary = summarize_session_messages(
        [
            ChatMessage(
                id="msg_user",
                session_id="sess_test",
                role=MessageRole.USER,
                content="Who is Jordan Lee?",
                created_at="2026-03-16T09:59:00Z",
            ),
            tool_message,
            ChatMessage(
                id="msg_assistant",
                session_id="sess_test",
                role=MessageRole.ASSISTANT,
                content="Jordan Lee is a product designer and engineer.",
                created_at="2026-03-16T10:01:00Z",
            ),
        ]
    )

    assert "Retained grounded fact:" in summary
    assert "[who is Jordan Lee] Jordan Lee is a product designer and engineer" in summary
    assert "Retained uncertainty:" in summary
    assert "@jordancode" in summary
    assert "Session size: 3 messages (1 user, 1 assistant, 1 tool)." in summary


def test_memory_manager_persists_structured_summary_in_existing_summary_column(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    memory_manager = MemoryManager(session_manager=session_manager)

    tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search query: who is Jordan Lee\n",
        payload={
            "query": "who is Jordan Lee",
            "summary_points": [
                "Jordan Lee is a product designer and engineer.",
                "One profile lists the handle @jordancode.",
            ],
            "display_sources": [
                {
                    "title": "Company Bio",
                    "url": "https://company.example.com/jordan-lee",
                },
            ],
            "synthesized_findings": [
                {
                    "text": "Jordan Lee is a product designer and engineer.",
                    "score": 8.1,
                    "support_count": 2,
                    "source_titles": ["Company Bio"],
                    "source_urls": ["https://company.example.com/jordan-lee"],
                },
                {
                    "text": "One profile lists the handle @jordancode.",
                    "score": 4.2,
                    "support_count": 1,
                    "source_titles": ["Community Q&A"],
                    "source_urls": ["https://community.example.com/jordan-qa"],
                },
            ],
        },
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Who is Jordan Lee?",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.TOOL,
            build_tool_history_content(
                tool_result,
                tool_call=ToolCall(
                    tool_name="search_web",
                    arguments={"query": "who is Jordan Lee"},
                ),
            ),
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Jordan Lee is a product designer and engineer.",
            session_id=session.id,
        )

        rendered_summary = memory_manager.build_or_refresh_session_summary(session.id)
        stored_summary = session_manager.session_repository.get_summary_text(session.id)

        assert stored_summary is not None
        assert stored_summary.startswith("{")

        parsed_summary = parse_persisted_session_memory(stored_summary)
        assert parsed_summary is not None
        assert parsed_summary.summary_text == rendered_summary
        assert parsed_summary.recent_user_intents == ("Who is Jordan Lee",)
        assert parsed_summary.retained_facts[0].query == "who is Jordan Lee"
        assert (
            parsed_summary.retained_facts[0].text
            == "Jordan Lee is a product designer and engineer"
        )
        assert parsed_summary.retained_uncertainties[0].text == (
            "One profile lists the handle @jordancode"
        )
    finally:
        session_manager.close()


def test_memory_manager_reads_legacy_plain_text_summary_without_breaking_context_note(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    memory_manager = MemoryManager(session_manager=session_manager)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Remember this old summary.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Stored legacy summaries should keep working.",
            session_id=session.id,
        )
        session_manager.session_repository.update_summary_text(
            session.id,
            "Legacy summary text from an older session.",
        )

        state = memory_manager.get_session_state(session.id)
        note = memory_manager.build_context_note(session.id)

        assert state.summary_text == "Legacy summary text from an older session."
        assert state.summary.recent_user_intents == ()
        assert note is not None
        assert "Legacy summary text from an older session." in note
    finally:
        session_manager.close()


def test_summary_command_and_new_session_keep_session_memory_isolated(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    memory_manager = MemoryManager(session_manager=session_manager)
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=memory_manager,
    )

    try:
        first_session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Keep this in the first session.",
            session_id=first_session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "First-session answer.",
            session_id=first_session.id,
        )
        memory_manager.build_or_refresh_session_summary(first_session.id)

        first_summary = command_handler.handle("/summary")
        new_session_result = command_handler.handle("/new")
        second_summary = command_handler.handle("/summary")

        assert first_summary.lines[0] == "Session summary:"
        assert "First-session answer" in first_summary.lines[1]
        assert second_summary.lines == ("Session summary:", "No messages yet.")
        assert new_session_result.session_id is not None
        assert new_session_result.session_id != first_session.id
        assert memory_manager.get_session_summary(first_session.id) != "No messages yet."
        assert memory_manager.get_session_summary(new_session_result.session_id) == (
            "No messages yet."
        )
    finally:
        session_manager.close()
