"""Tests for the deterministic exact session-recall shortcut.

Covers:
1.  match_exact_recall_intent: correct intents for the mission's required inputs
2.  match_exact_recall_intent: no false positives on unrelated queries
3.  build_exact_recall_reply: correct output for all four intent kinds
4.  build_exact_recall_reply: graceful out-of-range nth
5.  Runtime integration: exact recall path bypasses the model
6.  Runtime integration: non-recall queries still reach the model
7.  Works correctly after file reads, tool calls, web search, long sessions
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.core.exact_recall import (
    ExactRecallIntent,
    ExactRecallKind,
    build_exact_recall_reply,
    match_exact_recall_intent,
)
from unclaw.memory.chat_store import ChatMemoryRecord, ChatMemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _records(*pairs: tuple[str, str]) -> list[ChatMemoryRecord]:
    """Build a list of ChatMemoryRecord from (role, content) pairs."""
    return [
        ChatMemoryRecord(seq=i + 1, role=role, content=content, created_at=f"2026-01-01T00:00:{i:02d}Z")
        for i, (role, content) in enumerate(pairs)
    ]


# ---------------------------------------------------------------------------
# match_exact_recall_intent — mission required inputs (must match)
# ---------------------------------------------------------------------------

class TestMatchIntentRequired:
    """Every pattern explicitly required by the session-recall mission spec."""

    def test_premiere_question(self) -> None:
        intent = match_exact_recall_intent("première question ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.NTH_USER_PROMPT
        assert intent.nth == 1

    def test_seconde_question(self) -> None:
        intent = match_exact_recall_intent("seconde question ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.NTH_USER_PROMPT
        assert intent.nth == 2

    def test_second_prompt_en(self) -> None:
        intent = match_exact_recall_intent("second prompt ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.NTH_USER_PROMPT
        assert intent.nth == 2

    def test_numeric_2e_prompt(self) -> None:
        intent = match_exact_recall_intent("si je parle de prompt, c'était quoi mon 2e prompt ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.NTH_USER_PROMPT
        assert intent.nth == 2

    def test_continuation_et_la_3eme(self) -> None:
        # Continuation shorthand — no context word, must still match
        intent = match_exact_recall_intent("et la 3eme ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.NTH_USER_PROMPT
        assert intent.nth == 3

    def test_continuation_premiere_alone(self) -> None:
        intent = match_exact_recall_intent("première ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.NTH_USER_PROMPT
        assert intent.nth == 1

    def test_continuation_seconde_alone(self) -> None:
        intent = match_exact_recall_intent("seconde ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.NTH_USER_PROMPT
        assert intent.nth == 2

    def test_count_messages(self) -> None:
        intent = match_exact_recall_intent("combien de messages ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.COUNT_ALL_MESSAGES

    def test_count_messages_slash_prompts(self) -> None:
        # The exact failing example from the mission
        intent = match_exact_recall_intent("combien de messages / prompts ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.COUNT_ALL_MESSAGES

    def test_count_prompts_only(self) -> None:
        intent = match_exact_recall_intent("combien de prompts ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.COUNT_USER_PROMPTS

    def test_list_prompts_fr(self) -> None:
        intent = match_exact_recall_intent("liste tous mes prompts dans l'ordre")
        assert intent is not None
        assert intent.kind is ExactRecallKind.LIST_USER_PROMPTS

    def test_list_prompts_original_failure(self) -> None:
        # Verbatim from the original manual failure log
        intent = match_exact_recall_intent("peux tu me faire la liste de mes prompts dans l'ordre")
        assert intent is not None
        assert intent.kind is ExactRecallKind.LIST_USER_PROMPTS

    def test_tu_as_un_historique_de_combien(self) -> None:
        # Verbatim from the original manual failure log
        intent = match_exact_recall_intent("tu as un historique de combien de message ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.COUNT_ALL_MESSAGES

    def test_first_prompt_en(self) -> None:
        intent = match_exact_recall_intent("first prompt ?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.NTH_USER_PROMPT
        assert intent.nth == 1

    def test_third_message_en(self) -> None:
        intent = match_exact_recall_intent("what was my third message?")
        assert intent is not None
        assert intent.kind is ExactRecallKind.NTH_USER_PROMPT
        assert intent.nth == 3

    def test_numeric_3rd(self) -> None:
        intent = match_exact_recall_intent("3rd prompt")
        assert intent is not None
        assert intent.kind is ExactRecallKind.NTH_USER_PROMPT
        assert intent.nth == 3

    def test_list_my_prompts_en(self) -> None:
        intent = match_exact_recall_intent("list my prompts")
        assert intent is not None
        assert intent.kind is ExactRecallKind.LIST_USER_PROMPTS


# ---------------------------------------------------------------------------
# match_exact_recall_intent — must NOT match (no false positives)
# ---------------------------------------------------------------------------

class TestMatchIntentNoFalsePositives:
    """Queries from the mission flow that must NOT trigger exact recall."""

    def test_greeting(self) -> None:
        assert match_exact_recall_intent("salut") is None

    def test_read_file(self) -> None:
        assert match_exact_recall_intent("read hello.txt") is None

    def test_web_search_overwrite(self) -> None:
        assert match_exact_recall_intent("web search + overwrite hello.txt") is None

    def test_math_combien(self) -> None:
        # "combien" present but no "de message" / "de prompt"
        assert match_exact_recall_intent("combien font 4x5") is None

    def test_capitale(self) -> None:
        assert match_exact_recall_intent("capitale de la france") is None

    def test_meteo(self) -> None:
        assert match_exact_recall_intent("météo demain à Lille") is None

    def test_definition(self) -> None:
        assert match_exact_recall_intent("définition du mot blague") is None

    def test_ordinal_in_other_context(self) -> None:
        # "premier" present but about something unrelated and no context word
        assert match_exact_recall_intent("Napoléon était premier consul") is None

    def test_long_sentence_with_ordinal(self) -> None:
        # Continuation pattern must NOT fire on long sentences
        assert match_exact_recall_intent("le troisième jour de la semaine") is None

    def test_empty_string(self) -> None:
        assert match_exact_recall_intent("") is None

    def test_plain_number(self) -> None:
        # Bare digit with no ordinal suffix — no match
        assert match_exact_recall_intent("3 ?") is None


# ---------------------------------------------------------------------------
# build_exact_recall_reply — NTH_USER_PROMPT
# ---------------------------------------------------------------------------

class TestBuildReplyNth:
    def test_first_prompt(self) -> None:
        records = _records(
            ("user", "salut"),
            ("assistant", "Bonjour !"),
            ("user", "read hello.txt"),
            ("assistant", "Contenu : Hello."),
        )
        reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.NTH_USER_PROMPT, nth=1),
        )
        assert "User message #1" in reply
        assert "salut" in reply
        assert "read hello.txt" not in reply

    def test_second_prompt(self) -> None:
        records = _records(
            ("user", "salut"),
            ("assistant", "Bonjour !"),
            ("user", "read hello.txt"),
            ("assistant", "Contenu : Hello."),
        )
        reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.NTH_USER_PROMPT, nth=2),
        )
        assert "User message #2" in reply
        assert "read hello.txt" in reply
        assert "salut" not in reply

    def test_nth_out_of_range(self) -> None:
        records = _records(("user", "only one prompt"))
        reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.NTH_USER_PROMPT, nth=5),
        )
        assert "1" in reply   # "only 1 user message"
        assert "5" in reply   # requested #5
        assert '"' not in reply  # no content quote — it's an error message

    def test_nth_zero_treated_as_out_of_range(self) -> None:
        records = _records(("user", "prompt"))
        reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.NTH_USER_PROMPT, nth=0),
        )
        # nth=0 < 1 → out-of-range path
        assert "0" in reply

    def test_interspersed_tool_messages_do_not_shift_user_count(self) -> None:
        records = _records(
            ("user", "first user"),
            ("tool", "some tool output"),
            ("assistant", "reply"),
            ("user", "second user"),
        )
        reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.NTH_USER_PROMPT, nth=2),
        )
        assert "second user" in reply
        assert "tool output" not in reply


# ---------------------------------------------------------------------------
# build_exact_recall_reply — COUNT_USER_PROMPTS
# ---------------------------------------------------------------------------

class TestBuildReplyCountPrompts:
    def test_count_user_prompts(self) -> None:
        records = _records(
            ("user", "A"), ("assistant", "B"),
            ("user", "C"), ("assistant", "D"),
            ("user", "E"), ("assistant", "F"),
        )
        reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.COUNT_USER_PROMPTS),
        )
        assert "3" in reply      # 3 user prompts
        assert "6" in reply      # 6 total

    def test_count_empty_session(self) -> None:
        reply = build_exact_recall_reply(
            [],
            ExactRecallIntent(kind=ExactRecallKind.COUNT_USER_PROMPTS),
        )
        assert "0" in reply


# ---------------------------------------------------------------------------
# build_exact_recall_reply — COUNT_ALL_MESSAGES
# ---------------------------------------------------------------------------

class TestBuildReplyCountAll:
    def test_all_counts_shown(self) -> None:
        records = _records(
            ("user", "U1"),
            ("tool", "T1"),
            ("assistant", "A1"),
            ("user", "U2"),
            ("assistant", "A2"),
            ("tool", "T2"),
            ("tool", "T3"),
        )
        reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.COUNT_ALL_MESSAGES),
        )
        assert "7" in reply      # total
        assert "2" in reply      # user count (and assistant)
        assert "3" in reply      # tool count

    def test_zero_counts(self) -> None:
        reply = build_exact_recall_reply(
            [],
            ExactRecallIntent(kind=ExactRecallKind.COUNT_ALL_MESSAGES),
        )
        assert "0" in reply


# ---------------------------------------------------------------------------
# build_exact_recall_reply — LIST_USER_PROMPTS
# ---------------------------------------------------------------------------

class TestBuildReplyList:
    def test_ordered_list_exact(self) -> None:
        prompts = ["Alpha", "Beta", "Gamma", "Delta"]
        pairs: list[tuple[str, str]] = []
        for p in prompts:
            pairs.extend([("user", p), ("assistant", f"Reply to {p}")])
        records = _records(*pairs)
        reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.LIST_USER_PROMPTS),
        )
        # All prompts present in order
        for p in prompts:
            assert p in reply
        assert reply.index("Alpha") < reply.index("Beta") < reply.index("Gamma") < reply.index("Delta")
        # Numbered
        assert "1." in reply
        assert "4." in reply

    def test_list_excludes_non_user_messages(self) -> None:
        records = _records(
            ("user", "user prompt"),
            ("tool", "tool result — should not appear"),
            ("assistant", "assistant reply — should not appear"),
        )
        reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.LIST_USER_PROMPTS),
        )
        assert "user prompt" in reply
        assert "tool result" not in reply
        assert "assistant reply" not in reply

    def test_list_empty_session(self) -> None:
        reply = build_exact_recall_reply(
            [],
            ExactRecallIntent(kind=ExactRecallKind.LIST_USER_PROMPTS),
        )
        assert "No user prompts" in reply

    def test_list_includes_recall_questions_themselves(self) -> None:
        """Recall questions that have been persisted appear in the list — this is exact behavior."""
        records = _records(
            ("user", "salut"),
            ("assistant", "Bonjour"),
            ("user", "première question ?"),
            ("assistant", "salut"),
            ("user", "liste tous mes prompts dans l'ordre"),
        )
        reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.LIST_USER_PROMPTS),
        )
        assert "salut" in reply
        assert "première question ?" in reply
        assert "liste tous mes prompts dans l'ordre" in reply
        assert "3." in reply  # three user messages


# ---------------------------------------------------------------------------
# build_exact_recall_reply — works after file reads / tool calls / long sessions
# ---------------------------------------------------------------------------

class TestBuildReplyLongSession:
    def test_exact_recall_after_long_session(self, tmp_path: Path) -> None:
        """First prompt remains accessible regardless of session length."""
        store = ChatMemoryStore(tmp_path / "chats")
        session_id = "long_session"
        for i in range(60):
            store.append_message(
                session_id=session_id,
                role="user",
                content=f"Prompt number {i + 1}",
                created_at=f"2026-03-18T10:00:{i % 60:02d}Z",
            )
            store.append_message(
                session_id=session_id,
                role="assistant",
                content=f"Answer {i + 1}",
                created_at=f"2026-03-18T10:01:{i % 60:02d}Z",
            )
        records = store.read_messages(session_id)

        first = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.NTH_USER_PROMPT, nth=1),
        )
        assert "Prompt number 1" in first

        tenth = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.NTH_USER_PROMPT, nth=10),
        )
        assert "Prompt number 10" in tenth

        count_reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.COUNT_ALL_MESSAGES),
        )
        assert "120" in count_reply  # 60 user + 60 assistant
        assert "60" in count_reply   # user count

        list_reply = build_exact_recall_reply(
            records,
            ExactRecallIntent(kind=ExactRecallKind.LIST_USER_PROMPTS),
        )
        assert "Prompt number 1" in list_reply
        assert "Prompt number 60" in list_reply
        assert "1." in list_reply
        assert "60." in list_reply


# ---------------------------------------------------------------------------
# Runtime integration — exact recall path bypasses the model
# ---------------------------------------------------------------------------

class TestRuntimeIntegration:
    """
    Verify that run_user_turn returns a deterministic reply for recall queries
    without ever calling the model.

    Uses a minimal stub to avoid SQLite / Ollama dependencies.
    """

    def _make_stub_session_manager(self, tmp_path: Path) -> SimpleNamespace:
        """Return a minimal stub SessionManager with a real ChatMemoryStore."""
        store = ChatMemoryStore(tmp_path / "chats")
        session_id = "sess_rt"

        # Populate several turns
        turn_data = [
            ("user", "salut"),
            ("assistant", "Bonjour !"),
            ("user", "read hello.txt"),
            ("assistant", "Contenu : Hello world."),
            ("tool", "file content: Hello world."),
            ("user", "combien font 4x5"),
            ("assistant", "20"),
            ("user", "capitale de la france"),
            ("assistant", "Paris."),
        ]
        for i, (role, content) in enumerate(turn_data):
            store.append_message(
                session_id=session_id,
                role=role,
                content=content,
                created_at=f"2026-03-18T10:00:{i:02d}Z",
            )

        stub = SimpleNamespace(
            chat_store=store,
            current_session_id=session_id,
        )
        return stub

    def test_shortcut_returns_correct_first_prompt(self, tmp_path: Path) -> None:
        from unclaw.core.runtime import _try_exact_recall_shortcut

        stub = self._make_stub_session_manager(tmp_path)
        reply = _try_exact_recall_shortcut(
            session_manager=stub,  # type: ignore[arg-type]
            session_id=stub.current_session_id,
            user_input="première question ?",
        )
        assert reply is not None
        assert "salut" in reply
        assert "User message #1" in reply

    def test_shortcut_returns_correct_third_prompt(self, tmp_path: Path) -> None:
        from unclaw.core.runtime import _try_exact_recall_shortcut

        stub = self._make_stub_session_manager(tmp_path)
        reply = _try_exact_recall_shortcut(
            session_manager=stub,  # type: ignore[arg-type]
            session_id=stub.current_session_id,
            user_input="et la 3eme ?",
        )
        assert reply is not None
        assert "combien font 4x5" in reply
        assert "User message #3" in reply

    def test_shortcut_count_all_messages(self, tmp_path: Path) -> None:
        from unclaw.core.runtime import _try_exact_recall_shortcut

        stub = self._make_stub_session_manager(tmp_path)
        reply = _try_exact_recall_shortcut(
            session_manager=stub,  # type: ignore[arg-type]
            session_id=stub.current_session_id,
            user_input="combien de messages / prompts ?",
        )
        assert reply is not None
        assert "9" in reply   # 9 total messages
        assert "4" in reply   # 4 user prompts (salut, read hello.txt, combien font 4x5, capitale...)

    def test_shortcut_list_all_prompts(self, tmp_path: Path) -> None:
        from unclaw.core.runtime import _try_exact_recall_shortcut

        stub = self._make_stub_session_manager(tmp_path)
        reply = _try_exact_recall_shortcut(
            session_manager=stub,  # type: ignore[arg-type]
            session_id=stub.current_session_id,
            user_input="liste tous mes prompts dans l'ordre",
        )
        assert reply is not None
        assert "salut" in reply
        assert "read hello.txt" in reply
        assert "combien font 4x5" in reply
        assert "capitale de la france" in reply
        # Ordered correctly
        assert reply.index("salut") < reply.index("read hello.txt")
        assert reply.index("read hello.txt") < reply.index("combien font 4x5")

    def test_shortcut_returns_none_for_unrelated_query(self, tmp_path: Path) -> None:
        from unclaw.core.runtime import _try_exact_recall_shortcut

        stub = self._make_stub_session_manager(tmp_path)
        # Normal queries must NOT be short-circuited
        for query in ("salut", "météo demain", "définition du mot blague", "combien font 4x5"):
            result = _try_exact_recall_shortcut(
                session_manager=stub,  # type: ignore[arg-type]
                session_id=stub.current_session_id,
                user_input=query,
            )
            assert result is None, f"Expected None for {query!r}, got {result!r}"

    def test_shortcut_returns_none_when_no_chat_store(self, tmp_path: Path) -> None:
        from unclaw.core.runtime import _try_exact_recall_shortcut

        stub = SimpleNamespace(chat_store=None, current_session_id="sess_noop")
        result = _try_exact_recall_shortcut(
            session_manager=stub,  # type: ignore[arg-type]
            session_id="sess_noop",
            user_input="première question ?",
        )
        assert result is None

    def test_run_user_turn_does_not_call_model_for_recall(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_temp_project,
    ) -> None:
        """Full run_user_turn: model must NOT be called for an exact recall query."""
        from types import SimpleNamespace as NS

        from unclaw.core.command_handler import CommandHandler
        from unclaw.core.runtime import run_user_turn
        from unclaw.core.session_manager import SessionManager
        from unclaw.llm.base import LLMResponse
        from unclaw.logs.event_bus import EventBus
        from unclaw.logs.tracer import Tracer
        from unclaw.schemas.chat import MessageRole
        from unclaw.settings import load_settings
        from unclaw.tools.registry import ToolRegistry

        model_call_count = 0

        class _NeverCallProvider:
            provider_name = "ollama"

            def __init__(self, *, base_url: str = "", default_timeout_seconds: float = 60.0) -> None:
                pass

            def chat(self, profile, messages, **kwargs) -> LLMResponse:  # type: ignore[no-untyped-def]
                nonlocal model_call_count
                model_call_count += 1
                # Force a test failure with a descriptive message
                raise AssertionError("Model must NOT be called for an exact recall query")

        monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", _NeverCallProvider)

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)
        tracer = Tracer(event_bus=EventBus(), event_repository=session_manager.event_repository)
        command_handler = CommandHandler(
            settings=settings,
            session_manager=session_manager,
            memory_manager=NS(),
        )

        try:
            session = session_manager.ensure_current_session()
            # Populate history
            for i, msg in enumerate(["salut", "read hello.txt", "capitale de la france"]):
                session_manager.add_message(MessageRole.USER, msg, session_id=session.id)
                session_manager.add_message(
                    MessageRole.ASSISTANT, f"Reply {i}", session_id=session.id
                )

            # This recall question must NOT reach the model
            recall_input = "première question ?"
            session_manager.add_message(MessageRole.USER, recall_input, session_id=session.id)

            reply = run_user_turn(
                session_manager=session_manager,
                command_handler=command_handler,
                user_input=recall_input,
                tracer=tracer,
                tool_registry=ToolRegistry(),
            )

            assert model_call_count == 0, "Model was called despite being an exact recall query"
            assert "salut" in reply
            assert "User message #1" in reply
        finally:
            session_manager.close()

    def test_run_user_turn_still_calls_model_for_normal_queries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_temp_project,
    ) -> None:
        """Verify non-recall queries still reach the model (no regression)."""
        from types import SimpleNamespace as NS

        from unclaw.core.command_handler import CommandHandler
        from unclaw.core.runtime import run_user_turn
        from unclaw.core.session_manager import SessionManager
        from unclaw.llm.base import LLMResponse
        from unclaw.logs.event_bus import EventBus
        from unclaw.logs.tracer import Tracer
        from unclaw.schemas.chat import MessageRole
        from unclaw.settings import load_settings
        from unclaw.tools.registry import ToolRegistry

        model_call_count = 0

        class _CountingProvider:
            provider_name = "ollama"

            def __init__(self, *, base_url: str = "", default_timeout_seconds: float = 60.0) -> None:
                pass

            def chat(self, profile, messages, **kwargs) -> LLMResponse:  # type: ignore[no-untyped-def]
                nonlocal model_call_count
                model_call_count += 1
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="Paris.",
                    created_at="2026-03-18T10:00:00Z",
                    finish_reason="stop",
                )

        monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", _CountingProvider)

        project_root = make_temp_project()
        settings = load_settings(project_root=project_root)
        session_manager = SessionManager.from_settings(settings)
        tracer = Tracer(event_bus=EventBus(), event_repository=session_manager.event_repository)
        command_handler = CommandHandler(
            settings=settings,
            session_manager=session_manager,
            memory_manager=NS(),
        )

        try:
            session = session_manager.ensure_current_session()
            normal_input = "capitale de la france"
            session_manager.add_message(MessageRole.USER, normal_input, session_id=session.id)

            reply = run_user_turn(
                session_manager=session_manager,
                command_handler=command_handler,
                user_input=normal_input,
                tracer=tracer,
                tool_registry=ToolRegistry(),
            )

            assert model_call_count == 1, "Model should be called for normal queries"
            assert reply == "Paris."
        finally:
            session_manager.close()
