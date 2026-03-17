from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future

import pytest

from unclaw import async_utils
from unclaw.async_utils import run_blocking
from unclaw.channels.telegram_api import TelegramApiClient
from unclaw.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMRole,
    ModelCapabilities,
    ResolvedModelProfile,
)
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools import web_tools


class _ImmediateExecutor:
    def __init__(self) -> None:
        self.submit_calls: list[object] = []

    def submit(self, fn):  # type: ignore[no-untyped-def]
        self.submit_calls.append(fn)
        future: Future[object] = Future()
        try:
            future.set_result(fn())
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_run_blocking_reuses_shared_executor_for_repeated_calls(monkeypatch) -> None:
    executor = _ImmediateExecutor()
    monkeypatch.setattr(async_utils, "_BLOCKING_EXECUTOR", executor)

    async def exercise() -> tuple[str, int]:
        return (
            await run_blocking(str.upper, "shared"),
            await run_blocking(pow, 2, 5),
        )

    assert asyncio.run(exercise()) == ("SHARED", 32)
    assert len(executor.submit_calls) == 2


def test_run_blocking_propagates_exceptions(monkeypatch) -> None:
    monkeypatch.setattr(async_utils, "_BLOCKING_EXECUTOR", _ImmediateExecutor())

    def fail() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(run_blocking(fail))


def test_base_provider_chat_async_runs_sync_chat_in_worker_thread() -> None:
    caller_thread = threading.get_ident()
    observed: dict[str, int] = {}

    class FakeProvider(BaseLLMProvider):
        provider_name = "fake"

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del profile, messages, timeout_seconds, thinking_enabled, content_callback, tools
            observed["thread"] = threading.get_ident()
            return LLMResponse(
                provider="fake",
                model_name="fake-model",
                content="async boundary",
                created_at="2026-03-16T12:00:00Z",
            )

        def is_available(self, *, timeout_seconds=None) -> bool:  # type: ignore[no-untyped-def]
            del timeout_seconds
            return True

    provider = FakeProvider()
    response = asyncio.run(
        provider.chat_async(
            profile=_build_profile(),
            messages=[LLMMessage(role=LLMRole.USER, content="hello")],
        )
    )

    assert response.content == "async boundary"
    assert observed["thread"] != caller_thread


def test_ollama_provider_list_models_async_runs_in_worker_thread(monkeypatch) -> None:
    caller_thread = threading.get_ident()
    observed: dict[str, int] = {}

    def fake_list_models(self, *, timeout_seconds=None):  # type: ignore[no-untyped-def]
        del self, timeout_seconds
        observed["thread"] = threading.get_ident()
        return ("qwen3.5:4b",)

    monkeypatch.setattr(OllamaProvider, "list_models", fake_list_models)

    provider = OllamaProvider()
    model_names = asyncio.run(provider.list_models_async())

    assert model_names == ("qwen3.5:4b",)
    assert observed["thread"] != caller_thread


def test_telegram_api_client_get_updates_async_runs_in_worker_thread(
    monkeypatch,
) -> None:
    caller_thread = threading.get_ident()
    observed: dict[str, object] = {}

    def fake_request(self, method: str, payload: dict[str, object]):  # type: ignore[no-untyped-def]
        del self
        observed["thread"] = threading.get_ident()
        observed["method"] = method
        observed["payload"] = payload
        return [{"update_id": 123}]

    monkeypatch.setattr(TelegramApiClient, "_request", fake_request)

    client = TelegramApiClient(bot_token="123:ABC")
    updates = asyncio.run(client.get_updates_async(offset=99, timeout_seconds=30))

    assert updates == [{"update_id": 123}]
    assert observed["method"] == "getUpdates"
    assert observed["payload"] == {
        "timeout": 30,
        "allowed_updates": ["message"],
        "offset": 99,
    }
    assert observed["thread"] != caller_thread


def test_search_web_async_runs_sync_search_in_worker_thread(monkeypatch) -> None:
    caller_thread = threading.get_ident()
    observed: dict[str, object] = {}
    expected = ToolResult.ok(
        tool_name="search_web",
        output_text="search result",
        payload={"query": "local ai"},
    )

    def fake_search_web(call: ToolCall) -> ToolResult:
        observed["thread"] = threading.get_ident()
        observed["call"] = call
        return expected

    monkeypatch.setattr(web_tools, "search_web", fake_search_web)

    result = asyncio.run(
        web_tools.search_web_async(
            ToolCall(tool_name="search_web", arguments={"query": "local ai"})
        )
    )

    assert result == expected
    assert observed["call"] == ToolCall(
        tool_name="search_web",
        arguments={"query": "local ai"},
    )
    assert observed["thread"] != caller_thread


def _build_profile() -> ResolvedModelProfile:
    return ResolvedModelProfile(
        name="main",
        provider="fake",
        model_name="fake-model",
        temperature=0.0,
        capabilities=ModelCapabilities(
            thinking_supported=False,
            tool_mode="none",
            supports_tools=False,
            supports_native_tool_calling=False,
        ),
    )
