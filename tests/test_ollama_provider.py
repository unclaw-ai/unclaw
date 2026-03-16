from __future__ import annotations

import io
import json
import re
from urllib.error import HTTPError, URLError

import pytest

from unclaw.llm import ollama_provider
from unclaw.llm.base import (
    LLMConnectionError,
    LLMMessage,
    LLMProviderError,
    LLMResponseError,
    LLMRole,
    ModelCapabilities,
    ResolvedModelProfile,
)
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.tools.contracts import ToolCall, ToolDefinition, ToolPermissionLevel


def test_chat_sends_boolean_think_flag_and_captures_reasoning(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {
                        "content": "Hello",
                        "thinking": "Need a short answer.",
                    },
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(thinking_supported=True),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
        thinking_enabled=True,
    )

    assert captured["payload"] == {
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"temperature": 0.3},
        "think": True,
    }
    assert response.content == "Hello"
    assert response.reasoning == "Need a short answer."
    assert response.finish_reason == "stop"


def test_chat_forces_think_off_for_profiles_without_thinking_support(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "llama3.2:3b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {"content": "Fast reply"},
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(
            model_name="llama3.2:3b",
            thinking_supported=False,
        ),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
        thinking_enabled=True,
    )

    assert captured["payload"]["think"] is False
    assert response.reasoning is None


def test_chat_strips_leaked_think_block_from_non_streaming_content(monkeypatch) -> None:
    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        del request, timeout
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {
                        "content": (
                            "<think>Need private reasoning.</think>\n\n"
                            "Visible answer."
                        ),
                    },
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(thinking_supported=False),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
    )

    assert response.content == "Visible answer."


def test_chat_streams_real_incremental_content_when_callback_is_provided(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeStreamResponse(
            (
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "message": {"thinking": "Need "},
                },
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:01Z",
                    "message": {"thinking": "speed."},
                },
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:01Z",
                    "message": {"content": "Hel"},
                },
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:01Z",
                    "message": {"content": "lo"},
                },
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:02Z",
                    "done_reason": "stop",
                },
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    streamed_chunks: list[str] = []
    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(thinking_supported=True),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
        thinking_enabled=True,
        content_callback=streamed_chunks.append,
    )

    assert captured["payload"]["stream"] is True
    assert captured["payload"]["think"] is True
    assert streamed_chunks == ["Hel", "lo"]
    assert response.content == "Hello"
    assert response.reasoning == "Need speed."
    assert response.finish_reason == "stop"


def test_chat_sends_keep_alive_when_profile_configured(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {"content": "Warm reply"},
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    provider.chat(
        profile=_build_profile(thinking_supported=True, keep_alive="10m"),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
    )

    assert captured["payload"]["keep_alive"] == "10m"  # type: ignore[index]


def test_chat_sends_tool_definitions_in_ollama_format(monkeypatch) -> None:
    """When tools are provided, the payload includes them in Ollama's native format."""
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {"content": "I'll help with that."},
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    tools = [
        ToolDefinition(
            name="search_web",
            description="Search the web for information.",
            permission_level=ToolPermissionLevel.NETWORK,
            arguments={"query": "The search query string"},
        ),
        ToolDefinition(
            name="read_file",
            description="Read a local file.",
            permission_level=ToolPermissionLevel.LOCAL_READ,
            arguments={"path": "Absolute file path", "max_lines": "Maximum lines to read"},
        ),
    ]

    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(thinking_supported=False),
        messages=[LLMMessage(role=LLMRole.USER, content="search for python")],
        tools=tools,
    )

    sent_tools = captured["payload"]["tools"]  # type: ignore[index]
    assert len(sent_tools) == 2
    assert sent_tools[0] == {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string"},
                },
                "required": ["query"],
            },
        },
    }
    assert sent_tools[1]["function"]["name"] == "read_file"
    assert len(sent_tools[1]["function"]["parameters"]["required"]) == 2
    assert response.content == "I'll help with that."
    assert response.tool_calls is None


def test_chat_parses_tool_calls_from_response(monkeypatch) -> None:
    """When Ollama returns tool_calls, they are parsed into ToolCall objects."""

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_web",
                                    "arguments": {"query": "python tutorial"},
                                }
                            },
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": {"path": "/tmp/test.py"},
                                }
                            },
                        ],
                    },
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(thinking_supported=False),
        messages=[LLMMessage(role=LLMRole.USER, content="search for python")],
    )

    assert response.tool_calls is not None
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0] == ToolCall(
        tool_name="search_web", arguments={"query": "python tutorial"}
    )
    assert response.tool_calls[1] == ToolCall(
        tool_name="read_file", arguments={"path": "/tmp/test.py"}
    )


def test_chat_without_tools_does_not_include_tools_key(monkeypatch) -> None:
    """When no tools are provided, the payload must not contain a 'tools' key."""
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {"content": "Hello"},
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    provider.chat(
        profile=_build_profile(thinking_supported=False),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
    )

    assert "tools" not in captured["payload"]  # type: ignore[operator]


def test_chat_returns_none_tool_calls_when_model_does_not_call_tools(monkeypatch) -> None:
    """When the model returns no tool_calls, LLMResponse.tool_calls is None."""

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {"content": "Just text."},
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(thinking_supported=False),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
    )

    assert response.tool_calls is None
    assert response.content == "Just text."


def test_chat_handles_malformed_tool_calls_gracefully(monkeypatch) -> None:
    """Malformed tool_calls entries are silently skipped."""

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"not_function": {}},
                            {"function": {"name": ""}},
                            {"function": {"name": "valid_tool", "arguments": {"k": "v"}}},
                            "not_a_dict",
                        ],
                    },
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    response = provider.chat(
        profile=_build_profile(thinking_supported=False),
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
    )

    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "valid_tool"


@pytest.mark.parametrize("streaming", [False, True])
def test_chat_surfaces_explicit_connection_errors(monkeypatch, streaming) -> None:
    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        del request, timeout
        raise URLError("[Errno 111] Connection refused")

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    streamed_chunks: list[str] = []

    with pytest.raises(
        LLMConnectionError,
        match=re.escape(
            "Could not connect to Ollama at http://127.0.0.1:11434. "
            "Make sure the Ollama server is running."
        ),
    ):
        provider.chat(
            profile=_build_profile(thinking_supported=False),
            messages=[LLMMessage(role=LLMRole.USER, content="hi")],
            content_callback=streamed_chunks.append if streaming else None,
        )


@pytest.mark.parametrize("streaming", [False, True])
def test_chat_surfaces_explicit_timeout_errors(monkeypatch, streaming) -> None:
    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        del request, timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()
    streamed_chunks: list[str] = []

    with pytest.raises(
        LLMConnectionError,
        match=re.escape("Ollama request timed out after 7 seconds."),
    ):
        provider.chat(
            profile=_build_profile(thinking_supported=False),
            messages=[LLMMessage(role=LLMRole.USER, content="hi")],
            timeout_seconds=7.0,
            content_callback=streamed_chunks.append if streaming else None,
        )


def test_chat_surfaces_compact_http_error_details(monkeypatch) -> None:
    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        del request, timeout
        raise HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=io.BytesIO(b"{\"error\":\"model 'missing' not found\"}"),
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()

    with pytest.raises(
        LLMProviderError,
        match=re.escape("Ollama request failed with HTTP 404: model 'missing' not found"),
    ):
        provider.chat(
            profile=_build_profile(thinking_supported=False),
            messages=[LLMMessage(role=LLMRole.USER, content="hi")],
        )


def test_chat_surfaces_invalid_response_errors_cleanly(monkeypatch) -> None:
    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        del request, timeout
        return _FakeJsonResponse(
            json.dumps(
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-13T12:00:00Z",
                    "done_reason": "stop",
                    "message": {},
                }
            )
        )

    monkeypatch.setattr(ollama_provider, "urlopen", fake_urlopen)

    provider = ollama_provider.OllamaProvider()

    with pytest.raises(
        LLMResponseError,
        match=re.escape("Ollama returned an invalid response."),
    ):
        provider.chat(
            profile=_build_profile(thinking_supported=False),
            messages=[LLMMessage(role=LLMRole.USER, content="hi")],
        )


def _build_profile(
    *,
    model_name: str = "qwen3.5:4b",
    thinking_supported: bool,
    keep_alive: str | None = None,
) -> ResolvedModelProfile:
    return ResolvedModelProfile(
        name="main",
        provider="ollama",
        model_name=model_name,
        temperature=0.3,
        capabilities=ModelCapabilities(
            thinking_supported=thinking_supported,
            tool_mode="json_plan",
            supports_tools=True,
            supports_native_tool_calling=False,
        ),
        keep_alive=keep_alive,
    )


class _FakeJsonResponse:
    def __init__(self, body: str) -> None:
        self._body = body

    def __enter__(self) -> _FakeJsonResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False

    def read(self) -> bytes:
        return self._body.encode("utf-8")


class _FakeStreamResponse:
    def __init__(self, payloads: tuple[dict[str, object], ...]) -> None:
        self._lines = [
            json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
            for payload in payloads
        ]

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False

    def __iter__(self):
        return iter(self._lines)


# ---------------------------------------------------------------------------
# Real Ollama integration tests (skipped when Ollama is unavailable)
# ---------------------------------------------------------------------------

# Models known to support Ollama native tool calling.
_TOOL_CAPABLE_MODEL_PREFIXES = (
    "qwen",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "mistral",
    "command-r",
    "firefunction",
    "hermes",
)


def _find_tool_capable_model() -> str | None:
    """Return the name of a locally installed model that supports tool calling,
    or *None* if Ollama is unreachable or no suitable model is found."""
    provider = OllamaProvider()
    if not provider.is_available(timeout_seconds=5):
        return None
    try:
        models = provider.list_models(timeout_seconds=5)
    except Exception:
        return None
    for model in models:
        base = model.split(":")[0].lower()
        if any(base.startswith(prefix) for prefix in _TOOL_CAPABLE_MODEL_PREFIXES):
            return model
    return None


_INTEGRATION_MODEL = _find_tool_capable_model()

_skip_no_ollama = pytest.mark.skipif(
    _INTEGRATION_MODEL is None,
    reason="Ollama unavailable or no tool-capable model installed locally",
)


def _build_integration_profile(model_name: str) -> ResolvedModelProfile:
    return ResolvedModelProfile(
        name="integration",
        provider="ollama",
        model_name=model_name,
        temperature=0.0,
        capabilities=ModelCapabilities(
            thinking_supported=False,
            tool_mode="native",
            supports_tools=True,
            supports_native_tool_calling=True,
        ),
    )


_SIMPLE_TOOL = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a given city.",
    permission_level=ToolPermissionLevel.NETWORK,
    arguments={"city": "The city name to look up weather for"},
)


@pytest.mark.integration
@pytest.mark.real_ollama
@_skip_no_ollama
def test_integration_tool_definitions_accepted_by_ollama() -> None:
    """Ollama accepts a chat request with tool definitions and returns
    a structurally valid response (content string, valid model name,
    valid created_at)."""
    assert _INTEGRATION_MODEL is not None
    provider = OllamaProvider()
    profile = _build_integration_profile(_INTEGRATION_MODEL)

    response = provider.chat(
        profile=profile,
        messages=[LLMMessage(role=LLMRole.USER, content="Hello, what can you do?")],
        tools=[_SIMPLE_TOOL],
        timeout_seconds=120,
    )

    assert response.provider == "ollama"
    assert isinstance(response.model_name, str) and response.model_name
    assert isinstance(response.created_at, str) and response.created_at
    assert isinstance(response.content, str)
    assert isinstance(response.raw_payload, dict)


@pytest.mark.integration
@pytest.mark.real_ollama
@_skip_no_ollama
def test_integration_tool_call_response_parsed_into_contract() -> None:
    """When prompted to use a tool, a real Ollama model returns a tool_call
    that is correctly parsed into the ToolCall dataclass contract.

    If the model does not produce a tool call (model-dependent), the test
    verifies the response still conforms to the LLMResponse contract and
    skips the tool_call-specific assertions with a clear message."""
    assert _INTEGRATION_MODEL is not None
    provider = OllamaProvider()
    profile = _build_integration_profile(_INTEGRATION_MODEL)

    response = provider.chat(
        profile=profile,
        messages=[
            LLMMessage(
                role=LLMRole.USER,
                content="What is the weather in Paris? Use the get_weather tool.",
            ),
        ],
        tools=[_SIMPLE_TOOL],
        timeout_seconds=120,
    )

    # The response must always conform to the LLMResponse contract.
    assert response.provider == "ollama"
    assert isinstance(response.content, str)
    assert isinstance(response.raw_payload, dict)

    if response.tool_calls is None:
        pytest.skip(
            f"Model {_INTEGRATION_MODEL} did not produce a tool call for this prompt "
            "(model-dependent behavior); structural contract still validated."
        )

    # Validate parsed tool calls match the ToolCall contract.
    assert isinstance(response.tool_calls, tuple)
    assert len(response.tool_calls) >= 1

    first_call = response.tool_calls[0]
    assert isinstance(first_call, ToolCall)
    assert isinstance(first_call.tool_name, str) and first_call.tool_name
    assert isinstance(first_call.arguments, dict)
    # The model should have called our get_weather tool.
    assert first_call.tool_name == "get_weather"
    assert "city" in first_call.arguments
