"""Minimal orchestration for one chat turn."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from unclaw.constants import DEFAULT_CONTEXT_HISTORY_MESSAGE_LIMIT
from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.core.context_builder import build_context_messages
from unclaw.core.session_manager import SessionManager
from unclaw.core.timing import elapsed_ms
from unclaw.errors import UnclawError
from unclaw.llm.base import LLMContentCallback, LLMError, LLMMessage, LLMResponse
from unclaw.llm.model_profiles import resolve_model_profile
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.logs.tracer import Tracer
from unclaw.settings import Settings
from unclaw.tools.contracts import ToolDefinition


class OrchestratorError(UnclawError):
    """Raised when the minimal orchestrator cannot run a turn."""


class ModelCallFailedError(OrchestratorError):
    """Raised when the provider call fails after the route has been selected."""

    def __init__(
        self,
        *,
        provider: str,
        model_profile_name: str,
        model_name: str,
        duration_ms: int,
        error: str,
    ) -> None:
        super().__init__(error)
        self.provider = provider
        self.model_profile_name = model_profile_name
        self.model_name = model_name
        self.duration_ms = duration_ms
        self.error = error


@dataclass(frozen=True, slots=True)
class OrchestratorTurnResult:
    """Completed model turn metadata returned to the runtime."""

    response: LLMResponse
    model_duration_ms: int
    context_messages: tuple[LLMMessage, ...] = ()


@dataclass(slots=True)
class Orchestrator:
    """Coordinate the minimal model call for one user turn."""

    settings: Settings
    session_manager: SessionManager
    tracer: Tracer

    def run_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        model_profile_name: str,
        max_history_size: int | None = DEFAULT_CONTEXT_HISTORY_MESSAGE_LIMIT,
        capability_summary: RuntimeCapabilitySummary | None = None,
        system_context_notes: Sequence[str] | None = None,
        thinking_enabled: bool = False,
        content_callback: LLMContentCallback | None = None,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> OrchestratorTurnResult:
        """Resolve the model, build context, and call the provider."""
        profile = resolve_model_profile(self.settings, model_profile_name)
        provider = self._create_provider(profile.provider)
        context_messages = build_context_messages(
            session_manager=self.session_manager,
            session_id=session_id,
            user_message=user_message,
            max_history_size=max_history_size,
            capability_summary=capability_summary,
            system_context_notes=system_context_notes,
            model_profile_name=model_profile_name,
        )

        self.tracer.trace_model_called(
            session_id=session_id,
            provider=profile.provider,
            model_profile_name=profile.name,
            model_name=profile.model_name,
            message_count=len(context_messages),
        )

        model_started_at = perf_counter()
        try:
            response = provider.chat(
                profile=profile,
                messages=context_messages,
                thinking_enabled=thinking_enabled,
                content_callback=content_callback,
                tools=tools,
            )
        except LLMError as exc:
            raise ModelCallFailedError(
                provider=profile.provider,
                model_profile_name=profile.name,
                model_name=profile.model_name,
                duration_ms=elapsed_ms(model_started_at),
                error=str(exc),
            ) from exc

        return OrchestratorTurnResult(
            response=response,
            model_duration_ms=elapsed_ms(model_started_at),
            context_messages=tuple(context_messages),
        )

    def call_model(
        self,
        *,
        session_id: str,
        messages: Sequence[LLMMessage],
        model_profile_name: str,
        thinking_enabled: bool = False,
        content_callback: LLMContentCallback | None = None,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> OrchestratorTurnResult:
        """Call the model with pre-built messages (for agent loop iterations)."""
        profile = resolve_model_profile(self.settings, model_profile_name)
        provider = self._create_provider(profile.provider)

        self.tracer.trace_model_called(
            session_id=session_id,
            provider=profile.provider,
            model_profile_name=profile.name,
            model_name=profile.model_name,
            message_count=len(messages),
        )

        model_started_at = perf_counter()
        try:
            response = provider.chat(
                profile=profile,
                messages=messages,
                thinking_enabled=thinking_enabled,
                content_callback=content_callback,
                tools=tools,
            )
        except LLMError as exc:
            raise ModelCallFailedError(
                provider=profile.provider,
                model_profile_name=profile.name,
                model_name=profile.model_name,
                duration_ms=elapsed_ms(model_started_at),
                error=str(exc),
            ) from exc

        return OrchestratorTurnResult(
            response=response,
            model_duration_ms=elapsed_ms(model_started_at),
            context_messages=tuple(messages),
        )

    def _create_provider(self, provider_name: str) -> OllamaProvider:
        if provider_name == OllamaProvider.provider_name:
            return OllamaProvider(
                default_timeout_seconds=(
                    self.settings.app.providers.ollama.timeout_seconds
                )
            )

        raise OrchestratorError(
            f"Provider '{provider_name}' is not supported by the minimal runtime."
        )
