"""Neutral helper for model-assisted search grounding calls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from unclaw.core.orchestrator import (
    ModelCallFailedError,
    Orchestrator,
    OrchestratorError,
)
from unclaw.errors import ConfigurationError
from unclaw.llm.base import LLMMessage


class _NoopGroundingTracer:
    def trace_model_called(
        self,
        *,
        session_id: str,
        provider: str,
        model_profile_name: str,
        model_name: str,
        message_count: int,
    ) -> None:
        del session_id, provider, model_profile_name, model_name, message_count


def run_grounding_model_call(
    *,
    settings: Any,
    model_profile_name: str,
    messages: Sequence[LLMMessage],
    timeout_seconds: float,
) -> str | None:
    """Run a grounding-only model call through the standard orchestrator path."""
    try:
        model_profile = settings.models[model_profile_name]
    except (AttributeError, KeyError, TypeError):
        return None

    semantic_settings = replace(
        settings,
        app=replace(
            settings.app,
            providers=replace(
                settings.app.providers,
                ollama=replace(
                    settings.app.providers.ollama,
                    timeout_seconds=timeout_seconds,
                ),
            ),
        ),
        models={
            **settings.models,
            model_profile_name: replace(model_profile, temperature=0.0),
        },
    )
    orchestrator = Orchestrator(
        settings=semantic_settings,
        session_manager=None,
        tracer=_NoopGroundingTracer(),
    )
    try:
        response = orchestrator.call_model(
            session_id="search_grounding",
            messages=messages,
            model_profile_name=model_profile_name,
            thinking_enabled=False,
        )
    except (ConfigurationError, ModelCallFailedError, OrchestratorError):
        return None

    return response.response.content
