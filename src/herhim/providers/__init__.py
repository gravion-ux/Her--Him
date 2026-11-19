"""LLM providers: the offline engine and the OpenAI-compatible HTTP client."""

from __future__ import annotations

from herhim.providers.base import LLMProvider, LLMResponse, TurnContext
from herhim.providers.offline import OfflineProvider
from herhim.providers.openai_compat import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    OpenAICompatibleProvider,
)
from herhim.providers.router import ProviderRouter

__all__ = [
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "LLMProvider",
    "LLMResponse",
    "OfflineProvider",
    "OpenAICompatibleProvider",
    "ProviderRouter",
    "TurnContext",
    "build_router",
]


def build_router(provider: LLMProvider | None = None) -> ProviderRouter:
    """Build a router around a provider, defaulting to the offline engine."""
    return ProviderRouter(provider or OfflineProvider())
