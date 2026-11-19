"""The provider router.

The router refuses silent substitution. If the configured provider fails, the
failure is surfaced with the endpoint and model that were attempted. It never
quietly swaps a paid model for a free one, or a local model for a remote one.

This behaviour is adapted from the provider-routing pattern in
``gravion-ux/aegis`` (MIT).
"""

from __future__ import annotations

from herhim.errors import ProviderError
from herhim.providers.base import LLMProvider, LLMResponse, TurnContext

__all__ = ["ProviderRouter"]


class ProviderRouter:
    """Routes requests to one provider without silent fallback."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        allowed_fallbacks: tuple[str, ...] = (),
    ) -> None:
        """
        Args:
            provider: The provider to use.
            allowed_fallbacks: Names of providers that may be used *only* when
                explicitly listed here. Empty by default, which means no
                substitution is ever permitted.
        """
        self._provider = provider
        self._allowed_fallbacks = allowed_fallbacks
        self.attempts: list[str] = []

    @property
    def provider(self) -> LLMProvider:
        """The primary provider."""
        return self._provider

    @property
    def name(self) -> str:
        """The name of the primary provider."""
        return self._provider.name

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.9,
        max_tokens: int = 400,
        context: TurnContext | None = None,
    ) -> LLMResponse:
        """Send the request to the primary provider.

        Raises:
            ProviderError: When the primary provider fails. The error names
                the provider that was attempted.
        """
        self.attempts.append(self._provider.name)
        try:
            return self._provider.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                context=context,
            )
        except ProviderError as exc:
            raise ProviderError(
                f"provider {self._provider.name!r} failed and silent substitution is "
                f"disabled: {exc}"
            ) from exc

    def describe(self) -> dict[str, object]:
        """Return a JSON-serialisable description of the routing setup."""
        return {
            "provider": self._provider.name,
            "allowed_fallbacks": list(self._allowed_fallbacks),
            "attempts": list(self.attempts),
        }
