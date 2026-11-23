"""The provider router.

The router refuses silent substitution. If the configured provider fails, the
failure is surfaced with the endpoint and model that were attempted. It never
quietly swaps a paid model for a free one, or a local model for a remote one.

This behaviour is adapted from the provider-routing pattern in
``gravion-ux/aegis`` (MIT).
"""

from __future__ import annotations

from collections.abc import Mapping

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
        fallbacks: Mapping[str, LLMProvider] | None = None,
    ) -> None:
        """
        Args:
            provider: The provider to use.
            allowed_fallbacks: Names of providers that may be used *only* when
                explicitly listed here. Empty by default, which means no
                substitution is ever permitted.
            fallbacks: The providers the allowed names refer to. A name in
                ``allowed_fallbacks`` with no provider behind it is a
                configuration error and is refused.
        """
        self._provider = provider
        self._allowed_fallbacks = allowed_fallbacks
        self._fallbacks = dict(fallbacks or {})
        unknown = [name for name in allowed_fallbacks if name not in self._fallbacks]
        if unknown:
            raise ValueError(
                "allowed_fallbacks names providers that were not supplied: "
                + ", ".join(unknown)
            )
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

        When the primary fails, the providers named in ``allowed_fallbacks``
        are tried in that order. A substitution only ever happens to a
        provider the caller listed explicitly, so a paid model is never
        quietly swapped for a free one.

        Raises:
            ProviderError: When every candidate fails. The error names each
                provider that was attempted.
        """
        candidates = [self._provider]
        candidates.extend(self._fallbacks[name] for name in self._allowed_fallbacks)

        failures: list[str] = []
        for candidate in candidates:
            self.attempts.append(candidate.name)
            try:
                return candidate.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context=context,
                )
            except ProviderError as exc:
                failures.append("{} ({})".format(candidate.name, exc))
        raise ProviderError(
            "provider {!r} failed and silent substitution is disabled: {}".format(self._provider.name, "; ".join(failures))
        )

    def describe(self) -> dict[str, object]:
        """Return a JSON-serialisable description of the routing setup."""
        return {
            "provider": self._provider.name,
            "allowed_fallbacks": list(self._allowed_fallbacks),
            "fallback_providers": sorted(self._fallbacks),
            "attempts": list(self.attempts),
        }
