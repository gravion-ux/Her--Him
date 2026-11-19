"""An OpenAI-compatible HTTP provider.

Works unchanged against Ollama, LM Studio, vLLM, OpenRouter, and OpenAI
itself. Only the standard library is used, so the package keeps its
zero-runtime-dependency property.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from herhim.errors import ProviderError
from herhim.providers.base import LLMResponse, TurnContext

__all__ = ["ENV_BASE_URL", "ENV_MODEL", "OpenAICompatibleProvider"]

ENV_BASE_URL = "HERHIM_LLM_BASE_URL"
ENV_MODEL = "HERHIM_LLM_MODEL"
ENV_API_KEY = "HERHIM_LLM_API_KEY"

_DEFAULT_TIMEOUT = 120.0


class OpenAICompatibleProvider:
    """Talks to any server that speaks the ``/chat/completions`` dialect."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = (base_url or os.environ.get(ENV_BASE_URL, "")).rstrip("/")
        self.model = model or os.environ.get(ENV_MODEL, "")
        self._api_key = api_key if api_key is not None else os.environ.get(ENV_API_KEY, "")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        """Whether the provider has enough configuration to attempt a call."""
        return bool(self.base_url and self.model)

    @property
    def endpoint(self) -> str:
        """The full URL this provider posts to."""
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.9,
        max_tokens: int = 400,
        context: TurnContext | None = None,
    ) -> LLMResponse:
        """Post a chat completion request and return the reply."""
        if not self.configured:
            raise ProviderError(
                f"{self.name} provider is not configured: set {ENV_BASE_URL} and {ENV_MODEL}"
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(  # noqa: S310 - explicit http(s) endpoint
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(
                f"{self.name} request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"{self.name} request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.name} returned a non-JSON response") from exc

        choices = parsed.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name} returned no choices")
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "").strip()
        if not text:
            raise ProviderError(f"{self.name} returned an empty message")
        return LLMResponse(text=text, provider=self.name, model=self.model, raw=parsed)
