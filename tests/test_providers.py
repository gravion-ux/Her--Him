"""Tests for the provider protocol, the offline provider, the router, and HTTP."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from herhim.affect.moods import MoodLabel
from herhim.errors import ProviderError
from herhim.models.events import EventKind
from herhim.providers import build_router
from herhim.providers import offline as offline_module
from herhim.providers.base import LLMProvider, LLMResponse, TurnContext
from herhim.providers.offline import OfflineProvider
from herhim.providers.openai_compat import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    OpenAICompatibleProvider,
)
from herhim.providers.router import ProviderRouter


class _FakeResponse:
    """A minimal stand-in for an HTTP response body."""

    def __init__(self, body: bytes) -> None:
        """Store the body the fake response will return."""
        self._body = body

    def read(self) -> bytes:
        """Return the stored body."""
        return self._body

    def __enter__(self) -> _FakeResponse:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Leave the context manager."""


class TestProviderProtocol:
    """Tests for the shared provider types."""

    def test_turn_context_serialises(self) -> None:
        """The turn context serialises every field a provider may read."""
        context = TurnContext(persona_id="p-1", mood="calm", events=("TEASE",), turn_index=2)
        data = context.to_dict()
        assert data["persona_id"] == "p-1"
        assert data["events"] == ["TEASE"]
        assert data["turn_index"] == 2

    def test_turn_context_defaults(self) -> None:
        """A bare turn context is still usable."""
        context = TurnContext()
        assert context.events == ()
        assert context.stage == "STRANGER"

    def test_response_serialises(self) -> None:
        """A response serialises its text, provider, and model."""
        data = LLMResponse("hello", "offline", "model-1").to_dict()
        assert data == {"text": "hello", "provider": "offline", "model": "model-1"}

    def test_the_offline_provider_satisfies_the_protocol(self) -> None:
        """The offline provider is a valid provider."""
        assert isinstance(OfflineProvider(), LLMProvider)


class TestOfflineProvider:
    """Tests for the deterministic offline provider."""

    def test_identity(self) -> None:
        """The provider names itself."""
        provider = OfflineProvider()
        assert provider.name == "offline"
        assert provider.model

    def test_an_event_selects_the_reply(self) -> None:
        """The primary event picks the reply line."""
        provider = OfflineProvider()
        response = provider.complete(
            [{"role": "user", "content": "no"}],
            context=TurnContext(events=("REJECTION",), mood="calm"),
        )
        assert response.text
        assert response.provider == "offline"

    def test_the_ordinal_rotates_the_variants(self) -> None:
        """The same event says something different on each turn."""
        provider = OfflineProvider()
        first = provider.complete([], context=TurnContext(events=("REJECTION",), event_ordinal=0))
        second = provider.complete([], context=TurnContext(events=("REJECTION",), event_ordinal=1))
        assert first.text != second.text

    def test_an_unknown_event_falls_back_to_the_mood(self) -> None:
        """An event the provider does not know falls back to the mood."""
        provider = OfflineProvider()
        response = provider.complete([], context=TurnContext(events=("NONSENSE",), mood="joyful"))
        assert response.text

    def test_an_unknown_mood_falls_back_to_calm(self) -> None:
        """A mood the provider does not know falls back to calm."""
        provider = OfflineProvider()
        response = provider.complete([], context=TurnContext(events=(), mood="nonsense"))
        assert response.text == "Okay."

    def test_without_a_context_the_reply_is_neutral(self) -> None:
        """A provider call without a context still answers."""
        response = OfflineProvider().complete([])
        assert response.text
        assert response.model

    def test_every_mood_has_a_fallback_line(self) -> None:
        """Every mood label has at least one offline line."""
        for label in MoodLabel:
            response = OfflineProvider().complete([], context=TurnContext(mood=label.value))
            assert response.text

    def test_event_without_lines_falls_back_to_mood(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When an event has no lines, fallback to mood selection."""
        original_lines = offline_module._EVENT_LINES.copy()
        # Add an event kind with no lines (empty tuple is falsy)
        offline_module._EVENT_LINES[EventKind.APOLOGY] = ()  # type: ignore[assignment]

        provider = OfflineProvider()
        response = provider.complete(
            [], context=TurnContext(events=("APOLOGY",), mood="calm")
        )
        assert response.text  # Should fall back to calm mood

        # Restore
        offline_module._EVENT_LINES.clear()
        offline_module._EVENT_LINES.update(original_lines)


class TestProviderRouter:
    """Tests for the provider router."""

    def test_identity(self) -> None:
        """The router reports the provider it wraps."""
        provider = OfflineProvider()
        router = ProviderRouter(provider)
        assert router.provider is provider
        assert router.name == "offline"

    def test_complete_delegates(self) -> None:
        """The router forwards the request to the provider."""
        router = ProviderRouter(OfflineProvider())
        response = router.complete([{"role": "user", "content": "hi"}])
        assert response.provider == "offline"
        assert router.attempts == ["offline"]

    def test_a_failure_is_surfaced_with_the_provider_name(self) -> None:
        """A provider failure names the provider that was attempted."""

        class _Broken:
            name = "broken"

            def complete(self, messages, **kwargs):
                raise ProviderError("down")

        router = ProviderRouter(_Broken())
        with pytest.raises(ProviderError, match="broken"):
            router.complete([])

    def test_describe(self) -> None:
        """The router describes itself."""
        router = ProviderRouter(OfflineProvider())
        router.complete([])
        data = router.describe()
        assert data["provider"] == "offline"
        assert data["attempts"] == ["offline"]
        assert data["allowed_fallbacks"] == []


class TestOpenAICompatibleProvider:
    """Tests for the OpenAI-compatible HTTP provider."""

    def test_not_configured_without_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A provider with no configuration refuses to run."""
        monkeypatch.delenv(ENV_BASE_URL, raising=False)
        monkeypatch.delenv(ENV_MODEL, raising=False)
        provider = OpenAICompatibleProvider()
        assert provider.configured is False

    def test_configured_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The environment supplies the endpoint and the model."""
        monkeypatch.setenv(ENV_BASE_URL, "http://localhost:11434/v1/")
        monkeypatch.setenv(ENV_MODEL, "llama3")
        provider = OpenAICompatibleProvider()
        assert provider.configured is True
        assert provider.endpoint == "http://localhost:11434/v1/chat/completions"

    def test_explicit_arguments_win(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit arguments override the environment."""
        monkeypatch.setenv(ENV_BASE_URL, "http://from-env/v1")
        provider = OpenAICompatibleProvider(base_url="http://explicit/v1", model="m")
        assert provider.endpoint == "http://explicit/v1/chat/completions"

    def test_the_api_key_is_sent_as_a_bearer_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The key becomes an authorisation header."""
        monkeypatch.setenv(ENV_API_KEY, "abc123")
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m")
        assert provider._headers()["Authorization"] == "Bearer abc123"

    def test_no_api_key_means_no_authorization_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a key only the content type is sent."""
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m")
        assert "Authorization" not in provider._headers()

    def test_an_unconfigured_provider_refuses(self) -> None:
        """An unconfigured provider raises a provider error."""
        with pytest.raises(ProviderError, match="not configured"):
            OpenAICompatibleProvider(base_url="", model="").complete([])

    def test_a_successful_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A well-formed response becomes an LLM response."""
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["request"] = request
            captured["timeout"] = timeout
            body = json.dumps(
                {"choices": [{"message": {"content": "Hello there"}}], "id": "abc"}
            ).encode("utf-8")
            return _FakeResponse(body)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m", timeout=5.0)
        response = provider.complete([{"role": "user", "content": "hi"}])
        assert response.text == "Hello there"
        assert response.provider == "openai-compatible"
        assert captured["timeout"] == 5.0
        request = captured["request"]
        assert request.get_method() == "POST"

    def test_an_http_error_is_surfaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An HTTP error becomes a provider error."""

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(
                "http://x/v1",
                500,
                "Server Error",
                {},
                io.BytesIO(b"detail"),
            )

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m")
        with pytest.raises(ProviderError, match="HTTP 500"):
            provider.complete([])

    def test_a_connection_error_is_surfaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A connection error becomes a provider error."""

        def fake_urlopen(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m")
        with pytest.raises(ProviderError, match="connection refused"):
            provider.complete([])

    def test_a_non_json_body_is_surfaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A body that is not JSON becomes a provider error."""
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeResponse(b"<html>nope</html>"),
        )
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m")
        with pytest.raises(ProviderError, match="non-JSON"):
            provider.complete([])

    def test_an_empty_choice_list_is_surfaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A response without choices becomes a provider error."""
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeResponse(json.dumps({"choices": []}).encode()),
        )
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m")
        with pytest.raises(ProviderError, match="no choices"):
            provider.complete([])

    def test_an_empty_message_is_surfaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A response with no content becomes a provider error."""
        payload = json.dumps({"choices": [{"message": {"content": "  "}}]}).encode("utf-8")
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeResponse(payload),
        )
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m")
        with pytest.raises(ProviderError, match="empty message"):
            provider.complete([])


def test_build_router_defaults_to_the_offline_provider() -> None:
    """A router built without a provider uses the offline engine."""
    router = build_router()
    assert isinstance(router.provider, OfflineProvider)
