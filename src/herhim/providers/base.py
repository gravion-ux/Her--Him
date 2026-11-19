"""The provider protocol and shared response type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["LLMProvider", "LLMResponse", "TurnContext"]


@dataclass(frozen=True)
class TurnContext:
    """Everything a provider may need beyond the raw messages.

    The offline provider uses this to stay state-aware without an LLM. The
    HTTP provider ignores it.

    Attributes:
        persona_id: The persona speaking.
        mood: The mood the engine derived for this turn.
        events: The event kinds that fired, primary first.
        event_ordinal: How many times the primary event has fired in this
            session. Used to rotate deterministic response variants.
        stage: The relationship stage.
        turn_index: How many turns have happened in this session.
        user_text: The raw user turn.
    """

    persona_id: str = ""
    mood: str = "calm"
    events: tuple[str, ...] = ()
    event_ordinal: int = 0
    stage: str = "STRANGER"
    turn_index: int = 0
    user_text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "persona_id": self.persona_id,
            "mood": self.mood,
            "events": list(self.events),
            "event_ordinal": self.event_ordinal,
            "stage": self.stage,
            "turn_index": self.turn_index,
            "user_text": self.user_text,
        }


@dataclass(frozen=True)
class LLMResponse:
    """A generated reply.

    Attributes:
        text: The reply text.
        provider: The provider that produced it.
        model: The model identifier, or a synthetic label for offline output.
        raw: The provider's raw payload, when there is one.
    """

    text: str
    provider: str
    model: str = ""
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
        }


@runtime_checkable
class LLMProvider(Protocol):
    """The interface every backend implements."""

    name: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.9,
        max_tokens: int = 400,
        context: TurnContext | None = None,
    ) -> LLMResponse:
        """Generate a reply for ``messages``."""
        ...
