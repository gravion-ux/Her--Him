"""A conversation session: the persona, its engine, memory, provider, and voice.

This is where the pieces meet. The order of operations is deliberate:

1. The engine classifies the user's turn and moves the emotional state.
2. The engine compiles a style directive from that state.
3. The provider is asked to express the state in words.
4. The same state is mapped into voice parameters.

The provider is never asked what the persona feels. It is told.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from herhim.affect.directive import build_directive
from herhim.affect.engine import AffectEngine, TurnRationale, persona_engine
from herhim.catalog.worldbook import scan_lorebook
from herhim.guardrails.audit import AuditLog
from herhim.guardrails.consent import assert_consent
from herhim.guardrails.scope import assert_adult_scope
from herhim.memory.recall import recall
from herhim.memory.store import MemoryStore
from herhim.models.persona import Persona
from herhim.models.voice import VoiceClip, VoiceParams
from herhim.providers.base import LLMProvider, LLMResponse, TurnContext
from herhim.providers.offline import OfflineProvider
from herhim.providers.router import ProviderRouter
from herhim.voice.emotion_map import map_to_params
from herhim.voice.samples import SamplesVoiceEngine

__all__ = ["Session", "TurnResult"]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TurnResult:
    """Everything one turn produced.

    Attributes:
        persona_id: The persona who spoke.
        text: The reply.
        mood: The mood the engine derived.
        rationale: The full explanation of the turn.
        state: The emotional and relational state after the turn.
        voice: The synthesis parameters for this turn.
        segments: The reply split into speakable segments.
        provider: The provider that generated the reply.
    """

    persona_id: str
    text: str
    mood: str
    rationale: dict[str, Any]
    state: dict[str, Any]
    voice: VoiceParams | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "persona_id": self.persona_id,
            "text": self.text,
            "mood": self.mood,
            "provider": self.provider,
            "rationale": self.rationale,
            "state": self.state,
            "voice": self.voice.to_dict() if self.voice else None,
            "segments": self.segments,
        }


class Session:
    """One conversation with one persona."""

    def __init__(
        self,
        persona: Persona,
        *,
        provider: LLMProvider | None = None,
        memory: MemoryStore | None = None,
        audit: AuditLog | None = None,
        voice_engine: Any | None = None,
    ) -> None:
        assert_adult_scope(persona)
        self.persona = persona
        self.engine: AffectEngine = persona_engine(persona)
        self.memory = memory or MemoryStore()
        self.audit = audit or AuditLog()
        self.router = ProviderRouter(provider or OfflineProvider())
        self.voice_engine = voice_engine or SamplesVoiceEngine()
        self.session_id = uuid.uuid4().hex
        self._event_counts: Counter[str] = Counter()

    # -- state ----------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Return the current state of the session."""
        snapshot = self.engine.snapshot()
        snapshot["session_id"] = self.session_id
        snapshot["turns"] = len(self.memory.recent())
        return snapshot

    def reset(self) -> None:
        """Return the session to its opening state."""
        self.engine.reset()
        self.memory.clear()
        self._event_counts.clear()
        self.audit.record("session.reset", self.persona.persona_id, "record")

    # -- prompting ------------------------------------------------------

    def _relationship_summary(self) -> str:
        """Render the relationship state as prompt text."""
        relationship = self.engine.relationship
        return (
            f"Relationship: {relationship.stage().value}. "
            f"trust {relationship.trust:.2f}, intimacy {relationship.intimacy:.2f}, "
            f"tension {relationship.tension:.2f}, jealousy {relationship.jealousy:.2f}, "
            f"dependency {relationship.dependency:.2f}, commitment {relationship.commitment:.2f}."
        )

    def _memory_block(self, user_text: str) -> str:
        """Render recalled memory as prompt text, keeping sources visible."""
        recalled = recall(self.memory, user_text)
        if not recalled:
            return ""
        lines = []
        for item in recalled:
            source = item.record.source or "unknown"
            lines.append(f"- ({source}) {item.record.text}")
        return "WHAT YOU REMEMBER\n" + "\n".join(lines)

    def build_messages(self, user_text: str, rationale: TurnRationale) -> list[dict[str, str]]:
        """Assemble the prompt for one turn."""
        triggered = scan_lorebook(self.persona.lorebook, user_text)
        lore_block = ""
        if triggered:
            entries = "\n".join(f"- {entry.content}" for entry in triggered)
            lore_block = f"WHAT YOU KNOW\n{entries}"

        directive = build_directive(
            rationale.mood,
            rationale.after,
            rationale.stage_after,
            self.persona.traits,
            boundary_held=rationale.boundary_held,
        )

        system_parts = [
            self.persona.system_prompt,
            lore_block,
            self._relationship_summary(),
            self._memory_block(user_text),
            directive,
        ]
        system = "\n\n".join(part for part in system_parts if part)

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for record in self.memory.recent(limit=12):
            role = "assistant" if record.role == "assistant" else "user"
            messages.append({"role": role, "content": record.text})
        messages.append({"role": "user", "content": user_text})
        return messages

    # -- the turn -------------------------------------------------------

    def turn(self, user_text: str) -> TurnResult:
        """Process one user turn and return everything it produced."""
        rationale = self.engine.apply_turn(user_text)

        primary_event = rationale.events[0].kind.value if rationale.events else ""
        ordinal = self._event_counts[primary_event]
        self._event_counts[primary_event] += 1

        messages = self.build_messages(user_text, rationale)
        context = TurnContext(
            persona_id=self.persona.persona_id,
            mood=rationale.mood.value,
            events=tuple(event.kind.value for event in rationale.events),
            event_ordinal=ordinal,
            stage=rationale.stage_after.value,
            turn_index=self.engine.turn_index - 1,
            user_text=user_text,
        )

        try:
            response: LLMResponse = self.router.complete(messages, context=context)
        except Exception as exc:
            self.audit.record(
                "provider.failure",
                self.persona.persona_id,
                "refuse",
                {"error": str(exc)},
            )
            raise

        assert_consent(
            self.engine.relationship,
            tuple(event.kind for event in rationale.events),
        )

        self.memory.add_turn("user", user_text, source="user")
        self.memory.add_turn("assistant", response.text, source=self.persona.persona_id)
        self.memory.add_episode(
            f"Turn {self.engine.turn_index}: {primary_event or 'neutral'} -> {rationale.mood.value}",
            source="affect-engine",
        )

        if rationale.stage_before is not rationale.stage_after:
            self.audit.record(
                "relationship.stage",
                self.persona.persona_id,
                "record",
                {
                    "from": rationale.stage_before.value,
                    "to": rationale.stage_after.value,
                    "mood": rationale.mood.value,
                },
            )

        params = map_to_params(
            rationale.after,
            self.persona.traits,
            rationale.mood,
            self.persona.voice,
        )
        segments = self._segments(response.text, params)

        return TurnResult(
            persona_id=self.persona.persona_id,
            text=response.text,
            mood=rationale.mood.value,
            rationale=rationale.to_dict(),
            state=self.state(),
            voice=params,
            segments=segments,
            provider=response.provider,
        )

    def _segments(self, text: str, params: VoiceParams) -> list[dict[str, Any]]:
        """Split a reply into speakable segments."""
        parts = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
        if not parts:
            return []
        return [
            {
                "text": part,
                "display_text": part,
                "voice": params.to_dict(),
            }
            for part in parts
        ]

    # -- voice ----------------------------------------------------------

    def speak(self, text: str | None = None) -> VoiceClip:
        """Render a clip for the persona's current state."""
        params = map_to_params(
            self.engine.vector,
            self.persona.traits,
            self.engine.mood,
            self.persona.voice,
        )
        return self.voice_engine.synthesize(text or "", params)
