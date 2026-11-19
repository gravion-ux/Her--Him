"""The persona model and its Character Card v3 round trip."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from herhim.models.affect import AffectVector
from herhim.models.card import CARD_SPEC_VERSION, HERHIM_EXTENSION_KEY
from herhim.models.relationship import RelationshipState
from herhim.models.traits import TraitProfile
from herhim.models.voice import VoiceBinding

__all__ = ["LoreEntry", "Persona"]


@dataclass(frozen=True)
class LoreEntry:
    """A keyword-triggered worldbook entry.

    Attributes:
        keys: Lowercase trigger keywords.
        content: The text injected when any key matches.
        position: Where in the prompt the content is placed.
        constant: When true the entry is always injected.
    """

    keys: tuple[str, ...]
    content: str
    position: str = "before_char"
    constant: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "keys": list(self.keys),
            "content": self.content,
            "position": self.position,
            "constant": self.constant,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoreEntry:
        """Build an entry from a mapping, ignoring unknown keys."""
        keys = data.get("keys", [])
        return cls(
            keys=tuple(str(key).lower() for key in keys),
            content=str(data.get("content", "")),
            position=str(data.get("position", "before_char")),
            constant=bool(data.get("constant", False)),
        )


@dataclass(frozen=True)
class Persona:
    """A complete companion persona.

    A persona is a Character Card v3 document plus the herhim extension block.
    The canonical card fields describe *who* the persona is; the extension
    block describes *how* she reacts.
    """

    persona_id: str
    catalogue: str
    gender: str
    display_name: str
    tagline: str
    description: str
    personality: str
    scenario: str
    first_mes: str
    mes_example: str
    system_prompt: str
    alternate_greetings: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    creator: str = "gravion-ux"
    character_version: str = "1.0.0"
    creator_notes: str = ""
    post_history_instructions: str = ""
    affect_baseline: AffectVector = field(default_factory=AffectVector)
    traits: TraitProfile = field(default_factory=TraitProfile)
    event_sensitivity: dict[str, float] = field(default_factory=dict)
    relationship_baseline: RelationshipState = field(default_factory=RelationshipState)
    voice: VoiceBinding = field(default_factory=lambda: VoiceBinding(voice_id=""))
    lorebook: tuple[LoreEntry, ...] = ()

    def to_card(self) -> dict[str, Any]:
        """Render the persona as a Character Card v3 document."""
        return {
            "spec_version": CARD_SPEC_VERSION,
            "name": self.display_name,
            "description": self.description,
            "personality": self.personality,
            "scenario": self.scenario,
            "first_mes": self.first_mes,
            "mes_example": self.mes_example,
            "creator_notes": self.creator_notes,
            "system_prompt": self.system_prompt,
            "post_history_instructions": self.post_history_instructions,
            "alternate_greetings": list(self.alternate_greetings),
            "tags": list(self.tags),
            "creator": self.creator,
            "character_version": self.character_version,
            "extensions": {
                HERHIM_EXTENSION_KEY: {
                    "persona_id": self.persona_id,
                    "catalogue": self.catalogue,
                    "gender": self.gender,
                    "tagline": self.tagline,
                    "affect_baseline": self.affect_baseline.to_dict(),
                    "traits": self.traits.to_dict(),
                    "event_sensitivity": dict(self.event_sensitivity),
                    "relationship_baseline": {
                        "trust": self.relationship_baseline.trust,
                        "intimacy": self.relationship_baseline.intimacy,
                        "tension": self.relationship_baseline.tension,
                        "jealousy": self.relationship_baseline.jealousy,
                        "dependency": self.relationship_baseline.dependency,
                        "commitment": self.relationship_baseline.commitment,
                    },
                    "voice": self.voice.to_dict(),
                    "lorebook": {
                        "entries": [entry.to_dict() for entry in self.lorebook],
                    },
                },
            },
        }

    @classmethod
    def from_card(cls, card: dict[str, Any]) -> Persona:
        """Build a persona from a Character Card v3 document."""
        extensions = card.get("extensions") or {}
        block = extensions.get(HERHIM_EXTENSION_KEY) or {}
        relationship = block.get("relationship_baseline") or {}
        lorebook = (block.get("lorebook") or {}).get("entries") or []
        return cls(
            persona_id=str(block.get("persona_id", "")),
            catalogue=str(block.get("catalogue", "")),
            gender=str(block.get("gender", "")),
            display_name=str(card.get("name", "")),
            tagline=str(block.get("tagline", "")),
            description=str(card.get("description", "")),
            personality=str(card.get("personality", "")),
            scenario=str(card.get("scenario", "")),
            first_mes=str(card.get("first_mes", "")),
            mes_example=str(card.get("mes_example", "")),
            system_prompt=str(card.get("system_prompt", "")),
            alternate_greetings=tuple(str(g) for g in card.get("alternate_greetings", [])),
            tags=tuple(str(t) for t in card.get("tags", [])),
            creator=str(card.get("creator", "gravion-ux")),
            character_version=str(card.get("character_version", "1.0.0")),
            creator_notes=str(card.get("creator_notes", "")),
            post_history_instructions=str(card.get("post_history_instructions", "")),
            affect_baseline=AffectVector.from_dict(block.get("affect_baseline") or {}),
            traits=TraitProfile.from_dict(block.get("traits") or {}),
            event_sensitivity={
                str(k): float(v) for k, v in (block.get("event_sensitivity") or {}).items()
            },
            relationship_baseline=RelationshipState.from_dict(relationship),
            voice=VoiceBinding.from_dict(block.get("voice") or {}),
            lorebook=tuple(LoreEntry.from_dict(entry) for entry in lorebook),
        )

    def sensitivity_for(self, event_kind: str) -> float:
        """Return the persona's sensitivity multiplier for an event kind."""
        return float(self.event_sensitivity.get(event_kind, 1.0))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the full card."""
        return self.to_card()

    def summary(self) -> dict[str, Any]:
        """Return a compact listing entry for catalogue views."""
        return {
            "persona_id": self.persona_id,
            "catalogue": self.catalogue,
            "gender": self.gender,
            "display_name": self.display_name,
            "tagline": self.tagline,
            "locale": self.voice.locale,
            "voice_id": self.voice.voice_id,
            "sample_id": self.voice.sample_id,
            "tags": list(self.tags),
        }
