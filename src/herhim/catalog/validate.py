"""Structural validation of persona cards.

The rules here are the validity half of the catalogue quality gate. The
balance, contradiction, and coverage rules live in :mod:`herhim.catalog.quality`.
"""

from __future__ import annotations

from herhim.models.card import validate_card_shape
from herhim.models.events import EventKind
from herhim.models.persona import Persona
from herhim.models.traits import TRAIT_NAMES

__all__ = ["validate_persona", "validate_persona_card"]

_REQUIRED_TEXT = (
    "display_name",
    "tagline",
    "description",
    "personality",
    "scenario",
    "first_mes",
    "system_prompt",
)


def validate_persona(persona: Persona) -> list[str]:
    """Return a list of problems with a persona. Empty means valid."""
    problems: list[str] = []

    if not persona.persona_id:
        problems.append("persona_id is empty")
    if persona.catalogue not in ("her", "him"):
        problems.append(f"unknown catalogue: {persona.catalogue!r}")
    if persona.gender not in ("female", "male"):
        problems.append(f"unknown gender: {persona.gender!r}")

    for field in _REQUIRED_TEXT:
        if not getattr(persona, field).strip():
            problems.append(f"{field} is empty")

    if not persona.lorebook:
        problems.append("lorebook is empty")
    for entry in persona.lorebook:
        if not entry.keys:
            problems.append(f"lorebook entry has no keys: {entry.content[:40]!r}")
        if not entry.content.strip():
            problems.append("lorebook entry has empty content")

    for name in TRAIT_NAMES:
        value = persona.traits.get(name)
        if not 0.0 <= value <= 1.0:
            problems.append(f"trait {name} out of range: {value}")

    if not -1.0 <= persona.affect_baseline.valence <= 1.0:
        problems.append("affect baseline valence out of range")
    for name in ("arousal", "assertion", "tenderness"):
        value = getattr(persona.affect_baseline, name)
        if not 0.0 <= value <= 1.0:
            problems.append(f"affect baseline {name} out of range")

    for kind, value in persona.event_sensitivity.items():
        if kind not in {event.value for event in EventKind}:
            problems.append(f"unknown event kind in sensitivity: {kind}")
        if not 0.0 <= value <= 3.0:
            problems.append(f"event sensitivity for {kind} out of range: {value}")

    if not persona.voice.voice_id:
        problems.append("voice binding has no voice_id")
    if not persona.voice.locale:
        problems.append("voice binding has no locale")

    return problems


def validate_persona_card(card: dict[str, object]) -> list[str]:
    """Return a list of problems with a raw card document."""
    problems = validate_card_shape(card)
    if problems:
        return problems
    return validate_persona(Persona.from_card(card))
