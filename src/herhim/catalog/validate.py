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

#: The largest per-event sensitivity a persona may declare.
MAX_EVENT_SENSITIVITY = 3.0


def _validate_identity(persona: Persona) -> list[str]:
    """Check the persona's identity fields."""
    problems: list[str] = []
    if not persona.persona_id:
        problems.append("persona_id is empty")
    if persona.catalogue not in ("her", "him"):
        problems.append(f"unknown catalogue: {persona.catalogue!r}")
    if persona.gender not in ("female", "male"):
        problems.append(f"unknown gender: {persona.gender!r}")
    return problems


def _validate_text(persona: Persona) -> list[str]:
    """Check that every required text field carries content."""
    problems: list[str] = []
    for field in _REQUIRED_TEXT:
        if not getattr(persona, field).strip():
            problems.append(f"{field} is empty")
    return problems


def _validate_lorebook(persona: Persona) -> list[str]:
    """Check the lorebook entries."""
    problems: list[str] = []
    if not persona.lorebook:
        problems.append("lorebook is empty")
    for entry in persona.lorebook:
        if not entry.keys:
            problems.append(f"lorebook entry has no keys: {entry.content[:40]!r}")
        if not entry.content.strip():
            problems.append("lorebook entry has empty content")
    return problems


def _validate_traits(persona: Persona) -> list[str]:
    """Check that every trait sits inside the unit interval."""
    problems: list[str] = []
    for name in TRAIT_NAMES:
        value = persona.traits.get(name)
        if not 0.0 <= value <= 1.0:
            problems.append(f"trait {name} out of range: {value}")
    return problems


def _validate_affect(persona: Persona) -> list[str]:
    """Check the affect baseline values."""
    problems: list[str] = []
    if not -1.0 <= persona.affect_baseline.valence <= 1.0:
        problems.append("affect baseline valence out of range")
    for name in ("arousal", "assertion", "tenderness"):
        value = getattr(persona.affect_baseline, name)
        if not 0.0 <= value <= 1.0:
            problems.append(f"affect baseline {name} out of range")
    return problems


def _validate_sensitivity(persona: Persona) -> list[str]:
    """Check the per-event sensitivity map."""
    problems: list[str] = []
    for kind, value in persona.event_sensitivity.items():
        if kind not in {event.value for event in EventKind}:
            problems.append(f"unknown event kind in sensitivity: {kind}")
        if not 0.0 <= value <= MAX_EVENT_SENSITIVITY:
            problems.append(f"event sensitivity for {kind} out of range: {value}")
    return problems


def _validate_voice(persona: Persona) -> list[str]:
    """Check the voice binding."""
    problems: list[str] = []
    if not persona.voice.voice_id:
        problems.append("voice binding has no voice_id")
    if not persona.voice.locale:
        problems.append("voice binding has no locale")
    return problems


def validate_persona(persona: Persona) -> list[str]:
    """Return a list of problems with a persona. Empty means valid."""
    problems: list[str] = []
    problems.extend(_validate_identity(persona))
    problems.extend(_validate_text(persona))
    problems.extend(_validate_lorebook(persona))
    problems.extend(_validate_traits(persona))
    problems.extend(_validate_affect(persona))
    problems.extend(_validate_sensitivity(persona))
    problems.extend(_validate_voice(persona))
    return problems


def validate_persona_card(card: dict[str, object]) -> list[str]:
    """Return a list of problems with a raw card document."""
    problems = validate_card_shape(card)
    if problems:
        return problems
    return validate_persona(Persona.from_card(card))
