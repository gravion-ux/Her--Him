"""The deterministic affect engine: classification, mood, decay, directive."""

from __future__ import annotations

from herhim.affect.classifier import EVENT_PRECEDENCE, EventClassifier, base_deltas, classify
from herhim.affect.directive import MoodGuidance, build_directive, mood_guidance
from herhim.affect.engine import (
    DECAY_PER_TURN,
    OFFLINE_TAU_MINUTES,
    AffectEngine,
    TurnRationale,
    persona_engine,
)
from herhim.affect.moods import (
    MOOD_ALPHA,
    MOOD_LABELS,
    MoodLabel,
    MoodResult,
    derive_mood,
    event_mood_bias,
    raw_scores,
)

__all__ = [
    "DECAY_PER_TURN",
    "EVENT_PRECEDENCE",
    "MOOD_ALPHA",
    "MOOD_LABELS",
    "OFFLINE_TAU_MINUTES",
    "AffectEngine",
    "EventClassifier",
    "MoodGuidance",
    "MoodLabel",
    "MoodResult",
    "TurnRationale",
    "base_deltas",
    "build_directive",
    "classify",
    "derive_mood",
    "event_mood_bias",
    "mood_guidance",
    "persona_engine",
    "raw_scores",
]
