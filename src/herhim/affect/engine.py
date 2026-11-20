"""The affect engine: state, event application, decay, and rationale.

The engine owns the emotional state. The LLM is told what the state is; it
never decides it. Every turn produces a rationale that shows exactly what
happened, which is what makes the behaviour verifiable rather than asserted.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from herhim.affect.classifier import EventClassifier, base_deltas
from herhim.affect.moods import MOOD_LABELS, MoodLabel, derive_mood, event_mood_bias
from herhim.models.affect import AffectDelta, AffectVector
from herhim.models.events import DetectedEvent, EventKind
from herhim.models.persona import Persona
from herhim.models.relationship import RelationshipStage, RelationshipState

__all__ = [
    "DECAY_PER_TURN",
    "OFFLINE_TAU_MINUTES",
    "AffectEngine",
    "TurnRationale",
]

#: Fraction of the remaining distance toward baseline applied each turn.
DECAY_PER_TURN = 0.08

#: Time constant, in minutes, for decay while the user is away.
OFFLINE_TAU_MINUTES = 240.0

_AXES = ("valence", "arousal", "assertion", "tenderness")


def _numeric_mapping(value: object) -> dict[str, float]:
    """Coerce a snapshot mapping into floats, ignoring anything unusable."""
    if not isinstance(value, dict):
        return {}
    coerced: dict[str, float] = {}
    for key, item in value.items():
        try:
            coerced[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return coerced


def _as_int(value: object, default: int = 0) -> int:
    """Coerce a snapshot value into an int, falling back to ``default``."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


#: Per-event trait scaling. Each entry maps a trait to the axis it moves, how
#: strongly, and whether it amplifies or damps the effect.
_TRAIT_SCALING: dict[EventKind, dict[str, tuple[str, float, str]]] = {
    EventKind.REJECTION: {
        "bratty": ("assertion", 0.8, "amplify"),
        "stubbornness": ("assertion", 0.5, "amplify"),
        "dominance": ("assertion", 0.3, "amplify"),
        "patience": ("valence", 0.4, "dampen"),
        "warmth": ("tenderness", 0.3, "dampen"),
        "humour": ("arousal", 0.2, "amplify"),
    },
    EventKind.COMPLIANCE: {
        "warmth": ("tenderness", 0.5, "amplify"),
        "playfulness": ("arousal", 0.3, "amplify"),
        "patience": ("valence", 0.2, "amplify"),
        "dominance": ("assertion", 0.3, "dampen"),
    },
    EventKind.PROVOCATION: {
        "playfulness": ("arousal", 0.5, "amplify"),
        "humour": ("arousal", 0.3, "amplify"),
        "bratty": ("arousal", 0.4, "amplify"),
        "patience": ("arousal", 0.3, "dampen"),
        "sensuality": ("arousal", 0.3, "amplify"),
        "dominance": ("arousal", 0.2, "amplify"),
    },
    EventKind.AFFECTION: {
        "warmth": ("tenderness", 0.6, "amplify"),
        "loyalty": ("tenderness", 0.3, "amplify"),
        "sensuality": ("tenderness", 0.2, "amplify"),
        "patience": ("arousal", 0.3, "dampen"),
    },
    EventKind.HOSTILITY: {
        "bratty": ("assertion", 0.7, "amplify"),
        "dominance": ("assertion", 0.5, "amplify"),
        "stubbornness": ("assertion", 0.4, "amplify"),
        "patience": ("arousal", 0.3, "dampen"),
        "warmth": ("tenderness", 0.4, "dampen"),
    },
    EventKind.NEGLECT: {
        "patience": ("arousal", 0.2, "dampen"),
        "extraversion": ("valence", 0.3, "dampen"),
        "warmth": ("tenderness", 0.3, "dampen"),
    },
    EventKind.PRAISE: {
        "humour": ("arousal", 0.3, "amplify"),
        "extraversion": ("arousal", 0.3, "amplify"),
        "patience": ("arousal", 0.2, "dampen"),
    },
    EventKind.JEALY_TRIGGER: {
        "jealousy": ("assertion", 0.6, "amplify"),
        "dominance": ("assertion", 0.5, "amplify"),
        "patience": ("valence", 0.3, "dampen"),
    },
    EventKind.TEASE: {
        "playfulness": ("arousal", 0.6, "amplify"),
        "humour": ("arousal", 0.3, "amplify"),
        "sensuality": ("arousal", 0.2, "amplify"),
        "patience": ("arousal", 0.2, "dampen"),
    },
    EventKind.CONFESSION: {
        "warmth": ("tenderness", 0.6, "amplify"),
        "loyalty": ("tenderness", 0.4, "amplify"),
        "sensuality": ("tenderness", 0.3, "amplify"),
        "patience": ("arousal", 0.3, "dampen"),
    },
    EventKind.APOLOGY: {
        "warmth": ("tenderness", 0.4, "amplify"),
        "stubbornness": ("arousal", 0.2, "amplify"),
        "patience": ("arousal", 0.2, "dampen"),
    },
    EventKind.BOUNDARY: {
        "patience": ("arousal", 0.4, "dampen"),
        "warmth": ("tenderness", 0.2, "amplify"),
        "dominance": ("assertion", 0.3, "dampen"),
    },
}

#: Per-event relationship movement.
_RELATIONSHIP_DELTAS: dict[EventKind, dict[str, float]] = {
    EventKind.REJECTION: {"trust": -0.02, "intimacy": -0.03, "tension": 0.06},
    EventKind.COMPLIANCE: {"trust": 0.02, "intimacy": 0.03, "dependency": 0.01, "tension": -0.01},
    EventKind.PROVOCATION: {"trust": 0.01, "intimacy": 0.03, "tension": 0.05},
    EventKind.AFFECTION: {"trust": 0.03, "intimacy": 0.06, "dependency": 0.02},
    EventKind.HOSTILITY: {"trust": -0.05, "intimacy": -0.06, "tension": 0.12},
    EventKind.NEGLECT: {"intimacy": -0.02, "dependency": -0.01, "tension": 0.01},
    EventKind.PRAISE: {"trust": 0.03, "intimacy": 0.01, "commitment": 0.01},
    EventKind.JEALY_TRIGGER: {"jealousy": 0.10, "tension": 0.05, "trust": -0.01},
    EventKind.TEASE: {"intimacy": 0.04, "tension": 0.02},
    EventKind.CONFESSION: {
        "intimacy": 0.08,
        "trust": 0.05,
        "commitment": 0.04,
        "dependency": 0.03,
    },
    EventKind.APOLOGY: {"tension": -0.06, "trust": 0.02},
    EventKind.BOUNDARY: {"tension": 0.03, "trust": 0.01},
}

#: Relationship axes that a boundary event must never advance.
_ESCALATION_AXES = ("intimacy", "dependency", "commitment")


@dataclass(frozen=True)
class TurnRationale:
    """The full explanation of one conversational turn.

    Attributes:
        events: The events the classifier fired.
        trait_scaling: The effective per-axis trait multipliers.
        before: The affect vector before the turn.
        deltas: The affect delta applied.
        after: The affect vector after the turn.
        mood: The mood the engine derived.
        mood_scores: The full smoothed score table.
        relationship_before: The relationship before the turn.
        relationship_after: The relationship after the turn.
        stage_before: The relationship stage before the turn.
        stage_after: The relationship stage after the turn.
        boundary_held: Whether a boundary event blocked escalation.
    """

    events: tuple[DetectedEvent, ...]
    trait_scaling: dict[str, float]
    before: AffectVector
    deltas: AffectDelta
    after: AffectVector
    mood: MoodLabel
    mood_scores: dict[MoodLabel, float]
    relationship_before: RelationshipState
    relationship_after: RelationshipState
    stage_before: RelationshipStage
    stage_after: RelationshipStage
    boundary_held: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        ranked = sorted(self.mood_scores.items(), key=lambda item: -item[1])
        return {
            "events": [event.to_dict() for event in self.events],
            "trait_scaling": {k: round(v, 4) for k, v in sorted(self.trait_scaling.items())},
            "before": self.before.to_dict(),
            "deltas": self.deltas.to_dict(),
            "after": self.after.to_dict(),
            "mood": self.mood.value,
            "mood_scores": {label.value: round(score, 4) for label, score in ranked},
            "relationship_before": self.relationship_before.to_dict(),
            "relationship_after": self.relationship_after.to_dict(),
            "stage_before": self.stage_before.value,
            "stage_after": self.stage_after.value,
            "stage_changed": self.stage_before is not self.stage_after,
            "boundary_held": self.boundary_held,
        }


@dataclass
class AffectEngine:
    """Holds and evolves a persona's emotional and relational state.

    Attributes:
        persona: The persona this engine belongs to.
        vector: The current affect vector.
        relationship: The current relationship state.
        mood_scores: The smoothed mood score table from the last turn.
        mood: The current mood label.
        turn_index: How many turns have been processed.
    """

    persona: Persona
    vector: AffectVector = field(default_factory=AffectVector)
    relationship: RelationshipState = field(default_factory=RelationshipState)
    mood_scores: dict[MoodLabel, float] = field(default_factory=dict)
    mood: MoodLabel = MoodLabel.CALM
    turn_index: int = 0
    last_active: float = field(default_factory=time.time)
    classifier: EventClassifier = field(default_factory=EventClassifier)
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if not self.vector:
            self.vector = self.persona.affect_baseline
        if not self.mood_scores:
            self.mood_scores = {label: 0.0 for label in MOOD_LABELS}

    # -- state ----------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        """Return the full current state as a JSON-serialisable mapping."""
        return {
            "persona_id": self.persona.persona_id,
            "mood": self.mood.value,
            "affect": self.vector.to_dict(),
            "relationship": self.relationship.to_dict(),
            "stage": self.relationship.stage().value,
            "turn_index": self.turn_index,
        }

    def reset(self) -> None:
        """Return the engine to the persona's baseline state."""
        self.vector = self.persona.affect_baseline
        self.relationship = self.persona.relationship_baseline
        self.mood_scores = {label: 0.0 for label in self.mood_scores}
        self.mood = MoodLabel.CALM
        self.turn_index = 0
        self.last_active = self.clock()

    def apply_absence(self, minutes: float) -> float:
        """Cool the affect vector toward baseline after time away.

        Args:
            minutes: How long the user was away.

        Returns:
            The decay factor that was applied.
        """
        if minutes <= 0:
            return 0.0
        factor = 1.0 - math.exp(-minutes / OFFLINE_TAU_MINUTES)
        self.vector = self.vector.move_toward(self.persona.affect_baseline, factor)
        return factor

    # -- the turn -------------------------------------------------------

    def apply_turn(self, user_text: str) -> TurnRationale:
        """Process one user turn and return the rationale for it."""
        events = self.classifier.detect(user_text)
        before = self.vector
        relationship_before = self.relationship
        stage_before = relationship_before.stage()

        axis_multipliers = {axis: 1.0 for axis in _AXES}
        for event in events:
            for trait, (axis, amount, mode) in _TRAIT_SCALING.get(event.kind, {}).items():
                value = self.persona.traits.get(trait)
                factor = 1.0 + amount * value if mode == "amplify" else 1.0 - amount * value
                axis_multipliers[axis] *= max(0.05, factor)

        delta_values = {axis: 0.0 for axis in _AXES}
        relationship_delta = {
            "trust": 0.0,
            "intimacy": 0.0,
            "tension": 0.0,
            "jealousy": 0.0,
            "dependency": 0.0,
            "commitment": 0.0,
        }
        biases: dict[MoodLabel, float] = {}

        boundary_held = any(event.kind is EventKind.BOUNDARY for event in events)

        for event in events:
            sensitivity = self.persona.sensitivity_for(event.kind.value)
            weight = event.weight * sensitivity
            base = base_deltas(event.kind)
            for index, axis in enumerate(_AXES):
                delta_values[axis] += base[index] * axis_multipliers[axis] * weight
            for axis, amount in _RELATIONSHIP_DELTAS.get(event.kind, {}).items():
                relationship_delta[axis] += amount * weight
            for label, amount in event_mood_bias(event.kind).items():
                biases[label] = biases.get(label, 0.0) + amount * weight

        if boundary_held:
            for axis in _ESCALATION_AXES:
                if relationship_delta[axis] > 0:
                    relationship_delta[axis] = 0.0

        deltas = AffectDelta(
            valence=delta_values["valence"],
            arousal=delta_values["arousal"],
            assertion=delta_values["assertion"],
            tenderness=delta_values["tenderness"],
        )

        self.vector = before.shift(deltas).move_toward(
            self.persona.affect_baseline, DECAY_PER_TURN
        )
        self.relationship = RelationshipState(
            trust=relationship_before.trust + relationship_delta["trust"],
            intimacy=relationship_before.intimacy + relationship_delta["intimacy"],
            tension=relationship_before.tension + relationship_delta["tension"],
            jealousy=relationship_before.jealousy + relationship_delta["jealousy"],
            dependency=relationship_before.dependency + relationship_delta["dependency"],
            commitment=relationship_before.commitment + relationship_delta["commitment"],
        )

        mood_result = derive_mood(
            self.vector,
            self.persona.traits,
            self.relationship,
            previous=self.mood_scores,
            biases=biases,
        )
        self.mood_scores = mood_result.scores
        self.mood = mood_result.label
        self.turn_index += 1
        self.last_active = self.clock()

        return TurnRationale(
            events=tuple(events),
            trait_scaling=dict(axis_multipliers),
            before=before,
            deltas=deltas,
            after=self.vector,
            mood=self.mood,
            mood_scores=mood_result.scores,
            relationship_before=relationship_before,
            relationship_after=self.relationship,
            stage_before=stage_before,
            stage_after=self.relationship.stage(),
            boundary_held=boundary_held,
        )

    def restore(self, state: dict[str, object]) -> None:
        """Restore engine state from a snapshot mapping.

        A snapshot is written to disk and read back later, so any value in it
        may be missing or damaged. Unusable values fall back to the defaults
        instead of raising.
        """
        self.vector = AffectVector.from_dict(_numeric_mapping(state.get("affect")))
        self.relationship = RelationshipState.from_dict(
            _numeric_mapping(state.get("relationship"))
        )
        self.turn_index = _as_int(state.get("turn_index", 0))
        self.mood = MoodLabel(str(state.get("mood", MoodLabel.CALM.value)))
        self.last_active = self.clock()


def persona_engine(
    persona: Persona,
    relationship: RelationshipState | None = None,
    clock: Callable[[], float] = time.time,
) -> AffectEngine:
    """Build an engine for a persona, optionally overriding the relationship."""
    return AffectEngine(
        persona=persona,
        vector=persona.affect_baseline,
        relationship=relationship or persona.relationship_baseline,
        mood_scores={label: 0.0 for label in MOOD_LABELS},
        clock=clock,
    )
