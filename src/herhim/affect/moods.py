"""Mood derivation.

Eighteen mood labels, each with an explicit scoring function over the affect
vector, the trait profile, and the relationship state. Scores are smoothed
across turns with an exponential moving average so a single word cannot make
a persona flicker between moods.

Every score is deterministic. Given the same state, the same mood always
comes out.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from herhim.models.affect import AffectVector
from herhim.models.events import EventKind
from herhim.models.relationship import RelationshipState
from herhim.models.traits import TraitProfile

__all__ = [
    "MOOD_ALPHA",
    "MOOD_LABELS",
    "MoodLabel",
    "MoodResult",
    "derive_mood",
    "event_mood_bias",
    "raw_scores",
]


class MoodLabel(StrEnum):
    """The eighteen attitude labels a persona can hold."""

    FIERCE = "fierce"
    SULKY = "sulky"
    DEFIANT = "defiant"
    PLAYFUL = "playful"
    FLIRTY = "flirty"
    SULTRY = "sultry"
    AFFECTIONATE = "affectionate"
    JOYFUL = "joyful"
    BORED = "bored"
    DETACHED = "detached"
    MELANCHOLY = "melancholy"
    POSSESSIVE = "possessive"
    ADMIRING = "admiring"
    APOLOGETIC = "apologetic"
    PROVOCATIVE = "provocative"
    SATED = "sated"
    CALM = "calm"
    SMUG = "smug"


#: Smoothing factor for the exponential moving average across turns.
MOOD_ALPHA = 0.30

#: Tie-break order when two labels score identically. Earlier wins.
MOOD_LABELS: tuple[MoodLabel, ...] = (
    MoodLabel.FIERCE,
    MoodLabel.SULKY,
    MoodLabel.DEFIANT,
    MoodLabel.PLAYFUL,
    MoodLabel.FLIRTY,
    MoodLabel.SULTRY,
    MoodLabel.AFFECTIONATE,
    MoodLabel.JOYFUL,
    MoodLabel.BORED,
    MoodLabel.DETACHED,
    MoodLabel.MELANCHOLY,
    MoodLabel.POSSESSIVE,
    MoodLabel.ADMIRING,
    MoodLabel.APOLOGETIC,
    MoodLabel.PROVOCATIVE,
    MoodLabel.SATED,
    MoodLabel.CALM,
    MoodLabel.SMUG,
)

#: Additive nudges applied to raw scores when an event fires. This is the
#: explicit link between what the user did and how the persona reacts.
_EVENT_BIAS: dict[EventKind, dict[MoodLabel, float]] = {
    EventKind.REJECTION: {
        MoodLabel.DEFIANT: 0.25,
        MoodLabel.SULKY: 0.15,
        MoodLabel.FIERCE: 0.10,
    },
    EventKind.COMPLIANCE: {
        MoodLabel.AFFECTIONATE: 0.15,
        MoodLabel.JOYFUL: 0.15,
        MoodLabel.PLAYFUL: 0.10,
    },
    EventKind.PROVOCATION: {
        MoodLabel.PLAYFUL: 0.30,
        MoodLabel.FLIRTY: 0.15,
        MoodLabel.SULTRY: 0.05,
    },
    EventKind.AFFECTION: {
        MoodLabel.AFFECTIONATE: 0.30,
        MoodLabel.SULTRY: 0.10,
    },
    EventKind.HOSTILITY: {
        MoodLabel.FIERCE: 0.35,
        MoodLabel.DETACHED: 0.10,
        MoodLabel.SULKY: 0.10,
    },
    EventKind.NEGLECT: {
        MoodLabel.BORED: 0.30,
        MoodLabel.DETACHED: 0.20,
        MoodLabel.MELANCHOLY: 0.10,
    },
    EventKind.PRAISE: {
        MoodLabel.JOYFUL: 0.25,
        MoodLabel.ADMIRING: 0.10,
    },
    EventKind.JEALY_TRIGGER: {
        MoodLabel.POSSESSIVE: 0.35,
        MoodLabel.FIERCE: 0.10,
        MoodLabel.SULKY: 0.10,
    },
    EventKind.TEASE: {
        MoodLabel.PLAYFUL: 0.20,
        MoodLabel.FLIRTY: 0.20,
        MoodLabel.PROVOCATIVE: 0.10,
    },
    EventKind.CONFESSION: {
        MoodLabel.AFFECTIONATE: 0.25,
        MoodLabel.SULTRY: 0.15,
        MoodLabel.JOYFUL: 0.10,
    },
    EventKind.APOLOGY: {
        MoodLabel.APOLOGETIC: 0.30,
        MoodLabel.CALM: 0.10,
    },
    EventKind.BOUNDARY: {
        MoodLabel.DETACHED: 0.15,
        MoodLabel.CALM: 0.15,
        MoodLabel.SULKY: 0.05,
    },
}


def event_mood_bias(kind: EventKind) -> dict[MoodLabel, float]:
    """Return the mood nudges an event kind applies."""
    return dict(_EVENT_BIAS.get(kind, {}))


def raw_scores(
    vector: AffectVector,
    traits: TraitProfile,
    relationship: RelationshipState,
) -> dict[MoodLabel, float]:
    """Return the unsmoothed score for every mood label."""
    valence = vector.valence
    arousal = vector.arousal
    assertion = vector.assertion
    tenderness = vector.tenderness
    positive = max(0.0, valence)
    negative = max(0.0, -valence)

    return {
        MoodLabel.FIERCE: arousal * assertion * (1.0 + negative) * (1.0 - tenderness),
        MoodLabel.SULKY: negative * (1.0 - traits.patience) * 1.6 + (1.0 - arousal) * 0.2,
        MoodLabel.DEFIANT: (
            assertion
            * (0.4 + traits.bratty)
            * (1.2 - traits.patience * 0.4)
            * (1.0 - traits.warmth * 0.2)
            * (1.0 - negative * 0.5)
        ),
        MoodLabel.PLAYFUL: (
            arousal * (0.2 + traits.playfulness * 0.8) * (1.0 - traits.dominance * 0.5)
        ),
        MoodLabel.FLIRTY: (
            arousal * (traits.sensuality**1.3) * (1.0 - traits.patience * 0.3) + tenderness * 0.25
        ),
        MoodLabel.SULTRY: traits.sensuality * assertion * arousal * 1.5,
        MoodLabel.AFFECTIONATE: tenderness * (0.4 + traits.warmth),
        MoodLabel.JOYFUL: positive * (0.4 + traits.extraversion) * (0.3 + arousal),
        MoodLabel.BORED: (1.0 - arousal) * (1.0 - traits.extraversion) * 1.2,
        MoodLabel.DETACHED: (1.0 - traits.warmth) * (1.0 - tenderness) * (1.0 - positive) * 1.1,
        MoodLabel.MELANCHOLY: negative * (1.0 - traits.extraversion * 0.6) * 1.3,
        MoodLabel.POSSESSIVE: relationship.jealousy * traits.dominance * (1.0 - relationship.trust),
        MoodLabel.ADMIRING: traits.warmth * positive * (0.4 + traits.extraversion) * 1.1,
        MoodLabel.APOLOGETIC: (1.0 - assertion) * tenderness * 1.2,
        MoodLabel.PROVOCATIVE: assertion * traits.playfulness * arousal * traits.sensuality * 1.4,
        MoodLabel.SATED: tenderness * assertion * (1.0 - arousal * 0.5) * 0.8,
        MoodLabel.CALM: (1.0 - arousal) * (1.0 - abs(valence)) * (0.3 + traits.patience) * 1.2,
        MoodLabel.SMUG: assertion * positive * (1.0 - traits.extraversion * 0.3) * 1.1,
    }


@dataclass(frozen=True)
class MoodResult:
    """The derived mood and the scores behind it."""

    label: MoodLabel
    scores: dict[MoodLabel, float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        ranked = sorted(self.scores.items(), key=lambda item: -item[1])
        return {
            "mood": self.label.value,
            "mood_scores": {label.value: round(score, 4) for label, score in ranked},
        }


def derive_mood(
    vector: AffectVector,
    traits: TraitProfile,
    relationship: RelationshipState,
    previous: dict[MoodLabel, float] | None = None,
    biases: dict[MoodLabel, float] | None = None,
) -> MoodResult:
    """Derive the current mood.

    Args:
        vector: The current affect vector.
        traits: The persona's trait profile.
        relationship: The current relationship state.
        previous: Scores from the previous turn, for EMA smoothing.
        biases: Extra nudges, typically from the events that just fired.

    Returns:
        The winning mood label and the full smoothed score table.
    """
    raw = raw_scores(vector, traits, relationship)
    for label, amount in (biases or {}).items():
        raw[label] = raw.get(label, 0.0) + amount

    prior = previous or {}
    smoothed: dict[MoodLabel, float] = {}
    for label in MOOD_LABELS:
        before = prior.get(label, 0.0)
        smoothed[label] = (1.0 - MOOD_ALPHA) * before + MOOD_ALPHA * raw[label]

    winner = max(
        MOOD_LABELS,
        key=lambda label: (smoothed[label], -MOOD_LABELS.index(label)),
    )
    return MoodResult(label=winner, scores=smoothed)
