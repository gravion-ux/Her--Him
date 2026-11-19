"""The four-dimensional affect vector.

Four dimensions were chosen because they map cleanly onto what voice synthesis
can express and onto what a persona's behavior can express. More dimensions
would be less legible; fewer would lose the distinction between *sultry* and
*angry*, which are both high-arousal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["AffectDelta", "AffectVector"]

_VALENCE_RANGE = (-1.0, 1.0)
_UNIT_RANGE = (0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class AffectVector:
    """A point in affect space.

    Attributes:
        valence: pleasant (-1.0) through unpleasant (+1.0 is not used; negative
            means unpleasant).
        arousal: calm (0.0) through activated (1.0).
        assertion: yielding (0.0) through assertive (1.0).
        tenderness: guarded (0.0) through warm (1.0).
    """

    valence: float = 0.0
    arousal: float = 0.5
    assertion: float = 0.5
    tenderness: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", _clamp(self.valence, *_VALENCE_RANGE))
        for name in ("arousal", "assertion", "tenderness"):
            object.__setattr__(self, name, _clamp(getattr(self, name), *_UNIT_RANGE))

    def __add__(self, other: AffectVector) -> AffectVector:
        return replace(
            self,
            valence=self.valence + other.valence,
            arousal=self.arousal + other.arousal,
            assertion=self.assertion + other.assertion,
            tenderness=self.tenderness + other.tenderness,
        )

    def __sub__(self, other: AffectVector) -> AffectVector:
        return replace(
            self,
            valence=self.valence - other.valence,
            arousal=self.arousal - other.arousal,
            assertion=self.assertion - other.assertion,
            tenderness=self.tenderness - other.tenderness,
        )

    def __mul__(self, factor: float) -> AffectVector:
        return replace(
            self,
            valence=self.valence * factor,
            arousal=self.arousal * factor,
            assertion=self.assertion * factor,
            tenderness=self.tenderness * factor,
        )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable mapping of the four axes."""
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "assertion": round(self.assertion, 4),
            "tenderness": round(self.tenderness, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> AffectVector:
        """Build a vector from a mapping, ignoring unknown keys."""
        return cls(
            valence=float(data.get("valence", 0.0)),
            arousal=float(data.get("arousal", 0.5)),
            assertion=float(data.get("assertion", 0.5)),
            tenderness=float(data.get("tenderness", 0.5)),
        )

    def move_toward(self, target: AffectVector, factor: float) -> AffectVector:
        """Move a fraction of the remaining distance toward ``target``."""
        ratio = _clamp(factor, 0.0, 1.0)
        return replace(
            self,
            valence=self.valence + (target.valence - self.valence) * ratio,
            arousal=self.arousal + (target.arousal - self.arousal) * ratio,
            assertion=self.assertion + (target.assertion - self.assertion) * ratio,
            tenderness=self.tenderness + (target.tenderness - self.tenderness) * ratio,
        )


    def shift(self, delta: AffectDelta) -> AffectVector:
        """Return this vector moved by a raw, unclamped delta."""
        return AffectVector(
            valence=self.valence + delta.valence,
            arousal=self.arousal + delta.arousal,
            assertion=self.assertion + delta.assertion,
            tenderness=self.tenderness + delta.tenderness,
        )


@dataclass(frozen=True)
class AffectDelta:
    """A signed change to an affect vector.

    Unlike :class:`AffectVector`, a delta is not clamped. Deltas on the unit
    axes are routinely negative, and clamping them at construction time would
    silently erase half of every emotional reaction.
    """

    valence: float = 0.0
    arousal: float = 0.0
    assertion: float = 0.0
    tenderness: float = 0.0

    def __add__(self, other: AffectDelta) -> AffectDelta:
        return AffectDelta(
            valence=self.valence + other.valence,
            arousal=self.arousal + other.arousal,
            assertion=self.assertion + other.assertion,
            tenderness=self.tenderness + other.tenderness,
        )

    def to_vector(self) -> AffectVector:
        """Return the delta as a clamped :class:`AffectVector`."""
        return AffectVector(
            valence=self.valence,
            arousal=self.arousal,
            assertion=self.assertion,
            tenderness=self.tenderness,
        )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable mapping of the four axes."""
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "assertion": round(self.assertion, 4),
            "tenderness": round(self.tenderness, 4),
        }
