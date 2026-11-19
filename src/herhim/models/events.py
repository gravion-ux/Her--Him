"""Event kinds and the deterministic classifier's output types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["DetectedEvent", "EventKind"]


class EventKind(str, Enum):
    """The twelve conversational events the classifier recognises."""

    REJECTION = "REJECTION"
    COMPLIANCE = "COMPLIANCE"
    PROVOCATION = "PROVOCATION"
    AFFECTION = "AFFECTION"
    HOSTILITY = "HOSTILITY"
    NEGLECT = "NEGLECT"
    PRAISE = "PRAISE"
    JEALY_TRIGGER = "JEALY_TRIGGER"
    TEASE = "TEASE"
    CONFESSION = "CONFESSION"
    APOLOGY = "APOLOGY"
    BOUNDARY = "BOUNDARY"


@dataclass(frozen=True)
class DetectedEvent:
    """One event the classifier matched in a user turn.

    Attributes:
        kind: The event that fired.
        intensity: Strength of the match, in ``0.0 .. 1.0``.
        matched: The literal fragments that triggered the match.
        scale: Multiplier applied to this event's base deltas. The primary
            event uses ``1.0``; secondary events use ``0.5``.
    """

    kind: EventKind
    intensity: float
    matched: tuple[str, ...] = ()
    scale: float = 1.0

    @property
    def weight(self) -> float:
        """The effective delta multiplier for this event."""
        return self.intensity * self.scale

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "kind": self.kind.value,
            "intensity": round(self.intensity, 4),
            "matched": list(self.matched),
            "scale": self.scale,
        }
