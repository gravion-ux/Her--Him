"""Relationship state and its six stages.

The relationship model is deliberately separate from the affect model. Mood
describes how a persona feels right now; the relationship describes what has
built between the persona and the user, and it can regress.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["RelationshipStage", "RelationshipState"]


class RelationshipStage(str, Enum):
    """The six relationship stages, in ascending order of closeness."""

    STRANGER = "STRANGER"
    ACQUAINTED = "ACQUAINTED"
    WARM = "WARM"
    FLIRTING = "FLIRTING"
    INTIMATE = "INTIMATE"
    BONDED = "BONDED"


_STAGE_THRESHOLDS: tuple[tuple[float, RelationshipStage], ...] = (
    (0.20, RelationshipStage.STRANGER),
    (0.40, RelationshipStage.ACQUAINTED),
    (0.58, RelationshipStage.WARM),
    (0.75, RelationshipStage.FLIRTING),
    (0.90, RelationshipStage.INTIMATE),
)
_TOP_STAGE = RelationshipStage.BONDED

#: Tension above this level caps the stage one level below the score-derived
#: stage, so a bond broken by hostility visibly drops.
_TENSION_CAP_THRESHOLD = 0.70


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class RelationshipState:
    """Six axes describing what has built between persona and user."""

    trust: float = 0.05
    intimacy: float = 0.03
    tension: float = 0.0
    jealousy: float = 0.05
    dependency: float = 0.04
    commitment: float = 0.03

    def __post_init__(self) -> None:
        for field_name in (
            "trust",
            "intimacy",
            "tension",
            "jealousy",
            "dependency",
            "commitment",
        ):
            object.__setattr__(self, field_name, _clamp(float(getattr(self, field_name))))

    def score(self) -> float:
        """Return the composite closeness score used to derive the stage."""
        return (
            self.intimacy * 0.35
            + self.trust * 0.30
            + self.commitment * 0.20
            + self.dependency * 0.15
            - self.tension * 0.10
        )

    def stage(self) -> RelationshipStage:
        """Return the current stage, applying the tension regression cap."""
        raw = self.score()
        stage = _TOP_STAGE
        for threshold, candidate in _STAGE_THRESHOLDS:
            if raw < threshold:
                stage = candidate
                break
        if self.tension > _TENSION_CAP_THRESHOLD:
            stage = _demote(stage)
        return stage

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-serialisable mapping including the derived stage."""
        return {
            "trust": round(self.trust, 4),
            "intimacy": round(self.intimacy, 4),
            "tension": round(self.tension, 4),
            "jealousy": round(self.jealousy, 4),
            "dependency": round(self.dependency, 4),
            "commitment": round(self.commitment, 4),
            "stage": self.stage().value,
            "score": round(self.score(), 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> RelationshipState:
        """Build a state from a mapping, ignoring unknown keys."""
        return cls(
            trust=float(data.get("trust", 0.05)),
            intimacy=float(data.get("intimacy", 0.03)),
            tension=float(data.get("tension", 0.0)),
            jealousy=float(data.get("jealousy", 0.05)),
            dependency=float(data.get("dependency", 0.04)),
            commitment=float(data.get("commitment", 0.03)),
        )


_STAGE_ORDER = (
    RelationshipStage.STRANGER,
    RelationshipStage.ACQUAINTED,
    RelationshipStage.WARM,
    RelationshipStage.FLIRTING,
    RelationshipStage.INTIMATE,
    RelationshipStage.BONDED,
)


def _demote(stage: RelationshipStage) -> RelationshipStage:
    """Return the stage one level below ``stage``."""
    index = _STAGE_ORDER.index(stage)
    return _STAGE_ORDER[max(0, index - 1)]
