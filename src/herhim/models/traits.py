"""The eleven-axis trait profile that makes two personas react differently."""

from __future__ import annotations

from dataclasses import dataclass, fields

__all__ = ["TRAIT_NAMES", "TraitProfile"]

TRAIT_NAMES = (
    "extraversion",
    "playfulness",
    "bratty",
    "stubbornness",
    "warmth",
    "jealousy",
    "dominance",
    "loyalty",
    "patience",
    "sensuality",
    "humour",
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class TraitProfile:
    """A persona's standing temperament.

    Every axis is in ``0.0 .. 1.0``. Traits are what make a rejection land
    differently on a bratty persona than on a patient one.
    """

    extraversion: float = 0.5
    playfulness: float = 0.5
    bratty: float = 0.5
    stubbornness: float = 0.5
    warmth: float = 0.5
    jealousy: float = 0.5
    dominance: float = 0.5
    loyalty: float = 0.5
    patience: float = 0.5
    sensuality: float = 0.5
    humour: float = 0.5

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, _clamp(float(getattr(self, field.name))))

    def get(self, name: str) -> float:
        """Return one axis by name, or ``0.0`` for an unknown name."""
        if name not in TRAIT_NAMES:
            return 0.0
        return float(getattr(self, name))

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable mapping of every axis."""
        return {name: round(self.get(name), 4) for name in TRAIT_NAMES}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> TraitProfile:
        """Build a profile from a mapping, ignoring unknown keys."""
        values = {name: float(data.get(name, 0.5)) for name in TRAIT_NAMES}
        return cls(**values)

    def dominant(self) -> tuple[str, float]:
        """Return the highest axis and its value."""
        best_name = max(TRAIT_NAMES, key=self.get)
        return best_name, self.get(best_name)
