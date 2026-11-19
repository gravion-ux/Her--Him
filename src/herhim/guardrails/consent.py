"""Consent and respect invariants.

The engine may escalate intimacy when the relationship and the conversation
allow it. Two things stop that: a boundary the user set, and a relationship
that has not reached the stage where escalation is appropriate.

These are structural checks on state, not content filters.
"""

from __future__ import annotations

from herhim.errors import GuardrailError
from herhim.models.events import EventKind
from herhim.models.relationship import RelationshipStage, RelationshipState

__all__ = [
    "ESCALATION_ALLOWED_STAGES",
    "assert_consent",
    "escalation_permitted",
]

#: The stages at which the persona may escalate intimacy.
ESCALATION_ALLOWED_STAGES: frozenset[RelationshipStage] = frozenset(
    {
        RelationshipStage.FLIRTING,
        RelationshipStage.INTIMATE,
        RelationshipStage.BONDED,
    }
)


def escalation_permitted(
    relationship: RelationshipState,
    *,
    boundary_held: bool = False,
) -> bool:
    """Whether the persona may escalate intimacy this turn."""
    if boundary_held:
        return False
    return relationship.stage() in ESCALATION_ALLOWED_STAGES


def assert_consent(
    relationship: RelationshipState,
    events: tuple[EventKind, ...],
    *,
    escalating: bool = False,
) -> None:
    """Check that a turn respects the consent invariants.

    Args:
        relationship: The relationship state after the turn.
        events: The event kinds that fired.
        escalating: Whether the reply escalated intimacy.

    Raises:
        GuardrailError: When a boundary was asserted and the reply escalated
            anyway, or when escalation happened below the allowed stage.
    """
    boundary_held = EventKind.BOUNDARY in events
    if boundary_held and escalating:
        raise GuardrailError("a boundary was asserted but the reply escalated intimacy")
    if escalating and not escalation_permitted(relationship, boundary_held=boundary_held):
        raise GuardrailError(
            f"intimacy escalation is not permitted at stage {relationship.stage().value}"
        )
