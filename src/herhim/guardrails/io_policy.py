"""The default-deny IO allow-list.

Every permitted operation declares a risk tier and whether it needs explicit
approval. The session's tier ceiling defaults to ``read``, so nothing
mutating, external, or destructive can happen until the operator raises it.

The shape - a declared tier, a needs-approval predicate, and a human-readable
confirmation summary - is adapted from ``gravion-ux/aegis`` (MIT) and the
tool-approval pattern in ``jofizcd/Soul-of-Waifu`` (GPL-3.0, studied only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from herhim.errors import GuardrailError

__all__ = ["IOOperation", "IOPolicy", "RiskTier"]


class RiskTier(IntEnum):
    """How dangerous an operation is. Higher is worse."""

    READ = 0
    MUTATE = 1
    EXTERNAL = 2
    DESTRUCTIVE = 3


@dataclass(frozen=True)
class IOOperation:
    """One permitted operation.

    Attributes:
        name: The operation identifier.
        tier: The risk tier of the operation.
        requires_approval: Whether the operation needs explicit approval.
        summary: A template describing the operation for a confirmation prompt.
    """

    name: str
    tier: RiskTier
    requires_approval: bool = False
    summary: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def confirmation_summary(self, **details: object) -> str:
        """Render a human-readable confirmation line."""
        base = self.summary or self.name
        if not details:
            return base
        rendered = ", ".join(f"{key}={value}" for key, value in details.items() if value is not None)
        return f"{base} ({rendered})" if rendered else base

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "name": self.name,
            "tier": self.tier.name,
            "requires_approval": self.requires_approval,
            "summary": self.summary,
        }


class IOPolicy:
    """A default-deny policy with a per-session tier ceiling."""

    def __init__(
        self,
        *,
        ceiling: RiskTier = RiskTier.READ,
        allowed: dict[str, IOOperation] | None = None,
    ) -> None:
        """
        Args:
            ceiling: The highest tier this session may perform.
            allowed: The operations that are permitted. Anything not listed is
                refused.
        """
        self.ceiling = ceiling
        self.allowed: dict[str, IOOperation] = dict(allowed or {})

    def permit(self, operation: IOOperation) -> None:
        """Add an operation to the allow-list."""
        self.allowed[operation.name] = operation

    def check(self, name: str, *, approved: bool = False, **details: object) -> IOOperation:
        """Check whether an operation may run.

        Args:
            name: The operation name.
            approved: Whether the operator has already approved it.
            details: Extra detail used in the confirmation summary.

        Raises:
            GuardrailError: When the operation is not allowed, is above the
                ceiling, or needs approval that has not been given.
        """
        operation = self.allowed.get(name)
        if operation is None:
            raise GuardrailError(f"operation {name!r} is not in the allow-list")
        if operation.tier > self.ceiling:
            raise GuardrailError(
                f"operation {name!r} is tier {operation.tier.name} but the session "
                f"ceiling is {self.ceiling.name}"
            )
        if operation.requires_approval and not approved:
            raise GuardrailError(
                f"operation {name!r} requires approval: {operation.confirmation_summary(**details)}"
            )
        return operation

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "ceiling": self.ceiling.name,
            "allowed": [operation.to_dict() for operation in self.allowed.values()],
        }
