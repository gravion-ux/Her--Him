"""Catalogue-level quality rules.

These rules reuse the diagnostic discipline of ``gravion-ux/preference-data-doctor``
(MIT): a named rule, a severity, and an explanation of what is wrong.

Three families of rule run here:

* **Balance** - no catalogue may be dominated by a single trait archetype.
* **Contradiction** - no two personas may be effectively the same person.
* **Coverage** - every event kind must have at least one persona that is
  especially sensitive to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from herhim.models.events import EventKind
from herhim.models.persona import Persona
from herhim.models.traits import TRAIT_NAMES

__all__ = ["QualityFinding", "QualityReport", "analyse_catalogue"]

#: A trait may not be the dominant axis of more than this share of a catalogue.
BALANCE_LIMIT = 0.40

#: A persona must differ from every other by at least this much on the
#: combined trait profile.
CONTRAST_MIN_DISTANCE = 0.45

#: An event kind needs at least one persona at or above this sensitivity.
COVERAGE_MIN_SENSITIVITY = 1.30


@dataclass(frozen=True)
class QualityFinding:
    """One rule violation.

    Attributes:
        rule: The rule identifier, such as ``CAT-BAL-001``.
        severity: ``critical``, ``high``, ``medium``, ``low``, or ``info``.
        subject: The persona or catalogue the finding is about.
        message: What is wrong.
    """

    rule: str
    severity: str
    subject: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable mapping."""
        return {
            "rule": self.rule,
            "severity": self.severity,
            "subject": self.subject,
            "message": self.message,
        }


@dataclass(frozen=True)
class QualityReport:
    """The findings for one catalogue."""

    catalogue: str
    findings: tuple[QualityFinding, ...]

    @property
    def ok(self) -> bool:
        """Whether the catalogue passed every rule."""
        return not any(finding.severity in ("critical", "high") for finding in self.findings)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "catalogue": self.catalogue,
            "ok": self.ok,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _trait_distance(left: Persona, right: Persona) -> float:
    """Return the mean absolute difference between two trait profiles."""
    total = 0.0
    for name in TRAIT_NAMES:
        total += abs(left.traits.get(name) - right.traits.get(name))
    return total / len(TRAIT_NAMES)


def analyse_catalogue(catalogue: str, personas: list[Persona]) -> QualityReport:
    """Run every catalogue-level rule and return the findings."""
    findings: list[QualityFinding] = []

    if not personas:
        findings.append(
            QualityFinding("CAT-VAL-001", "critical", catalogue, "catalogue has no personas")
        )
        return QualityReport(catalogue=catalogue, findings=tuple(findings))

    dominant_counts: dict[str, int] = {}
    for persona in personas:
        name, _ = persona.traits.dominant()
        dominant_counts[name] = dominant_counts.get(name, 0) + 1

    limit = int(BALANCE_LIMIT * len(personas))
    for name in sorted(dominant_counts):
        count = dominant_counts[name]
        if count > limit:
            findings.append(
                QualityFinding(
                    "CAT-BAL-001",
                    "high",
                    catalogue,
                    f"{count} of {len(personas)} personas are dominated by {name!r} "
                    f"(limit {limit})",
                )
            )

    for index, persona in enumerate(personas):
        for other in personas[index + 1 :]:
            if _trait_distance(persona, other) < CONTRAST_MIN_DISTANCE:
                findings.append(
                    QualityFinding(
                        "CAT-CON-001",
                        "high",
                        persona.persona_id,
                        f"too similar to {other.persona_id} "
                        f"(distance {_trait_distance(persona, other):.3f} "
                        f"< {CONTRAST_MIN_DISTANCE})",
                    )
                )

    for event in EventKind:
        best = max(personas, key=lambda p: p.sensitivity_for(event.value))
        if best.sensitivity_for(event.value) < COVERAGE_MIN_SENSITIVITY:
            findings.append(
                QualityFinding(
                    "CAT-COV-001",
                    "medium",
                    catalogue,
                    f"no persona is especially sensitive to {event.value} "
                    f"(best is {best.persona_id} at {best.sensitivity_for(event.value):.2f})",
                )
            )

    return QualityReport(catalogue=catalogue, findings=tuple(findings))
