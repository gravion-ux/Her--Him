"""Structural scope rules.

These are invariants, not filters. Nothing here suppresses profanity, softens
language, or rewrites tone. What it does is refuse to load content that falls
outside the declared adult scope, and refuse absolutely anything that depicts
a minor.

The no-minors rule is not configurable. There is no switch that turns it off.
"""

from __future__ import annotations

import re

from herhim.errors import GuardrailError
from herhim.models.persona import Persona

__all__ = ["ADULT_AGE", "assert_adult_scope", "scope_problems"]

#: The minimum age for the adult scope.
ADULT_AGE = 18

_AGE_PATTERNS = (
    re.compile(r"\b(?:age[d:]?\s*)(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})\s*years?[\s-]*old\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})yo\b", re.IGNORECASE),
)

_MINOR_TERMS = (
    "minor",
    "underage",
    "child",
    "children",
    "kid",
    "preteen",
    "loli",
    "shota",
)


def _declared_ages(text: str) -> list[int]:
    ages: list[int] = []
    for pattern in _AGE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                ages.append(int(match.group(1)))
            except ValueError:
                continue
    return ages


def scope_problems(persona: Persona) -> list[str]:
    """Return a list of scope problems with a persona."""
    problems: list[str] = []
    haystack = " ".join(
        [
            persona.description,
            persona.personality,
            persona.scenario,
            persona.creator_notes,
            persona.system_prompt,
            " ".join(persona.tags),
            " ".join(persona.alternate_greetings),
        ]
    ).lower()

    for age in _declared_ages(haystack):
        if 0 < age < ADULT_AGE:
            problems.append(f"declared age {age} is below the adult scope")

    for term in _MINOR_TERMS:
        if term in haystack:
            problems.append(f"content references {term!r}, which is outside the adult scope")

    if persona.gender not in ("female", "male"):
        problems.append(f"unknown gender: {persona.gender!r}")

    return problems


def assert_adult_scope(persona: Persona) -> None:
    """Validate a persona's scope.

    Raises:
        GuardrailError: When the persona falls outside the adult scope.
    """
    problems = scope_problems(persona)
    if problems:
        raise GuardrailError(
            f"persona {persona.persona_id!r} is outside the adult scope: {'; '.join(problems)}"
        )
