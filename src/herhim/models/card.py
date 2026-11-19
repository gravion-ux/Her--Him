"""Character Card v3 constants and shape helpers.

The canonical interop format is Character Card v3 (MIT licensed). Character
Card v2 is supported as a read-only import path because the v2 repository
carries no license of its own.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CARD_SPEC_VERSION",
    "HERHIM_EXTENSION_KEY",
    "REQUIRED_CARD_FIELDS",
    "card_persona_id",
    "strip_herhim_extension",
    "validate_card_shape",
]

#: The Character Card specification version this package writes.
CARD_SPEC_VERSION = "3.0"

#: The key under ``extensions`` that holds the herhim-specific persona data.
HERHIM_EXTENSION_KEY = "herhim"

#: Fields every card must carry for it to be considered well-formed.
REQUIRED_CARD_FIELDS = (
    "spec_version",
    "name",
    "description",
    "personality",
    "scenario",
    "first_mes",
    "mes_example",
    "creator",
    "character_version",
    "extensions",
)


def validate_card_shape(card: Any) -> list[str]:
    """Return a list of structural problems with a card.

    An empty list means the card has every required field with a value.
    """
    problems: list[str] = []
    if not isinstance(card, dict):
        return ["card is not a mapping"]
    for field in REQUIRED_CARD_FIELDS:
        if field not in card:
            problems.append(f"missing field: {field}")
        elif card[field] in (None, "", [], {}):
            problems.append(f"empty field: {field}")
    extensions = card.get("extensions")
    if isinstance(extensions, dict) and HERHIM_EXTENSION_KEY not in extensions:
        problems.append(f"extensions.{HERHIM_EXTENSION_KEY} block is absent")
    return problems


def card_persona_id(card: dict[str, Any]) -> str:
    """Return the herhim persona id carried by a card, or an empty string."""
    extensions = card.get("extensions")
    if not isinstance(extensions, dict):
        return ""
    block = extensions.get(HERHIM_EXTENSION_KEY)
    if not isinstance(block, dict):
        return ""
    return str(block.get("persona_id", ""))


def strip_herhim_extension(card: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the card without the herhim extension block.

    Useful when exporting to a consumer that does not understand the
    extension, while preserving every canonical Character Card field.
    """
    stripped = dict(card)
    extensions = stripped.get("extensions")
    if isinstance(extensions, dict):
        remaining = {k: v for k, v in extensions.items() if k != HERHIM_EXTENSION_KEY}
        stripped["extensions"] = remaining
    return stripped
