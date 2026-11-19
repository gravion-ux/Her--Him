"""Keyword-triggered worldbook scanning.

Implements the Character Card v3 rule that a Character Book stacks with a
World Book and is used first: character-book entries are scanned before any
external world book, and a world-book entry is only consulted for content the
character book did not already supply.
"""

from __future__ import annotations

import re

from herhim.models.persona import LoreEntry

__all__ = ["scan_lorebook", "scan_worldbook"]

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _normalise(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _matches(entry: LoreEntry, words: set[str], text: str) -> bool:
    if entry.constant:
        return True
    for key in entry.keys:
        if not key:
            continue
        if " " in key:
            if key in text:
                return True
        elif key in words:
            return True
    return False


def scan_lorebook(entries: tuple[LoreEntry, ...], text: str) -> list[LoreEntry]:
    """Return the character-book entries triggered by ``text``.

    Constant entries are always included and come first, which keeps the
    prompt stable no matter what the user says.
    """
    lowered = text.lower()
    words = set(_normalise(text))
    constant = [entry for entry in entries if entry.constant]
    triggered = [entry for entry in entries if not entry.constant and _matches(entry, words, lowered)]
    return constant + triggered


def scan_worldbook(
    entries: tuple[LoreEntry, ...],
    text: str,
    already_supplied: set[str] | None = None,
) -> list[LoreEntry]:
    """Return the world-book entries that add something new.

    Args:
        entries: The world-book entries to consider.
        text: The user's turn.
        already_supplied: Keys the character book already covered. Entries
            whose keys all appear in this set are skipped, so the character
            book is always used first.
    """
    supplied = already_supplied or set()
    lowered = text.lower()
    words = set(_normalise(text))
    result: list[LoreEntry] = []
    for entry in entries:
        if entry.constant:
            result.append(entry)
            continue
        if entry.keys and set(entry.keys).issubset(supplied):
            continue
        if _matches(entry, words, lowered):
            result.append(entry)
    return result
