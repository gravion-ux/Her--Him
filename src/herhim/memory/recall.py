"""Recall: choose what to put in front of the model, and say where it came from.

Facts are scored by keyword overlap and recency. Every recalled item keeps
its source so the prompt can distinguish a verified fact from an assumption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from herhim.memory.store import MemoryRecord, MemoryStore

__all__ = ["Recall", "recall"]

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: How much a keyword hit contributes to the score.
_KEYWORD_WEIGHT = 0.55

#: How much recency contributes to the score.
_RECENCY_WEIGHT = 0.25


@dataclass(frozen=True)
class Recall:
    """One recalled item and why it was recalled."""

    record: MemoryRecord
    score: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "text": self.record.text,
            "layer": self.record.layer,
            "source": self.record.source,
            "confidence": self.record.confidence,
            "score": round(self.score, 4),
        }


def _keywords(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def recall(store: MemoryStore, query: str, *, limit: int = 8) -> list[Recall]:
    """Return the records most relevant to ``query``.

    Only verified facts, profile notes, and episode summaries are considered.
    Verbatim turns are handled separately by the session.
    """
    query_words = _keywords(query)
    if not query_words:
        return []

    candidates = [*store.facts(), *store.profile(), *store.episodes()]
    if not candidates:
        return []

    newest = max(record.created_at for record in candidates)
    oldest = min(record.created_at for record in candidates)
    span = max(1.0, newest - oldest)

    scored: list[Recall] = []
    for record in candidates:
        record_words = _keywords(record.text)
        overlap = len(query_words & record_words)
        if overlap == 0:
            continue
        keyword_score = min(1.0, overlap / max(1, len(query_words))) * _KEYWORD_WEIGHT
        recency = (record.created_at - oldest) / span
        recency_score = recency * _RECENCY_WEIGHT
        scored.append(Recall(record=record, score=keyword_score + recency_score))

    scored.sort(key=lambda item: (-item.score, item.record.created_at))
    return scored[:limit]
