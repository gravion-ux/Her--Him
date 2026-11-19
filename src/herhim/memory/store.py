"""The layered memory store.

Four layers, following the memory-hub model used by ``morettt/my-neuro``
(MIT) and the fact/assumption separation used by ``gravion-ux/aegis`` (MIT):

* ``facts`` - verified, durable statements.
* ``profile`` - conservatively accumulated preferences and landmines.
* ``episodes`` - summarised significant events.
* ``recent`` - verbatim turns, bounded.

Verified facts are kept strictly separate from assumptions and open
questions, so a guess is never presented to the model as a fact.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CONFIDENCE_LEVELS", "MemoryRecord", "MemoryStore"]

#: The confidence levels a record can carry.
CONFIDENCE_LEVELS: tuple[str, ...] = ("verified", "assumed", "open_question")

#: How many verbatim turns are retained.
RECENT_LIMIT = 40


@dataclass(frozen=True)
class MemoryRecord:
    """One remembered thing.

    Attributes:
        record_id: A unique identifier.
        layer: ``facts``, ``profile``, ``episodes``, or ``recent``.
        text: The content.
        source: Where the content came from.
        confidence: ``verified``, ``assumed``, or ``open_question``.
        created_at: Unix timestamp of creation.
        role: For ``recent`` records, who said it.
    """

    record_id: str
    layer: str
    text: str
    source: str = ""
    confidence: str = "verified"
    created_at: float = field(default_factory=time.time)
    role: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "record_id": self.record_id,
            "layer": self.layer,
            "text": self.text,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MemoryRecord:
        """Build a record from a mapping, ignoring unknown keys."""
        return cls(
            record_id=str(data.get("record_id", "")),
            layer=str(data.get("layer", "facts")),
            text=str(data.get("text", "")),
            source=str(data.get("source", "")),
            confidence=str(data.get("confidence", "verified")),
            created_at=float(data.get("created_at", 0.0) or 0.0),  # type: ignore[arg-type]
            role=str(data.get("role", "")),
        )


class MemoryStore:
    """A durable, layered memory for one session."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[MemoryRecord] = []

    # -- writing ---------------------------------------------------------

    def add(
        self,
        layer: str,
        text: str,
        *,
        source: str = "",
        confidence: str = "verified",
        role: str = "",
    ) -> MemoryRecord:
        """Store one record and return it."""
        if confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"unknown confidence level: {confidence}")
        record = MemoryRecord(
            record_id=uuid.uuid4().hex,
            layer=layer,
            text=text,
            source=source,
            confidence=confidence,
            role=role,
        )
        self.records.append(record)
        if layer == "recent":
            self._trim_recent()
        return record

    def add_fact(self, text: str, *, source: str = "") -> MemoryRecord:
        """Store a verified, durable fact."""
        return self.add("facts", text, source=source, confidence="verified")

    def add_profile(self, text: str, *, source: str = "") -> MemoryRecord:
        """Store a conservatively accumulated profile note."""
        return self.add("profile", text, source=source, confidence="verified")

    def add_episode(self, text: str, *, source: str = "") -> MemoryRecord:
        """Store a summarised event."""
        return self.add("episodes", text, source=source, confidence="verified")

    def add_turn(self, role: str, text: str, *, source: str = "") -> MemoryRecord:
        """Store one verbatim turn."""
        return self.add("recent", text, source=source, confidence="verified", role=role)

    def _trim_recent(self) -> None:
        """Keep the recent layer bounded."""
        recent = [record for record in self.records if record.layer == "recent"]
        if len(recent) <= RECENT_LIMIT:
            return
        excess = len(recent) - RECENT_LIMIT
        doomed = {record.record_id for record in recent[:excess]}
        self.records = [record for record in self.records if record.record_id not in doomed]

    # -- reading ---------------------------------------------------------

    def layer(self, name: str) -> list[MemoryRecord]:
        """Return every record in one layer, oldest first."""
        return [record for record in self.records if record.layer == name]

    def facts(self) -> list[MemoryRecord]:
        """Return the verified facts."""
        return [record for record in self.layer("facts") if record.confidence == "verified"]

    def profile(self) -> list[MemoryRecord]:
        """Return the profile notes."""
        return self.layer("profile")

    def episodes(self) -> list[MemoryRecord]:
        """Return the episode summaries."""
        return self.layer("episodes")

    def recent(self, limit: int = RECENT_LIMIT) -> list[MemoryRecord]:
        """Return the most recent verbatim turns, oldest first."""
        recent = self.layer("recent")
        return recent[-limit:]

    def clear(self) -> None:
        """Forget everything."""
        self.records.clear()

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping of the whole store."""
        return {"records": [record.to_dict() for record in self.records]}

    def save(self, path: Path | None = None) -> Path | None:
        """Write the store to disk."""
        target = path or self.path
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path) -> MemoryStore:
        """Read a store from disk."""
        store = cls(path)
        if not path.exists():
            return store
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return store
        records = payload.get("records", []) if isinstance(payload, dict) else []
        store.records = [MemoryRecord.from_dict(record) for record in records]
        return store
