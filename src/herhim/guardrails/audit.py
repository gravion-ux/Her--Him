"""The guardrail audit log.

Every guardrail decision, every relationship stage transition, and every
refused operation is written here. The log is append-only and JSONL, so it
can be tailed and inspected without loading the whole file.

The rendering is injection-resistant and redacts credential-shaped values,
following the pattern used by ``gravion-ux/context-bridge`` (MIT).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["AuditEvent", "AuditLog"]

_REDACT_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
)

_CREDENTIAL_VALUE = re.compile(
    r"(?i)\b(?:bearer\s+[a-z0-9._\-]+|(?:sk|pk|ghp|gho|xox[baprs])[-_][a-z0-9]{8,})\b"
)


def redact(value: object) -> object:
    """Redact credential-shaped content from a value."""
    if isinstance(value, dict):
        cleaned: dict[str, object] = {}
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in _REDACT_KEYS):
                cleaned[str(key)] = "[redacted]"
            else:
                cleaned[str(key)] = redact(item)
        return cleaned
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _CREDENTIAL_VALUE.sub("[redacted]", value)
    return value


@dataclass(frozen=True)
class AuditEvent:
    """One audited decision.

    Attributes:
        kind: What happened, such as ``guardrail.scope`` or ``relationship.stage``.
        subject: What it happened to.
        decision: ``allow``, ``refuse``, or ``record``.
        detail: Extra context.
        timestamp: Unix timestamp.
    """

    kind: str
    subject: str
    decision: str
    detail: dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "kind": self.kind,
            "subject": self.subject,
            "decision": self.decision,
            "detail": redact(self.detail),
            "timestamp": self.timestamp,
        }

    def to_line(self) -> str:
        """Return the event as one JSONL line."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AuditLog:
    """An append-only audit trail."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[AuditEvent] = []

    def record(
        self,
        kind: str,
        subject: str,
        decision: str,
        detail: dict[str, object] | None = None,
    ) -> AuditEvent:
        """Append an event and return it."""
        event = AuditEvent(kind=kind, subject=subject, decision=decision, detail=detail or {})
        self.events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.to_line() + "\n")
        return event

    def by_kind(self, kind: str) -> list[AuditEvent]:
        """Return every event of one kind."""
        return [event for event in self.events if event.kind == kind]

    def refusals(self) -> list[AuditEvent]:
        """Return every refused event."""
        return [event for event in self.events if event.decision == "refuse"]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {"events": [event.to_dict() for event in self.events]}

    def save(self, path: Path | None = None) -> Path | None:
        """Write the log to disk as JSONL."""
        target = path or self.path
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(event.to_line() + "\n" for event in self.events),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path) -> AuditLog:
        """Read a log from disk."""
        log = cls(path)
        if not path.exists():
            return log
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            log.events.append(
                AuditEvent(
                    kind=str(payload.get("kind", "")),
                    subject=str(payload.get("subject", "")),
                    decision=str(payload.get("decision", "")),
                    detail=dict(payload.get("detail") or {}),
                    timestamp=float(payload.get("timestamp", 0.0) or 0.0),
                )
            )
        return log
