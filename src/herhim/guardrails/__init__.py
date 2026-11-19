"""Structural guardrails: scope, consent, IO policy, and the audit trail."""

from __future__ import annotations

from herhim.guardrails.audit import AuditEvent, AuditLog, redact
from herhim.guardrails.consent import (
    ESCALATION_ALLOWED_STAGES,
    assert_consent,
    escalation_permitted,
)
from herhim.guardrails.io_policy import IOOperation, IOPolicy, RiskTier
from herhim.guardrails.scope import ADULT_AGE, assert_adult_scope, scope_problems

__all__ = [
    "ADULT_AGE",
    "ESCALATION_ALLOWED_STAGES",
    "AuditEvent",
    "AuditLog",
    "IOOperation",
    "IOPolicy",
    "RiskTier",
    "assert_adult_scope",
    "assert_consent",
    "escalation_permitted",
    "redact",
    "scope_problems",
]
