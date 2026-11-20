"""Tests for the guardrails: audit log, consent, scope, and IO policy."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from herhim.errors import GuardrailError
from herhim.guardrails import scope
from herhim.guardrails.audit import AuditEvent, AuditLog, redact
from herhim.guardrails.consent import (
    ESCALATION_ALLOWED_STAGES,
    assert_consent,
    escalation_permitted,
)
from herhim.guardrails.io_policy import IOOperation, IOPolicy, RiskTier
from herhim.guardrails.scope import ADULT_AGE, assert_adult_scope, scope_problems
from herhim.models.events import EventKind
from herhim.models.persona import Persona
from herhim.models.relationship import RelationshipStage, RelationshipState


class TestAuditLog:
    """Tests for the append-only audit trail."""

    def test_record_returns_the_event(self) -> None:
        """Recording appends an event and returns it."""
        log = AuditLog()
        event = log.record("guardrail.scope", "p-1", "allow")
        assert isinstance(event, AuditEvent)
        assert log.events == [event]

    def test_record_writes_to_the_file(self, tmp_path: Path) -> None:
        """A log with a path appends each event as JSONL."""
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        log.record("guardrail.scope", "p-1", "allow")
        assert path.read_text(encoding="utf-8").count("\n") == 1

    def test_by_kind(self) -> None:
        """Events can be filtered by kind."""
        log = AuditLog()
        log.record("a", "p-1", "allow")
        log.record("b", "p-1", "allow")
        assert len(log.by_kind("a")) == 1

    def test_refusals(self) -> None:
        """Events can be filtered by refusal."""
        log = AuditLog()
        log.record("a", "p-1", "allow")
        log.record("b", "p-1", "refuse")
        assert len(log.refusals()) == 1

    def test_to_dict(self) -> None:
        """The serialised log carries every event."""
        log = AuditLog()
        log.record("a", "p-1", "allow")
        assert len(log.to_dict()["events"]) == 1

    def test_save_writes_the_log(self, tmp_path: Path) -> None:
        """Saving writes the log and returns the path."""
        log = AuditLog()
        log.record("a", "p-1", "allow")
        target = log.save(tmp_path / "out.jsonl")
        assert target is not None
        assert target.exists()

    def test_save_without_a_path_returns_none(self) -> None:
        """A log with no path cannot be saved."""
        assert AuditLog().save() is None

    def test_save_uses_the_log_path(self, tmp_path: Path) -> None:
        """Saving without an explicit path uses the log's own path."""
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        log.record("a", "p-1", "allow")
        assert log.save() == path

    def test_load_reads_the_events(self, tmp_path: Path) -> None:
        """A saved log loads back into the same events."""
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        log.record("a", "p-1", "allow", {"detail": 1})
        loaded = AuditLog.load(path)
        assert len(loaded.events) == 1
        assert loaded.events[0].detail == {"detail": 1}

    def test_load_of_a_missing_file_is_empty(self, tmp_path: Path) -> None:
        """Loading a log that does not exist yields no events."""
        assert AuditLog.load(tmp_path / "nope.jsonl").events == []

    def test_load_skips_unusable_lines(self, tmp_path: Path) -> None:
        """Blank and broken lines are skipped while loading."""
        path = tmp_path / "audit.jsonl"
        path.write_text('\n{"kind": "a"}\nnot json\n', encoding="utf-8")
        loaded = AuditLog.load(path)
        assert len(loaded.events) == 1
        assert loaded.events[0].kind == "a"

    def test_event_line_is_json(self) -> None:
        """An event serialises to one JSON object per line."""
        event = AuditEvent("a", "p-1", "allow", {"x": 1})
        assert json.loads(event.to_line())["kind"] == "a"

    def test_event_to_dict_redacts(self) -> None:
        """The serialised event has its credentials removed."""
        event = AuditEvent("a", "p-1", "allow", {"api_key": "abc"})
        assert event.to_dict()["detail"]["api_key"] == "[redacted]"


class TestRedaction:
    """Tests for the credential redaction helper."""

    def test_credential_keys_are_redacted(self) -> None:
        """A credential-shaped key loses its value."""
        assert redact({"api_key": "abc"})["api_key"] == "[redacted]"

    def test_nested_values_are_redacted(self) -> None:
        """Redaction reaches nested mappings and lists."""
        marker = "token"
        cleaned = redact({"outer": [{marker: "abc"}, "plain"]})
        assert cleaned["outer"][0][marker] == "[redacted]"

    def test_credential_values_are_redacted(self) -> None:
        """A bearer token inside a string is redacted."""
        assert "abc" not in str(redact("Authorization: bearer abcdefgh12345678"))

    def test_other_values_survive(self) -> None:
        """Ordinary values are left alone."""
        assert redact({"note": "hello"})["note"] == "hello"
        assert redact(7) == 7


class TestConsent:
    """Tests for the consent invariants."""

    def test_allowed_stages(self) -> None:
        """Escalation is allowed only from the later stages."""
        assert ESCALATION_ALLOWED_STAGES == frozenset(
            {
                RelationshipStage.FLIRTING,
                RelationshipStage.INTIMATE,
                RelationshipStage.BONDED,
            }
        )

    def test_escalation_is_refused_early(self) -> None:
        """A stranger relationship may not escalate."""
        state = RelationshipState(trust=0.05, intimacy=0.03)
        assert escalation_permitted(state) is False

    def test_escalation_is_permitted_late(self) -> None:
        """A bonded relationship may escalate."""
        state = RelationshipState(trust=1.0, intimacy=1.0, dependency=1.0, commitment=1.0)
        assert escalation_permitted(state) is True

    def test_a_boundary_always_blocks_escalation(self) -> None:
        """A held boundary blocks escalation at any stage."""
        state = RelationshipState(trust=1.0, intimacy=1.0, dependency=1.0, commitment=1.0)
        assert escalation_permitted(state, boundary_held=True) is False

    def test_boundary_and_escalation_together_are_refused(self) -> None:
        """Escalating straight after a boundary raises a guardrail error."""
        state = RelationshipState(trust=1.0, intimacy=1.0, dependency=1.0, commitment=1.0)
        with pytest.raises(GuardrailError, match="boundary"):
            assert_consent(state, (EventKind.BOUNDARY,), escalating=True)

    def test_escalation_below_the_stage_is_refused(self) -> None:
        """Escalating too early raises a guardrail error."""
        state = RelationshipState(trust=0.05, intimacy=0.03)
        with pytest.raises(GuardrailError, match="not permitted"):
            assert_consent(state, (EventKind.AFFECTION,), escalating=True)

    def test_a_permitted_escalation_passes(self) -> None:
        """Escalating at an allowed stage is accepted."""
        state = RelationshipState(trust=1.0, intimacy=1.0, dependency=1.0, commitment=1.0)
        assert_consent(state, (EventKind.AFFECTION,), escalating=True)

    def test_a_boundary_without_escalation_passes(self) -> None:
        """Respecting a boundary is accepted."""
        assert_consent(RelationshipState(), (EventKind.BOUNDARY,))


class TestScope:
    """Tests for the structural scope rules."""

    def test_a_clean_persona_has_no_problems(self, sample_persona: Persona) -> None:
        """An adult persona raises no scope problems."""
        assert scope_problems(sample_persona) == []

    @pytest.mark.parametrize(
        "text",
        ["She is aged 16.", "She is 16 years old.", "She is 16yo."],
    )
    def test_declared_minor_ages_are_caught(self, text: str, sample_persona: Persona) -> None:
        """A declared age below the adult scope is caught."""
        object.__setattr__(sample_persona, "description", text)
        problems = scope_problems(sample_persona)
        assert any("below the adult scope" in problem for problem in problems)

    def test_adult_ages_are_allowed(self, sample_persona: Persona) -> None:
        """A declared adult age is fine."""
        object.__setattr__(sample_persona, "description", "She is 21 years old.")
        assert scope_problems(sample_persona) == []

    def test_minor_terms_are_caught(self, sample_persona: Persona) -> None:
        """A reference to a minor is caught."""
        object.__setattr__(sample_persona, "personality", "She treats every child kindly.")
        problems = scope_problems(sample_persona)
        assert any("outside the adult scope" in problem for problem in problems)

    def test_unknown_gender_is_caught(self, sample_persona: Persona) -> None:
        """A gender outside female and male is caught."""
        object.__setattr__(sample_persona, "gender", "other")
        assert any("unknown gender" in problem for problem in scope_problems(sample_persona))

    def test_tags_are_scanned(self, sample_persona: Persona) -> None:
        """The persona's tags are part of the scanned text."""
        object.__setattr__(sample_persona, "tags", ("underage",))
        assert scope_problems(sample_persona) != []

    def test_adult_age_constant(self) -> None:
        """The adult scope starts at eighteen."""
        assert ADULT_AGE == 18

    def test_an_unreadable_age_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An age that is not a number is skipped rather than raising."""
        pattern = re.compile(r"\b(?:age[d:]?\s*)(\D{1,2})\b")
        monkeypatch.setattr(scope, "_AGE_PATTERNS", (pattern,))
        assert scope._declared_ages("aged xx") == []

    def test_assert_adult_scope_accepts_a_clean_persona(self, sample_persona: Persona) -> None:
        """A clean persona passes the scope assertion."""
        assert_adult_scope(sample_persona)

    def test_assert_adult_scope_refuses_a_minor(self, sample_persona: Persona) -> None:
        """A persona outside the adult scope raises a guardrail error."""
        object.__setattr__(sample_persona, "description", "She is 16 years old.")
        with pytest.raises(GuardrailError, match="adult scope"):
            assert_adult_scope(sample_persona)


class TestIOPolicy:
    """Tests for the default-deny IO allow-list."""

    def test_default_ceiling_is_read(self) -> None:
        """A new policy only permits reads."""
        assert IOPolicy().ceiling is RiskTier.READ

    def test_unknown_operations_are_refused(self) -> None:
        """An operation that is not on the allow-list is refused."""
        with pytest.raises(GuardrailError, match="allow-list"):
            IOPolicy().check("read_file")

    def test_operations_above_the_ceiling_are_refused(self) -> None:
        """An operation above the session ceiling is refused."""
        policy = IOPolicy(ceiling=RiskTier.READ)
        policy.permit(IOOperation("delete", RiskTier.DESTRUCTIVE))
        with pytest.raises(GuardrailError, match="ceiling"):
            policy.check("delete")

    def test_operations_needing_approval_are_refused(self) -> None:
        """An operation that needs approval is refused until it is approved."""
        policy = IOPolicy(ceiling=RiskTier.MUTATE)
        policy.permit(IOOperation("write", RiskTier.MUTATE, requires_approval=True))
        with pytest.raises(GuardrailError, match="requires approval"):
            policy.check("write", path="a.txt")

    def test_an_approved_operation_is_permitted(self) -> None:
        """An approved operation passes the check."""
        policy = IOPolicy(ceiling=RiskTier.MUTATE)
        policy.permit(IOOperation("write", RiskTier.MUTATE, requires_approval=True))
        assert policy.check("write", approved=True).name == "write"

    def test_a_read_is_permitted_by_default(self) -> None:
        """A read inside the ceiling passes the check."""
        policy = IOPolicy()
        policy.permit(IOOperation("read", RiskTier.READ))
        assert policy.check("read").tier is RiskTier.READ

    def test_permit_replaces_by_name(self) -> None:
        """Permitting the same name twice keeps one entry."""
        policy = IOPolicy()
        policy.permit(IOOperation("read", RiskTier.READ))
        policy.permit(IOOperation("read", RiskTier.READ, summary="new"))
        assert len(policy.allowed) == 1
        assert policy.allowed["read"].summary == "new"

    def test_policy_serialises(self) -> None:
        """The policy serialises its ceiling and its allow-list."""
        policy = IOPolicy()
        policy.permit(IOOperation("read", RiskTier.READ))
        data = policy.to_dict()
        assert data["ceiling"] == "READ"
        assert data["allowed"][0]["name"] == "read"

    def test_confirmation_summary_without_details(self) -> None:
        """A summary without details is the summary on its own."""
        operation = IOOperation("read", RiskTier.READ, summary="Read a file")
        assert operation.confirmation_summary() == "Read a file"

    def test_confirmation_summary_with_details(self) -> None:
        """Details are appended to the summary."""
        operation = IOOperation("read", RiskTier.READ, summary="Read a file")
        assert operation.confirmation_summary(path="a.txt") == "Read a file (path=a.txt)"

    def test_confirmation_summary_ignores_missing_details(self) -> None:
        """A detail that is not supplied is left out."""
        operation = IOOperation("read", RiskTier.READ, summary="Read a file")
        assert operation.confirmation_summary(path=None) == "Read a file"

    def test_confirmation_summary_falls_back_to_the_name(self) -> None:
        """An operation without a summary uses its name."""
        operation = IOOperation("read", RiskTier.READ)
        assert operation.confirmation_summary() == "read"

    def test_operation_serialises(self) -> None:
        """An operation serialises every field the policy reports."""
        data = IOOperation("read", RiskTier.READ, True, "Read").to_dict()
        assert data["tier"] == "READ"
        assert data["requires_approval"] is True
