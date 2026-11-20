"""Tests for the affect subsystem: classifier, moods, directive, and engine."""

from __future__ import annotations

import pytest

from herhim.affect.classifier import (
    EVENT_PRECEDENCE,
    EventClassifier,
    Rule,
    base_deltas,
    classify,
)
from herhim.affect.directive import MoodGuidance, build_directive, mood_guidance
from herhim.affect.engine import (
    DECAY_PER_TURN,
    OFFLINE_TAU_MINUTES,
    AffectEngine,
    TurnRationale,
    persona_engine,
)
from herhim.affect.moods import (
    MOOD_ALPHA,
    MOOD_LABELS,
    MoodLabel,
    MoodResult,
    derive_mood,
    event_mood_bias,
    raw_scores,
)
from herhim.models.affect import AffectVector
from herhim.models.events import EventKind
from herhim.models.persona import Persona
from herhim.models.relationship import RelationshipStage, RelationshipState
from herhim.models.traits import TraitProfile


class _FalsyAffectVector(AffectVector):
    """An affect vector that reports itself as empty.

    The engine uses this to exercise its "no explicit vector" fallback.
    """

    def __bool__(self) -> bool:
        """Always false, so the engine falls back to the persona baseline."""
        return False


class TestEventClassifier:
    """Tests for the deterministic event classifier."""

    def test_empty_text_detects_nothing(self) -> None:
        """An empty turn fires no events."""
        assert classify("") == []
        assert classify("    ") == []

    def test_rejection_is_detected(self) -> None:
        """A flat refusal is a rejection."""
        kinds = [event.kind for event in classify("No.")]
        assert EventKind.REJECTION in kinds

    def test_hostility_is_detected(self) -> None:
        """A hostile dismissal is a hostility event."""
        kinds = [event.kind for event in classify("I hate you")]
        assert EventKind.HOSTILITY in kinds

    def test_matching_is_case_insensitive(self) -> None:
        """Rules ignore case."""
        kinds = [event.kind for event in classify("SHUT UP")]
        assert EventKind.HOSTILITY in kinds

    def test_primary_event_is_strongest(self) -> None:
        """The strongest match leads and carries the full scale."""
        events = classify("I hate you")
        assert events[0].kind is EventKind.HOSTILITY
        assert events[0].scale == 1.0

    def test_secondary_event_is_scaled(self) -> None:
        """A close secondary event is kept at half scale."""
        events = classify("No, I'm sorry")
        kinds = [event.kind for event in events]
        assert EventKind.APOLOGY in kinds
        assert EventKind.REJECTION in kinds
        secondary = next(e for e in events if e.kind is EventKind.REJECTION)
        assert secondary.scale == 0.5

    def test_repeated_matches_raise_the_intensity(self) -> None:
        """More matches of the same rule make the event stronger."""
        events = classify("no no no")
        assert events[0].intensity == 1.0

    def test_strongest_rule_per_kind_wins(self) -> None:
        """Two rules for one kind keep the stronger match."""
        events = classify("no never")
        rejections = [event for event in events if event.kind is EventKind.REJECTION]
        assert len(rejections) == 1
        assert rejections[0].intensity == 0.75

    def test_weak_secondary_is_dropped(self) -> None:
        """A secondary event far weaker than the primary is discarded."""
        classifier = EventClassifier(
            (
                Rule(EventKind.REJECTION, r"\balpha\b", 1.0, "strong"),
                Rule(EventKind.PRAISE, r"\bbeta\b", 0.3, "weak"),
            )
        )
        events = classifier.detect("alpha beta")
        assert [event.kind for event in events] == [EventKind.REJECTION]

    def test_moderate_secondary_is_kept_at_half_scale(self) -> None:
        """A secondary event within the band is kept at half scale."""
        classifier = EventClassifier(
            (
                Rule(EventKind.REJECTION, r"\balpha\b", 1.0, "strong"),
                Rule(EventKind.PRAISE, r"\bbeta\b", 0.6, "medium"),
            )
        )
        events = classifier.detect("alpha beta")
        assert [event.kind for event in events] == [EventKind.REJECTION, EventKind.PRAISE]
        assert events[1].scale == 0.5

    def test_custom_rules_without_matches(self) -> None:
        """A rule set that matches nothing fires nothing."""
        classifier = EventClassifier((Rule(EventKind.TEASE, r"\bnothing here\b", 0.9, "x"),))
        assert classifier.detect("something else") == []

    def test_rule_compiles_case_insensitively(self) -> None:
        """The compiled pattern honours the ignore-case flag."""
        pattern = Rule(EventKind.TEASE, r"\btease\b", 0.9, "teasing").compiled()
        assert pattern.search("TEASE ME") is not None

    def test_base_deltas_cover_every_kind(self) -> None:
        """Every event kind has a four-axis base delta."""
        for kind in EventKind:
            deltas = base_deltas(kind)
            assert len(deltas) == 4

    def test_precedence_covers_every_kind(self) -> None:
        """The tie-break table lists every event kind exactly once."""
        assert set(EVENT_PRECEDENCE) == set(EventKind)
        assert len(EVENT_PRECEDENCE) == len(EventKind)


class TestMoodDerivation:
    """Tests for mood scoring and smoothing."""

    def test_every_label_has_a_raw_score(self) -> None:
        """The raw score table covers all eighteen labels."""
        scores = raw_scores(AffectVector(), TraitProfile(), RelationshipState())
        assert set(scores) == set(MOOD_LABELS)
        assert len(MOOD_LABELS) == 18

    def test_derive_mood_returns_a_known_label(self) -> None:
        """The derived mood is always one of the declared labels."""
        result = derive_mood(AffectVector(), TraitProfile(), RelationshipState())
        assert result.label in MOOD_LABELS
        assert set(result.scores) == set(MOOD_LABELS)

    def test_affection_bias_wins(self) -> None:
        """An affection event pushes the mood towards affectionate."""
        result = derive_mood(
            vector=AffectVector(valence=0.8, arousal=0.6, assertion=0.4, tenderness=0.9),
            traits=TraitProfile(warmth=0.9),
            relationship=RelationshipState(
                trust=0.5,
                intimacy=0.5,
                dependency=0.5,
                commitment=0.5,
            ),
            biases=event_mood_bias(EventKind.AFFECTION),
        )
        assert result.label is MoodLabel.AFFECTIONATE

    def test_previous_scores_are_smoothed(self) -> None:
        """A previous score table is carried into the result."""
        previous = dict.fromkeys(MOOD_LABELS, 1.0)
        result = derive_mood(
            vector=AffectVector(),
            traits=TraitProfile(),
            relationship=RelationshipState(),
            previous=previous,
        )
        assert all(score > MOOD_ALPHA for score in result.scores.values())

    def test_event_mood_bias_is_a_copy(self) -> None:
        """The bias table is copied so callers cannot mutate it."""
        bias = event_mood_bias(EventKind.REJECTION)
        bias[MoodLabel.DEFIANT] = 99.0
        assert event_mood_bias(EventKind.REJECTION)[MoodLabel.DEFIANT] != 99.0

    def test_mood_result_to_dict_runs_sorted(self) -> None:
        """The mapping lists the moods from strongest to weakest."""
        result = MoodResult(label=MoodLabel.CALM, scores=dict.fromkeys(MOOD_LABELS, 0.1))
        data = result.to_dict()
        assert data["mood"] == MoodLabel.CALM.value
        assert "mood_scores" in data


class TestDirective:
    """Tests for the style directive compiler."""

    def test_every_mood_has_guidance(self) -> None:
        """Each mood carries a summary, a delivery, and a set of avoidances."""
        for label in MOOD_LABELS:
            guidance = mood_guidance(label)
            assert isinstance(guidance, MoodGuidance)
            assert guidance.summary
            assert guidance.delivery
            assert guidance.avoid

    def test_directive_contains_the_state(self) -> None:
        """The directive names the mood and the relationship stage."""
        text = build_directive(
            MoodLabel.CALM,
            AffectVector(),
            RelationshipStage.STRANGER,
            TraitProfile(),
        )
        assert "CURRENT STATE" in text
        assert "RULES" in text
        assert MoodLabel.CALM.value in text
        assert RelationshipStage.STRANGER.value in text

    def test_low_tenderness_is_called_out(self) -> None:
        """A cold affect vector adds an explicit warmth warning."""
        text = build_directive(
            MoodLabel.SULKY,
            AffectVector(tenderness=0.1),
            RelationshipStage.STRANGER,
            TraitProfile(),
        )
        assert "not feeling tender" in text

    def test_stubborn_assertion_is_called_out(self) -> None:
        """Assertive and stubborn adds an immovability warning."""
        text = build_directive(
            MoodLabel.DEFIANT,
            AffectVector(assertion=0.8),
            RelationshipStage.STRANGER,
            TraitProfile(stubbornness=0.8),
        )
        assert "will not back down" in text

    def test_boundary_is_called_out(self) -> None:
        """A held boundary adds a warning and an avoidance rule."""
        text = build_directive(
            MoodLabel.DETACHED,
            AffectVector(),
            RelationshipStage.STRANGER,
            TraitProfile(),
            boundary_held=True,
        )
        assert "A boundary was just set" in text
        assert "escalate anything physical or intimate" in text

    def test_every_stage_has_guidance(self) -> None:
        """Each relationship stage contributes its own guidance."""
        for stage in RelationshipStage:
            text = build_directive(MoodLabel.CALM, AffectVector(), stage, TraitProfile())
            assert stage.value in text


class TestAffectEngine:
    """Tests for the affect engine's state transitions."""

    def test_persona_engine_starts_at_the_baseline(self, sample_persona: Persona) -> None:
        """A fresh engine sits on the persona's baseline."""
        engine = persona_engine(sample_persona)
        assert engine.vector == sample_persona.affect_baseline
        assert engine.relationship == sample_persona.relationship_baseline
        assert engine.turn_index == 0
        assert engine.mood is MoodLabel.CALM

    def test_persona_engine_accepts_a_relationship_override(
        self, sample_persona: Persona
    ) -> None:
        """An explicit relationship replaces the persona's baseline."""
        override = RelationshipState(trust=0.7, intimacy=0.7)
        engine = persona_engine(sample_persona, relationship=override)
        assert engine.relationship == override

    def test_engine_falls_back_to_the_baseline_vector(self, sample_persona: Persona) -> None:
        """An empty vector falls back to the persona's affect baseline."""
        engine = AffectEngine(persona=sample_persona, vector=_FalsyAffectVector())
        assert engine.vector == sample_persona.affect_baseline

    def test_engine_builds_the_mood_table(self, sample_persona: Persona) -> None:
        """The mood score table is populated on construction."""
        engine = AffectEngine(persona=sample_persona)
        assert set(engine.mood_scores) == set(MOOD_LABELS)
        assert all(score == 0.0 for score in engine.mood_scores.values())

    def test_neutral_turn_fires_no_events(self, sample_persona: Persona) -> None:
        """A neutral turn changes nothing observable."""
        engine = persona_engine(sample_persona)
        rationale = engine.apply_turn("The weather is fine today")
        assert rationale.events == ()
        assert engine.turn_index == 1

    def test_rejection_moves_the_state(self, sample_persona: Persona) -> None:
        """A rejection lowers valence and raises tension."""
        engine = persona_engine(sample_persona)
        rationale = engine.apply_turn("No, I don't want that")
        assert rationale.events
        assert rationale.after.valence < rationale.before.valence
        assert rationale.relationship_after.tension > rationale.relationship_before.tension

    def test_boundary_blocks_escalation(self, sample_persona: Persona) -> None:
        """A boundary turn stops intimacy from advancing."""
        engine = persona_engine(sample_persona)
        before = engine.relationship.intimacy
        rationale = engine.apply_turn("Don't touch me. I love you")
        assert rationale.boundary_held is True
        assert engine.relationship.intimacy == pytest.approx(before)

    def test_snapshot_shape(self, sample_persona: Persona) -> None:
        """The snapshot carries the state the API exposes."""
        engine = persona_engine(sample_persona)
        snapshot = engine.snapshot()
        assert snapshot["persona_id"] == sample_persona.persona_id
        assert snapshot["stage"] == RelationshipStage.ACQUAINTED.value
        assert "affect" in snapshot
        assert "relationship" in snapshot

    def test_reset_restores_the_baseline(self, sample_persona: Persona) -> None:
        """Resetting returns the engine to the persona's opening state."""
        engine = persona_engine(sample_persona)
        engine.apply_turn("No.")
        engine.reset()
        assert engine.vector == sample_persona.affect_baseline
        assert engine.relationship == sample_persona.relationship_baseline
        assert engine.turn_index == 0
        assert engine.mood is MoodLabel.CALM

    def test_absence_of_zero_minutes_is_a_no_op(self, sample_persona: Persona) -> None:
        """No time away means no decay."""
        engine = persona_engine(sample_persona)
        before = engine.vector
        assert engine.apply_absence(0.0) == 0.0
        assert engine.vector == before

    def test_absence_cools_toward_the_baseline(self, sample_persona: Persona) -> None:
        """Time away moves the vector back toward the persona baseline."""
        engine = persona_engine(sample_persona)
        engine.apply_turn("No.")
        away = engine.vector
        factor = engine.apply_absence(OFFLINE_TAU_MINUTES)
        assert 0.0 < factor < 1.0
        assert engine.vector != away

    def test_restore_rebuilds_the_state(self, sample_persona: Persona) -> None:
        """A snapshot restores the vector, relationship, and counters."""
        engine = persona_engine(sample_persona)
        engine.apply_turn("No.")
        snapshot = engine.snapshot()
        restored = persona_engine(sample_persona)
        restored.restore(snapshot)
        assert restored.turn_index == 1
        assert restored.vector.to_dict() == engine.vector.to_dict()

    def test_restore_ignores_unusable_values(self, sample_persona: Persona) -> None:
        """Rubbish in a snapshot falls back to safe defaults."""
        engine = persona_engine(sample_persona)
        engine.restore({"affect": "nonsense", "relationship": 5, "turn_index": "abc"})
        assert engine.turn_index == 0
        assert engine.vector == AffectVector()

    def test_restore_accepts_a_numeric_string_turn_index(self, sample_persona: Persona) -> None:
        """A turn index stored as a string is still restored."""
        engine = persona_engine(sample_persona)
        engine.restore({"turn_index": "7"})
        assert engine.turn_index == 7

    def test_rationale_reports_the_stage_change(self, sample_persona: Persona) -> None:
        """The rationale says whether the relationship stage moved."""
        engine = persona_engine(sample_persona)
        rationale = engine.apply_turn("No.")
        data = rationale.to_dict()
        assert "stage_changed" in data
        assert data["stage_before"] == rationale.stage_before.value
        assert isinstance(rationale, TurnRationale)

    def test_decay_constant_is_used(self, sample_persona: Persona) -> None:
        """Each turn pulls the vector back toward the baseline."""
        engine = persona_engine(sample_persona)
        engine.apply_turn("I love you so much")
        assert engine.vector.to_dict() != sample_persona.affect_baseline.to_dict()
        assert DECAY_PER_TURN > 0.0


class TestEngineRestore:
    """Tests for restoring an engine from a snapshot."""

    def test_a_snapshot_round_trips(self, sample_persona: Persona) -> None:
        """Restoring a snapshot brings the engine back to that state."""
        engine = persona_engine(sample_persona)
        engine.apply_turn("I love you so much")
        snapshot = engine.snapshot()
        restored = persona_engine(sample_persona)
        restored.restore(snapshot)
        assert restored.vector.to_dict() == engine.vector.to_dict()
        assert restored.turn_index == engine.turn_index
        assert restored.mood is engine.mood

    def test_an_empty_snapshot_falls_back_to_the_defaults(
        self, sample_persona: Persona
    ) -> None:
        """Restoring nothing leaves the engine at the documented defaults."""
        engine = persona_engine(sample_persona)
        engine.restore({})
        assert engine.turn_index == 0
        assert engine.vector.to_dict() == AffectVector().to_dict()
        assert engine.mood is MoodLabel.CALM
