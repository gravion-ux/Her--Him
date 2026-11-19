"""Tests for the affect engine and mood derivation."""

from __future__ import annotations

import pytest

from herhim.affect.engine import AffectEngine, persona_engine
from herhim.affect.moods import MoodLabel, derive_mood, event_mood_bias
from herhim.models.affect import AffectDelta, AffectVector, _clamp
from herhim.models.events import EventKind
from herhim.models.persona import LoreEntry, Persona
from herhim.models.relationship import RelationshipState
from herhim.models.traits import TraitProfile


class TestAffectVector:
    """Tests for the AffectVector model."""

    def test_clamping(self) -> None:
        """Values are clamped to valid ranges."""
        assert _clamp(-0.5, 0.0, 1.0) == 0.0
        assert _clamp(0.5, 0.0, 1.0) == 0.5
        assert _clamp(1.5, 0.0, 1.0) == 1.0

    def test_shift(self) -> None:
        """Vector can be shifted by a delta."""
        v = AffectVector(valence=0.5, arousal=0.5, assertion=0.5, tenderness=0.5)
        d = AffectDelta(valence=-0.3, arousal=0.2, assertion=0.1, tenderness=-0.1)
        result = v.shift(d)
        assert abs(result.valence - 0.2) < 0.01
        assert abs(result.arousal - 0.7) < 0.01
        assert abs(result.assertion - 0.6) < 0.01
        assert abs(result.tenderness - 0.4) < 0.01


class TestMoodDerivation:
    """Tests for mood derivation from affect vectors and traits."""

    def test_defiant_from_rejection(self) -> None:
        """A rejection with high assertion should produce defiant mood."""
        vector = AffectVector(valence=-0.3, arousal=0.6, assertion=0.6, tenderness=0.4)
        traits = TraitProfile(bratty=0.55, playfulness=0.65)
        relationship = RelationshipState(
            trust=0.5,
            intimacy=0.5,
            tension=0.1,
            jealousy=0.1,
            dependency=0.1,
            commitment=0.1,
        )

        # REJECTION event should apply bias toward DEFIANT
        biases = event_mood_bias(EventKind.REJECTION)
        mood_result = derive_mood(
            vector=vector,
            traits=traits,
            relationship=relationship,
            biases=biases,
        )

        # REJECTION with negative valence and high assertion should bias toward DEFIANT
        assert mood_result.label == MoodLabel.DEFIANT

    def test_calm_at_baseline(self) -> None:
        """At baseline, mood should be calm."""
        vector = AffectVector(valence=0.15, arousal=0.40, assertion=0.35, tenderness=0.50)
        traits = TraitProfile()
        relationship = RelationshipState(
            trust=0.5,
            intimacy=0.5,
            tension=0.1,
            jealousy=0.1,
            dependency=0.1,
            commitment=0.1,
        )

        mood_result = derive_mood(
            vector=vector,
            traits=traits,
            relationship=relationship,
        )

        assert mood_result.label == MoodLabel.CALM


class TestAffectEngine:
    """Tests for the AffectEngine class."""

    @pytest.fixture
    def sample_persona(self) -> Persona:
        """Create a sample persona for testing."""
        return Persona(
            persona_id="test-001",
            catalogue="her",
            gender="female",
            display_name="Test Persona",
            tagline="A test persona",
            description="A test description",
            personality="Test personality",
            scenario="Test scenario",
            first_mes="Hello",
            mes_example="Hi",
            system_prompt="You are a test persona.",
            affect_baseline=AffectVector(
                valence=0.15,
                arousal=0.40,
                assertion=0.35,
                tenderness=0.50,
            ),
            traits=TraitProfile(
                extraversion=0.70,
                playfulness=0.65,
                bratty=0.55,
                stubbornness=0.45,
                warmth=0.55,
                jealousy=0.25,
                dominance=0.40,
                loyalty=0.50,
                patience=0.40,
                sensuality=0.45,
                humour=0.60,
            ),
            event_sensitivity={"REJECTION": 1.40, "PROVOCATION": 1.25},
            relationship_baseline=RelationshipState(
                trust=0.32,
                intimacy=0.25,
                tension=0.03,
                jealousy=0.06,
                dependency=0.12,
                commitment=0.10,
            ),
            lorebook=(
                LoreEntry(
                    keys=("test",),
                    content="Test lore entry for unit testing.",
                    position="before_char",
                    constant=True,
                ),
            ),
        )

    @pytest.fixture
    def engine(self, sample_persona: Persona) -> AffectEngine:
        """Create an affect engine for testing."""
        return persona_engine(sample_persona)

    def test_engine_reset(self, engine: AffectEngine) -> None:
        """Engine can be reset to baseline state."""
        # Apply a turn first
        engine.apply_turn("This is a test")

        # Reset
        engine.reset()

        # Check state is back to baseline
        assert engine.vector == engine.persona.affect_baseline
        assert engine.turn_index == 0
        assert engine.mood == MoodLabel.CALM

    def test_apply_turn(self, engine: AffectEngine) -> None:
        """Apply a turn and check state changes."""
        rationale = engine.apply_turn("No. I don't want that.")

        # Should have detected REJECTION
        assert len(rationale.events) > 0
        assert any(e.kind.value == "REJECTION" for e in rationale.events)


class TestGoldenScenario:
    """Golden scenario tests for validation."""

    def test_rejection_then_provocation(self) -> None:
        """Validate the golden scenario from the spec."""
        # Create a persona matching her-001-sassy
        persona = Persona(
            persona_id="her-001-sassy",
            catalogue="her",
            gender="female",
            display_name="Sassy Maya",
            tagline="Quick-witted with plenty of attitude",
            description="Test persona for golden scenario",
            personality="Test",
            scenario="Test",
            first_mes="Oh, you finally showed up.",
            mes_example="I like you.",
            system_prompt="You are Sassy Maya.",
            affect_baseline=AffectVector(
                valence=0.15,
                arousal=0.40,
                assertion=0.35,
                tenderness=0.50,
            ),
            traits=TraitProfile(
                extraversion=0.70,
                playfulness=0.65,
                bratty=0.55,
                stubbornness=0.45,
                warmth=0.55,
                jealousy=0.25,
                dominance=0.40,
                loyalty=0.50,
                patience=0.40,
                sensuality=0.45,
                humour=0.60,
            ),
            event_sensitivity={"REJECTION": 1.40, "PROVOCATION": 1.25},
            relationship_baseline=RelationshipState(
                trust=0.32,
                intimacy=0.25,
                tension=0.03,
                jealousy=0.06,
                dependency=0.12,
                commitment=0.10,
            ),
            lorebook=(
                LoreEntry(
                    keys=("sassy",),
                    content="Sassy Maya profile content.",
                    position="before_char",
                    constant=True,
                ),
            ),
        )

        engine = persona_engine(persona)

        # Turn 1: Rejection
        r1 = engine.apply_turn("No. I don't want that.")
        assert any(e.kind.value == "REJECTION" for e in r1.events)

        # After rejection: valence down, arousal up, assertion up, tenderness down
        assert r1.after.valence < 0  # Negative
        assert r1.after.arousal > 0.5  # High
        assert r1.after.assertion > 0.5  # High
        assert r1.mood == MoodLabel.DEFIANT

        # Turn 2: Provocation
        r2 = engine.apply_turn("You can't, but I can.")
        assert any(e.kind.value == "PROVOCATION" for e in r2.events)
        assert r2.mood == MoodLabel.PLAYFUL