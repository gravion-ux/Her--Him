"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

from herhim.models.affect import AffectVector
from herhim.models.persona import LoreEntry, Persona
from herhim.models.relationship import RelationshipState
from herhim.models.traits import TraitProfile
from herhim.models.voice import VoiceBinding


@pytest.fixture
def sample_persona() -> Persona:
    """Create a sample persona for testing."""
    return Persona(
        persona_id="test-001",
        catalogue="her",
        gender="female",
        display_name="Test Persona",
        tagline="A test persona",
        description="A test description for unit testing",
        personality="Curious, intelligent, and occasionally sarcastic. Enjoys deep conversations.",
        scenario="Testing environment for Her & Him components",
        first_mes="Hello! I'm glad we're talking. What's on your mind?",
        mes_example="I like you. Let's keep talking.",
        system_prompt=(
            "You are Test Persona, a curious AI companion. You're curious, intelligent, "
            "and occasionally sarcastic. You enjoy deep conversations."
        ),
        affect_baseline=AffectVector(valence=0.15, arousal=0.40, assertion=0.35, tenderness=0.50),
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
        event_sensitivity={
            "REJECTION": 1.40,
            "PROVOCATION": 1.25,
            "AFFECTION": 1.10,
            "HOSTILITY": 1.15,
            "PRAISE": 1.05,
            "TEASE": 1.30,
            "CONFESSION": 1.20,
            "APOLOGY": 0.95,
            "COMPLIANCE": 0.85,
            "NEGLECT": 0.80,
            "JEALY_TRIGGER": 1.15,
            "BOUNDARY": 1.10,
        },
        relationship_baseline=RelationshipState(
            trust=0.32,
            intimacy=0.25,
            tension=0.03,
            jealousy=0.06,
            dependency=0.12,
            commitment=0.10,
        ),
        voice=VoiceBinding(voice_id="test-voice", voice_name="Test Voice", locale="en_US"),
        lorebook=(
            LoreEntry(
                keys=("test",),
                content="Test lore entry for unit testing.",
                position="before_char",
                constant=True,
            ),
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: tests that take more than a second")