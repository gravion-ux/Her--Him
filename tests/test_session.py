"""Tests for the conversation session and the catalogue surfaces."""

from __future__ import annotations

import pytest

from herhim.errors import PersonaError, ProviderError
from herhim.experience.her import CATALOGUE as HER_CATALOGUE
from herhim.experience.her import load_her_surface
from herhim.experience.him import CATALOGUE as HIM_CATALOGUE
from herhim.experience.him import load_him_surface
from herhim.experience.session import Session
from herhim.experience.surface import CatalogueSurface, load_surface
from herhim.guardrails.audit import AuditLog
from herhim.memory.store import MemoryStore
from herhim.models.affect import AffectVector
from herhim.models.persona import LoreEntry, Persona
from herhim.models.relationship import RelationshipState
from herhim.models.traits import TraitProfile
from herhim.models.voice import VoiceBinding, VoiceParams
from herhim.providers.offline import OfflineProvider


# Helper to create a persona without lorebook for the empty-lore branch
def _persona_no_lore(**overrides: object) -> Persona:
    """Build a minimal persona with no lorebook for testing empty lore branch."""
    base: dict[str, object] = {
        "persona_id": "test-no-lore",
        "catalogue": "her",
        "gender": "female",
        "display_name": "No Lore",
        "tagline": "A test without lore",
        "description": "No lore here",
        "personality": "Friendly",
        "scenario": "Test",
        "first_mes": "Hi",
        "mes_example": "Hello",
        "system_prompt": "You are friendly",
        "affect_baseline": AffectVector(
            valence=0.15, arousal=0.40, assertion=0.35, tenderness=0.50
        ),
        "traits": TraitProfile(
            extraversion=0.70, playfulness=0.65, bratty=0.55, stubbornness=0.45,
            warmth=0.55, jealousy=0.25, dominance=0.40, loyalty=0.50,
            patience=0.40, sensuality=0.45, humour=0.60,
        ),
        "event_sensitivity": {"REJECTION": 1.40, "COMPLIANCE": 0.85},
        "relationship_baseline": RelationshipState(
            trust=0.32, intimacy=0.25, tension=0.03,
            jealousy=0.06, dependency=0.12, commitment=0.10,
        ),
        "voice": VoiceBinding(voice_id="test-voice", voice_name="Test Voice", locale="en_US"),
        "lorebook": (),  # Empty - triggers the else branch in build_messages
    }
    base.update(overrides)
    return Persona(**base)  # type: ignore[arg-type]


class _FailingProvider:
    """A provider that always fails, for the error path."""

    name = "failing"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.9,
        max_tokens: int = 400,
        context: object = None,
    ) -> object:
        """Raise the way a real provider failure would."""
        raise ProviderError("the provider is down")


def _voice_persona(sample_persona: Persona) -> Persona:
    """Return a copy of the persona bound to a committed sample clip."""
    object.__setattr__(sample_persona, "voice", VoiceBinding(voice_id="sassy-maya"))
    return sample_persona


class TestSessionState:
    """Tests for the session's state handling."""

    def test_defaults_are_wired_up(self, sample_persona: Persona) -> None:
        """A session builds every collaborator it needs."""
        session = Session(sample_persona)
        assert isinstance(session.router.provider, OfflineProvider)
        assert isinstance(session.memory, MemoryStore)
        assert isinstance(session.audit, AuditLog)
        assert session.session_id

    def test_state_snapshot(self, sample_persona: Persona) -> None:
        """The state carries the session id and the turn count."""
        session = Session(sample_persona)
        state = session.state()
        assert state["session_id"] == session.session_id
        assert state["turns"] == 0
        assert "affect" in state

    def test_reset(self, sample_persona: Persona) -> None:
        """Resetting clears the memory and records the reset."""
        session = Session(sample_persona)
        session.memory.add_fact("a fact")
        session.reset()
        assert session.memory.records == []
        assert session.audit.by_kind("session.reset")

    def test_relationship_summary(self, sample_persona: Persona) -> None:
        """The relationship is rendered into prompt text."""
        session = Session(sample_persona)
        summary = session._relationship_summary()
        assert "Relationship:" in summary
        assert "trust" in summary


class TestSessionPrompts:
    """Tests for the prompt the session assembles."""

    def test_lore_and_relationship_reach_the_prompt(self, sample_persona: Persona) -> None:
        """The lorebook and the relationship are part of the system prompt."""
        session = Session(sample_persona)
        rationale = session.engine.apply_turn("No.")
        messages = session.build_messages("No.", rationale)
        system = messages[0]["content"]
        assert "WHAT YOU KNOW" in system
        assert "Relationship:" in system
        assert "CURRENT STATE" in system

    def test_the_user_turn_is_last(self, sample_persona: Persona) -> None:
        """The user's turn is the final message."""
        session = Session(sample_persona)
        rationale = session.engine.apply_turn("hello")
        messages = session.build_messages("hello", rationale)
        assert messages[-1] == {"role": "user", "content": "hello"}

    def test_recent_turns_reach_the_prompt(self, sample_persona: Persona) -> None:
        """Earlier turns are replayed to the provider in order."""
        session = Session(sample_persona)
        session.memory.add_turn("user", "I live in Berlin", source="user")
        session.memory.add_turn("assistant", "Berlin is lovely", source="test-001")
        rationale = session.engine.apply_turn("hello")
        messages = session.build_messages("hello", rationale)
        assert {"role": "user", "content": "I live in Berlin"} in messages
        assert {"role": "assistant", "content": "Berlin is lovely"} in messages

    def test_remembered_facts_reach_the_prompt(self, sample_persona: Persona) -> None:
        """A recalled fact appears in the prompt with its source."""
        session = Session(sample_persona)
        session.memory.add_fact("The user lives in Berlin", source="profile")
        rationale = session.engine.apply_turn("Berlin")
        system = session.build_messages("Berlin", rationale)[0]["content"]
        assert "WHAT YOU REMEMBER" in system
        assert "Berlin" in system

    def test_no_memory_means_no_memory_block(self, sample_persona: Persona) -> None:
        """An empty memory contributes nothing to the prompt."""
        session = Session(sample_persona)
        assert session._memory_block("anything") == ""


class TestSessionTurn:
    """Tests for the turn itself."""

    def test_turn_produces_a_result(self, sample_persona: Persona) -> None:
        """A turn returns the reply, the mood, and the rationale."""
        session = Session(sample_persona)
        result = session.turn("No, I don't want that")
        assert result.persona_id == sample_persona.persona_id
        assert result.text
        assert result.mood
        assert result.segments
        assert result.voice is not None

    def test_turn_serialises(self, sample_persona: Persona) -> None:
        """The turn result serialises to a JSON-friendly mapping."""
        session = Session(sample_persona)
        data = session.turn("No.").to_dict()
        assert data["persona_id"] == sample_persona.persona_id
        assert data["voice"]["voice_id"] == sample_persona.voice.voice_id
        assert data["segments"][0]["text"]

    def test_turn_records_the_exchange(self, sample_persona: Persona) -> None:
        """The turn is written to memory as two verbatim records."""
        session = Session(sample_persona)
        session.turn("No.")
        assert len(session.memory.recent()) == 2
        assert session.memory.episodes()

    def test_a_stage_change_is_audited(self, sample_persona: Persona) -> None:
        """A relationship stage change is written to the audit log."""
        session = Session(sample_persona)
        session.turn("No, I don't want that")
        assert session.audit.by_kind("relationship.stage")

    def test_a_provider_failure_is_audited_and_raised(self, sample_persona: Persona) -> None:
        """A provider failure is recorded and re-raised."""
        session = Session(sample_persona, provider=_FailingProvider())
        with pytest.raises(ProviderError):
            session.turn("No.")
        assert session.audit.by_kind("provider.failure")

    def test_segments_split_on_sentences(self, sample_persona: Persona) -> None:
        """A reply is split into speakable sentences."""
        session = Session(sample_persona)
        segments = session._segments("One. Two! Three?", VoiceParams("v1"))
        assert len(segments) == 3
        assert segments[0]["text"] == "One."

    def test_an_empty_reply_has_no_segments(self, sample_persona: Persona) -> None:
        """An empty reply produces no segments."""
        session = Session(sample_persona)
        assert session._segments("   ", VoiceParams("v1")) == []

    def test_speak_renders_a_clip(self, sample_persona: Persona) -> None:
        """Speaking returns the clip bound to the persona's voice."""
        session = Session(_voice_persona(sample_persona))
        clip = session.speak("Hello there")
        assert clip.clip_id == "sassy-maya"
        assert clip.transcript

    def test_turn_state_is_a_dict(self, sample_persona: Persona) -> None:
        """Regression: result.state is a dict, not a method."""
        session = Session(sample_persona)
        result = session.turn("Hello")
        assert isinstance(result.state, dict)
        assert "stage" in result.state

    def test_turn_without_loreblock(self) -> None:
        """A persona with no lorebook has no WHAT YOU KNOW block."""
        persona = _persona_no_lore()
        session = Session(persona)
        rationale = session.engine.apply_turn("hello")
        messages = session.build_messages("hello", rationale)
        system = messages[0]["content"]
        assert "WHAT YOU KNOW" not in system

    def test_turn_without_stage_change(self) -> None:
        """A turn that stays in the same stage doesn't audit a change."""
        # Create a persona already at BONDED stage
        persona = Persona(
            persona_id="test-bonded",
            catalogue="her",
            gender="female",
            display_name="Bonded",
            tagline="Already bonded",
            description="High intimacy test",
            personality="Friendly",
            scenario="Test",
            first_mes="Hi",
            mes_example="Hello",
            system_prompt="You are friendly",
            affect_baseline=AffectVector(
                valence=0.5, arousal=0.4, assertion=0.3, tenderness=0.5
            ),
            traits=TraitProfile(
                extraversion=0.70, playfulness=0.65, bratty=0.55, stubbornness=0.45,
                warmth=0.55, jealousy=0.25, dominance=0.40, loyalty=0.50,
                patience=0.40, sensuality=0.45, humour=0.60,
            ),
            relationship_baseline=RelationshipState(
                trust=0.99, intimacy=0.99, tension=0.0,
                jealousy=0.0, dependency=0.99, commitment=0.99
            ),
            voice=VoiceBinding(voice_id="test-voice", voice_name="Test Voice", locale="en_US"),
            lorebook=(LoreEntry(keys=("test",), content="Test lore.", constant=True),),
        )
        session = Session(persona)
        # Apply a few turns without events that would change stage
        session.turn("Hello")
        # Check that no relationship.stage events were recorded (stage didn't change)
        stage_events = [e for e in session.audit.events if e.kind == "relationship.stage"]
        assert stage_events == []


class TestSurfaces:
    """Tests for the catalogue surfaces."""

    def test_her_surface(self) -> None:
        """The her surface loads from its manifest."""
        surface = load_her_surface()
        assert surface.catalogue == "her"
        assert surface.title
        assert surface.default_persona
        assert surface.theme

    def test_him_surface(self) -> None:
        """The him surface loads from its manifest."""
        surface = load_him_surface()
        assert surface.catalogue == "him"
        assert surface.default_persona

    def test_catalogue_names(self) -> None:
        """Each surface module declares its catalogue."""
        assert HER_CATALOGUE == "her"
        assert HIM_CATALOGUE == "him"

    def test_surface_serialises(self) -> None:
        """A surface serialises every field it carries."""
        data = load_her_surface().to_dict()
        assert data["catalogue"] == "her"
        assert "theme" in data
        assert "onboarding" in data

    def test_a_non_mapping_theme_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A theme that is not an object is reported as a malformed manifest."""
        monkeypatch.setattr(
            "herhim.experience.surface.load_manifest",
            lambda catalogue: {"catalogue": catalogue, "theme": "nonsense"},
        )
        with pytest.raises(PersonaError, match="malformed"):
            load_surface("her")

    def test_a_missing_title_falls_back_to_the_catalogue_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest without a title uses the catalogue name."""
        monkeypatch.setattr("herhim.experience.surface.load_manifest", lambda catalogue: {})
        assert load_surface("her").title == "Her"

    def test_a_malformed_manifest_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A manifest that cannot be rendered raises a persona error."""

        class _Unstringable:
            def __str__(self) -> str:
                raise ValueError("nope")

        monkeypatch.setattr(
            "herhim.experience.surface.load_manifest",
            lambda catalogue: {"title": _Unstringable()},
        )
        with pytest.raises(PersonaError, match="malformed"):
            load_surface("her")

    def test_surface_type(self) -> None:
        """The loader returns the documented type."""
        assert isinstance(load_her_surface(), CatalogueSurface)
