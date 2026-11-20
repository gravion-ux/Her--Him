"""Tests for the data models: affect, traits, relationship, voice, and cards."""

from __future__ import annotations

import pytest

from herhim.models.affect import AffectDelta, AffectVector
from herhim.models.card import (
    CARD_SPEC_VERSION,
    HERHIM_EXTENSION_KEY,
    REQUIRED_CARD_FIELDS,
    card_persona_id,
    strip_herhim_extension,
    validate_card_shape,
)
from herhim.models.events import DetectedEvent, EventKind
from herhim.models.persona import LoreEntry, Persona
from herhim.models.relationship import RelationshipStage, RelationshipState
from herhim.models.traits import TRAIT_NAMES, TraitProfile
from herhim.models.voice import VoiceBinding, VoiceClip, VoiceInfo, VoiceParams


class TestAffectVector:
    """Tests for the four-dimensional affect vector."""

    def test_defaults(self) -> None:
        """The default vector sits in the middle of every axis."""
        vector = AffectVector()
        assert vector.valence == 0.0
        assert vector.arousal == 0.5
        assert vector.assertion == 0.5
        assert vector.tenderness == 0.5

    def test_valence_is_clamped(self) -> None:
        """Valence cannot leave the -1.0 .. 1.0 range."""
        assert AffectVector(valence=5.0).valence == 1.0
        assert AffectVector(valence=-5.0).valence == -1.0

    def test_unit_axes_are_clamped(self) -> None:
        """Arousal, assertion, and tenderness cannot leave 0.0 .. 1.0."""
        vector = AffectVector(arousal=9.0, assertion=-3.0, tenderness=4.0)
        assert vector.arousal == 1.0
        assert vector.assertion == 0.0
        assert vector.tenderness == 1.0

    def test_addition(self) -> None:
        """Vectors add axis by axis, and the result is clamped."""
        left = AffectVector(valence=0.4, arousal=0.6, assertion=0.2, tenderness=0.8)
        right = AffectVector(valence=0.1, arousal=0.2, assertion=0.3, tenderness=0.4)
        total = left + right
        assert total.valence == pytest.approx(0.5)
        assert total.arousal == pytest.approx(0.8)
        assert total.assertion == pytest.approx(0.5)
        assert total.tenderness == 1.0

    def test_subtraction(self) -> None:
        """Vectors subtract axis by axis."""
        left = AffectVector(valence=0.4, arousal=0.6, assertion=0.2, tenderness=0.8)
        right = AffectVector(valence=0.1, arousal=0.2, assertion=0.3, tenderness=0.4)
        difference = left - right
        assert difference.valence == pytest.approx(0.3)
        assert difference.arousal == pytest.approx(0.4)
        assert difference.assertion == 0.0
        assert difference.tenderness == pytest.approx(0.4)

    def test_multiplication(self) -> None:
        """Scaling a vector scales every axis."""
        vector = AffectVector(valence=0.4, arousal=0.6, assertion=0.2, tenderness=0.8)
        doubled = vector * 2.0
        assert doubled.valence == pytest.approx(0.8)
        assert doubled.arousal == 1.0
        assert doubled.assertion == pytest.approx(0.4)
        assert doubled.tenderness == 1.0

    def test_to_dict_rounds(self) -> None:
        """The mapping is rounded to four decimals."""
        vector = AffectVector(valence=0.123456, arousal=0.5, assertion=0.5, tenderness=0.5)
        assert vector.to_dict()["valence"] == 0.1235

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown keys are ignored and missing keys take their defaults."""
        vector = AffectVector.from_dict({"valence": 0.25, "nonsense": 1.0})
        assert vector.valence == 0.25
        assert vector.arousal == 0.5

    def test_move_toward(self) -> None:
        """Moving toward a target closes part of the distance."""
        start = AffectVector(valence=0.0, arousal=0.5, assertion=0.5, tenderness=0.5)
        target = AffectVector(valence=1.0, arousal=0.5, assertion=0.5, tenderness=0.5)
        moved = start.move_toward(target, 0.25)
        assert moved.valence == pytest.approx(0.25)

    def test_move_toward_clamps_the_ratio(self) -> None:
        """A ratio above 1.0 lands exactly on the target."""
        start = AffectVector(valence=0.0, arousal=0.5, assertion=0.5, tenderness=0.5)
        target = AffectVector(valence=1.0, arousal=0.5, assertion=0.5, tenderness=0.5)
        assert start.move_toward(target, 4.0).valence == 1.0

    def test_shift_applies_a_delta(self) -> None:
        """Shifting adds a raw, unclamped delta."""
        vector = AffectVector(valence=0.5, arousal=0.5, assertion=0.5, tenderness=0.5)
        delta = AffectDelta(valence=-0.3, arousal=0.2, assertion=0.1, tenderness=-0.1)
        shifted = vector.shift(delta)
        assert shifted.valence == pytest.approx(0.2)
        assert shifted.arousal == pytest.approx(0.7)
        assert shifted.assertion == pytest.approx(0.6)
        assert shifted.tenderness == pytest.approx(0.4)


class TestAffectDelta:
    """Tests for the unclamped affect delta."""

    def test_delta_is_not_clamped(self) -> None:
        """A delta may leave the unit range on purpose."""
        delta = AffectDelta(valence=-2.0, arousal=3.0)
        assert delta.valence == -2.0
        assert delta.arousal == 3.0

    def test_addition(self) -> None:
        """Deltas add component by component."""
        total = AffectDelta(valence=0.1, arousal=0.2) + AffectDelta(valence=-0.3, arousal=0.1)
        assert total.valence == pytest.approx(-0.2)
        assert total.arousal == pytest.approx(0.3)

    def test_to_vector_clamps(self) -> None:
        """Converting to a vector clamps it."""
        vector = AffectDelta(valence=5.0, arousal=-1.0).to_vector()
        assert vector.valence == 1.0
        assert vector.arousal == 0.0

    def test_to_dict(self) -> None:
        """The mapping carries all four axes."""
        assert set(AffectDelta(valence=0.5).to_dict()) == {
            "valence",
            "arousal",
            "assertion",
            "tenderness",
        }


class TestTraitProfile:
    """Tests for the eleven-axis trait profile."""

    def test_defaults(self) -> None:
        """Every axis defaults to the midpoint."""
        profile = TraitProfile()
        assert profile.get("extraversion") == 0.5
        assert profile.get("humour") == 0.5

    def test_values_are_clamped(self) -> None:
        """Trait values cannot leave 0.0 .. 1.0."""
        profile = TraitProfile(extraversion=4.0, humour=-1.0)
        assert profile.get("extraversion") == 1.0
        assert profile.get("humour") == 0.0

    def test_get_unknown_axis(self) -> None:
        """An unknown axis reads as zero rather than raising."""
        assert TraitProfile().get("charisma") == 0.0

    def test_to_dict_covers_every_axis(self) -> None:
        """The mapping carries exactly the declared axes."""
        assert set(TraitProfile().to_dict()) == set(TRAIT_NAMES)

    def test_from_dict_defaults_missing_axes(self) -> None:
        """Missing axes take their default and unknown axes are ignored."""
        profile = TraitProfile.from_dict({"warmth": 0.9, "charisma": 1.0})
        assert profile.get("warmth") == 0.9
        assert profile.get("humour") == 0.5

    def test_dominant(self) -> None:
        """The dominant axis is the highest one."""
        profile = TraitProfile(warmth=0.9, humour=0.2)
        assert profile.dominant() == ("warmth", 0.9)


class TestRelationshipState:
    """Tests for the relationship state and its six stages."""

    def test_defaults(self) -> None:
        """A fresh relationship starts almost at zero."""
        state = RelationshipState()
        assert state.trust == 0.05
        assert state.intimacy == 0.03
        assert state.tension == 0.0

    def test_values_are_clamped(self) -> None:
        """Every axis lives in 0.0 .. 1.0."""
        state = RelationshipState(trust=5.0, intimacy=-1.0)
        assert state.trust == 1.0
        assert state.intimacy == 0.0

    def test_score(self) -> None:
        """The score weights closeness axes and subtracts tension."""
        state = RelationshipState(
            trust=1.0,
            intimacy=1.0,
            commitment=1.0,
            dependency=1.0,
            tension=0.0,
        )
        assert state.score() == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, RelationshipStage.STRANGER),
            (0.35, RelationshipStage.ACQUAINTED),
            (0.50, RelationshipStage.WARM),
            (0.60, RelationshipStage.FLIRTING),
            (0.70, RelationshipStage.FLIRTING),
            (0.80, RelationshipStage.INTIMATE),
            (0.90, RelationshipStage.BONDED),
            (1.0, RelationshipStage.BONDED),
        ],
    )
    def test_stage_thresholds(self, value: float, expected: RelationshipStage) -> None:
        """The stage follows the composite score."""
        state = RelationshipState(
            trust=value,
            intimacy=value,
            commitment=value,
            dependency=value,
        )
        assert state.stage() is expected

    def test_tension_caps_the_stage(self) -> None:
        """High tension demotes the stage by one level."""
        state = RelationshipState(
            trust=1.0,
            intimacy=1.0,
            commitment=1.0,
            dependency=1.0,
            tension=0.8,
        )
        assert state.stage() is RelationshipStage.INTIMATE

    def test_to_dict(self) -> None:
        """The mapping carries the derived stage and score."""
        data = RelationshipState().to_dict()
        assert data["stage"] == RelationshipStage.STRANGER.value
        assert "score" in data

    def test_from_dict(self) -> None:
        """Building from a mapping ignores unknown keys."""
        state = RelationshipState.from_dict({"trust": 0.4, "nonsense": 1.0})
        assert state.trust == 0.4
        assert state.intimacy == 0.03


class TestVoiceModels:
    """Tests for the voice value objects."""

    def test_params_to_dict(self) -> None:
        """The parameters round-trip through a mapping."""
        params = VoiceParams(voice_id="v1", emotion="happy", pitch=1.0, speed=1.2, volume=7.0)
        data = params.to_dict()
        assert data["voice_id"] == "v1"
        assert data["emotion"] == "happy"
        assert data["pitch"] == 1.0

    def test_info_to_dict(self) -> None:
        """A voice info carries its identifier, name, and locale."""
        assert VoiceInfo("v1", "Voice One", "en_US").to_dict() == {
            "voice_id": "v1",
            "voice_name": "Voice One",
            "locale": "en_US",
        }

    def test_clip_to_dict_without_params(self) -> None:
        """A clip without parameters serialises with a null payload."""
        assert VoiceClip("c1", "clip.wav").to_dict()["params"] is None

    def test_clip_to_dict_with_params(self) -> None:
        """A clip with parameters serialises the parameter mapping."""
        clip = VoiceClip("c1", "clip.wav", 120, VoiceParams("v1"))
        data = clip.to_dict()
        assert data["params"]["voice_id"] == "v1"
        assert data["duration_ms"] == 120

    def test_binding_to_dict(self) -> None:
        """The binding carries every field the card format needs."""
        data = VoiceBinding("v1").to_dict()
        assert data["voice_id"] == "v1"
        assert data["locale"] == "en_US"

    def test_binding_from_dict_defaults(self) -> None:
        """Missing fields take their defaults and unknown keys are ignored."""
        binding = VoiceBinding.from_dict({"voice_id": "v2", "nonsense": 1.0})
        assert binding.voice_id == "v2"
        assert binding.locale == "en_US"
        assert binding.volume == 6.0


class TestCardHelpers:
    """Tests for the Character Card shape helpers."""

    def test_non_mapping(self) -> None:
        """A non-mapping card is rejected outright."""
        assert validate_card_shape(["not", "a", "card"]) == ["card is not a mapping"]

    def test_missing_fields(self) -> None:
        """Every missing required field is reported."""
        problems = validate_card_shape({})
        assert len(problems) == len(REQUIRED_CARD_FIELDS)
        assert "missing field: name" in problems

    def test_empty_fields(self) -> None:
        """An empty required field is reported as empty."""
        card = dict.fromkeys(REQUIRED_CARD_FIELDS, "x")
        card["extensions"] = {}
        problems = validate_card_shape(card)
        assert "empty field: extensions" in problems

    def test_missing_extension_block(self) -> None:
        """A card without the herhim block is reported."""
        card = dict.fromkeys(REQUIRED_CARD_FIELDS, "x")
        card["extensions"] = {"other": {}}
        assert validate_card_shape(card) == ["extensions.herhim block is absent"]

    def test_valid_card(self, sample_persona: Persona) -> None:
        """A generated card has no structural problems."""
        assert validate_card_shape(sample_persona.to_dict()) == []

    def test_card_persona_id(self, sample_persona: Persona) -> None:
        """The persona id is read from the extension block."""
        assert card_persona_id(sample_persona.to_dict()) == "test-001"

    def test_card_persona_id_without_extensions(self) -> None:
        """A card without extensions carries no persona id."""
        assert card_persona_id({"name": "x"}) == ""

    def test_card_persona_id_with_bad_block(self) -> None:
        """A malformed extension block carries no persona id."""
        assert card_persona_id({"extensions": {HERHIM_EXTENSION_KEY: "nope"}}) == ""

    def test_strip_extension(self, sample_persona: Persona) -> None:
        """The herhim block is removed and the canonical fields survive."""
        stripped = strip_herhim_extension(sample_persona.to_dict())
        assert HERHIM_EXTENSION_KEY not in stripped["extensions"]
        assert stripped["name"] == sample_persona.display_name

    def test_strip_extension_keeps_other_blocks(self) -> None:
        """Extension blocks the package does not own are preserved."""
        card = {"extensions": {HERHIM_EXTENSION_KEY: {"persona_id": "x"}, "other": {"a": 1}}}
        stripped = strip_herhim_extension(card)
        assert stripped["extensions"] == {"other": {"a": 1}}

    def test_strip_extension_without_extensions(self) -> None:
        """A card without extensions is returned unchanged."""
        card = {"name": "x"}
        assert strip_herhim_extension(card) == card


class TestPersonaModel:
    """Tests for the persona model itself."""

    def test_to_card_shape(self, sample_persona: Persona) -> None:
        """The card declares the v3 spec version and the extension block."""
        card = sample_persona.to_card()
        assert card["spec_version"] == CARD_SPEC_VERSION
        assert HERHIM_EXTENSION_KEY in card["extensions"]
        assert card["extensions"][HERHIM_EXTENSION_KEY]["persona_id"] == "test-001"

    def test_round_trip(self, sample_persona: Persona) -> None:
        """A card survives a full round trip."""
        restored = Persona.from_card(sample_persona.to_card())
        assert restored.persona_id == sample_persona.persona_id
        assert restored.traits.to_dict() == sample_persona.traits.to_dict()
        assert restored.event_sensitivity == sample_persona.event_sensitivity
        assert restored.lorebook == sample_persona.lorebook

    def test_from_card_defaults(self) -> None:
        """An empty card yields an empty but well-formed persona."""
        persona = Persona.from_card({})
        assert persona.persona_id == ""
        assert persona.display_name == ""
        assert persona.creator == "gravion-ux"
        assert persona.lorebook == ()

    def test_lore_entry_from_dict(self) -> None:
        """Lore keys are lowercased and unknown keys are ignored."""
        entry = LoreEntry.from_dict({"keys": ["Alpha"], "content": "c", "nonsense": 1})
        assert entry.keys == ("alpha",)
        assert entry.position == "before_char"
        assert entry.constant is False

    def test_lore_entry_to_dict(self) -> None:
        """The lore entry serialises every field."""
        entry = LoreEntry(keys=("a",), content="c", position="after_char", constant=True)
        assert entry.to_dict() == {
            "keys": ["a"],
            "content": "c",
            "position": "after_char",
            "constant": True,
        }

    def test_sensitivity_for(self, sample_persona: Persona) -> None:
        """A declared sensitivity is returned verbatim."""
        assert sample_persona.sensitivity_for("REJECTION") == 1.4

    def test_sensitivity_default(self, sample_persona: Persona) -> None:
        """An undeclared event kind uses a neutral multiplier."""
        assert sample_persona.sensitivity_for("NOT_AN_EVENT") == 1.0

    def test_summary(self, sample_persona: Persona) -> None:
        """The summary carries the fields the catalogue listing needs."""
        summary = sample_persona.summary()
        assert summary["persona_id"] == "test-001"
        assert summary["locale"] == "en_US"
        assert summary["catalogue"] == "her"


class TestEvents:
    """Tests for the event kinds and detected events."""

    def test_event_kind_is_a_string(self) -> None:
        """Event kinds behave like plain strings."""
        assert EventKind.REJECTION == "REJECTION"
        assert isinstance(EventKind.REJECTION, str)

    def test_detected_event_weight(self) -> None:
        """The weight is intensity scaled by the event scale."""
        event = DetectedEvent(EventKind.TEASE, 0.8, ("tease",), 0.5)
        assert event.weight == pytest.approx(0.4)

    def test_detected_event_to_dict(self) -> None:
        """The detected event serialises every field."""
        event = DetectedEvent(EventKind.TEASE, 0.8, ("tease",))
        data = event.to_dict()
        assert data["kind"] == "TEASE"
        assert data["matched"] == ["tease"]
        assert data["scale"] == 1.0
