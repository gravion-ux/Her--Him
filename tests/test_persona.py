"""Tests for the Persona model and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from herhim.catalog.loader import load_card_file
from herhim.catalog.validate import validate_persona, validate_persona_card
from herhim.errors import PersonaError
from herhim.models.persona import Persona


class TestPersonaValidation:
    """Tests for persona validation."""

    def test_valid_persona(self, sample_persona: Persona) -> None:
        """A valid persona passes all checks."""
        problems = validate_persona(sample_persona)
        assert not problems, f"Unexpected problems: {problems}"

    def test_missing_persona_id(self, sample_persona: Persona) -> None:
        """Missing persona_id is caught."""
        # Can't modify frozen dataclass, so test via card validation
        card = sample_persona.to_dict()
        card["extensions"]["herhim"]["persona_id"] = ""
        problems = validate_persona(Persona.from_card(card))
        assert "persona_id is empty" in problems

    def test_missing_lorebook(self, sample_persona: Persona) -> None:
        """Empty lorebook is caught."""
        card = sample_persona.to_dict()
        card["extensions"]["herhim"]["lorebook"] = {"entries": []}
        problems = validate_persona_card(card)
        assert any("lorebook" in p for p in problems)

    def test_invalid_gender(self, sample_persona: Persona) -> None:
        """Invalid gender is caught."""
        card = sample_persona.to_dict()
        card["extensions"]["herhim"]["gender"] = "other"
        problems = validate_persona(Persona.from_card(card))
        assert any("unknown gender" in p for p in problems)

    def test_out_of_range_trait(self, sample_persona: Persona) -> None:
        """Out of range trait values are clamped."""
        card = sample_persona.to_dict()
        card["extensions"]["herhim"]["traits"]["extraversion"] = 1.5
        restored = Persona.from_card(card)
        # TraitProfile clamps values to 0.0-1.0 range in __post_init__
        assert restored.traits.get("extraversion") == 1.0, "Trait should be clamped to 1.0"


class TestPersonaCardRoundTrip:
    """Tests for Persona serialization/deserialization."""

    def test_round_trip(self, sample_persona: Persona) -> None:
        """Persona can be serialized and deserialized without loss."""
        original = sample_persona
        card = original.to_dict()
        restored = Persona.from_card(card)

        assert restored.persona_id == original.persona_id
        assert restored.display_name == original.display_name
        assert restored.catalogue == original.catalogue
        assert restored.gender == original.gender
        assert restored.traits.to_dict() == original.traits.to_dict()
        assert restored.voice.voice_id == original.voice.voice_id

    def test_card_shape_validation(self, sample_persona: Persona) -> None:
        """Card has all required fields."""
        card = sample_persona.to_dict()
        problems = validate_persona_card(card)
        assert not problems, f"Card validation failed: {problems}"


class TestCatalogueLoading:
    """Tests for loading persona cards from disk."""

    def test_load_her_catalogue(self) -> None:
        """Can load the her catalogue."""
        from herhim.catalog.registry import PersonaRegistry

        registry = PersonaRegistry.load(("her",))
        assert len(registry) >= 24

    def test_load_him_catalogue(self) -> None:
        """Can load the him catalogue."""
        from herhim.catalog.registry import PersonaRegistry

        registry = PersonaRegistry.load(("him",))
        assert len(registry) >= 24

    def test_load_both_catalogues(self) -> None:
        """Can load both catalogues together."""
        from herhim.catalog.registry import PersonaRegistry

        registry = PersonaRegistry.load(("her", "him"))
        assert len(registry) >= 48

    def test_persona_not_found(self) -> None:
        """Looking up non-existent persona raises error."""
        from herhim.catalog.registry import PersonaRegistry

        registry = PersonaRegistry.load()
        with pytest.raises(PersonaError):
            registry.get("nonexistent-id")


class TestManifestLoading:
    """Tests for manifest file loading."""

    def test_load_her_manifest(self) -> None:
        """Can load the her manifest."""
        from herhim.catalog.loader import load_manifest

        manifest = load_manifest("her")
        assert manifest["catalogue"] == "her"
        assert "title" in manifest
        assert "persona_count" in manifest

    def test_load_him_manifest(self) -> None:
        """Can load the him manifest."""
        from herhim.catalog.loader import load_manifest

        manifest = load_manifest("him")
        assert manifest["catalogue"] == "him"
        assert "title" in manifest