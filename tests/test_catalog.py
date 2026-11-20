"""Tests for catalogue loading, validation, the registry, and quality rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from herhim.catalog.loader import load_card_file, load_catalogue_cards, load_manifest
from herhim.catalog.quality import (
    BALANCE_LIMIT,
    CONTRAST_MIN_DISTANCE,
    COVERAGE_MIN_SENSITIVITY,
    QualityFinding,
    QualityReport,
    analyse_catalogue,
)
from herhim.catalog.registry import CATALOGUES, PersonaRegistry
from herhim.catalog.validate import validate_persona, validate_persona_card
from herhim.catalog.worldbook import _matches, scan_lorebook, scan_worldbook
from herhim.errors import PersonaError
from herhim.models.events import EventKind
from herhim.models.persona import LoreEntry, Persona
from herhim.models.traits import TRAIT_NAMES, TraitProfile
from herhim.models.voice import VoiceBinding
from herhim.paths import catalogue_dir


def _traits(*high: str) -> TraitProfile:
    """Build a trait profile with the named axes at 1.0 and the rest at 0.0."""
    values = {name: (1.0 if name in high else 0.0) for name in TRAIT_NAMES}
    return TraitProfile(**values)


def _persona(persona_id: str, traits: TraitProfile, **overrides: object) -> Persona:
    """Build a minimal but valid persona for catalogue-level tests."""
    fields: dict[str, object] = {
        "persona_id": persona_id,
        "catalogue": "her",
        "gender": "female",
        "display_name": persona_id,
        "tagline": "A tagline",
        "description": "A description",
        "personality": "A personality",
        "scenario": "A scenario",
        "first_mes": "Hello",
        "mes_example": "Hi",
        "system_prompt": "You are a test persona.",
        "traits": traits,
        "lorebook": (LoreEntry(keys=("x",), content="Some lore."),),
        "voice": VoiceBinding(voice_id="v1"),
    }
    fields.update(overrides)
    return Persona(**fields)  # type: ignore[arg-type]


class TestCardLoading:
    """Tests for reading persona cards off disk."""

    def test_load_a_bundled_card(self) -> None:
        """A bundled card loads into a persona."""
        path = catalogue_dir("her") / "her-001-sassy.json"
        persona = load_card_file(path)
        assert persona.persona_id == "her-001-sassy"
        assert persona.catalogue == "her"

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        """A card that does not exist raises a persona error."""
        with pytest.raises(PersonaError):
            load_card_file(tmp_path / "nope.json")

    def test_unreadable_json_is_reported(self, tmp_path: Path) -> None:
        """A card that is not valid JSON raises a persona error."""
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(PersonaError):
            load_card_file(path)

    def test_malformed_card_is_reported(self, tmp_path: Path) -> None:
        """A card missing required fields raises a persona error."""
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"name": "x"}), encoding="utf-8")
        with pytest.raises(PersonaError, match="malformed"):
            load_card_file(path)


class TestCatalogueLoading:
    """Tests for loading whole catalogues."""

    def test_her_catalogue_loads(self) -> None:
        """The her catalogue loads and is ordered by persona id."""
        personas = load_catalogue_cards("her")
        assert len(personas) > 0
        assert [p.persona_id for p in personas] == sorted(p.persona_id for p in personas)

    def test_him_catalogue_loads(self) -> None:
        """The him catalogue loads."""
        assert len(load_catalogue_cards("him")) > 0

    def test_missing_directory_is_reported(self, tmp_path: Path) -> None:
        """A catalogue directory that does not exist raises a persona error."""
        with pytest.raises(PersonaError, match="not found"):
            load_catalogue_cards("her", root=tmp_path / "nope")

    def test_manifest_is_not_a_persona(self, tmp_path: Path) -> None:
        """The manifest file is skipped when loading personas."""
        source = catalogue_dir("her") / "her-001-sassy.json"
        (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        (tmp_path / "manifest.json").write_text(json.dumps({"catalogue": "her"}), encoding="utf-8")
        assert len(load_catalogue_cards("her", root=tmp_path)) == 1


class TestManifestLoading:
    """Tests for reading catalogue manifests."""

    def test_manifest_loads(self) -> None:
        """The her manifest loads as a mapping."""
        manifest = load_manifest("her")
        assert manifest["catalogue"] == "her"

    def test_missing_manifest_is_reported(self, tmp_path: Path) -> None:
        """A manifest that does not exist raises a persona error."""
        with pytest.raises(PersonaError, match="not found"):
            load_manifest("her", root=tmp_path)

    def test_unreadable_manifest_is_reported(self, tmp_path: Path) -> None:
        """A manifest that is not valid JSON raises a persona error."""
        (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(PersonaError, match="cannot read"):
            load_manifest("her", root=tmp_path)

    def test_non_mapping_manifest_is_reported(self, tmp_path: Path) -> None:
        """A manifest that is not an object raises a persona error."""
        (tmp_path / "manifest.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(PersonaError, match="not a mapping"):
            load_manifest("her", root=tmp_path)


class TestPersonaRegistry:
    """Tests for the persona registry."""

    def test_loads_both_catalogues(self) -> None:
        """The default registry holds both catalogues."""
        registry = PersonaRegistry.load()
        assert len(registry) == len(load_catalogue_cards("her")) + len(
            load_catalogue_cards("him")
        )

    def test_loads_one_catalogue(self) -> None:
        """A registry can be limited to one catalogue."""
        registry = PersonaRegistry.load(("her",))
        assert registry.catalogues() == ["her"]

    def test_catalogue_names(self) -> None:
        """Both catalogue names are declared."""
        assert CATALOGUES == ("her", "him")

    def test_contains(self) -> None:
        """Membership is checked by persona id."""
        registry = PersonaRegistry.load(("her",))
        assert "her-001-sassy" in registry
        assert "nope" not in registry

    def test_unknown_persona_is_reported(self) -> None:
        """Looking up an unknown id raises a persona error."""
        with pytest.raises(PersonaError):
            PersonaRegistry.load().get("nope")

    def test_add_replaces_by_id(self) -> None:
        """Adding the same id twice keeps one entry."""
        persona = _persona("p-1", TraitProfile())
        registry = PersonaRegistry([persona])
        registry.add(_persona("p-1", TraitProfile(warmth=0.9)))
        assert len(registry) == 1
        assert registry.get("p-1").traits.get("warmth") == 0.9

    def test_all_is_ordered(self) -> None:
        """Every persona comes back ordered by id."""
        registry = PersonaRegistry.load(("her",))
        ids = [persona.persona_id for persona in registry.all()]
        assert ids == sorted(ids)

    def test_by_catalogue(self) -> None:
        """A catalogue filter returns only that catalogue."""
        registry = PersonaRegistry.load()
        assert all(p.catalogue == "him" for p in registry.by_catalogue("him"))

    def test_manifest_is_reachable(self) -> None:
        """The registry exposes the catalogue manifests."""
        assert PersonaRegistry.load().manifest("her")["catalogue"] == "her"

    def test_summaries(self) -> None:
        """Summaries are produced for the whole registry or one catalogue."""
        registry = PersonaRegistry.load()
        assert len(registry.summaries()) == len(registry)
        assert len(registry.summaries("her")) == len(registry.by_catalogue("her"))


class TestPersonaValidation:
    """Tests for the structural persona validator."""

    def test_valid_persona(self, sample_persona: Persona) -> None:
        """A complete persona passes every rule."""
        assert validate_persona(sample_persona) == []

    def test_missing_persona_id(self, sample_persona: Persona) -> None:
        """An empty persona id is caught."""
        object.__setattr__(sample_persona, "persona_id", "")
        assert "persona_id is empty" in validate_persona(sample_persona)

    def test_unknown_catalogue(self, sample_persona: Persona) -> None:
        """A catalogue outside her and him is caught."""
        object.__setattr__(sample_persona, "catalogue", "other")
        assert any("unknown catalogue" in p for p in validate_persona(sample_persona))

    def test_unknown_gender(self, sample_persona: Persona) -> None:
        """A gender outside female and male is caught."""
        object.__setattr__(sample_persona, "gender", "other")
        assert any("unknown gender" in p for p in validate_persona(sample_persona))

    def test_blank_text_field(self, sample_persona: Persona) -> None:
        """A whitespace-only free-text field is caught."""
        object.__setattr__(sample_persona, "tagline", "   ")
        assert "tagline is empty" in validate_persona(sample_persona)

    def test_empty_lorebook(self, sample_persona: Persona) -> None:
        """A persona without lorebook entries is caught."""
        object.__setattr__(sample_persona, "lorebook", ())
        assert "lorebook is empty" in validate_persona(sample_persona)

    def test_lorebook_entry_without_keys(self, sample_persona: Persona) -> None:
        """A lorebook entry without trigger keys is caught."""
        entry = LoreEntry(keys=(), content="Some lore.")
        object.__setattr__(sample_persona, "lorebook", (entry,))
        assert any("no keys" in p for p in validate_persona(sample_persona))

    def test_lorebook_entry_without_content(self, sample_persona: Persona) -> None:
        """A lorebook entry without content is caught."""
        entry = LoreEntry(keys=("x",), content="   ")
        object.__setattr__(sample_persona, "lorebook", (entry,))
        assert "lorebook entry has empty content" in validate_persona(sample_persona)

    def test_out_of_range_trait(self, sample_persona: Persona) -> None:
        """A trait outside 0.0 .. 1.0 is caught."""
        object.__setattr__(sample_persona.traits, "extraversion", 1.5)
        assert any("out of range" in p for p in validate_persona(sample_persona))

    def test_out_of_range_affect_baseline(self, sample_persona: Persona) -> None:
        """An affect baseline outside its range is caught."""
        object.__setattr__(sample_persona.affect_baseline, "valence", 2.0)
        assert any("valence out of range" in p for p in validate_persona(sample_persona))

    def test_out_of_range_affect_unit_axis(self, sample_persona: Persona) -> None:
        """An affect unit axis outside its range is caught."""
        object.__setattr__(sample_persona.affect_baseline, "arousal", 1.5)
        assert any("arousal out of range" in p for p in validate_persona(sample_persona))

    def test_unknown_sensitivity_event(self, sample_persona: Persona) -> None:
        """A sensitivity for an unknown event kind is caught."""
        object.__setattr__(sample_persona, "event_sensitivity", {"NOT_AN_EVENT": 1.0})
        assert any("unknown event kind" in p for p in validate_persona(sample_persona))

    def test_out_of_range_sensitivity(self, sample_persona: Persona) -> None:
        """A sensitivity multiplier above the ceiling is caught."""
        object.__setattr__(sample_persona, "event_sensitivity", {"REJECTION": 9.0})
        assert any("out of range" in p for p in validate_persona(sample_persona))

    def test_missing_voice_id(self, sample_persona: Persona) -> None:
        """A voice binding without an id is caught."""
        object.__setattr__(sample_persona.voice, "voice_id", "")
        assert "voice binding has no voice_id" in validate_persona(sample_persona)

    def test_missing_voice_locale(self, sample_persona: Persona) -> None:
        """A voice binding without a locale is caught."""
        object.__setattr__(sample_persona.voice, "locale", "")
        assert "voice binding has no locale" in validate_persona(sample_persona)

    def test_card_validation_stops_at_shape_problems(self) -> None:
        """A structurally broken card is not validated any further."""
        problems = validate_persona_card({"name": "x"})
        assert any("missing field" in p for p in problems)

    def test_card_validation_of_a_good_card(self, sample_persona: Persona) -> None:
        """A good card passes both the shape and the persona rules."""
        assert validate_persona_card(sample_persona.to_dict()) == []


class TestQualityRules:
    """Tests for the catalogue-level quality rules."""

    def test_empty_catalogue_is_critical(self) -> None:
        """A catalogue without personas is a critical finding."""
        report = analyse_catalogue("her", [])
        assert report.ok is False
        assert report.findings[0].rule == "CAT-VAL-001"
        assert report.findings[0].severity == "critical"

    def test_dominant_archetype_is_flagged(self) -> None:
        """Too many personas sharing one dominant trait is flagged."""
        personas = [_persona(f"p-{i}", TraitProfile(extraversion=0.9)) for i in range(3)]
        report = analyse_catalogue("her", personas)
        balance = [f for f in report.findings if f.rule == "CAT-BAL-001"]
        assert balance
        assert balance[0].severity == "high"

    def test_similar_personas_are_flagged(self) -> None:
        """Two personas that are effectively the same are flagged."""
        personas = [
            _persona("p-1", TraitProfile()),
            _persona("p-2", TraitProfile()),
        ]
        report = analyse_catalogue("her", personas)
        contrast = [f for f in report.findings if f.rule == "CAT-CON-001"]
        assert contrast
        assert str(CONTRAST_MIN_DISTANCE) in contrast[0].message

    def test_missing_coverage_is_flagged(self) -> None:
        """An event kind nobody is sensitive to is flagged."""
        report = analyse_catalogue("her", [_persona("p-1", TraitProfile())])
        coverage = [f for f in report.findings if f.rule == "CAT-COV-001"]
        assert len(coverage) == 12
        assert COVERAGE_MIN_SENSITIVITY > 1.0

    def test_a_good_catalogue_passes(self) -> None:
        """A catalogue that satisfies every rule reports no problems."""
        personas = [
            _persona(
                "p-1",
                _traits("extraversion", "playfulness", "bratty", "stubbornness", "warmth"),
            ),
            _persona(
                "p-2",
                _traits("jealousy", "dominance", "loyalty", "patience", "sensuality"),
            ),
            _persona(
                "p-3",
                _traits("playfulness", "stubbornness", "jealousy", "dominance"),
                event_sensitivity={event.value: 1.5 for event in EventKind},
            ),
        ]
        report = analyse_catalogue("her", personas)
        assert report.ok is True
        assert report.findings == ()

    def test_balance_limit_is_derived_from_the_catalogue_size(self) -> None:
        """The balance limit scales with the number of personas."""
        personas = [_persona(f"p-{i}", TraitProfile(extraversion=0.9)) for i in range(4)]
        report = analyse_catalogue("her", personas)
        balance = [f for f in report.findings if f.rule == "CAT-BAL-001"]
        assert balance
        assert BALANCE_LIMIT * 4 < 4

    def test_finding_serialises(self) -> None:
        """A finding serialises every field."""
        finding = QualityFinding("CAT-VAL-001", "critical", "her", "empty")
        assert finding.to_dict()["rule"] == "CAT-VAL-001"

    def test_report_serialises(self) -> None:
        """A report serialises its findings and its verdict."""
        report = QualityReport("her", (QualityFinding("R", "low", "her", "m"),))
        data = report.to_dict()
        assert data["catalogue"] == "her"
        assert data["ok"] is True

    def test_bundled_catalogues_have_no_critical_findings(self) -> None:
        """The shipped catalogues never fail a critical rule."""
        registry = PersonaRegistry.load()
        for catalogue in CATALOGUES:
            report = analyse_catalogue(catalogue, registry.by_catalogue(catalogue))
            critical = [f for f in report.findings if f.severity == "critical"]
            assert critical == []

    def test_bundled_personas_validate(self) -> None:
        """Every shipped persona satisfies the structural validator."""
        registry = PersonaRegistry.load()
        for persona in registry.all():
            assert validate_persona(persona) == [], persona.persona_id


class TestWorldbook:
    """Tests for keyword-triggered worldbook scanning."""

    def test_constant_entries_always_come_first(self) -> None:
        """A constant entry is returned ahead of the triggered ones."""
        entries = (
            LoreEntry(keys=("match",), content="Triggered."),
            LoreEntry(keys=("always",), content="Constant.", constant=True),
        )
        result = scan_lorebook(entries, "please match this")
        assert result[0].content == "Constant."
        assert len(result) == 2

    def test_single_word_key_matches(self) -> None:
        """A single-word key matches as a whole word."""
        entries = (LoreEntry(keys=("moon",), content="Lore."),)
        assert len(scan_lorebook(entries, "I saw the moon tonight")) == 1

    def test_single_word_key_does_not_match_inside_a_word(self) -> None:
        """A single-word key does not match inside a longer word."""
        entries = (LoreEntry(keys=("moon",), content="Lore."),)
        assert scan_lorebook(entries, "moonshine") == []

    def test_multi_word_key_matches_as_a_phrase(self) -> None:
        """A multi-word key matches as a phrase."""
        entries = (LoreEntry(keys=("old town",), content="Lore."),)
        assert len(scan_lorebook(entries, "I walked through the old town")) == 1

    def test_empty_key_is_ignored(self) -> None:
        """An empty key never triggers."""
        entries = (LoreEntry(keys=("",), content="Lore."),)
        assert scan_lorebook(entries, "anything") == []

    def test_no_match_returns_nothing(self) -> None:
        """A turn that matches no key returns nothing."""
        entries = (LoreEntry(keys=("moon",), content="Lore."),)
        assert scan_lorebook(entries, "nothing relevant") == []

    def test_worldbook_keeps_constant_entries(self) -> None:
        """The world book always includes its constant entries."""
        entries = (LoreEntry(keys=("x",), content="Always.", constant=True),)
        assert len(scan_worldbook(entries, "anything")) == 1

    def test_worldbook_skips_already_supplied_entries(self) -> None:
        """An entry the character book already covered is skipped."""
        entries = (LoreEntry(keys=("moon",), content="Lore."),)
        assert scan_worldbook(entries, "the moon", already_supplied={"moon"}) == []

    def test_worldbook_keeps_new_entries(self) -> None:
        """An entry that adds something new is returned."""
        entries = (LoreEntry(keys=("moon",), content="Lore."),)
        assert len(scan_worldbook(entries, "the moon", already_supplied={"sun"})) == 1

    def test_worldbook_ignores_unmatched_entries(self) -> None:
        """An entry that matches nothing is not returned."""
        entries = (LoreEntry(keys=("moon",), content="Lore."),)
        assert scan_worldbook(entries, "nothing relevant") == []

    def test_a_constant_entry_matches_anything(self) -> None:
        """A constant entry matches without any keywords."""
        entry = LoreEntry(keys=("moon",), content="Always.", constant=True)
        assert _matches(entry, set(), "nothing relevant") is True
