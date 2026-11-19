"""The persona catalogue: loading, validation, quality, and worldbooks."""

from __future__ import annotations

from herhim.catalog.loader import load_card_file, load_catalogue_cards, load_manifest
from herhim.catalog.quality import QualityFinding, QualityReport, analyse_catalogue
from herhim.catalog.registry import CATALOGUES, PersonaRegistry
from herhim.catalog.validate import validate_persona, validate_persona_card
from herhim.catalog.worldbook import scan_lorebook, scan_worldbook

__all__ = [
    "CATALOGUES",
    "PersonaRegistry",
    "QualityFinding",
    "QualityReport",
    "analyse_catalogue",
    "load_card_file",
    "load_catalogue_cards",
    "load_manifest",
    "scan_lorebook",
    "scan_worldbook",
    "validate_persona",
    "validate_persona_card",
]
