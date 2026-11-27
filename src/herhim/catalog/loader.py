"""Load persona cards from the bundled catalogue directories."""

from __future__ import annotations

import json
from pathlib import Path

from herhim.errors import PersonaError
from herhim.models.card import validate_card_shape
from herhim.models.persona import Persona
from herhim.paths import catalogue_dir

__all__ = ["load_card_file", "load_catalogue_cards", "load_manifest"]


def load_card_file(path: Path) -> Persona:
    """Load one persona card from a JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersonaError(f"cannot read card {path.name}: {exc}") from exc
    problems = validate_card_shape(payload)
    if problems:
        raise PersonaError(f"card {path.name} is malformed: {'; '.join(problems)}")
    return Persona.from_card(payload)


def load_catalogue_cards(catalogue: str, root: Path | None = None) -> list[Persona]:
    """Load every persona card in one catalogue, ordered by persona id."""
    base = root if root is not None else catalogue_dir(catalogue)
    if not base.exists():
        raise PersonaError(f"catalogue directory not found: {base}")
    personas: list[Persona] = []
    for path in sorted(base.glob("*.json")):
        if path.name == "manifest.json":
            continue
        personas.append(load_card_file(path))
    seen: set[str] = set()
    for persona in personas:
        if persona.persona_id in seen:
            raise PersonaError(f"duplicate persona id in catalogue: {persona.persona_id}")
        seen.add(persona.persona_id)
    return sorted(personas, key=lambda persona: persona.persona_id)


def load_manifest(catalogue: str, root: Path | None = None) -> dict[str, object]:
    """Load a catalogue's manifest."""
    base = root if root is not None else catalogue_dir(catalogue)
    path = base / "manifest.json"
    if not path.exists():
        raise PersonaError(f"catalogue manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersonaError(f"cannot read manifest {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PersonaError(f"catalogue manifest {path.name} is not a mapping")
    return payload
