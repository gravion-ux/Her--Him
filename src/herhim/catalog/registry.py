"""The persona registry."""

from __future__ import annotations

from herhim.catalog.loader import load_catalogue_cards, load_manifest
from herhim.errors import PersonaError
from herhim.models.persona import Persona

__all__ = ["CATALOGUES", "PersonaRegistry"]

#: The two parallel catalogues.
CATALOGUES: tuple[str, ...] = ("her", "him")


class PersonaRegistry:
    """Holds every persona and answers lookups against it."""

    def __init__(self, personas: list[Persona] | None = None) -> None:
        self._personas: dict[str, Persona] = {}
        for persona in personas or []:
            self._personas[persona.persona_id] = persona

    @classmethod
    def load(cls, catalogues: tuple[str, ...] = CATALOGUES) -> PersonaRegistry:
        """Load every persona from the bundled catalogues."""
        personas: list[Persona] = []
        for catalogue in catalogues:
            personas.extend(load_catalogue_cards(catalogue))
        return cls(personas)

    def __len__(self) -> int:
        return len(self._personas)

    def __contains__(self, persona_id: object) -> bool:
        return str(persona_id) in self._personas

    def add(self, persona: Persona) -> None:
        """Register a persona, replacing any existing one with the same id."""
        self._personas[persona.persona_id] = persona

    def get(self, persona_id: str) -> Persona:
        """Return one persona.

        Raises:
            PersonaError: When no persona has that id.
        """
        persona = self._personas.get(persona_id)
        if persona is None:
            raise PersonaError(f"unknown persona: {persona_id}")
        return persona

    def all(self) -> list[Persona]:
        """Return every persona, ordered by id."""
        return [self._personas[key] for key in sorted(self._personas)]

    def by_catalogue(self, catalogue: str) -> list[Persona]:
        """Return the personas of one catalogue."""
        return [persona for persona in self.all() if persona.catalogue == catalogue]

    def catalogues(self) -> list[str]:
        """Return the catalogue names present in the registry."""
        return sorted({persona.catalogue for persona in self._personas})

    def manifest(self, catalogue: str) -> dict[str, object]:
        """Return one catalogue's manifest."""
        return load_manifest(catalogue)

    def summaries(self, catalogue: str | None = None) -> list[dict[str, object]]:
        """Return compact listing entries."""
        personas = self.by_catalogue(catalogue) if catalogue else self.all()
        return [persona.summary() for persona in personas]
