"""Catalogue surfaces: the themed shell around each catalogue.

A surface carries the identity of one experience layer - its name, its theme,
its onboarding copy, and its default persona. The engine is shared; the
surface is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from herhim.catalog.loader import load_manifest
from herhim.errors import PersonaError

__all__ = ["CatalogueSurface", "load_surface"]


@dataclass(frozen=True)
class CatalogueSurface:
    """The presentation layer of one catalogue.

    Attributes:
        catalogue: ``her`` or ``him``.
        title: The surface title.
        subtitle: A one-line description of the catalogue.
        theme: CSS colour tokens for the surface.
        onboarding: The copy shown before the first message.
        default_persona: The persona selected on arrival.
        default_locale: The locale of the default persona's voice.
    """

    catalogue: str
    title: str
    subtitle: str
    theme: dict[str, str]
    onboarding: str
    default_persona: str
    default_locale: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "catalogue": self.catalogue,
            "title": self.title,
            "subtitle": self.subtitle,
            "theme": dict(self.theme),
            "onboarding": self.onboarding,
            "default_persona": self.default_persona,
            "default_locale": self.default_locale,
        }


def _theme_tokens(value: object) -> dict[str, str]:
    """Coerce a manifest theme into a mapping of CSS tokens.

    Raises:
        TypeError: When the theme is not a mapping.
    """
    if not isinstance(value, dict):
        raise TypeError(f"theme must be a mapping, got {type(value).__name__}")
    return {str(key): str(item) for key, item in value.items()}


def load_surface(catalogue: str) -> CatalogueSurface:
    """Load one catalogue's surface from its manifest."""
    manifest = load_manifest(catalogue)
    try:
        return CatalogueSurface(
            catalogue=catalogue,
            title=str(manifest.get("title", catalogue.title())),
            subtitle=str(manifest.get("subtitle", "")),
            theme=_theme_tokens(manifest.get("theme", {})),
            onboarding=str(manifest.get("onboarding", "")),
            default_persona=str(manifest.get("default_persona", "")),
            default_locale=str(manifest.get("default_locale", "en_US")),
        )
    except (TypeError, ValueError) as exc:
        raise PersonaError(f"manifest for {catalogue!r} is malformed: {exc}") from exc
