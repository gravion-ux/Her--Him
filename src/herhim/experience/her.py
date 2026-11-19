"""The "her" experience layer."""

from __future__ import annotations

from herhim.experience.surface import CatalogueSurface, load_surface

__all__ = ["CATALOGUE", "SURFACE", "load_her_surface"]

#: The catalogue this surface belongs to.
CATALOGUE = "her"


def load_her_surface() -> CatalogueSurface:
    """Load the "her" surface."""
    return load_surface(CATALOGUE)


#: The surface loaded on demand; kept lazy so importing this module is cheap.
SURFACE: CatalogueSurface | None = None
