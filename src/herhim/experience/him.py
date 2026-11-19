"""The "him" experience layer."""

from __future__ import annotations

from herhim.experience.surface import CatalogueSurface, load_surface

__all__ = ["CATALOGUE", "SURFACE", "load_him_surface"]

#: The catalogue this surface belongs to.
CATALOGUE = "him"


def load_him_surface() -> CatalogueSurface:
    """Load the "him" surface."""
    return load_surface(CATALOGUE)


#: The surface loaded on demand; kept lazy so importing this module is cheap.
SURFACE: CatalogueSurface | None = None
