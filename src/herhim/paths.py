"""Locate the package's bundled data directory."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__all__ = ["catalogue_dir", "data_dir", "sample_dir"]


def data_dir() -> Path:
    """Return the path to the bundled ``data`` directory."""
    return Path(str(files("herhim").joinpath("data")))


def sample_dir() -> Path:
    """Return the path to the committed sample clips."""
    return data_dir() / "samples"


def catalogue_dir(catalogue: str) -> Path:
    """Return the path to one catalogue's persona directory."""
    return data_dir() / catalogue
