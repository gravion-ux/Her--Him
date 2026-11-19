"""Error types and process exit codes used across the package.

The exit-code vocabulary mirrors the house style already used across the
gravion-ux toolkit repositories so that shell callers can rely on a stable
contract.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "EXIT_INVALID_INPUT",
    "EXIT_INVALID_STATE",
    "EXIT_OK",
    "EXIT_USAGE",
    "HerhimError",
    "GuardrailError",
    "PersonaError",
    "ProviderError",
    "VoiceError",
]


class EXIT_OK(IntEnum):
    """Successful completion."""

    OK = 0


class EXIT_USAGE(IntEnum):
    """The command line was malformed."""

    USAGE = 2


class EXIT_INVALID_INPUT(IntEnum):
    """A supplied value was well-formed but semantically invalid."""

    INVALID_INPUT = 3


class EXIT_INVALID_STATE(IntEnum):
    """The runtime reached a state it cannot recover from."""

    INVALID_STATE = 4


class HerhimError(Exception):
    """Base class for every error raised by this package."""


class PersonaError(HerhimError):
    """A persona card is missing, malformed, or fails validation."""


class ProviderError(HerhimError):
    """An LLM provider could not fulfil a request."""


class VoiceError(HerhimError):
    """A voice engine could not produce or locate a clip."""


class GuardrailError(HerhimError):
    """A structural guardrail refused an operation."""
