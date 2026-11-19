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
    "ExitCode",
    "GuardrailError",
    "HerhimError",
    "PersonaError",
    "ProviderError",
    "VoiceError",
]


class ExitCode(IntEnum):
    """The stable process exit codes used by the command line entry points.

    These are plain integers at runtime, so they can be returned from a
    ``main()`` function or passed straight to :func:`sys.exit`.
    """

    OK = 0
    USAGE = 2
    INVALID_INPUT = 3
    INVALID_STATE = 4


#: Successful completion.
EXIT_OK = ExitCode.OK

#: The command line was malformed.
EXIT_USAGE = ExitCode.USAGE

#: A supplied value was well-formed but semantically invalid.
EXIT_INVALID_INPUT = ExitCode.INVALID_INPUT

#: The runtime reached a state it cannot recover from.
EXIT_INVALID_STATE = ExitCode.INVALID_STATE


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
