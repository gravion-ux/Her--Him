"""Resolve a voice engine by name."""

from __future__ import annotations

import os

from herhim.errors import VoiceError
from herhim.voice.base import VoiceEngine
from herhim.voice.kokoro import KokoroVoiceEngine
from herhim.voice.piper import PiperVoiceEngine
from herhim.voice.samples import SamplesVoiceEngine

__all__ = ["DEFAULT_ENGINE", "available_engines", "build_voice_engine"]

#: The engine used when nothing else is requested.
DEFAULT_ENGINE = "samples"

ENV_ENGINE = "HERHIM_VOICE_ENGINE"
ENV_PIPER_MODELS = "HERHIM_PIPER_MODELS"


def available_engines() -> tuple[str, ...]:
    """Return the engine names that can be resolved."""
    return ("samples", "piper", "kokoro")


def build_voice_engine(name: str | None = None) -> VoiceEngine:
    """Build a voice engine.

    Args:
        name: The engine name. Defaults to ``HERHIM_VOICE_ENGINE`` and then
            to the committed-sample engine.

    Raises:
        VoiceError: When the requested engine does not exist.
    """
    engine_name = (name or os.environ.get(ENV_ENGINE) or DEFAULT_ENGINE).strip().lower()
    if engine_name == "samples":
        return SamplesVoiceEngine()
    if engine_name == "piper":
        return PiperVoiceEngine(model_dir=os.environ.get(ENV_PIPER_MODELS, ""))
    if engine_name == "kokoro":
        return KokoroVoiceEngine()
    raise VoiceError(
        f"unknown voice engine {engine_name!r}; choose one of {', '.join(available_engines())}"
    )
