"""The voice engine protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from herhim.models.voice import VoiceClip, VoiceInfo, VoiceParams

__all__ = ["VoiceEngine"]


@runtime_checkable
class VoiceEngine(Protocol):
    """The interface every speech backend implements.

    The shape follows the TTS interface pattern used by Open-LLM-VTuber (MIT):
    one abstract contract, several engines, and a factory that resolves one
    from configuration.
    """

    name: str

    def synthesize(self, text: str, params: VoiceParams) -> VoiceClip:
        """Render ``text`` with ``params`` and return the resulting clip."""
        ...

    def list_voices(self) -> list[VoiceInfo]:
        """Return the voices this engine can speak with."""
        ...
