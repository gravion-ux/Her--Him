"""Voice bindings, synthesis parameters, and clips."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["VoiceBinding", "VoiceClip", "VoiceInfo", "VoiceParams"]


@dataclass(frozen=True)
class VoiceParams:
    """Audible parameters for one synthesis request.

    Attributes:
        voice_id: The engine-specific voice identifier.
        emotion: An emotion tag understood by the synthesis vocabulary.
        pitch: Pitch offset in ``-12 .. 12``.
        speed: Speaking rate in ``0.5 .. 2.0``.
        volume: Relative loudness in ``0 .. 10``.
    """

    voice_id: str
    emotion: str = "neutral"
    pitch: float = 0.0
    speed: float = 1.0
    volume: float = 6.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "voice_id": self.voice_id,
            "emotion": self.emotion,
            "pitch": round(self.pitch, 3),
            "speed": round(self.speed, 3),
            "volume": round(self.volume, 3),
        }


@dataclass(frozen=True)
class VoiceInfo:
    """A voice known to an engine."""

    voice_id: str
    voice_name: str
    locale: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable mapping."""
        return {
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
            "locale": self.locale,
        }


@dataclass(frozen=True)
class VoiceClip:
    """A rendered piece of speech."""

    clip_id: str
    path: str
    duration_ms: int = 0
    params: VoiceParams | None = None
    transcript: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "clip_id": self.clip_id,
            "path": self.path,
            "duration_ms": self.duration_ms,
            "params": self.params.to_dict() if self.params else None,
            "transcript": self.transcript,
        }


@dataclass(frozen=True)
class VoiceBinding:
    """The voice a persona speaks with.

    Attributes:
        voice_id: Identifier used by the sample engine and the catalogue index.
        voice_name: Human-readable name of the voice.
        locale: BCP-47-ish locale tag such as ``en_US`` or ``hi_IN``.
        piper_voice: Piper voice name, when one is bound.
        kokoro_voice: Kokoro voice name, when one is bound.
        sample_id: Identifier of the committed sample clip.
        pitch: Base pitch offset applied before mood mapping.
        speed: Base speaking rate applied before mood mapping.
        volume: Base loudness applied before mood mapping.
    """

    voice_id: str
    voice_name: str = ""
    locale: str = "en_US"
    piper_voice: str = ""
    kokoro_voice: str = ""
    sample_id: str = ""
    pitch: float = 0.0
    speed: float = 1.0
    volume: float = 6.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
            "locale": self.locale,
            "piper_voice": self.piper_voice,
            "kokoro_voice": self.kokoro_voice,
            "sample_id": self.sample_id,
            "pitch": round(self.pitch, 3),
            "speed": round(self.speed, 3),
            "volume": round(self.volume, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> VoiceBinding:
        """Build a binding from a mapping, ignoring unknown keys."""
        return cls(
            voice_id=str(data.get("voice_id", "")),
            voice_name=str(data.get("voice_name", "")),
            locale=str(data.get("locale", "en_US")),
            piper_voice=str(data.get("piper_voice", "")),
            kokoro_voice=str(data.get("kokoro_voice", "")),
            sample_id=str(data.get("sample_id", "")),
            pitch=float(data.get("pitch", 0.0)),  # type: ignore[arg-type]
            speed=float(data.get("speed", 1.0)),  # type: ignore[arg-type]
            volume=float(data.get("volume", 6.0)),  # type: ignore[arg-type]
        )
