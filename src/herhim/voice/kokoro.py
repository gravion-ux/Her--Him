"""The Kokoro voice engine adapter.

Kokoro is Apache-2.0 licensed and runs locally with no API key. It covers
eight languages with high-quality voices.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from herhim.errors import VoiceError
from herhim.models.voice import VoiceClip, VoiceInfo, VoiceParams

__all__ = ["KokoroVoiceEngine"]

#: A selection of the voices Kokoro ships.
KNOWN_VOICES: tuple[tuple[str, str, str], ...] = (
    ("af_heart", "Heart (American female)", "en_US"),
    ("af_bella", "Bella (American female)", "en_US"),
    ("af_nicole", "Nicole (American female)", "en_US"),
    ("am_michael", "Michael (American male)", "en_US"),
    ("am_adam", "Adam (American male)", "en_US"),
    ("bf_emma", "Emma (British female)", "en_GB"),
    ("bm_george", "George (British male)", "en_GB"),
    ("ef_dora", "Dora (Spanish female)", "es_ES"),
)


class KokoroVoiceEngine:
    """Renders speech with a local Kokoro installation."""

    name = "kokoro"

    def __init__(self, binary: str | None = None) -> None:
        self.binary = binary or shutil.which("kokoro") or ""

    @property
    def available(self) -> bool:
        """Whether a Kokoro backend can be found on this machine."""
        if self.binary:
            return True
        try:
            import kokoro  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def list_voices(self) -> list[VoiceInfo]:
        """Return the known Kokoro voices."""
        return [
            VoiceInfo(voice_id=voice_id, voice_name=name, locale=locale)
            for voice_id, name, locale in KNOWN_VOICES
        ]

    def synthesize(self, text: str, params: VoiceParams) -> VoiceClip:
        """Render ``text`` with Kokoro."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "clip.wav"
            if self.binary:
                command = [
                    self.binary,
                    "--voice",
                    params.voice_id,
                    "--output",
                    str(output),
                ]
                result = subprocess.run(  # noqa: S603 - fixed local command
                    command,
                    input=text.encode("utf-8"),
                    capture_output=True,
                    check=False,
                )
                if result.returncode != 0:
                    detail = result.stderr.decode("utf-8", errors="replace")[:400]
                    raise VoiceError(f"kokoro failed ({result.returncode}): {detail}")
            else:
                self._synthesize_with_package(text, output, params)
            if not output.exists():
                raise VoiceError("kokoro produced no output")
            return VoiceClip(
                clip_id=params.voice_id,
                path=str(output),
                params=params,
                transcript=text,
            )

    def _synthesize_with_package(self, text: str, output: Path, params: VoiceParams) -> None:
        """Render using the ``kokoro`` Python package."""
        try:
            import numpy as np  # noqa: PLC0415
            from kokoro import KPipeline  # noqa: PLC0415
        except ImportError as exc:
            raise VoiceError(
                "kokoro is not installed. Install the 'voice' extra: pip install 'herhim[voice]'"
            ) from exc
        pipeline = KPipeline(lang_code=params.voice_id[:2])
        chunks: list[np.ndarray] = []
        for _, _, audio in pipeline(text, voice=params.voice_id):
            chunks.append(audio)
        if not chunks:
            raise VoiceError("kokoro produced no audio")
        _write_wav(output, np.concatenate(chunks), 24000)


def _write_wav(path: Path, samples: Any, sample_rate: int) -> None:
    """Write a mono 16-bit WAV file."""
    data: Any = samples
    if hasattr(data, "tolist"):
        data = data.tolist()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for value in data:
            clipped = max(-1.0, min(1.0, float(value)))
            frames += int(clipped * 32767).to_bytes(2, "little", signed=True)
        handle.writeframes(bytes(frames))
