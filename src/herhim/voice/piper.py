"""The Piper voice engine adapter.

Piper is MIT licensed and runs entirely locally. It supports 42 locales and
roughly 120 voices, including Hindi (``hi_IN/priyamvada``, ``hi_IN/pratham``),
Malayalam (``ml_IN``), and Nepali (``ne_NP``).

The adapter prefers the ``piper`` Python package and falls back to the
``piper`` command line binary. When neither is available it raises a clear
error rather than pretending to work.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from herhim.errors import VoiceError
from herhim.models.voice import VoiceClip, VoiceInfo, VoiceParams

__all__ = ["PiperVoiceEngine"]

#: A small selection of the voices Piper ships, used for discovery.
KNOWN_VOICES: tuple[tuple[str, str, str], ...] = (
    ("en_US-amy-medium", "Amy", "en_US"),
    ("en_US-ryan-high", "Ryan", "en_US"),
    ("en_GB-alan-medium", "Alan", "en_GB"),
    ("es_ES-davefx-medium", "DaveFX", "es_ES"),
    ("fr_FR-siwis-medium", "Siwis", "fr_FR"),
    ("de_DE-thorsten_emotional-medium", "Thorsten (emotional)", "de_DE"),
    ("hi_IN-priyamvada-medium", "Priyamvada", "hi_IN"),
    ("hi_IN-pratham-medium", "Pratham", "hi_IN"),
    ("ml_IN-arjun-medium", "Arjun", "ml_IN"),
    ("ml_IN-meera-medium", "Meera", "ml_IN"),
    ("ne_NP-chitwan-medium", "Chitwan", "ne_NP"),
    ("zh_CN-huayan-medium", "Huayan", "zh_CN"),
    ("ja_JP-test-medium", "Test", "ja_JP"),
    ("ko_KR-kss-low", "KSS", "ko_KR"),
)


class PiperVoiceEngine:
    """Renders speech with a local Piper installation."""

    name = "piper"

    def __init__(self, binary: str | None = None, model_dir: str | None = None) -> None:
        self.binary = binary or shutil.which("piper") or ""
        self.model_dir = model_dir or ""

    @property
    def available(self) -> bool:
        """Whether a Piper backend can be found on this machine."""
        if self.binary:
            return True
        try:
            import piper  # noqa: F401
        except ImportError:
            return False
        return True

    def list_voices(self) -> list[VoiceInfo]:
        """Return the known Piper voices."""
        return [
            VoiceInfo(voice_id=voice_id, voice_name=name, locale=locale)
            for voice_id, name, locale in KNOWN_VOICES
        ]

    def _model_path(self, voice_id: str) -> Path:
        if not self.model_dir:
            raise VoiceError(
                "Piper needs a model directory. Install the piper voices and point "
                "HERHIM_PIPER_MODELS at the folder holding the .onnx files."
            )
        path = Path(self.model_dir) / f"{voice_id}.onnx"
        if not path.exists():
            raise VoiceError(f"Piper model not found: {path}")
        return path

    def synthesize(self, text: str, params: VoiceParams) -> VoiceClip:
        """Render ``text`` with Piper."""
        model = self._model_path(params.voice_id)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "clip.wav"
            if self.binary:
                command = [
                    self.binary,
                    "--model",
                    str(model),
                    "--output_file",
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
                    raise VoiceError(f"piper failed ({result.returncode}): {detail}")
            else:
                self._synthesize_with_package(text, model, output)
            if not output.exists():
                raise VoiceError("piper produced no output")
            target = Path(tmp) / f"{params.voice_id}.wav"
            target.write_bytes(output.read_bytes())
            return VoiceClip(
                clip_id=params.voice_id,
                path=str(target),
                params=params,
                transcript=text,
            )

    def _synthesize_with_package(self, text: str, model: Path, output: Path) -> None:
        """Render using the ``piper`` Python package."""
        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise VoiceError(
                "piper is not installed. Install the 'voice' extra: pip install 'herhim[voice]'"
            ) from exc
        voice = PiperVoice.load(str(model))
        with wave.open(str(output), "wb") as handle:
            voice.synthesize(text, handle)
