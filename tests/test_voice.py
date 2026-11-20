"""Tests for the voice engines: mapping, factory, samples, Piper, and Kokoro."""

from __future__ import annotations

import subprocess
import sys
import types
import wave
from pathlib import Path

import pytest

from herhim.affect.moods import MoodLabel
from herhim.errors import VoiceError
from herhim.models.affect import AffectVector
from herhim.models.traits import TraitProfile
from herhim.models.voice import VoiceBinding, VoiceParams
from herhim.voice.emotion_map import EMOTION_TAGS, emotion_for_mood, map_to_params
from herhim.voice.factory import DEFAULT_ENGINE, available_engines, build_voice_engine
from herhim.voice.kokoro import KokoroVoiceEngine, _write_wav
from herhim.voice.piper import PiperVoiceEngine
from herhim.voice.samples import SampleIndex, SamplesVoiceEngine, load_sample_index


def _fake_run(output_flag: str, *, returncode: int = 0, write: bool = True):
    """Build a stand-in for ``subprocess.run`` that writes the output file."""

    def run(command, **kwargs):
        if returncode != 0:
            return subprocess.CompletedProcess(command, returncode, b"", b"boom")
        if write:
            index = command.index(output_flag)
            Path(command[index + 1]).write_bytes(b"RIFFfake")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    return run


def _install_fake_piper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake ``piper`` package that renders a silent clip."""

    class _FakePiperVoice:
        @staticmethod
        def load(model: str) -> _FakePiperVoice:
            """Return a fake voice for the requested model."""
            return _FakePiperVoice()

        def synthesize(self, text: str, handle: object) -> None:
            """Write a silent frame to the output handle."""
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(22050)
            handle.writeframes(b"\x00\x00" * 64)

    piper_module = types.ModuleType("piper")
    piper_module.__path__ = []  # type: ignore[attr-defined]
    voice_module = types.ModuleType("piper.voice")
    voice_module.PiperVoice = _FakePiperVoice
    piper_module.voice = voice_module
    monkeypatch.setitem(sys.modules, "piper", piper_module)
    monkeypatch.setitem(sys.modules, "piper.voice", voice_module)


def _install_fake_kokoro(monkeypatch: pytest.MonkeyPatch, *, chunks: int = 1) -> None:
    """Install a fake ``kokoro`` and ``numpy`` pair that renders audio."""

    class _FakeArray:
        def tolist(self) -> list[float]:
            """Return the rendered samples."""
            return [0.0, 0.5, -0.5]

    numpy_module = types.ModuleType("numpy")
    numpy_module.concatenate = lambda chunks: _FakeArray()

    class _FakePipeline:
        def __init__(self, lang_code: str) -> None:
            """Store the requested language code."""
            self.lang_code = lang_code

        def __call__(self, text: str, voice: str | None = None):
            """Yield the rendered chunks."""
            return iter([(None, None, [0.0, 0.5]) for _ in range(chunks)])

    kokoro_module = types.ModuleType("kokoro")
    kokoro_module.KPipeline = _FakePipeline

    monkeypatch.setitem(sys.modules, "numpy", numpy_module)
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_module)


class TestEmotionMap:
    """Tests for mapping emotional state onto voice parameters."""

    def test_every_mood_maps_to_a_known_emotion(self) -> None:
        """Every mood has an emotion the speech backends understand."""
        for label in MoodLabel:
            assert emotion_for_mood(label) in EMOTION_TAGS

    def test_an_unknown_mood_is_neutral(self) -> None:
        """A mood with no mapping falls back to neutral."""
        assert emotion_for_mood("nonsense") == "neutral"

    def test_parameters_stay_inside_their_ranges(self) -> None:
        """The mapped parameters respect the synthesis limits."""
        params = map_to_params(
            AffectVector(valence=5.0, arousal=5.0, assertion=5.0, tenderness=5.0),
            TraitProfile(),
            MoodLabel.CALM,
            VoiceBinding("v1"),
        )
        assert -12.0 <= params.pitch <= 12.0
        assert 0.5 <= params.speed <= 2.0
        assert 0.0 <= params.volume <= 10.0

    def test_the_voice_id_comes_from_the_binding(self) -> None:
        """The mapped parameters speak with the persona's voice."""
        params = map_to_params(
            AffectVector(),
            TraitProfile(),
            MoodLabel.CALM,
            VoiceBinding("v1", pitch=1.0, speed=1.1, volume=5.0),
        )
        assert params.voice_id == "v1"
        assert params.pitch == pytest.approx(2.5)
        assert params.speed == pytest.approx(1.275)
        assert params.volume == pytest.approx(6.0)

    def test_the_emotion_follows_the_mood(self) -> None:
        """A joyful persona sounds happy."""
        params = map_to_params(
            AffectVector(),
            TraitProfile(),
            MoodLabel.JOYFUL,
            VoiceBinding("v1"),
        )
        assert params.emotion == "happy"


class TestVoiceFactory:
    """Tests for resolving a voice engine by name."""

    def test_available_engines(self) -> None:
        """The factory knows the three bundled engines."""
        assert available_engines() == ("samples", "piper", "kokoro")

    def test_default_engine(self) -> None:
        """The default engine is the committed-sample engine."""
        assert DEFAULT_ENGINE == "samples"

    def test_build_samples(self) -> None:
        """The samples engine can be built."""
        assert isinstance(build_voice_engine("samples"), SamplesVoiceEngine)

    def test_build_piper(self) -> None:
        """The Piper engine can be built."""
        assert isinstance(build_voice_engine("piper"), PiperVoiceEngine)

    def test_build_kokoro(self) -> None:
        """The Kokoro engine can be built."""
        assert isinstance(build_voice_engine("kokoro"), KokoroVoiceEngine)

    def test_the_environment_selects_the_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The engine can be chosen through the environment."""
        monkeypatch.setenv("HERHIM_VOICE_ENGINE", "samples")
        assert isinstance(build_voice_engine(), SamplesVoiceEngine)

    def test_an_unknown_engine_is_refused(self) -> None:
        """An engine that does not exist raises a voice error."""
        with pytest.raises(VoiceError, match="unknown voice engine"):
            build_voice_engine("nonsense")


class TestSampleEngine:
    """Tests for the committed-sample voice engine."""

    def test_the_index_loads(self) -> None:
        """The bundled sample index loads with clips in it."""
        assert load_sample_index().entries

    def test_a_missing_index_is_refused(self, tmp_path: Path) -> None:
        """A directory without an index raises a voice error."""
        with pytest.raises(VoiceError, match="sample index not found"):
            load_sample_index(str(tmp_path))

    def test_an_unreadable_index_is_refused(self, tmp_path: Path) -> None:
        """An index that is not valid JSON raises a voice error."""
        (tmp_path / "index.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(VoiceError, match="unreadable"):
            load_sample_index(str(tmp_path))

    def test_an_unknown_clip_is_refused(self) -> None:
        """Asking for a clip that does not exist raises a voice error."""
        with pytest.raises(VoiceError, match="unknown sample clip"):
            load_sample_index().get("nope")

    def test_the_path_of_a_clip(self) -> None:
        """A clip entry knows where its file lives."""
        index = load_sample_index()
        assert index.path_for("sassy-maya").name == "sassy-maya.wav"

    def test_a_clip_can_be_found_by_persona(self) -> None:
        """A persona's committed clip can be looked up."""
        index = load_sample_index()
        assert index.for_persona("her-001-sassy") is not None
        assert index.for_persona("nope") is None

    def test_the_index_is_built_directly(self, tmp_path: Path) -> None:
        """An index can be built from a list of entries."""
        index = SampleIndex([{"id": "a", "file": "a.wav"}], tmp_path)
        assert index.get("a")["file"] == "a.wav"

    def test_synthesize_returns_the_committed_clip(self) -> None:
        """Synthesis returns the committed clip with its transcript."""
        engine = SamplesVoiceEngine()
        clip = engine.synthesize("hello", VoiceParams("sassy-maya"))
        assert clip.clip_id == "sassy-maya"
        assert clip.duration_ms > 0
        assert clip.transcript

    def test_synthesize_falls_back_to_the_requested_text(self) -> None:
        """A clip without a transcript carries the requested text."""
        engine = SamplesVoiceEngine()
        clip = engine.synthesize("hello", VoiceParams("steady-ethan"))
        assert clip.transcript

    def test_list_voices(self) -> None:
        """Every committed clip is listed as a voice."""
        voices = SamplesVoiceEngine().list_voices()
        assert voices
        assert voices[0].voice_id

    def test_clip_for_persona(self) -> None:
        """The engine can find the clip bound to a persona."""
        assert SamplesVoiceEngine().clip_for_persona("her-001-sassy") is not None


class TestPiperEngine:
    """Tests for the Piper voice engine adapter."""

    def test_known_voices(self) -> None:
        """The adapter lists the voices Piper ships."""
        assert PiperVoiceEngine().list_voices()

    def test_unavailable_without_a_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a binary or the package the engine is unavailable."""
        monkeypatch.setitem(sys.modules, "piper", None)
        assert PiperVoiceEngine(binary="").available is False

    def test_available_with_a_binary(self) -> None:
        """A configured binary makes the engine available."""
        assert PiperVoiceEngine(binary="piper").available is True

    def test_available_with_the_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The installed package makes the engine available."""
        _install_fake_piper(monkeypatch)
        assert PiperVoiceEngine(binary="").available is True

    def test_a_model_directory_is_required(self) -> None:
        """Rendering without a model directory raises a voice error."""
        with pytest.raises(VoiceError, match="model directory"):
            PiperVoiceEngine(binary="piper").synthesize("hi", VoiceParams("v1"))

    def test_a_missing_model_is_refused(self, tmp_path: Path) -> None:
        """Rendering with a model that is not there raises a voice error."""
        with pytest.raises(VoiceError, match="model not found"):
            PiperVoiceEngine(binary="piper", model_dir=str(tmp_path)).synthesize(
                "hi", VoiceParams("v1")
            )

    def test_the_binary_renders_a_clip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The command-line backend renders a clip."""
        (tmp_path / "v1.onnx").write_bytes(b"model")
        monkeypatch.setattr(subprocess, "run", _fake_run("--output_file"))
        engine = PiperVoiceEngine(binary="piper", model_dir=str(tmp_path))
        clip = engine.synthesize("hi", VoiceParams("v1"))
        assert clip.clip_id == "v1"
        assert clip.transcript == "hi"

    def test_a_failing_binary_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-zero exit from the binary raises a voice error."""
        (tmp_path / "v1.onnx").write_bytes(b"model")
        monkeypatch.setattr(subprocess, "run", _fake_run("--output_file", returncode=1))
        engine = PiperVoiceEngine(binary="piper", model_dir=str(tmp_path))
        with pytest.raises(VoiceError, match="piper failed"):
            engine.synthesize("hi", VoiceParams("v1"))

    def test_a_binary_that_produces_nothing_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A silent binary that writes no file raises a voice error."""
        (tmp_path / "v1.onnx").write_bytes(b"model")
        monkeypatch.setattr(subprocess, "run", _fake_run("--output_file", write=False))
        engine = PiperVoiceEngine(binary="piper", model_dir=str(tmp_path))
        with pytest.raises(VoiceError, match="produced no output"):
            engine.synthesize("hi", VoiceParams("v1"))

    def test_the_package_renders_a_clip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Python package renders a clip into a WAV file."""
        (tmp_path / "v1.onnx").write_bytes(b"model")
        _install_fake_piper(monkeypatch)
        engine = PiperVoiceEngine(model_dir=str(tmp_path))
        clip = engine.synthesize("hi", VoiceParams("v1"))
        assert clip.clip_id == "v1"

    def test_a_missing_package_is_reported(self, tmp_path: Path) -> None:
        """Without the package the engine says so."""
        (tmp_path / "v1.onnx").write_bytes(b"model")
        with pytest.raises(VoiceError, match="piper is not installed"):
            PiperVoiceEngine(model_dir=str(tmp_path))._synthesize_with_package(
                "hi", tmp_path / "v1.onnx", tmp_path / "out.wav"
            )


class TestKokoroEngine:
    """Tests for the Kokoro voice engine adapter."""

    def test_known_voices(self) -> None:
        """The adapter lists the voices Kokoro ships."""
        assert KokoroVoiceEngine().list_voices()

    def test_unavailable_without_a_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a binary or the package the engine is unavailable."""
        monkeypatch.setitem(sys.modules, "kokoro", None)
        assert KokoroVoiceEngine(binary="").available is False

    def test_available_with_a_binary(self) -> None:
        """A configured binary makes the engine available."""
        assert KokoroVoiceEngine(binary="kokoro").available is True

    def test_available_with_the_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The installed package makes the engine available."""
        _install_fake_kokoro(monkeypatch)
        assert KokoroVoiceEngine(binary="").available is True

    def test_the_binary_renders_a_clip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The command-line backend renders a clip."""
        monkeypatch.setattr(subprocess, "run", _fake_run("--output"))
        clip = KokoroVoiceEngine(binary="kokoro").synthesize("hi", VoiceParams("af_heart"))
        assert clip.clip_id == "af_heart"

    def test_a_failing_binary_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-zero exit from the binary raises a voice error."""
        monkeypatch.setattr(subprocess, "run", _fake_run("--output", returncode=1))
        with pytest.raises(VoiceError, match="kokoro failed"):
            KokoroVoiceEngine(binary="kokoro").synthesize("hi", VoiceParams("af_heart"))

    def test_a_binary_that_produces_nothing_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A silent binary that writes no file raises a voice error."""
        monkeypatch.setattr(subprocess, "run", _fake_run("--output", write=False))
        with pytest.raises(VoiceError, match="produced no output"):
            KokoroVoiceEngine(binary="kokoro").synthesize("hi", VoiceParams("af_heart"))

    def test_the_package_renders_a_clip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Python package renders a clip into a WAV file."""
        _install_fake_kokoro(monkeypatch)
        clip = KokoroVoiceEngine().synthesize("hi", VoiceParams("af_heart"))
        assert clip.clip_id == "af_heart"

    def test_a_package_that_renders_nothing_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pipeline that yields no audio raises a voice error."""
        _install_fake_kokoro(monkeypatch, chunks=0)
        with pytest.raises(VoiceError, match="produced no audio"):
            KokoroVoiceEngine().synthesize("hi", VoiceParams("af_heart"))

    def test_a_missing_package_is_reported(self, tmp_path: Path) -> None:
        """Without the package the engine says so."""
        with pytest.raises(VoiceError, match="kokoro is not installed"):
            KokoroVoiceEngine()._synthesize_with_package(
                "hi", tmp_path / "out.wav", VoiceParams("af_heart")
            )

    def test_write_wav_produces_a_readable_file(self, tmp_path: Path) -> None:
        """The WAV writer produces a file the standard library can read."""
        target = tmp_path / "out.wav"
        _write_wav(target, [0.0, 0.5, -0.5, 2.0], 24000)
        with wave.open(str(target), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 24000
            assert handle.getnframes() == 4
