"""Voice engines and the emotion-to-audio mapping."""

from __future__ import annotations

from herhim.voice.base import VoiceEngine
from herhim.voice.emotion_map import EMOTION_TAGS, emotion_for_mood, map_to_params
from herhim.voice.factory import DEFAULT_ENGINE, available_engines, build_voice_engine
from herhim.voice.kokoro import KokoroVoiceEngine
from herhim.voice.piper import PiperVoiceEngine
from herhim.voice.samples import SampleIndex, SamplesVoiceEngine, load_sample_index

__all__ = [
    "DEFAULT_ENGINE",
    "EMOTION_TAGS",
    "KokoroVoiceEngine",
    "PiperVoiceEngine",
    "SampleIndex",
    "SamplesVoiceEngine",
    "VoiceEngine",
    "available_engines",
    "build_voice_engine",
    "emotion_for_mood",
    "load_sample_index",
    "map_to_params",
]
