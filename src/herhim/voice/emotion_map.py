"""Map emotional state onto audible parameters.

This is the module that puts the mood *in the voice* rather than only in the
words. The same affect vector that selects a mood label also selects pitch,
speed, volume, and an emotion tag, so a sulky companion is audibly different
from a joyful one even when the words are identical.

The parameter set - voice, pitch, speed, volume - matches the
``voice_setting`` block used by Open-LLM-VTuber (MIT) and the MiniMax speech
API.
"""

from __future__ import annotations

from herhim.affect.moods import MoodLabel
from herhim.models.affect import AffectVector
from herhim.models.traits import TraitProfile
from herhim.models.voice import VoiceBinding, VoiceParams

__all__ = ["EMOTION_TAGS", "emotion_for_mood", "map_to_params"]

#: The emotion vocabulary understood by the speech backends.
EMOTION_TAGS: tuple[str, ...] = (
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
    "neutral",
)

_MOOD_EMOTION: dict[MoodLabel, str] = {
    MoodLabel.JOYFUL: "happy",
    MoodLabel.PLAYFUL: "happy",
    MoodLabel.AFFECTIONATE: "happy",
    MoodLabel.ADMIRING: "happy",
    MoodLabel.FLIRTY: "happy",
    MoodLabel.SULTRY: "happy",
    MoodLabel.PROVOCATIVE: "happy",
    MoodLabel.FIERCE: "angry",
    MoodLabel.DEFIANT: "angry",
    MoodLabel.POSSESSIVE: "angry",
    MoodLabel.SMUG: "angry",
    MoodLabel.SULKY: "sad",
    MoodLabel.MELANCHOLY: "sad",
    MoodLabel.APOLOGETIC: "sad",
    MoodLabel.BORED: "neutral",
    MoodLabel.DETACHED: "neutral",
    MoodLabel.CALM: "neutral",
    MoodLabel.SATED: "neutral",
}


def emotion_for_mood(mood: MoodLabel) -> str:
    """Return the emotion tag for a mood."""
    return _MOOD_EMOTION.get(mood, "neutral")


def map_to_params(
    vector: AffectVector,
    traits: TraitProfile,
    mood: MoodLabel,
    binding: VoiceBinding,
) -> VoiceParams:
    """Compile the current state into synthesis parameters.

    Args:
        vector: The current affect vector.
        traits: The persona's trait profile.
        mood: The derived mood.
        binding: The persona's voice binding, supplying the base parameters.

    Returns:
        The synthesis parameters for this turn.
    """
    pitch = binding.pitch + vector.valence * 4.0 + vector.arousal * 2.0 + vector.assertion
    speed = binding.speed + vector.arousal * 0.25 + (1.0 - traits.patience) * 0.10
    volume = binding.volume + vector.assertion * 1.5 + vector.arousal * 0.5

    return VoiceParams(
        voice_id=binding.voice_id,
        emotion=emotion_for_mood(mood),
        pitch=max(-12.0, min(12.0, round(pitch, 2))),
        speed=max(0.5, min(2.0, round(speed, 3))),
        volume=max(0.0, min(10.0, round(volume, 2))),
    )
