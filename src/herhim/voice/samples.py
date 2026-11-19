"""The committed-sample voice engine.

Every persona ships with a short, pre-rendered clip. This engine is what
makes the voice experience demonstrable with no install, no network, and no
model. It is also the default engine for the web surface.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from herhim.errors import VoiceError
from herhim.models.voice import VoiceClip, VoiceInfo, VoiceParams
from herhim.paths import sample_dir

__all__ = ["SampleIndex", "SamplesVoiceEngine", "load_sample_index"]

_INDEX_FILE = "index.json"


class SampleIndex:
    """The catalogue of committed clips."""

    def __init__(self, entries: list[dict[str, object]], root: Path) -> None:
        self._entries = entries
        self._root = root
        self._by_id = {str(entry.get("id")): entry for entry in entries}

    @property
    def entries(self) -> list[dict[str, object]]:
        """Every clip entry."""
        return list(self._entries)

    def get(self, clip_id: str) -> dict[str, object]:
        """Return one clip entry, raising when it does not exist."""
        entry = self._by_id.get(clip_id)
        if entry is None:
            raise VoiceError(f"unknown sample clip: {clip_id}")
        return entry

    def path_for(self, clip_id: str) -> Path:
        """Return the on-disk path of one clip."""
        entry = self.get(clip_id)
        return self._root / str(entry.get("file", ""))

    def for_persona(self, persona_id: str) -> dict[str, object] | None:
        """Return the clip bound to a persona, if there is one."""
        for entry in self._entries:
            if entry.get("persona_id") == persona_id:
                return entry
        return None


@lru_cache(maxsize=4)
def load_sample_index(root: str | None = None) -> SampleIndex:
    """Load the sample index, caching the result per directory."""
    base = Path(root) if root else sample_dir()
    index_path = base / _INDEX_FILE
    if not index_path.exists():
        raise VoiceError(f"sample index not found at {index_path}")
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VoiceError(f"sample index is unreadable: {exc}") from exc
    entries = payload.get("clips", []) if isinstance(payload, dict) else []
    return SampleIndex(entries=list(entries), root=base)


class SamplesVoiceEngine:
    """Plays back the committed clips."""

    name = "samples"

    def __init__(self, index: SampleIndex | None = None) -> None:
        self._index = index or load_sample_index()

    def synthesize(self, text: str, params: VoiceParams) -> VoiceClip:
        """Return the clip bound to the requested voice.

        The committed clips are fixed recordings, so the requested text is
        carried on the clip as its transcript rather than being synthesised.
        """
        entry = self._index.get(params.voice_id)
        return VoiceClip(
            clip_id=str(entry.get("id", params.voice_id)),
            path=str(self._index.path_for(str(entry.get("id", params.voice_id)))),
            duration_ms=int(entry.get("duration_ms", 0) or 0),
            params=params,
            transcript=str(entry.get("transcript", text)),
        )

    def list_voices(self) -> list[VoiceInfo]:
        """Return every voice that has a committed clip."""
        voices: list[VoiceInfo] = []
        for entry in self._index.entries:
            voices.append(
                VoiceInfo(
                    voice_id=str(entry.get("id", "")),
                    voice_name=str(entry.get("voice_name", "")),
                    locale=str(entry.get("locale", "")),
                )
            )
        return voices

    def clip_for_persona(self, persona_id: str) -> dict[str, object] | None:
        """Return the index entry bound to a persona."""
        return self._index.for_persona(persona_id)
