"""The deterministic offline provider.

This provider needs no credentials and no network. It produces a real reply
that respects the mood the engine derived, which is what lets the emotional
behaviour be demonstrated, tested, and regression-guarded without a model.

Responses are chosen deterministically: the first time an event fires the
first variant for that event is used, the second time the second variant, and
so on. The same session always produces the same transcript.
"""

from __future__ import annotations

from herhim.affect.moods import MoodLabel
from herhim.models.events import EventKind
from herhim.providers.base import LLMResponse, TurnContext

__all__ = ["OfflineProvider"]

#: Primary replies, keyed by the event that caused them. The first variant of
#: each list is what a persona says the first time that event fires.
_EVENT_LINES: dict[EventKind, tuple[str, ...]] = {
    EventKind.REJECTION: (
        "Fuck you!",
        "No. Just... no.",
        "Don't flatter yourself.",
        "I asked you nicely. You said no. Fine.",
    ),
    EventKind.PROVOCATION: (
        "Fuck me, then. I'm all yours.",
        "Oh, you want that? Prove it.",
        "Careful. I bite.",
        "Keep talking like that and you'll get exactly what you asked for.",
    ),
    EventKind.AFFECTION: (
        "Come here. You felt that, didn't you?",
        "Say it again. Slowly.",
        "You have no idea what you do to me.",
    ),
    EventKind.HOSTILITY: (
        "Wow. Okay.",
        "You did not just say that to me.",
        "Get out of my face.",
    ),
    EventKind.NEGLECT: (
        "You're really going to just... sit there.",
        "Interesting. So we don't talk anymore.",
        "Fine. Suit yourself.",
    ),
    EventKind.PRAISE: (
        "Careful. Keep talking like that and I'll get used to it.",
        "You think you can just say things like that?",
        "Mm. Noted.",
    ),
    EventKind.JEALY_TRIGGER: (
        "Jealous? Of course I'm jealous. Don't play dumb.",
        "Tell me about her. Slowly.",
        "You're testing me. I see what you're doing.",
    ),
    EventKind.TEASE: (
        "You're going to get exactly what you deserve.",
        "Keep it up.",
        "You're enjoying this far too much.",
    ),
    EventKind.CONFESSION: (
        "Don't say things you can't take back.",
        "...Say it again. Please.",
        "I didn't think you meant it until now.",
    ),
    EventKind.APOLOGY: (
        "Okay. I hear you.",
        "Don't apologise. Just don't do it again.",
        "Hm. Accepted.",
    ),
    EventKind.BOUNDARY: (
        "Okay. I heard you.",
        "Then we don't.",
        "Understood. No.",
    ),
    EventKind.COMPLIANCE: (
        "Good.",
        "Finally. I was starting to think you'd never say yes.",
        "That's what I wanted to hear.",
    ),
}

#: Fallback replies, keyed by mood, for turns where no event fired.
_MOOD_LINES: dict[MoodLabel, tuple[str, ...]] = {
    MoodLabel.FIERCE: ("Don't.", "Say that again and see what happens.", "I'm at my limit."),
    MoodLabel.SULKY: ("...", "Fine.", "I don't have anything to say to you."),
    MoodLabel.DEFIANT: ("Try me.", "No.", "Make me."),
    MoodLabel.PLAYFUL: ("Go on then.", "You're cute when you're nervous.", "Mhm."),
    MoodLabel.FLIRTY: (
        "You're testing my patience.",
        "Come closer.",
        "I wonder what you're hinting at.",
    ),
    MoodLabel.SULTRY: ("Slow down.", "You have no idea what you're doing to me.", "Stay there."),
    MoodLabel.AFFECTIONATE: ("Come here.", "I missed you.", "Stay a minute."),
    MoodLabel.JOYFUL: ("Ha! Look at you.", "That was actually good.", "I like this."),
    MoodLabel.BORED: ("Mm.", "Is that it?", "You have my attention for about four more seconds."),
    MoodLabel.DETACHED: ("I suppose.", "Okay.", "That's one option."),
    MoodLabel.MELANCHOLY: ("It's fine.", "I just... yeah.", "Never mind."),
    MoodLabel.POSSESSIVE: (
        "You're mine, you know.",
        "Who else have you been talking to?",
        "Don't go anywhere.",
    ),
    MoodLabel.ADMIRING: ("You're not bad, you know.", "I like watching you.", "Say more."),
    MoodLabel.APOLOGETIC: ("I'm sorry.", "That was too much.", "I shouldn't have said that."),
    MoodLabel.PROVOCATIVE: ("What are you going to do about it?", "Go on.", "Prove it."),
    MoodLabel.SATED: ("Mm.", "That's enough for now.", "Stay."),
    MoodLabel.CALM: ("Okay.", "Take your time.", "I'm here."),
    MoodLabel.SMUG: ("I know.", "Told you.", "You're welcome."),
}

_NEUTRAL_LINES: tuple[str, ...] = (
    "Go on.",
    "I'm listening.",
    "Hm.",
    "Tell me more.",
    "And?",
)


class OfflineProvider:
    """A deterministic provider that needs nothing but the current state."""

    name = "offline"
    model = "herhim-offline-v1"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.9,
        max_tokens: int = 400,
        context: TurnContext | None = None,
    ) -> LLMResponse:
        """Return a deterministic, mood-aware reply."""
        ctx = context or TurnContext()
        lines = self._select(ctx)
        return LLMResponse(text=lines, provider=self.name, model=self.model)

    def _select(self, ctx: TurnContext) -> str:
        """Choose the reply line for the current context."""
        for event_name in ctx.events:
            try:
                kind = EventKind(event_name)
            except ValueError:
                continue
            options = _EVENT_LINES.get(kind)
            if options:
                return options[ctx.event_ordinal % len(options)]
        try:
            mood = MoodLabel(ctx.mood)
        except ValueError:
            mood = MoodLabel.CALM
        options = _MOOD_LINES.get(mood, _NEUTRAL_LINES)
        return options[ctx.turn_index % len(options)]
