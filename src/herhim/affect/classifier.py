"""The deterministic event classifier.

The classifier reads the *user's* turn and decides which conversational events
fired. It is never LLM-dependent, which is what makes emotional behaviour
reproducible and regression-testable.

Every pattern lives in one visible table. Nothing about a persona's reaction
is hidden in a prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from herhim.models.events import DetectedEvent, EventKind

__all__ = [
    "EVENT_PRECEDENCE",
    "EventClassifier",
    "base_deltas",
    "classify",
]

#: Tie-break order when two events match with the same intensity. Earlier
#: entries win.
EVENT_PRECEDENCE: tuple[EventKind, ...] = (
    EventKind.HOSTILITY,
    EventKind.BOUNDARY,
    EventKind.JEALY_TRIGGER,
    EventKind.REJECTION,
    EventKind.PROVOCATION,
    EventKind.TEASE,
    EventKind.CONFESSION,
    EventKind.APOLOGY,
    EventKind.AFFECTION,
    EventKind.NEGLECT,
    EventKind.COMPLIANCE,
    EventKind.PRAISE,
)


@dataclass(frozen=True)
class Rule:
    """One regex pattern that signals an event kind.

    Attributes:
        kind: The event this rule signals.
        pattern: A regular expression matched against the normalised text.
        intensity: The base strength of the match.
        label: A short human-readable description used in the rationale.
    """

    kind: EventKind
    pattern: str
    intensity: float
    label: str

    def compiled(self) -> re.Pattern[str]:
        """Return the compiled pattern."""
        return re.compile(self.pattern, re.IGNORECASE)


_RULES: tuple[Rule, ...] = (
    # --- hostility -------------------------------------------------------
    Rule(EventKind.HOSTILITY, r"\bshut up\b", 0.85, "hostile dismissal"),
    Rule(EventKind.HOSTILITY, r"\bgo away\b", 0.80, "hostile dismissal"),
    Rule(EventKind.HOSTILITY, r"\bleave me alone\b", 0.85, "hostile dismissal"),
    Rule(EventKind.HOSTILITY, r"\bi hate you\b", 0.95, "hostile dismissal"),
    Rule(EventKind.HOSTILITY, r"\bfuck (?:you|off)\b", 0.95, "hostile dismissal"),
    Rule(EventKind.HOSTILITY, r"\b(?:idiot|stupid|dumb|ugly)\b", 0.70, "hostile insult"),
    Rule(EventKind.HOSTILITY, r"\bget lost\b", 0.80, "hostile dismissal"),
    # --- boundary --------------------------------------------------------
    Rule(EventKind.BOUNDARY, r"\bdon'?t touch me\b", 0.95, "physical boundary"),
    Rule(EventKind.BOUNDARY, r"\bno touching\b", 0.95, "physical boundary"),
    Rule(EventKind.BOUNDARY, r"\bdon'?t (?:touch|go near|come closer|come back|do that)\b", 0.75, "physical boundary"),
    Rule(EventKind.BOUNDARY, r"\bback off\b", 0.85, "physical boundary"),
    Rule(EventKind.BOUNDARY, r"\bi'?m not comfortable\b", 0.90, "stated boundary"),
    Rule(EventKind.BOUNDARY, r"\bstop\b", 0.60, "stated boundary"),
    Rule(EventKind.BOUNDARY, r"\bboundar(?:y|ies)\b", 0.70, "stated boundary"),
    # --- jealousy --------------------------------------------------------
    Rule(EventKind.JEALY_TRIGGER, r"\bare you jealous\b", 0.95, "jealousy probe"),
    Rule(EventKind.JEALY_TRIGGER, r"\bjealous\b", 0.80, "jealousy probe"),
    Rule(EventKind.JEALY_TRIGGER, r"\bwho'?s (?:she|he|they)\b", 0.85, "jealousy probe"),
    Rule(EventKind.JEALY_TRIGGER, r"\bmy ex\b", 0.85, "jealousy probe"),
    Rule(EventKind.JEALY_TRIGGER, r"\bother (?:girl|guy|woman|man|person)\b", 0.70, "jealousy probe"),
    Rule(EventKind.JEALY_TRIGGER, r"\b(?:another|other) girlfriend\b", 0.90, "jealousy probe"),
    # --- rejection -------------------------------------------------------
    Rule(EventKind.REJECTION, r"\bno\b", 0.75, "flat refusal"),
    Rule(EventKind.REJECTION, r"\bnot (?:going to|doing that|like that|interested)\b", 0.80, "refusal"),
    Rule(EventKind.REJECTION, r"\bi (?:don'?t|do not) (?:want|feel like)\b", 0.80, "refusal"),
    Rule(EventKind.REJECTION, r"\bnever\b", 0.75, "refusal"),
    Rule(EventKind.REJECTION, r"\bforget it\b", 0.70, "refusal"),
    Rule(EventKind.REJECTION, r"\bnot a chance\b", 0.85, "refusal"),
    # --- provocation -----------------------------------------------------
    Rule(EventKind.PROVOCATION, r"\byou can'?t\b", 0.85, "challenge"),
    Rule(EventKind.PROVOCATION, r"\bcan'?t you\b", 0.75, "challenge"),
    Rule(EventKind.PROVOCATION, r"\bbut i can\b", 0.90, "challenge"),
    Rule(EventKind.PROVOCATION, r"\btry me\b", 0.85, "challenge"),
    Rule(EventKind.PROVOCATION, r"\bprove it\b", 0.80, "challenge"),
    Rule(EventKind.PROVOCATION, r"\bwhat if\b", 0.60, "challenge"),
    Rule(EventKind.PROVOCATION, r"\bkeep guessing\b", 0.70, "challenge"),
    Rule(EventKind.PROVOCATION, r"\b(?:come on|go on) then\b", 0.75, "challenge"),
    # --- tease -----------------------------------------------------------
    Rule(EventKind.TEASE, r"\btease\b", 0.85, "teasing"),
    Rule(EventKind.TEASE, r"\bkeep it up\b", 0.65, "teasing"),
    Rule(EventKind.TEASE, r"\bif you insist\b", 0.70, "teasing"),
    Rule(EventKind.TEASE, r"\bwhat if i (?:didn'?t|don'?t)\b", 0.65, "teasing"),
    Rule(EventKind.TEASE, r"[😏😉🙃😜]", 0.75, "teasing emoji"),
    # --- confession ------------------------------------------------------
    Rule(EventKind.CONFESSION, r"\bi(?:'m| am) in love\b", 0.95, "confession"),
    Rule(EventKind.CONFESSION, r"\bi confess\b", 0.95, "confession"),
    Rule(EventKind.CONFESSION, r"\bmarry me\b", 0.95, "confession"),
    Rule(EventKind.CONFESSION, r"\bi feel (?:something|deeply|so much)\b", 0.80, "confession"),
    Rule(EventKind.CONFESSION, r"\byou'?re the only one\b", 0.90, "confession"),
    # --- apology ---------------------------------------------------------
    Rule(EventKind.APOLOGY, r"\bi'?m sorry\b", 0.90, "apology"),
    Rule(EventKind.APOLOGY, r"\bsorry\b", 0.75, "apology"),
    Rule(EventKind.APOLOGY, r"\bi apolog(?:ize|ise)\b", 0.90, "apology"),
    Rule(EventKind.APOLOGY, r"\bmy bad\b", 0.70, "apology"),
    Rule(EventKind.APOLOGY, r"\bforgive me\b", 0.90, "apology"),
    # --- affection -------------------------------------------------------
    Rule(EventKind.AFFECTION, r"\bi (?:love|adore) you\b", 0.95, "affection"),
    Rule(EventKind.AFFECTION, r"\bmiss you\b", 0.85, "affection"),
    Rule(EventKind.AFFECTION, r"\bkiss (?:you|me)\b", 0.80, "affection"),
    Rule(EventKind.AFFECTION, r"\bcuddle\b", 0.80, "affection"),
    Rule(EventKind.AFFECTION, r"\bcome here\b", 0.70, "affection"),
    Rule(EventKind.AFFECTION, r"\bi'?m yours\b", 0.85, "affection"),
    Rule(EventKind.AFFECTION, r"\bhold me\b", 0.80, "affection"),
    # --- neglect ---------------------------------------------------------
    Rule(EventKind.NEGLECT, r"\bwhatever\b", 0.70, "indifference"),
    Rule(EventKind.NEGLECT, r"\bdon'?t care\b", 0.75, "indifference"),
    Rule(EventKind.NEGLECT, r"\bnot interested\b", 0.75, "indifference"),
    Rule(EventKind.NEGLECT, r"\bwho cares\b", 0.70, "indifference"),
    Rule(EventKind.NEGLECT, r"\bnot talking (?:to you|right now)\b", 0.85, "indifference"),
    # --- compliance ------------------------------------------------------
    Rule(EventKind.COMPLIANCE, r"\bokay\b", 0.60, "agreement"),
    Rule(EventKind.COMPLIANCE, r"\bsure\b", 0.60, "agreement"),
    Rule(EventKind.COMPLIANCE, r"\byes\b", 0.65, "agreement"),
    Rule(EventKind.COMPLIANCE, r"\bi agree\b", 0.75, "agreement"),
    Rule(EventKind.COMPLIANCE, r"\bgo ahead\b", 0.75, "agreement"),
    Rule(EventKind.COMPLIANCE, r"\bdo it\b", 0.75, "agreement"),
    Rule(EventKind.COMPLIANCE, r"\bthat'?s fine\b", 0.70, "agreement"),
    # --- praise ----------------------------------------------------------
    Rule(EventKind.PRAISE, r"\byou'?re (?:amazing|great|beautiful|gorgeous|awesome)\b", 0.85, "praise"),
    Rule(EventKind.PRAISE, r"\bi (?:like|love) (?:you|that|this)\b", 0.70, "praise"),
    Rule(EventKind.PRAISE, r"\bwell done\b", 0.75, "praise"),
    Rule(EventKind.PRAISE, r"\bgood (?:boy|girl)\b", 0.70, "praise"),
    Rule(EventKind.PRAISE, r"\bi'?m proud of you\b", 0.85, "praise"),
)


def base_deltas(kind: EventKind) -> tuple[float, float, float, float]:
    """Return the base ``(valence, arousal, assertion, tenderness)`` deltas."""
    return _BASE_DELTAS[kind]


_BASE_DELTAS: dict[EventKind, tuple[float, float, float, float]] = {
    EventKind.REJECTION: (-0.35, 0.20, 0.10, -0.15),
    EventKind.COMPLIANCE: (0.30, 0.05, -0.05, 0.10),
    EventKind.PROVOCATION: (0.05, 0.30, 0.05, 0.05),
    EventKind.AFFECTION: (0.35, 0.10, -0.10, 0.25),
    EventKind.HOSTILITY: (-0.45, 0.35, 0.25, -0.30),
    EventKind.NEGLECT: (-0.20, -0.10, -0.05, -0.15),
    EventKind.PRAISE: (0.25, 0.15, 0.05, 0.15),
    EventKind.JEALY_TRIGGER: (-0.25, 0.25, 0.15, -0.10),
    EventKind.TEASE: (0.10, 0.30, 0.10, 0.10),
    EventKind.CONFESSION: (0.40, 0.25, 0.15, 0.30),
    EventKind.APOLOGY: (0.20, -0.15, -0.15, 0.20),
    EventKind.BOUNDARY: (-0.10, 0.05, 0.05, 0.05),
}


class EventClassifier:
    """Detects conversational events in a user turn.

    The classifier is stateless: every call is a pure function of the text.
    """

    def __init__(self, rules: tuple[Rule, ...] = _RULES) -> None:
        self._rules = rules

    def detect(self, text: str) -> list[DetectedEvent]:
        """Return the events fired by ``text``, strongest first.

        The first entry is the primary event and carries ``scale`` ``1.0``.
        Secondary events - those within 55% of the primary intensity - are
        returned with ``scale`` ``0.5`` so a combined turn ("sorry, but no")
        still registers both halves.
        """
        normalised = text.strip()
        if not normalised:
            return []

        matches: dict[EventKind, tuple[float, list[str]]] = {}
        for rule in self._rules:
            found = rule.compiled().findall(normalised)
            if not found:
                continue
            intensity = min(1.0, rule.intensity + 0.15 * (len(found) - 1))
            current = matches.get(rule.kind)
            if current is None or intensity > current[0]:
                matches[rule.kind] = (intensity, [str(item) for item in found[:4]])

        if not matches:
            return []

        ordered = sorted(
            matches.items(),
            key=lambda item: (
                -item[1][0],
                EVENT_PRECEDENCE.index(item[0]),
            ),
        )
        primary_intensity = ordered[0][1][0]
        events: list[DetectedEvent] = []
        for kind, (intensity, fragments) in ordered:
            if kind is ordered[0][0]:
                scale = 1.0
            elif intensity >= primary_intensity * 0.55:
                scale = 0.5
            else:
                continue
            events.append(
                DetectedEvent(kind=kind, intensity=intensity, matched=tuple(fragments), scale=scale)
            )
        return events


_DEFAULT_CLASSIFIER = EventClassifier()


def classify(text: str) -> list[DetectedEvent]:
    """Classify a user turn with the default rule set."""
    return _DEFAULT_CLASSIFIER.detect(text)
