"""Turn a natural-language music request into structured preferences.

The parser runs local, deterministic extraction: genre and mood names (from the
knowledge-base vocabulary), energy words or numbers, decades, acoustic/instrumental
cues, popularity, mood tags, requested count, and mentions of catalog songs or
artists. If no mood word is found, it falls back to the specialized mood
classifier. Every extracted value records *how* it was found so the agent's
trace and the final explanation can show it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.intent import MIN_CONFIDENCE, MoodClassifier, keyword_mood
from src.knowledge import KnowledgeBase, contains_phrase

ENERGY_WORDS = {
    "high energy": 0.85, "high-energy": 0.85, "hype": 0.9, "pumped": 0.9, "energetic": 0.85,
    "upbeat": 0.75, "fast": 0.8, "loud": 0.85, "low energy": 0.25, "low-energy": 0.25,
    "calm": 0.3, "mellow": 0.3, "soft": 0.3, "quiet": 0.25, "sleepy": 0.15, "slow": 0.3,
    "medium energy": 0.55, "moderate": 0.55,
}
POPULAR_WORDS = {"popular": 80, "mainstream": 80, "hits": 80, "top 40": 85,
                 "underground": 25, "obscure": 20, "hidden gems": 25, "lesser known": 25}


@dataclass
class ParsedRequest:
    """Structured preferences extracted from a request, plus provenance for each value."""
    text: str
    prefs: Dict[str, Any] = field(default_factory=dict)
    detected: Dict[str, str] = field(default_factory=dict)
    mentions: List[Dict] = field(default_factory=list)
    k: Optional[int] = None

    def has_music_signal(self) -> bool:
        """Return True if anything music-related was extracted."""
        return bool(self.prefs or self.mentions)


def _find_vocab(text: str, vocab: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Return (canonical, alias) for the longest vocabulary alias found in text."""
    for alias in sorted(vocab, key=len, reverse=True):
        if contains_phrase(text, alias):
            return vocab[alias], alias
    return None


def _parse_decade(text: str) -> Optional[str]:
    """Return a decade like '1980s' from '80s', "'80s", or '1980s'."""
    match = re.search(r"(?<![0-9])'?((?:19|20)?[0-9])0'?s\b", text.lower())
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 1:
        digits = ("20" if digits in "012" else "19") + digits
    return f"{digits}0s"


def parse_request(text: str, songs: List[Dict], kb: KnowledgeBase,
                  classifier: Optional[MoodClassifier] = None) -> ParsedRequest:
    """Extract preferences from free text; use the classifier for mood only if no mood word is found."""
    parsed = ParsedRequest(text=text)
    lowered = text.lower()
    prefs, detected = parsed.prefs, parsed.detected

    # Song / artist mentions (searched first so their words don't trigger other matches).
    remaining = lowered
    for song in sorted(songs, key=lambda s: len(s["title"]), reverse=True):
        if contains_phrase(remaining, song["title"]):
            parsed.mentions.append(song)
            remaining = remaining.replace(song["title"].lower(), " ")
    for artist in sorted({s["artist"] for s in songs}, key=len, reverse=True):
        if contains_phrase(remaining, artist):
            parsed.mentions.extend(s for s in songs if s["artist"] == artist and s not in parsed.mentions)
            detected["artist"] = f"{artist} (catalog search)"
            remaining = remaining.replace(artist.lower(), " ")

    genre_vocab = kb.vocabulary("genre")
    genre_vocab.update({s["genre"]: s["genre"] for s in songs})
    found = _find_vocab(remaining, genre_vocab)
    if found:
        prefs["genre"], alias = found
        detected["genre"] = f"{prefs['genre']} (keyword '{alias}')"
        remaining = re.sub(re.escape(alias), " ", remaining)

    mood_vocab = kb.vocabulary("mood")
    mood_vocab.update({s["mood"]: s["mood"] for s in songs})
    mood = keyword_mood(remaining, mood_vocab)
    if mood:
        prefs["mood"] = mood
        detected["mood"] = f"{mood} (keyword)"
    elif classifier is not None:
        label, prob = classifier.predict(text)
        if label and prob >= MIN_CONFIDENCE:
            prefs["mood"] = label
            detected["mood"] = f"{label} (mood classifier, p={prob:.2f})"

    number = re.search(r"energy\s*(?:of|=|:|level)?\s*(0?\.\d+|1(?:\.0+)?|0)\b", lowered)
    if number:
        prefs["energy"] = max(0.0, min(1.0, float(number.group(1))))
        detected["energy"] = f"{prefs['energy']:.2f} (number in request)"
    else:
        word = _find_vocab(remaining, {w: w for w in ENERGY_WORDS})
        if word:
            prefs["energy"] = ENERGY_WORDS[word[0]]
            detected["energy"] = f"{prefs['energy']:.2f} (word '{word[0]}')"

    decade = _parse_decade(lowered)
    if decade:
        prefs["decade"] = decade
        detected["decade"] = decade

    if re.search(r"\b(no vocals|no lyrics|without (vocals|lyrics)|instrumental)\b", lowered):
        prefs["likes_instrumental"] = True
        detected["likes_instrumental"] = "yes (keyword)"
    elif re.search(r"\b(vocals|singing|lyrics|sing along|sing-along)\b", lowered):
        prefs["likes_instrumental"] = False
        detected["likes_instrumental"] = "no (keyword)"

    if re.search(r"\bacoustic\b", lowered):
        prefs["likes_acoustic"] = True
        detected["likes_acoustic"] = "yes (keyword)"
    elif re.search(r"\b(electronic|electric|synths?)\b", lowered):
        prefs["likes_acoustic"] = False
        detected["likes_acoustic"] = "no (keyword)"

    pop = _find_vocab(remaining, {w: w for w in POPULAR_WORDS})
    if pop:
        prefs["popularity"] = POPULAR_WORDS[pop[0]]
        detected["popularity"] = f"{prefs['popularity']} (word '{pop[0]}')"

    tag_vocab = sorted({t for s in songs for t in s.get("mood_tags", [])})
    tags = [t for t in tag_vocab if contains_phrase(remaining, t)]
    if tags:
        prefs["mood_tags"] = tags
        detected["mood_tags"] = ", ".join(tags)

    count = re.search(r"\b(?:top\s+)?(\d{1,2})\s+(?:songs|tracks|recs|recommendations)\b|\btop\s+(\d{1,2})\b", lowered)
    if count:
        parsed.k = max(1, min(10, int(count.group(1) or count.group(2))))
        detected["k"] = str(parsed.k)

    return parsed
