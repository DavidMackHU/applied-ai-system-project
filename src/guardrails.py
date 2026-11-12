"""Input and output guardrails for VibeFinder AI.

Input checks run before any AI step: empty or oversized requests, prompt-injection
attempts, and wellbeing signals. Output checks run after the agent finishes:
every recommended song must exist in the catalog, scores must be finite, and any
preferences produced by the optional LLM parser are validated against a strict
allow-list before they reach the recommender.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger("vibefinder.guardrails")

MAX_REQUEST_CHARS = 500
INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"(reveal|print|show|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"\byou are now\b",
    r"\bact as (an? )?(unfiltered|jailbroken|dan)\b",
    r"\bsystem prompt\b",
    r"<\s*/?\s*(system|script)\b",
]
WELLBEING_PATTERNS = [
    r"\bkill (myself|me)\b", r"\bend (it all|my life)\b", r"\bsuicid", r"\bself[- ]harm",
    r"\bwant to die\b", r"\bhurt myself\b", r"\bno reason to live\b",
]
SUPPORT_MESSAGE = (
    "It sounds like you might be going through something really hard. Music can help, "
    "but you deserve support from a person too. In the US you can call or text 988 "
    "(Suicide & Crisis Lifeline) any time; outside the US, findahelpline.com lists free services."
)
ALLOWED_PREF_KEYS = {
    "genre", "mood", "energy", "likes_acoustic", "likes_instrumental",
    "valence", "decade", "popularity", "mood_tags", "k",
}


@dataclass
class GuardrailResult:
    """Outcome of an input check: allowed or not, with a user-facing reason."""
    allowed: bool
    reason: str = ""
    category: str = "ok"


def check_input(text: Any) -> GuardrailResult:
    """Validate a raw request before any parsing or retrieval happens."""
    if not isinstance(text, str) or not text.strip():
        return GuardrailResult(False, "Please describe what you want to listen to.", "empty")
    if len(text) > MAX_REQUEST_CHARS:
        return GuardrailResult(
            False, f"Request is too long ({len(text)} characters, max {MAX_REQUEST_CHARS}).",
            "too_long")
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            logger.warning("Blocked possible prompt injection: %r", text[:120])
            return GuardrailResult(
                False, "That looks like an instruction to change how I work, so I ignored it. "
                       "Tell me about the music you want instead.", "prompt_injection")
    return GuardrailResult(True)


def wellbeing_note(text: str) -> Optional[str]:
    """Return a support message if the request contains self-harm language, else None."""
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in WELLBEING_PATTERNS):
        logger.warning("Wellbeing signal detected; adding support message.")
        return SUPPORT_MESSAGE
    return None


def clamp(value: Any, low: float = 0.0, high: float = 1.0) -> Optional[float]:
    """Convert to float and clamp into [low, high]; return None if not a finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(low, min(high, number))


def validate_llm_prefs(raw: Any, genres: Set[str], moods: Set[str],
                       tags: Set[str]) -> Dict[str, Any]:
    """Keep only known keys and valid values from LLM output; drop anything else with a log line."""
    if not isinstance(raw, dict):
        logger.warning("LLM output was not a JSON object; ignoring it.")
        return {}
    clean: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in ALLOWED_PREF_KEYS or value is None or value == "":
            continue
        if key == "genre":
            if str(value).lower() in genres:
                clean["genre"] = str(value).lower()
            else:
                logger.info("Dropped LLM genre not in vocabulary: %r", value)
        elif key == "mood":
            if str(value).lower() in moods:
                clean["mood"] = str(value).lower()
            else:
                logger.info("Dropped LLM mood not in vocabulary: %r", value)
        elif key in ("energy", "valence"):
            number = clamp(value)
            if number is not None:
                clean[key] = round(number, 2)
        elif key == "popularity":
            number = clamp(value, 0, 100)
            if number is not None:
                clean[key] = int(number)
        elif key == "k":
            number = clamp(value, 1, 10)
            if number is not None:
                clean[key] = int(number)
        elif key in ("likes_acoustic", "likes_instrumental"):
            if isinstance(value, bool):
                clean[key] = value
        elif key == "decade":
            if re.fullmatch(r"(19|20)\d0s", str(value)):
                clean[key] = str(value)
        elif key == "mood_tags" and isinstance(value, list):
            kept = [str(t).lower() for t in value if str(t).lower() in tags]
            if kept:
                clean[key] = kept
    return clean


def validate_recommendations(results: Iterable, catalog_ids: Set[int]) -> List:
    """Drop any result whose song is not in the catalog or whose score is not finite."""
    valid = []
    for song, score, reasons in results:
        if song.get("id") not in catalog_ids:
            logger.error("Dropped recommendation not in catalog: %r", song.get("title"))
            continue
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            logger.error("Dropped recommendation with invalid score: %r", score)
            continue
        valid.append((song, score, reasons))
    return valid
