"""Self-critique for VibeFinder's agent: score a result list and propose a revision.

The critic measures how well a top-k list fits the listener profile (genre fit,
mood fit, energy fit, artist diversity, and match strength), combines them into a
0-1 confidence score, and names concrete issues. ``suggest_revision`` maps each
issue to one change the agent can try next (switch scoring mode or turn on the
diversity penalty). The agent keeps whichever attempt the critic scores highest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Tuple

from src.recommender import max_possible_score

METRIC_WEIGHTS = {"genre_fit": 0.25, "mood_fit": 0.25, "energy_fit": 0.2,
                  "diversity": 0.1, "strength": 0.2}
INFERRED_WEIGHT = 0.5   # preferences the agent inferred count half as much as explicit ones
METRIC_PREF = {"genre_fit": "genre", "mood_fit": "mood", "energy_fit": "energy"}
HIGH_CONFIDENCE = 0.8
MEDIUM_CONFIDENCE = 0.6


@dataclass
class Critique:
    """The critic's verdict on one result list."""
    confidence: float
    label: str
    metrics: Dict[str, Optional[float]]
    issues: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def confidence_label(score: float) -> str:
    """Map a 0-1 confidence score to high / medium / low."""
    if score >= HIGH_CONFIDENCE:
        return "high"
    if score >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


def _label_fit(songs: List[Dict], key: str, target: Optional[str],
               related: Dict[str, float]) -> Optional[float]:
    """Average per-song credit: 1 for an exact label match, the similarity for a related label."""
    if not target and not related:
        return None
    scores = []
    for song in songs:
        value = str(song.get(key, "")).lower()
        scores.append(1.0 if target and value == target else related.get(value, 0.0))
    return round(mean(scores), 3) if scores else 0.0


def evaluate(prefs: Dict, results: List[Tuple[Dict, float, List[str]]], catalog: List[Dict],
             mode: str, diversity: bool, exclude_ids: Optional[set] = None) -> Critique:
    """Score a result list against the profile and list the problems found."""
    if not results:
        return Critique(0.0, "low", {}, ["no_results"], ["No songs could be scored."])
    songs = [song for song, _, _ in results]
    catalog_genres = {s["genre"] for s in catalog}
    catalog_moods = {s["mood"] for s in catalog}
    genre, mood = prefs.get("genre"), prefs.get("mood")

    metrics: Dict[str, Optional[float]] = {
        "genre_fit": _label_fit(songs, "genre", genre, prefs.get("related_genres") or {}),
        "mood_fit": _label_fit(songs, "mood", mood, prefs.get("related_moods") or {}),
        "energy_fit": None,
        "diversity": round(len({s["artist"] for s in songs}) / len(songs), 3),
        "strength": None,
    }
    if prefs.get("energy") is not None:
        gap = mean(abs(s["energy"] - prefs["energy"]) for s in songs)
        metrics["energy_fit"] = round(max(0.0, 1.0 - gap / 0.5), 3)
    ideal = max_possible_score(prefs, mode, catalog_genres, catalog_moods)
    if ideal > 0:
        metrics["strength"] = round(min(1.0, results[0][1] / ideal), 3)

    inferred = set(prefs.get("inferred") or [])
    weights = {k: METRIC_WEIGHTS[k] * (INFERRED_WEIGHT if METRIC_PREF.get(k) in inferred else 1.0)
               for k, v in metrics.items() if v is not None}
    weight_sum = sum(weights.values())
    confidence = round(sum(w * metrics[k] for k, w in weights.items()) / weight_sum, 3)

    critique = Critique(confidence, confidence_label(confidence), metrics)
    if genre and genre not in catalog_genres:
        critique.notes.append(f"The catalog has no '{genre}' songs, so related genres stand in for it.")
    elif genre:
        available_count = sum(1 for s in catalog if s["genre"] == genre and s["id"] not in (exclude_ids or set()))
        if available_count == 0:
            critique.notes.append(
                f"No other '{genre}' songs are in the catalog, so related genres fill the list.")
        elif available_count < len(results):
            critique.notes.append(
                f"Only {available_count} '{genre}' song(s) are in the catalog, "
                "so related genres fill the rest of the list.")
    # Issue order is revision priority: gaps in what the listener explicitly asked for come
    # first, then artist repetition, then gaps in preferences the agent only inferred.
    gaps = []
    if metrics["genre_fit"] is not None and metrics["genre_fit"] < 0.4:
        gaps.append(("genre_gap", "genre"))
    if metrics["mood_fit"] is not None and metrics["mood_fit"] < 0.4:
        gaps.append(("mood_gap", "mood"))
    if metrics["energy_fit"] is not None and metrics["energy_fit"] < 0.6:
        gaps.append(("energy_gap", "energy"))
    critique.issues += [issue for issue, pref in gaps if pref not in inferred]
    if metrics["diversity"] < 0.8 and not diversity:
        critique.issues.append("low_diversity")
    critique.issues += [issue for issue, pref in gaps if pref in inferred]
    if metrics["strength"] is not None and metrics["strength"] < 0.5:
        critique.notes.append("Even the top song matches only part of the request.")

    if genre and mood and prefs.get("energy") is not None:
        both = [s for s in catalog
                if (s["genre"] == genre or (prefs.get("related_genres") or {}).get(s["genre"], 0) >= 0.8)
                and (s["mood"] == mood or (prefs.get("related_moods") or {}).get(s["mood"], 0) >= 0.8)
                and abs(s["energy"] - prefs["energy"]) <= 0.2]
        if not both:
            critique.notes.append(
                f"No song in the catalog is {genre} + {mood} at energy near {prefs['energy']:.2f}; "
                "the request has a built-in conflict.")
    return critique


REVISIONS = {
    "genre_gap": ({"mode": "genre_first"}, "genre fit is low, so try weighting genre more"),
    "mood_gap": ({"mode": "mood_first"}, "mood fit is low, so try weighting mood more"),
    "energy_gap": ({"mode": "energy_focused"}, "energy fit is low, so try weighting energy more"),
    "low_diversity": ({"diversity": True}, "artists repeat too often, so turn on the diversity penalty"),
}


def suggest_revision(critique: Critique, tried: List[Dict]) -> Optional[Tuple[Dict, str]]:
    """Return the first untried (change, reason) that addresses one of the critique's issues."""
    for issue in critique.issues:
        if issue in REVISIONS:
            change, reason = REVISIONS[issue]
            if change not in tried:
                return change, reason
    return None
