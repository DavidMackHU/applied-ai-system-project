"""VibeFinder AI agent: plan -> retrieve -> recommend -> critique -> revise.

The agent turns a free-text request into ranked, explained recommendations:

1. Guardrails check the raw request (empty, too long, prompt injection, wellbeing).
2. Plan: parse the request (local parser, plus Claude if enabled) and decide which
   tools to use (knowledge retrieval, catalog search) based on what was found.
3. Act: retrieve knowledge-base guides and fold their facts into the profile
   (related genres/moods, energy targets, instrumental preference), seed from any
   mentioned song, then run the recommender.
4. Check: the critic scores the result list and names issues.
5. Revise: try one change per issue (scoring mode or diversity penalty), up to
   ``max_iterations`` attempts, and keep the best-scoring attempt.

Every step is recorded in ``AgentResult.trace`` so the reasoning can be printed,
saved as JSONL, and audited.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src import guardrails
from src.critic import HIGH_CONFIDENCE, Critique, evaluate, suggest_revision
from src.intent import MoodClassifier, default_classifier
from src.knowledge import Chunk, KnowledgeBase
from src.parser import ParsedRequest, parse_request
from src.recommender import rank_songs

logger = logging.getLogger("vibefinder.agent")
DIVERSITY_PREFERENCE = 0.05


@dataclass
class TraceStep:
    """One recorded step of the agent's reasoning."""
    step: int
    phase: str
    action: str
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Everything the agent produced for one request."""
    request: str
    status: str
    message: str
    recommendations: List = field(default_factory=list)
    confidence: float = 0.0
    confidence_label: str = "low"
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    understood: Dict[str, str] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "balanced"
    diversity: bool = False
    iterations: int = 0
    safety_note: Optional[str] = None
    trace: List[TraceStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of the result."""
        data = asdict(self)
        data["recommendations"] = [
            {"title": s["title"], "artist": s["artist"], "genre": s["genre"], "mood": s["mood"],
             "energy": s["energy"], "score": score, "reasons": reasons}
            for s, score, reasons in self.recommendations
        ]
        return data


class VibeAgent:
    """Agentic, retrieval-augmented music recommender with guardrails and self-critique."""

    def __init__(self, songs: List[Dict], kb: Optional[KnowledgeBase] = None,
                 classifier: Optional[MoodClassifier] = None, use_rag: bool = True,
                 use_classifier: bool = True, llm_parser: Any = None,
                 max_iterations: int = 3, k: int = 5):
        """Configure the agent; RAG and the mood classifier can be switched off for ablations."""
        if not songs:
            raise ValueError("The song catalog is empty.")
        self.songs = songs
        self.kb = kb or KnowledgeBase()
        self.classifier = (classifier or default_classifier()) if use_classifier else None
        self.use_rag = use_rag
        self.llm_parser = llm_parser
        self.max_iterations = max(1, max_iterations)
        self.k = k
        self.catalog_ids = {s["id"] for s in songs}
        self._trace: List[TraceStep] = []

    # ------------------------------------------------------------------ helpers
    def _log(self, phase: str, action: str, detail: str, **data: Any) -> None:
        """Append a trace step and mirror it to the log file."""
        self._trace.append(TraceStep(len(self._trace) + 1, phase, action, detail, data))
        logger.info("[%s] %s: %s", phase, action, detail)

    @staticmethod
    def _merge(target: Dict[str, float], items: Dict[str, float], sources: Dict[str, str],
               prefix: str, source: str, skip: Optional[str] = None) -> List[str]:
        """Merge weighted labels into `target` (keeping the max weight); return labels added."""
        added = []
        for label, weight in items.items():
            if label == skip or weight <= target.get(label, 0.0):
                continue
            target[label] = weight
            sources[f"{prefix}:{label}"] = source
            added.append(f"{label} {weight:.1f}")
        return added

    def _apply_knowledge(self, prefs: Dict, understood: Dict[str, str],
                         citations: List[Dict], activity: Optional[Chunk]) -> None:
        """Fold retrieved guide facts into the profile, recording a citation for each use."""
        sources: Dict[str, str] = prefs.setdefault("sources", {})
        inferred: List[str] = prefs.setdefault("inferred", [])
        related_g: Dict[str, float] = prefs.setdefault("related_genres", {})
        related_m: Dict[str, float] = prefs.setdefault("related_moods", {})

        def cite(chunk: Chunk, used_for: str) -> None:
            citations.append({"id": chunk.id, "used_for": used_for})
            self._log("act", "retrieve_knowledge", f"{chunk.id} -> {used_for}")

        if activity is not None:
            label = f"{activity.title} guide"
            before = len(citations)
            if not prefs.get("genre") and activity.related_genres:
                added = self._merge(related_g, activity.related_genres, sources, "genre", label)
                if added:
                    inferred.append("genre")
                    cite(activity, "suggested genres: " + ", ".join(added))
            if not prefs.get("mood") and activity.moods:
                prefs["mood"] = activity.moods[0]
                inferred.append("mood")
                understood["mood"] = f"{prefs['mood']} ({label})"
                others = {m: 0.7 for m in activity.moods[1:]}
                self._merge(related_m, others, sources, "mood", label, skip=prefs["mood"])
                cite(activity, f"mood {prefs['mood']}" + (f" (+ related {', '.join(others)})" if others else ""))
            if prefs.get("energy") is None and activity.energy_range:
                low, high = activity.energy_range
                prefs["energy"] = round((low + high) / 2, 2)
                inferred.append("energy")
                understood["energy"] = f"{prefs['energy']:.2f} ({label})"
                cite(activity, f"energy target {prefs['energy']:.2f} from range {low}-{high}")
            if (prefs.get("likes_instrumental") is None
                    and activity.fields.get("instrumental", "").lower() == "preferred"):
                prefs["likes_instrumental"] = True
                understood["likes_instrumental"] = f"yes ({label})"
                cite(activity, "instrumental tracks preferred")
            if len(citations) == before:
                self._log("act", "retrieve_knowledge",
                          f"{activity.id} matched, but the profile already had genre, mood, and energy")

        genre = prefs.get("genre")
        genre_doc = self.kb.get("genre", genre) if genre else None
        if genre_doc:
            added = self._merge(related_g, genre_doc.related_genres, sources, "genre",
                                f"{genre} guide", skip=genre)
            if added:
                cite(genre_doc, "related genres: " + ", ".join(added))

        mood = prefs.get("mood")
        mood_doc = self.kb.get("mood", mood) if mood else None
        if mood_doc:
            added = self._merge(related_m, mood_doc.related_moods, sources, "mood",
                                f"{mood} guide", skip=mood)
            if added:
                cite(mood_doc, "related moods: " + ", ".join(added))

        if prefs.get("energy") is None:
            for doc in (mood_doc, genre_doc):
                if doc and doc.energy_range:
                    low, high = doc.energy_range
                    prefs["energy"] = round((low + high) / 2, 2)
                    inferred.append("energy")
                    understood["energy"] = f"{prefs['energy']:.2f} ({doc.title} guide)"
                    cite(doc, f"energy target {prefs['energy']:.2f} from range {low}-{high}")
                    break

    # ------------------------------------------------------------------ main loop
    def run(self, request: str, k: Optional[int] = None) -> AgentResult:
        """Run the full plan-act-check-revise loop for one request."""
        self._trace = []
        check = guardrails.check_input(request)
        if not check.allowed:
            self._log("guardrail", "check_input", f"blocked ({check.category}): {check.reason}")
            return AgentResult(request=str(request), status="blocked", message=check.reason,
                               trace=self._trace)
        self._log("guardrail", "check_input", "request passed input checks")
        safety = guardrails.wellbeing_note(request)
        if safety:
            self._log("guardrail", "wellbeing", "self-harm language detected; support message added")

        # ---- PLAN
        parsed: ParsedRequest = parse_request(request, self.songs, self.kb, self.classifier)
        understood = dict(parsed.detected)
        self._log("plan", "parse_request",
                  ", ".join(f"{key}={val}" for key, val in understood.items()) or "nothing explicit found",
                  prefs=dict(parsed.prefs))
        if self.llm_parser is not None:
            self._apply_llm(request, parsed, understood)

        query = request
        for song in parsed.mentions:
            query = re.sub(re.escape(song["title"]), " ", query, flags=re.IGNORECASE)
        hits = self.kb.search(query, top_k=5) if self.use_rag else []
        activity = next((h.chunk for h in hits if h.chunk.kind == "activity" and h.score >= 0.3), None)
        if hits:
            self._log("plan", "search_knowledge",
                      "; ".join(f"{h.chunk.id} ({h.score:.2f})" for h in hits))

        steps = []
        if parsed.mentions:
            steps.append(f"search_catalog: seed profile from '{parsed.mentions[0]['title']}'")
        if self.use_rag:
            if activity:
                steps.append(f"retrieve_knowledge: apply '{activity.title}' activity guide")
            if parsed.prefs.get("genre") or parsed.mentions:
                steps.append("retrieve_knowledge: add related genres from the genre guide")
            steps.append("retrieve_knowledge: add related moods from the mood guide")
        steps += ["recommend: score and rank the catalog", "critique: check fit and confidence",
                  f"revise: up to {self.max_iterations - 1} alternative strategies if confidence is not high"]
        self._log("plan", "make_plan", " | ".join(steps), steps=steps)

        if not parsed.has_music_signal() and activity is None:
            message = ("I couldn't find anything music-related in that request. Try naming a genre, "
                       "mood, activity, or a song you like (for example: 'chill lofi for studying').")
            self._log("final", "off_topic", "no music signal; returning help message")
            return AgentResult(request=request, status="off_topic", message=message,
                               understood=understood, safety_note=safety, trace=self._trace)

        # ---- ACT: build profile
        prefs: Dict[str, Any] = dict(parsed.prefs)
        citations: List[Dict] = []
        exclude = set()
        if parsed.mentions:
            seed = parsed.mentions[0]
            exclude = {s["id"] for s in parsed.mentions}
            for key, value in (("genre", seed["genre"]), ("mood", seed["mood"]),
                               ("energy", seed["energy"])):
                if prefs.get(key) is None:
                    prefs[key] = value
                    understood[key] = f"{value} (from '{seed['title']}')"
            if not prefs.get("mood_tags") and seed.get("mood_tags"):
                prefs["mood_tags"] = list(seed["mood_tags"])
                understood["mood_tags"] = f"{', '.join(seed['mood_tags'])} (from '{seed['title']}')"
            self._log("act", "search_catalog",
                      f"seeded from '{seed['title']}' by {seed['artist']}; excluding mentioned songs",
                      excluded=sorted(exclude))
        if self.use_rag:
            self._apply_knowledge(prefs, understood, citations, activity)
        k = k or parsed.k or self.k

        # ---- ACT / CHECK / REVISE loop
        state = {"mode": "balanced", "diversity": False}
        tried: List[Dict] = []
        attempts = []
        for iteration in range(1, self.max_iterations + 1):
            results = rank_songs(prefs, self.songs, k=k, mode=state["mode"],
                                 diversity=state["diversity"], exclude_ids=exclude)
            critique: Critique = evaluate(prefs, results, self.songs, state["mode"], state["diversity"], exclude)
            attempts.append((critique.confidence, iteration, dict(state), results, critique))
            self._log("act", "recommend",
                      f"attempt {iteration} (mode={state['mode']}, diversity={'on' if state['diversity'] else 'off'}): "
                      + ", ".join(s["title"] for s, _, _ in results))
            self._log("check", "critique",
                      f"confidence {critique.confidence:.2f} ({critique.label}); "
                      + ", ".join(f"{m}={v:.2f}" for m, v in critique.metrics.items() if v is not None)
                      + (f"; issues: {', '.join(critique.issues)}" if critique.issues else "; no issues"),
                      metrics=critique.metrics, issues=critique.issues)
            if critique.confidence >= HIGH_CONFIDENCE and not critique.issues:
                self._log("check", "stop", "confidence is high and no issues remain")
                break
            if iteration == self.max_iterations:
                self._log("check", "stop", "reached the attempt limit")
                break
            revision = suggest_revision(critique, tried)
            if revision is None:
                self._log("check", "stop", "no untried revision addresses the remaining issues")
                break
            change, reason = revision
            tried.append(change)
            state.update(change)
            self._log("revise", "change_strategy", f"{reason}: {change}", change=change)

        def selection_score(attempt) -> float:
            # Confidence decides, but an attempt that still repeats an artist pays a small
            # penalty, so a diverse list wins when it costs less than 0.05 confidence.
            return attempt[0] - (DIVERSITY_PREFERENCE if "low_diversity" in attempt[4].issues else 0.0)

        confidence, iteration, best_state, results, critique = max(
            attempts, key=lambda a: (selection_score(a), -a[1]))
        results = guardrails.validate_recommendations(results, self.catalog_ids)
        self._log("final", "select_best",
                  f"kept attempt {iteration} of {len(attempts)} (confidence {confidence:.2f}, "
                  f"mode={best_state['mode']}, diversity={'on' if best_state['diversity'] else 'off'}); "
                  f"rule: highest confidence, minus {DIVERSITY_PREFERENCE} if an artist repeats")

        message = self._summarize(results, understood, critique)
        return AgentResult(
            request=request, status="ok", message=message, recommendations=results,
            confidence=confidence, confidence_label=critique.label, metrics=critique.metrics,
            notes=critique.notes, understood=understood, citations=citations,
            mode=best_state["mode"], diversity=best_state["diversity"], iterations=len(attempts),
            safety_note=safety, trace=self._trace,
        )

    def _apply_llm(self, request: str, parsed: ParsedRequest, understood: Dict[str, str]) -> None:
        """Fill gaps in the local parse with validated Claude output (local keyword hits win)."""
        genres = sorted(set(self.kb.vocabulary("genre").values()) | {s["genre"] for s in self.songs})
        moods = sorted(set(self.kb.vocabulary("mood").values()) | {s["mood"] for s in self.songs})
        tags = sorted({t for s in self.songs for t in s.get("mood_tags", [])})
        raw = self.llm_parser.parse(request, genres, moods, tags)
        if raw is None:
            self._log("plan", "llm_parse", "Claude parser unavailable; continuing with local parse")
            return
        clean = guardrails.validate_llm_prefs(raw, set(genres), set(moods), set(tags))
        dropped = sorted(set(raw) - set(clean))
        added = []
        for key, value in clean.items():
            if key != "k" and parsed.prefs.get(key) is None:
                parsed.prefs[key] = value
                understood[key] = f"{value} (Claude)"
                added.append(key)
        self._log("plan", "llm_parse",
                  f"Claude added {', '.join(added) or 'nothing new'}"
                  + (f"; guardrail dropped {', '.join(dropped)}" if dropped else ""))

    @staticmethod
    def _summarize(results: List, understood: Dict[str, str], critique: Critique) -> str:
        """Write a short grounded explanation of the top pick and how the request was read."""
        if not results:
            return "No songs matched."
        top, _, reasons = results[0]
        lead = (f"Top pick: {top['title']} by {top['artist']} ({top['genre']}, {top['mood']}, "
                f"energy {top['energy']:.2f}). It scored highest because of: "
                + "; ".join(r.rsplit(" (", 1)[0] for r in reasons[:3]) + ".")
        reading = ", ".join(f"{k} = {v}" for k, v in understood.items())
        return lead + (f" I read your request as: {reading}." if reading else "")
