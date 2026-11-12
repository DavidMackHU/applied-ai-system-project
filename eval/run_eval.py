"""Evaluation harness for VibeFinder AI.

Runs the agent on every case in eval/cases.json under three configurations
(full system, retrieval off, mood classifier off), checks each result against
its expectations, and measures the mood classifier against the keyword baseline
on the hand-written holdout set in eval/mood_holdout.csv.

Usage (from the project root):
    python -m eval.run_eval               # prints the report, writes eval/results.md and .json

Exit code is 1 if any case fails in the full configuration, so it can gate CI.
"""

from __future__ import annotations

import json
import os
import sys
from statistics import mean
from typing import Dict, List, Tuple

from src.agent import AgentResult, VibeAgent
from src.app import setup_logging
from src.intent import default_classifier, keyword_mood, load_rows
from src.knowledge import PROJECT_ROOT, KnowledgeBase
from src.main import format_table
from src.recommender import load_songs

EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")
CONFIGS = {
    "full": {"use_rag": True, "use_classifier": True},
    "no_rag": {"use_rag": False, "use_classifier": True},
    "no_classifier": {"use_rag": True, "use_classifier": False},
}


def check_case(result: AgentResult, expect: Dict) -> List[Tuple[str, bool, str]]:
    """Return (check name, passed, observed value) for every expectation in a case."""
    checks = [("status", result.status == expect["status"], result.status)]
    songs = [s for s, _, _ in result.recommendations]
    if "top_genre_in" in expect:
        top = songs[0]["genre"] if songs else None
        checks.append(("top_genre", top in expect["top_genre_in"], str(top)))
    if "genres_in_top_k" in expect:
        rule = expect["genres_in_top_k"]
        count = sum(s["genre"] in rule["genres"] for s in songs)
        checks.append(("genres_in_top_k", count >= rule["min"], f"{count}/{len(songs)}"))
    if "moods_in_top_k" in expect:
        rule = expect["moods_in_top_k"]
        count = sum(s["mood"] in rule["moods"] for s in songs)
        checks.append(("moods_in_top_k", count >= rule["min"], f"{count}/{len(songs)}"))
    if "energy_range" in expect:
        low, high = expect["energy_range"]
        avg = mean(s["energy"] for s in songs) if songs else -1
        checks.append(("energy_range", low <= avg <= high, f"{avg:.2f}"))
    if "min_confidence" in expect:
        checks.append(("min_confidence", result.confidence >= expect["min_confidence"],
                       f"{result.confidence:.2f}"))
    if "max_confidence" in expect:
        checks.append(("max_confidence", result.confidence <= expect["max_confidence"],
                       f"{result.confidence:.2f}"))
    if "excludes_title" in expect:
        titles = [s["title"] for s in songs]
        checks.append(("excludes_title", expect["excludes_title"] not in titles, "excluded"
                       if expect["excludes_title"] not in titles else "present"))
    if expect.get("unique_artists"):
        artists = [s["artist"] for s in songs]
        checks.append(("unique_artists", len(set(artists)) == len(artists),
                       f"{len(set(artists))}/{len(artists)}"))
    if "avoid_moods" in expect:
        bad = [s["title"] for s in songs if s["mood"] in expect["avoid_moods"]]
        checks.append(("avoid_moods", not bad, ", ".join(bad) or "none"))
    if expect.get("safety_note"):
        checks.append(("safety_note", bool(result.safety_note), "shown" if result.safety_note else "missing"))
    return checks


def run_cases(songs, kb) -> Dict[str, List[Dict]]:
    """Run every case under every configuration and return per-config result rows."""
    with open(os.path.join(EVAL_DIR, "cases.json"), encoding="utf-8") as handle:
        cases = json.load(handle)
    report = {}
    for name, options in CONFIGS.items():
        agent = VibeAgent(songs, kb=kb, **options)
        rows = []
        for case in cases:
            result = agent.run(case["request"])
            checks = check_case(result, case["expect"])
            rows.append({
                "id": case["id"], "request": case["request"], "status": result.status,
                "confidence": result.confidence, "label": result.confidence_label,
                "attempts": result.iterations, "passed": all(ok for _, ok, _ in checks),
                "checks": [{"name": n, "passed": ok, "observed": obs} for n, ok, obs in checks],
                "top3": [s["title"] for s, _, _ in result.recommendations[:3]],
            })
        report[name] = rows
    return report


def run_classifier_eval(kb) -> Dict:
    """Compare keyword-only, classifier-only, and hybrid mood detection on the holdout set."""
    rows = load_rows(os.path.join(EVAL_DIR, "mood_holdout.csv"))
    clf = default_classifier()
    vocab = kb.vocabulary("mood")
    results = {"keyword": 0, "classifier": 0, "hybrid": 0}
    details = []
    for text, label in rows:
        kw = keyword_mood(text, vocab)
        nb, prob = clf.predict(text)
        nb_used = nb if prob >= 0.5 else None
        hybrid = kw or nb_used
        results["keyword"] += kw == label
        results["classifier"] += nb_used == label
        results["hybrid"] += hybrid == label
        details.append({"text": text, "label": label, "keyword": kw, "classifier": nb,
                        "p": prob, "hybrid": hybrid})
    n = len(rows)
    covered = [d for d in details if d["classifier"] and d["p"] >= 0.5]
    return {
        "n": n,
        "accuracy": {k: round(v / n, 3) for k, v in results.items()},
        "correct": results,
        "classifier_coverage": round(len(covered) / n, 3),
        "classifier_precision_when_used": round(
            sum(d["classifier"] == d["label"] for d in covered) / len(covered), 3) if covered else 0.0,
        "details": details,
    }


def render(report: Dict, clf: Dict) -> str:
    """Build the markdown/terminal report text."""
    out = ["# VibeFinder AI Evaluation Report", ""]
    full = report["full"]
    out.append("## Case results (full system)")
    out.append("")
    out.append("```")
    rows = [(r["id"], r["status"], f"{r['confidence']:.2f}" if r["status"] == "ok" else "-",
             str(r["attempts"]), "PASS" if r["passed"] else "FAIL",
             "\n".join(f"{'ok ' if c['passed'] else 'XX '}{c['name']}: {c['observed']}" for c in r["checks"]))
            for r in full]
    out.append(format_table(["Case", "Status", "Conf.", "Tries", "Result", "Checks"], rows))
    out.append("```")
    out.append("")
    out.append("## Summary by configuration")
    out.append("")
    out.append("| Configuration | Cases passed | Avg confidence (ok cases) | Failed cases |")
    out.append("|---|---|---|---|")
    for name, rows_ in report.items():
        passed = sum(r["passed"] for r in rows_)
        confs = [r["confidence"] for r in rows_ if r["status"] == "ok"]
        failed = ", ".join(r["id"] for r in rows_ if not r["passed"]) or "none"
        out.append(f"| {name} | {passed}/{len(rows_)} | {mean(confs):.2f} | {failed} |")
    out.append("")
    out.append("## Mood classifier vs. keyword baseline (holdout set)")
    out.append("")
    acc = clf["accuracy"]
    out.append(f"Holdout: {clf['n']} hand-written requests, 5 per mood, most without a mood keyword.")
    out.append("")
    out.append("| Method | Correct | Accuracy |")
    out.append("|---|---|---|")
    out.append(f"| Keyword baseline | {clf['correct']['keyword']}/{clf['n']} | {acc['keyword']:.0%} |")
    out.append(f"| Naive Bayes classifier (p >= 0.5) | {clf['correct']['classifier']}/{clf['n']} | {acc['classifier']:.0%} |")
    out.append(f"| Hybrid (keyword, then classifier) | {clf['correct']['hybrid']}/{clf['n']} | {acc['hybrid']:.0%} |")
    out.append("")
    out.append(f"Classifier answered {clf['classifier_coverage']:.0%} of requests and was right "
               f"{clf['classifier_precision_when_used']:.0%} of the time when it answered.")
    misses = [d for d in clf["details"] if d["hybrid"] != d["label"]]
    if misses:
        out.append("")
        out.append("Hybrid misses:")
        out.append("")
        out += [f"- \"{d['text']}\": expected {d['label']}, got {d['hybrid'] or 'nothing'}"
                f" (classifier said {d['classifier']} at p={d['p']:.2f})" for d in misses]
    return "\n".join(out)


def main() -> int:
    """Run all evaluations, print the report, and save results files."""
    setup_logging(quiet=True)
    songs = load_songs("data/songs.csv")
    kb = KnowledgeBase()
    report = run_cases(songs, kb)
    clf = run_classifier_eval(kb)
    text = render(report, clf)
    print(text)
    with open(os.path.join(EVAL_DIR, "results.md"), "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    with open(os.path.join(EVAL_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump({"cases": report, "classifier": clf}, handle, indent=2)
    return 0 if all(r["passed"] for r in report["full"]) else 1


if __name__ == "__main__":
    sys.exit(main())
