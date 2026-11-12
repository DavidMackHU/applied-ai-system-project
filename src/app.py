"""Command-line interface for VibeFinder AI.

Examples (run from the project root):

    python -m src.app "chill lofi for studying"
    python -m src.app "something like Storm Runner" --trace
    python -m src.app "k-pop dance hits" --no-rag          # ablation: retrieval off
    python -m src.app --demo                              # three sample requests
    python -m src.app --interactive                       # type requests one by one
    python -m src.app "sad songs" --llm                   # use Claude to parse (needs credentials)

Logs go to logs/vibefinder.log. ``--save-trace`` also writes the agent's step-by-step
trace as JSON Lines to logs/traces/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import List, Optional

from src.agent import AgentResult, VibeAgent
from src.knowledge import PROJECT_ROOT
from src.main import format_table
from src.recommender import load_songs

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DEMO_REQUESTS = [
    "chill lofi for studying",
    "something like Storm Runner for the gym",
    "k-pop dance hits",
]


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Log everything to logs/vibefinder.log; show warnings on screen (info if verbose, none if quiet)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger("vibefinder")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "vibefinder.log"), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verbose else logging.CRITICAL if quiet else logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(file_handler)
    root.addHandler(console)


def save_trace(result: AgentResult) -> str:
    """Write the result and its trace to logs/traces/<time>_<slug>.jsonl and return the path."""
    trace_dir = os.path.join(LOG_DIR, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", result.request.lower()).strip("-")[:40] or "empty"
    path = os.path.join(trace_dir, f"{datetime.now():%Y%m%d-%H%M%S}_{slug}.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for step in result.trace:
            handle.write(json.dumps({"type": "step", **step.__dict__}) + "\n")
        summary = result.to_dict()
        summary.pop("trace")
        handle.write(json.dumps({"type": "result", **summary}) + "\n")
    return path


def render(result: AgentResult, show_trace: bool = False) -> str:
    """Format an AgentResult for the terminal."""
    lines = [f'Request: "{result.request}"']
    if result.safety_note:
        lines.append(f"[Support] {result.safety_note}")
    if result.status != "ok":
        lines.append(f"Status: {result.status.upper()} - {result.message}")
    else:
        lines.append(
            f"Status: OK | confidence {result.confidence:.2f} ({result.confidence_label}) | "
            f"mode {result.mode} | diversity {'on' if result.diversity else 'off'} | "
            f"attempts {result.iterations}")
        if result.understood:
            lines.append("Understood: " + "; ".join(f"{k} = {v}" for k, v in result.understood.items()))
        if result.citations:
            lines.append("Knowledge used:")
            lines += [f"  - {c['id']}: {c['used_for']}" for c in result.citations]
        rows = [(str(i), s["title"], s["artist"], f"{s['genre']} / {s['mood']}",
                 f"{s['energy']:.2f}", f"{score:.2f}", "\n".join(reasons))
                for i, (s, score, reasons) in enumerate(result.recommendations, start=1)]
        lines.append(format_table(["#", "Title", "Artist", "Genre / Mood", "Energy", "Score", "Reasons"], rows))
        lines.append("Critic: " + ", ".join(f"{k} {v:.2f}" for k, v in result.metrics.items() if v is not None))
        lines += [f"Note: {n}" for n in result.notes]
        lines.append(result.message)
    if show_trace:
        lines.append("Agent trace:")
        lines += [f"  {t.step:>2}. [{t.phase}] {t.action}: {t.detail}" for t in result.trace]
    return "\n".join(lines)


def build_agent(args: argparse.Namespace) -> VibeAgent:
    """Load the catalog and construct the agent from CLI flags."""
    songs = load_songs(args.data)
    llm_parser = None
    if args.llm:
        from src.llm import make_parser
        llm_parser = make_parser()
        if llm_parser is None:
            print("Claude parser unavailable; continuing with the local parser.", file=sys.stderr)
    return VibeAgent(songs, use_rag=not args.no_rag, use_classifier=not args.no_classifier,
                     llm_parser=llm_parser, max_iterations=args.max_iterations, k=args.k)


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, run the agent, and print results. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="VibeFinder AI: agentic, retrieval-augmented music recommender")
    parser.add_argument("request", nargs="?", help="what you want to listen to, in plain English")
    parser.add_argument("--k", type=int, default=5, help="number of songs (1-10, default 5)")
    parser.add_argument("--trace", action="store_true", help="print the agent's step-by-step trace")
    parser.add_argument("--save-trace", action="store_true", help="save the trace to logs/traces/")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--no-rag", action="store_true", help="ablation: turn off knowledge retrieval")
    parser.add_argument("--no-classifier", action="store_true", help="ablation: turn off the mood classifier")
    parser.add_argument("--llm", action="store_true", help="also parse with Claude (needs anthropic + credentials)")
    parser.add_argument("--max-iterations", type=int, default=3, help="max recommend/critique attempts")
    parser.add_argument("--demo", action="store_true", help="run three sample requests")
    parser.add_argument("--interactive", action="store_true", help="enter requests one at a time")
    parser.add_argument("--data", default="data/songs.csv", help="path to the song catalog CSV")
    parser.add_argument("--verbose", action="store_true", help="show info logs on screen")
    args = parser.parse_args(argv)
    args.k = max(1, min(10, args.k))
    setup_logging(args.verbose)
    log = logging.getLogger("vibefinder.app")

    try:
        agent = build_agent(args)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Startup failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.demo:
        requests = DEMO_REQUESTS
    elif args.interactive:
        requests = None
    elif args.request is not None:
        requests = [args.request]
    else:
        parser.print_help()
        return 1

    def handle(text: str) -> None:
        result = agent.run(text, k=args.k)
        log.info("request=%r status=%s confidence=%.2f", text, result.status, result.confidence)
        if args.json:
            data = result.to_dict()
            if not args.trace:
                data.pop("trace")
            print(json.dumps(data, indent=2))
        else:
            print(render(result, args.trace))
        if args.save_trace:
            print(f"Trace saved to {os.path.relpath(save_trace(result), PROJECT_ROOT)}")

    if requests is None:
        print("VibeFinder AI - describe what you want to hear (blank line to quit).")
        while True:
            try:
                text = input("> ")
            except EOFError:
                break
            if not text.strip():
                break
            handle(text)
            print()
    else:
        for i, text in enumerate(requests):
            if i:
                print("\n" + "-" * 100 + "\n")
            handle(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
