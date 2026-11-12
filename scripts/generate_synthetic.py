"""Generate the synthetic training set for VibeFinder's mood-intent classifier.

Each mood has a list of *situations* (how people describe a feeling without naming
it) that are slotted into request *templates*. The output is deterministic
(seeded) so the trained model, and every number reported in the docs, is
reproducible.

Usage:
    python scripts/generate_synthetic.py            # writes data/synthetic/mood_train.csv
"""

import csv
import os
import random

SEED = 110
PER_LABEL = 60
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "mood_train.csv")

TEMPLATES = [
    "{s}, play something",
    "music for when {s}",
    "{s} and I need songs",
    "songs for {s}",
    "what should I listen to, {s}",
    "give me a playlist because {s}",
    "{s}. any recommendations?",
    "I need music, {s}",
    "put on something, {s}",
    "{s} right now",
]

SITUATIONS = {
    "happy": [
        "I just got good news", "the sun is out and I feel great", "I aced my exam",
        "it's my birthday", "I'm in such a good mood", "everything is going right today",
        "I got the job", "I feel like smiling all day", "it's finally the weekend",
        "my friends are coming over and I'm excited", "life is good", "I'm celebrating",
    ],
    "chill": [
        "I want to relax on the couch", "it's a lazy sunday", "I'm winding down after work",
        "I just want to take it easy", "I'm lying in a hammock", "it's a slow quiet evening",
        "I'm sipping tea and doing nothing", "I need to decompress", "I'm hanging out at home",
        "I want background music while I unwind", "I'm cooling off", "I want low key music",
    ],
    "focused": [
        "I have a paper due tomorrow", "I need to get work done", "I'm cramming for a test",
        "I'm writing code all night", "I need to stay on task", "I have to finish this report",
        "I'm grinding through homework", "I need to lock in", "I'm doing deep work",
        "I can't get distracted", "I'm reading a textbook", "I need to be productive",
    ],
    "sad": [
        "my girlfriend left me", "I can't stop crying", "I miss my best friend",
        "my dog passed away", "I feel so alone", "nobody texted me back",
        "I lost someone I loved", "my heart hurts", "I had the worst day",
        "I feel empty inside", "we broke up last night", "I'm grieving",
    ],
    "intense": [
        "I'm about to hit a new personal record", "I need adrenaline", "I'm sprinting the last mile",
        "I want to go all out", "I'm pumping iron", "I need to get hyped before the game",
        "I want something that hits hard", "I'm doing a brutal hiit session", "push me to the limit",
        "I need maximum energy", "I'm in beast mode", "it's leg day",
    ],
    "angry": [
        "I'm so frustrated with everyone", "my boss yelled at me", "I want to punch a wall",
        "someone keyed my car", "I'm fed up", "I hate everything today",
        "I want to scream", "I'm boiling with rage", "people keep annoying me",
        "I need to let off steam", "I got cheated", "I'm sick of this",
    ],
    "romantic": [
        "I'm making dinner for my partner", "it's our anniversary", "I have a crush",
        "I'm slow dancing with my wife", "candlelight dinner tonight", "I'm falling for someone",
        "my boyfriend is coming over", "I want to set the mood", "we're cuddling on the couch",
        "I'm planning a proposal", "valentines evening", "I'm thinking about my husband",
    ],
    "nostalgic": [
        "I'm looking through old photos", "I miss high school", "I want to feel like a kid again",
        "remember the good old days", "I'm visiting my hometown", "I miss summers from years ago",
        "I'm thinking about my childhood", "throw me back in time", "I found my old yearbook",
        "I want songs my parents played", "I'm reminiscing", "I miss the way things used to be",
    ],
}


def generate(seed: int = SEED, per_label: int = PER_LABEL):
    """Return a shuffled list of (text, label) rows built from templates and situations."""
    rng = random.Random(seed)
    rows = []
    for label, situations in SITUATIONS.items():
        combos = [(t, s) for t in TEMPLATES for s in situations]
        for template, situation in rng.sample(combos, per_label):
            rows.append((template.format(s=situation), label))
    rng.shuffle(rows)
    return rows


def main() -> None:
    """Write the synthetic dataset to data/synthetic/mood_train.csv."""
    rows = generate()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
