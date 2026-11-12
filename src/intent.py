"""Specialized mood-intent model: a Naive Bayes classifier trained on synthetic requests.

People often describe a feeling without naming it ("my girlfriend left me").
Keyword matching misses those requests. This module trains a small multinomial
Naive Bayes model (unigrams + bigrams, Laplace smoothing) on the synthetic
dataset in ``data/synthetic/mood_train.csv`` so the parser can infer a mood when
no mood word is present. ``eval/run_eval.py`` measures it against the keyword
baseline on a hand-written holdout set.
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from src.knowledge import PROJECT_ROOT, TOKEN_RE, contains_phrase, stem

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic", "mood_train.csv")
MIN_CONFIDENCE = 0.5
_FILLER = {"a", "an", "the", "and", "to", "i", "im", "me", "my", "is", "it", "for", "of", "some"}


def features(text: str) -> List[str]:
    """Turn text into unigram and bigram features (stemmed, filler words dropped)."""
    words = [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in _FILLER]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class MoodClassifier:
    """Multinomial Naive Bayes over request text, predicting one of eight mood families."""

    def __init__(self, alpha: float = 1.0):
        """Create an untrained model with Laplace smoothing strength `alpha`."""
        self.alpha = alpha
        self.labels: List[str] = []
        self.priors: Dict[str, float] = {}
        self.counts: Dict[str, Counter] = {}
        self.totals: Dict[str, int] = {}
        self.vocab: set = set()

    def fit(self, rows: Iterable[Tuple[str, str]]) -> "MoodClassifier":
        """Train on (text, label) pairs and return self."""
        label_docs: Counter = Counter()
        self.counts = defaultdict(Counter)
        for text, label in rows:
            label_docs[label] += 1
            feats = features(text)
            self.counts[label].update(feats)
            self.vocab.update(feats)
        total_docs = sum(label_docs.values())
        if not total_docs:
            raise ValueError("Cannot train MoodClassifier on an empty dataset")
        self.labels = sorted(label_docs)
        self.priors = {lab: math.log(label_docs[lab] / total_docs) for lab in self.labels}
        self.totals = {lab: sum(self.counts[lab].values()) for lab in self.labels}
        return self

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Return a probability for every label (sums to 1)."""
        feats = [f for f in features(text) if f in self.vocab]
        vocab_size = len(self.vocab)
        log_scores = {}
        for lab in self.labels:
            denom = self.totals[lab] + self.alpha * vocab_size
            log_scores[lab] = self.priors[lab] + sum(
                math.log((self.counts[lab][f] + self.alpha) / denom) for f in feats
            )
        top = max(log_scores.values())
        exp = {lab: math.exp(s - top) for lab, s in log_scores.items()}
        total = sum(exp.values())
        return {lab: v / total for lab, v in exp.items()}

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        """Return (label, probability), or (None, p) when no known words were seen."""
        if not any(f in self.vocab for f in features(text)):
            return None, 0.0
        proba = self.predict_proba(text)
        label = max(proba, key=proba.get)
        return label, round(proba[label], 3)


def load_rows(path: str = TRAIN_PATH) -> List[Tuple[str, str]]:
    """Read (text, label) rows from a CSV with 'text' and 'label' columns."""
    with open(path, newline="", encoding="utf-8") as handle:
        return [(row["text"], row["label"]) for row in csv.DictReader(handle)]


_DEFAULT: Optional[MoodClassifier] = None


def default_classifier() -> MoodClassifier:
    """Return the classifier trained on the bundled synthetic dataset (cached)."""
    global _DEFAULT
    if _DEFAULT is None:
        if not os.path.exists(TRAIN_PATH):
            raise FileNotFoundError(
                f"{TRAIN_PATH} not found. Run: python scripts/generate_synthetic.py"
            )
        _DEFAULT = MoodClassifier().fit(load_rows())
    return _DEFAULT


def keyword_mood(text: str, mood_vocab: Dict[str, str]) -> Optional[str]:
    """Baseline: return the canonical mood whose name or alias appears in the text, longest first."""
    for alias in sorted(mood_vocab, key=len, reverse=True):
        if contains_phrase(text, alias):
            return mood_vocab[alias]
    return None
