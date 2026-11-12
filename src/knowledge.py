"""Retrieval over VibeFinder's custom knowledge base (the "R" in RAG).

The knowledge base is three markdown files in ``knowledge/``: genre guides, mood
families, and activity guides. Each ``## section`` becomes one retrievable chunk.
Retrieval is TF-IDF cosine similarity plus a phrase boost when a section title or
alias appears verbatim in the request. Retrieved chunks carry structured fields
(``Related genres``, ``Energy``, ``Moods``...) that the agent uses to build the
listener profile, so retrieval changes what gets recommended.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(PROJECT_ROOT, "knowledge")
KIND_BY_FILE = {"genres.md": "genre", "moods.md": "mood", "activities.md": "activity"}

TOKEN_RE = re.compile(r"[a-z0-9&]+")
STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with", "me", "my", "i",
    "im", "some", "something", "song", "songs", "music", "track", "tracks", "play", "give",
    "want", "need", "like", "that", "this", "is", "are", "be", "it", "while", "when", "at",
    "can", "you", "please", "get", "good", "best", "playlist", "vibe", "vibes", "just",
}


def stem(token: str) -> str:
    """Apply a tiny suffix stemmer so 'studying' matches 'study' and 'drives' matches 'drive'."""
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    """Lowercase, split into word tokens, drop stopwords, and stem."""
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def contains_phrase(text: str, phrase: str) -> bool:
    """Return True if `phrase` appears in `text` as whole words (case-insensitive)."""
    pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def parse_weighted_list(value: str) -> Dict[str, float]:
    """Parse 'indie pop 0.8, k-pop 0.8' into {'indie pop': 0.8, 'k-pop': 0.8}."""
    result: Dict[str, float] = {}
    for part in value.split(","):
        match = re.match(r"\s*(.+?)\s+([01](?:\.\d+)?)\s*$", part)
        if match:
            result[match.group(1).strip().lower()] = float(match.group(2))
    return result


def parse_range(value: str) -> Optional[Tuple[float, float]]:
    """Parse '0.25-0.45' into (0.25, 0.45)."""
    match = re.match(r"\s*([01](?:\.\d+)?)\s*-\s*([01](?:\.\d+)?)", value)
    return (float(match.group(1)), float(match.group(2))) if match else None


def parse_list(value: str) -> List[str]:
    """Parse 'focused, chill' into ['focused', 'chill']."""
    return [item.strip().lower() for item in value.split(",") if item.strip()]


@dataclass
class Chunk:
    """One retrievable section of the knowledge base."""
    id: str
    kind: str
    title: str
    text: str
    fields: Dict[str, str] = field(default_factory=dict)
    tokens: List[str] = field(default_factory=list)

    @property
    def aliases(self) -> List[str]:
        """Return the section title plus its listed aliases."""
        return [self.title] + parse_list(self.fields.get("aliases", ""))

    @property
    def energy_range(self) -> Optional[Tuple[float, float]]:
        """Return the (low, high) energy range, if the chunk lists one."""
        return parse_range(self.fields.get("energy", ""))

    @property
    def related_genres(self) -> Dict[str, float]:
        """Return related genres for a genre chunk, or suggested genres for an activity chunk."""
        return parse_weighted_list(self.fields.get("related genres", "") or self.fields.get("genres", ""))

    @property
    def related_moods(self) -> Dict[str, float]:
        """Return related moods with similarity weights."""
        return parse_weighted_list(self.fields.get("related moods", ""))

    @property
    def moods(self) -> List[str]:
        """Return the moods typically associated with this chunk."""
        return parse_list(self.fields.get("moods", ""))


@dataclass
class Retrieved:
    """A chunk returned by a search, with its relevance score and why it matched."""
    chunk: Chunk
    score: float
    matched_phrase: Optional[str] = None


class KnowledgeBase:
    """Loads the markdown knowledge files and answers TF-IDF + phrase-match queries."""

    def __init__(self, directory: str = KNOWLEDGE_DIR):
        """Load and index every knowledge file in `directory`."""
        self.directory = directory
        self.chunks: List[Chunk] = []
        for filename, kind in KIND_BY_FILE.items():
            path = os.path.join(directory, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Knowledge file missing: {path}")
            with open(path, encoding="utf-8") as handle:
                self.chunks.extend(self._split(handle.read(), filename, kind))
        self._build_index()

    @staticmethod
    def _split(markdown: str, filename: str, kind: str) -> List[Chunk]:
        """Split a markdown file into chunks at each '## ' heading."""
        chunks = []
        for block in re.split(r"^## ", markdown, flags=re.MULTILINE)[1:]:
            lines = block.strip().splitlines()
            title = lines[0].strip().lower()
            fields, prose = {}, []
            for line in lines[1:]:
                match = re.match(r"^([A-Za-z ]+):\s*(.+)$", line.strip())
                if match:
                    fields[match.group(1).strip().lower()] = match.group(2).strip()
                elif line.strip():
                    prose.append(line.strip())
            text = " ".join([title, fields.get("aliases", "")] + prose)
            chunks.append(Chunk(
                id=f"{filename}#{title.replace(' ', '-')}", kind=kind, title=title,
                text=" ".join(prose), fields=fields, tokens=tokenize(text),
            ))
        return chunks

    def _build_index(self) -> None:
        """Compute IDF weights and a normalized TF-IDF vector for every chunk."""
        n = len(self.chunks)
        doc_freq = Counter(token for chunk in self.chunks for token in set(chunk.tokens))
        self.idf = {token: math.log((1 + n) / (1 + df)) + 1.0 for token, df in doc_freq.items()}
        self.vectors = [self._vector(chunk.tokens) for chunk in self.chunks]

    def _vector(self, tokens: List[str]) -> Dict[str, float]:
        """Return a unit-length TF-IDF vector for a list of tokens."""
        counts = Counter(tokens)
        vec = {t: c * self.idf.get(t, 0.0) for t, c in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, kind: Optional[str] = None, top_k: int = 3,
               min_score: float = 0.15) -> List[Retrieved]:
        """Return the top chunks for a query, ranked by TF-IDF cosine plus phrase boost."""
        qvec = self._vector(tokenize(query))
        results = []
        for chunk, vec in zip(self.chunks, self.vectors):
            if kind and chunk.kind != kind:
                continue
            score = sum(weight * vec.get(token, 0.0) for token, weight in qvec.items())
            phrase = next((a for a in sorted(chunk.aliases, key=len, reverse=True)
                           if a and contains_phrase(query, a)), None)
            if phrase:
                score += 0.5
            if score >= min_score:
                results.append(Retrieved(chunk, round(score, 3), phrase))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get(self, kind: str, name: str) -> Optional[Chunk]:
        """Look up a chunk by exact title or alias (e.g. get('genre', 'kpop'))."""
        name = name.lower().strip()
        for chunk in self.chunks:
            if chunk.kind == kind and name in chunk.aliases:
                return chunk
        return None

    def vocabulary(self, kind: str) -> Dict[str, str]:
        """Map every title and alias of a kind to its canonical title, e.g. {'kpop': 'k-pop'}."""
        vocab = {}
        for chunk in self.chunks:
            if chunk.kind == kind:
                for alias in chunk.aliases:
                    vocab.setdefault(alias, chunk.title)
        return vocab
