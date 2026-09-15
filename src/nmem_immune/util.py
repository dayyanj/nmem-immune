"""Shared utilities for nmem-immune."""

from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np


def parse_embedding(raw) -> list[float]:
    """Parse embedding from various asyncpg/storage formats.

    Handles: list, ndarray, JSON string, memoryview (pgvector binary), None.
    Returns empty list if unparseable.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, np.ndarray)):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        try:
            return [float(x) for x in json.loads(raw)]
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(raw, memoryview):
        return [float(x) for x in np.frombuffer(raw, dtype=np.float32)]
    return []


def text_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two texts (word-level)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    a_np = np.array(a, dtype=np.float32).ravel()
    b_np = np.array(b, dtype=np.float32).ravel()
    if len(a_np) == 0 or len(b_np) == 0:
        return 0.0
    dot = float(np.dot(a_np, b_np))
    norm = float(np.linalg.norm(a_np) * np.linalg.norm(b_np))
    return dot / norm if norm > 0 else 0.0


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract top distinctive keywords from text using word boundaries."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "it",
        "this", "that", "these", "those", "and", "but", "or", "not",
    }
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    counts = Counter(w for w in words if w not in stop_words)
    return [w for w, _ in counts.most_common(top_n)]
