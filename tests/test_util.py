"""Tests for shared utility functions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nmem_immune.util import (
    cosine_similarity,
    extract_keywords,
    parse_embedding,
    text_similarity,
)


class TestParseEmbedding:
    def test_none(self):
        assert parse_embedding(None) == []

    def test_list(self):
        assert parse_embedding([1.0, 2.0]) == [1.0, 2.0]

    def test_ndarray(self):
        result = parse_embedding(np.array([1.0, 2.0], dtype=np.float32))
        assert result == pytest.approx([1.0, 2.0])

    def test_json_string(self):
        assert parse_embedding("[0.1, 0.2, 0.3]") == pytest.approx([0.1, 0.2, 0.3])

    def test_bad_json_string(self):
        assert parse_embedding("not json") == []

    def test_memoryview(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = parse_embedding(memoryview(arr))
        assert result == pytest.approx([1.0, 2.0])

    def test_unknown_type(self):
        assert parse_embedding(42) == []


class TestTextSimilarity:
    def test_identical(self):
        assert text_similarity("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert text_similarity("hello world", "foo bar") == 0.0

    def test_empty(self):
        assert text_similarity("", "hello") == 0.0


class TestCosineSimilarity:
    def test_identical(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_empty(self):
        assert cosine_similarity([], [1, 0]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0


class TestExtractKeywords:
    def test_extracts_top(self):
        text = "patient medication treatment medication treatment plan patient"
        kws = extract_keywords(text, top_n=2)
        assert "medication" in kws
        assert "patient" in kws or "treatment" in kws

    def test_filters_stop_words(self):
        text = "the is a an and but or not for with"
        assert extract_keywords(text) == []

    def test_empty(self):
        assert extract_keywords("") == []

    def test_short_words_excluded(self):
        text = "go do it me we"
        assert extract_keywords(text) == []
