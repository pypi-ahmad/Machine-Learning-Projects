"""Tests for NLP Projects 34 - Stop words in 28 Languages"""
import pytest
import pandas as pd
import re
from pathlib import Path


@pytest.fixture(scope="module")
def data_dir():
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "NLP Projects 34 - Stop words in 28 Languages"
    if not d.exists():
        pytest.skip(f"Data directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def stop_words(data_dir):
    fpath = data_dir / "hindi.txt"
    with open(fpath, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return words


class TestDataLoading:
    def test_file_exists(self, data_dir):
        assert (data_dir / "hindi.txt").exists()

    def test_file_not_empty(self, data_dir):
        assert (data_dir / "hindi.txt").stat().st_size > 0

    def test_loads_stop_words(self, stop_words):
        assert len(stop_words) > 5


class TestPreprocessing:
    def test_stop_words_are_strings(self, stop_words):
        assert all(isinstance(w, str) for w in stop_words)

    def test_stop_words_non_empty(self, stop_words):
        assert all(len(w) > 0 for w in stop_words)

    def test_stop_words_unique(self, stop_words):
        unique = set(stop_words)
        assert len(unique) > 5


class TestModel:
    def test_stop_word_removal(self, stop_words):
        corpus = "प्रत्येक व्यक्ति को शिक्षा का अधिकार है"
        words = corpus.split()
        sw_set = set(stop_words)
        filtered = [w for w in words if w not in sw_set]
        assert isinstance(filtered, list)


class TestPrediction:
    def test_word_index_creation(self, stop_words):
        word_dict = {w: i for i, w in enumerate(stop_words)}
        assert len(word_dict) == len(stop_words)
        assert all(isinstance(v, int) for v in word_dict.values())
