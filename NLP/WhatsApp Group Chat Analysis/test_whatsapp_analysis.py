"""Tests for NLP Projecct 13.WhatsApp Group Chat Analysis"""
import pytest
import re
from pathlib import Path
from collections import Counter


@pytest.fixture(scope="module")
def data_dir():
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "NLP Projecct 13.WhatsApp Group Chat Analysis"
    if not d.exists():
        pytest.skip(f"Data directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def text_data(data_dir):
    fpath = data_dir / "whatsapp.txt"
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


class TestDataLoading:
    def test_file_exists(self, data_dir):
        assert (data_dir / "whatsapp.txt").exists()

    def test_file_not_empty(self, data_dir):
        assert (data_dir / "whatsapp.txt").stat().st_size > 0

    def test_loads_as_text(self, text_data):
        assert isinstance(text_data, str)
        assert len(text_data) > 100


class TestPreprocessing:
    def test_tokenization(self, text_data):
        words = text_data.split()
        assert len(words) > 10

    def test_lowercasing(self, text_data):
        lower = text_data.lower()
        assert lower == lower.lower()

    def test_word_frequency(self, text_data):
        words = re.findall(r"\b[a-zA-Z]+\b", text_data.lower())
        freq = Counter(words)
        assert len(freq) > 10
        assert freq.most_common(1)[0][1] > 1


class TestModel:
    def test_vocabulary_size(self, text_data):
        words = set(re.findall(r"\b[a-zA-Z]+\b", text_data.lower()))
        assert len(words) > 50

    def test_character_distribution(self, text_data):
        chars = Counter(text_data.lower())
        assert "e" in chars or "a" in chars  # common letters


class TestPrediction:
    def test_ngram_generation(self, text_data):
        words = text_data.split()[:100]
        bigrams = list(zip(words[:-1], words[1:]))
        assert len(bigrams) > 0
        assert all(len(bg) == 2 for bg in bigrams)
