"""Tests for NLP Projecct 14.Next Word prediction Model"""
import pytest
import re
import numpy as np
from pathlib import Path
from collections import Counter


@pytest.fixture(scope="module")
def data_dir():
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "NLP Projecct 14.Next Word prediction Model"
    if not d.exists():
        pytest.skip(f"Data directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def text_data(data_dir):
    fpath = data_dir / "1661-0.txt"
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


class TestDataLoading:
    def test_file_exists(self, data_dir):
        assert (data_dir / "1661-0.txt").exists()

    def test_file_not_empty(self, data_dir):
        assert (data_dir / "1661-0.txt").stat().st_size > 0

    def test_loads_as_text(self, text_data):
        assert len(text_data) > 1000


class TestPreprocessing:
    def test_tokenization(self, text_data):
        words = text_data.split()
        assert len(words) > 100

    def test_unique_words(self, text_data):
        words = set(text_data.lower().split())
        assert len(words) > 50

    def test_sequence_creation(self, text_data):
        words = text_data.lower().split()[:200]
        word_to_idx = {w: i for i, w in enumerate(set(words))}
        sequences = []
        seq_len = 5
        for i in range(len(words) - seq_len):
            seq = [word_to_idx[w] for w in words[i:i + seq_len + 1]]
            sequences.append(seq)
        assert len(sequences) > 0
        assert all(len(s) == seq_len + 1 for s in sequences)


class TestModel:
    def test_input_output_split(self, text_data):
        words = text_data.lower().split()[:200]
        unique = sorted(set(words))
        word_to_idx = {w: i for i, w in enumerate(unique)}
        seq_len = 5
        X, y = [], []
        for i in range(len(words) - seq_len):
            X.append([word_to_idx[w] for w in words[i:i + seq_len]])
            y.append(word_to_idx[words[i + seq_len]])
        X = np.array(X)
        y = np.array(y)
        assert X.shape[1] == seq_len
        assert len(y) == len(X)
        assert y.max() < len(unique)


class TestPrediction:
    def test_probability_distribution(self, text_data):
        words = text_data.lower().split()[:200]
        unique = sorted(set(words))
        # Simulate softmax output
        logits = np.random.randn(len(unique))
        probs = np.exp(logits) / np.exp(logits).sum()
        assert abs(probs.sum() - 1.0) < 1e-6
        predicted = unique[np.argmax(probs)]
        assert predicted in unique
