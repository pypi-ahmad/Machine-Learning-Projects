"""Tests for NLP Projecct 10.TextSummarization"""
import pytest
import pandas as pd
import numpy as np
import re
from pathlib import Path


@pytest.fixture(scope="module")
def data_dir():
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "NLP Projecct 10.TextSummarization"
    if not d.exists():
        pytest.skip(f"Data directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def df(data_dir):
    return pd.read_csv(data_dir / 'tennis.csv')


class TestDataLoading:
    def test_file_exists(self, data_dir):
        assert (data_dir / "tennis.csv").exists()

    def test_loads_without_error(self, df):
        assert df is not None

    def test_not_empty(self, df):
        assert len(df) > 0

    def test_has_text_column(self, df):
        assert "article_text" in df.columns


class TestPreprocessing:
    def test_text_type(self, df):
        assert pd.api.types.is_string_dtype(df["article_text"])

    def test_articles_have_content(self, df):
        non_null = df["article_text"].dropna()
        assert non_null.str.len().mean() > 50

    def test_sentence_splitting(self, df):
        sample = str(df["article_text"].dropna().iloc[0])
        sentences = re.split(r"[.!?]+", sample)
        sentences = [s.strip() for s in sentences if s.strip()]
        assert len(sentences) >= 1


class TestModel:
    def test_word_frequency_extraction(self, df):
        sample = str(df["article_text"].dropna().iloc[0])
        words = re.findall(r"\b[a-zA-Z]{3,}\b", sample.lower())
        from collections import Counter
        freq = Counter(words)
        assert len(freq) > 0

    def test_sentence_scoring(self, df):
        sample = str(df["article_text"].dropna().iloc[0])
        sentences = [s.strip() for s in re.split(r"[.!?]+", sample) if s.strip()]
        words = re.findall(r"\b[a-zA-Z]{3,}\b", sample.lower())
        from collections import Counter
        freq = Counter(words)
        scores = {}
        for i, sent in enumerate(sentences):
            score = sum(freq.get(w, 0) for w in sent.lower().split())
            scores[i] = score
        assert len(scores) > 0


class TestPrediction:
    def test_summary_generation(self, df):
        sample = str(df["article_text"].dropna().iloc[0])
        sentences = [s.strip() for s in re.split(r"[.!?]+", sample) if s.strip()]
        if len(sentences) < 2:
            pytest.skip("Not enough sentences for summarization")
        words = re.findall(r"\b[a-zA-Z]{3,}\b", sample.lower())
        from collections import Counter
        freq = Counter(words)
        scored = [(sum(freq.get(w, 0) for w in s.lower().split()), s) for s in sentences]
        scored.sort(reverse=True)
        summary = ". ".join(s for _, s in scored[:3])
        assert len(summary) > 0
        assert len(summary) < len(sample)
