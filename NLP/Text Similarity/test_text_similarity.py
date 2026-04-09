"""Tests for NLP Projects 35 - Text Similarity"""
import pytest
import pandas as pd
import re
from pathlib import Path
from collections import Counter


@pytest.fixture(scope="module")
def data_dir():
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "NLP Projects 35 - Text Similarity"
    if not d.exists():
        pytest.skip(f"Data directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def df(data_dir):
    return pd.read_csv(data_dir / 'game of thrones.csv')


@pytest.fixture(scope="module")
def disney_text(data_dir):
    fpath = data_dir / "disney.txt"
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


class TestDataLoading:
    def test_csv_exists(self, data_dir):
        assert (data_dir / "game of thrones.csv").exists()

    def test_text_file_exists(self, data_dir):
        assert (data_dir / "disney.txt").exists()

    def test_csv_loads(self, df):
        assert len(df) > 0

    def test_csv_has_title(self, df):
        assert "Title" in df.columns

    def test_text_not_empty(self, disney_text):
        assert len(disney_text) > 50


class TestPreprocessing:
    def test_text_tokenization(self, disney_text):
        words = disney_text.split()
        assert len(words) > 10

    def test_csv_summary_column(self, df):
        assert "Summary" in df.columns
        non_null = df["Summary"].dropna()
        assert len(non_null) > 0

    def test_word_frequency(self, disney_text):
        words = re.findall(r"\b[a-zA-Z]{3,}\b", disney_text.lower())
        freq = Counter(words)
        assert len(freq) > 5


class TestModel:
    def test_combined_text(self, df):
        summaries = df["Summary"].dropna().astype(str)
        combined = " ".join(summaries)
        assert len(combined) > 100

    def test_tfidf_on_summaries(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        summaries = df["Summary"].dropna().astype(str)
        if len(summaries) < 2:
            pytest.skip("Not enough summaries")
        vec = TfidfVectorizer(max_features=50, stop_words="english")
        X = vec.fit_transform(summaries)
        assert X.shape[0] == len(summaries)


class TestPrediction:
    def test_top_words_extraction(self, disney_text):
        words = re.findall(r"\b[a-zA-Z]{3,}\b", disney_text.lower())
        freq = Counter(words)
        top = freq.most_common(10)
        assert len(top) >= 1
        assert all(isinstance(w, str) and isinstance(c, int) for w, c in top)
