"""Tests for NLP Projecct 4.Keyword Extraction"""
import pytest
import pandas as pd
import re
from pathlib import Path
from collections import Counter


@pytest.fixture(scope="module")
def data_dir():
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "NLP Projecct 4.Keyword Extraction"
    if not d.exists():
        pytest.skip(f"Data directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def df(data_dir):
    return pd.read_csv(data_dir / 'papers.csv', engine='python', on_bad_lines='skip')


class TestDataLoading:
    def test_file_exists(self, data_dir):
        assert (data_dir / "papers.csv").exists()

    def test_loads_without_error(self, df):
        assert df is not None

    def test_not_empty(self, df):
        assert len(df) > 0

    def test_expected_columns(self, df):
        for col in ['id', 'title', 'abstract', 'paper_text']:
            assert col in df.columns


class TestPreprocessing:
    def test_text_column_type(self, df):
        assert pd.api.types.is_string_dtype(df["abstract"])

    def test_text_not_empty(self, df):
        non_null = df["abstract"].dropna()
        assert len(non_null) > 0

    def test_basic_cleaning(self, df):
        sample = str(df["abstract"].dropna().iloc[0])
        cleaned = re.sub(r"[^a-zA-Z\s]", "", sample).lower()
        assert len(cleaned) > 0


class TestModel:
    def test_keyword_extraction_tfidf(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = df["abstract"].dropna().astype(str).head(50)
        vec = TfidfVectorizer(max_features=50, stop_words="english")
        X = vec.fit_transform(texts)
        feature_names = vec.get_feature_names_out()
        assert len(feature_names) > 0
        assert X.shape[0] == len(texts)

    def test_word_frequency_analysis(self, df):
        col = "paper_text" if "paper_text" in df.columns else "abstract"
        text = " ".join(df[col].dropna().astype(str).head(50))
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        freq = Counter(words)
        assert len(freq) >= 2


class TestPrediction:
    def test_top_keywords(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = df["abstract"].dropna().astype(str).head(50)
        vec = TfidfVectorizer(max_features=20, stop_words="english")
        X = vec.fit_transform(texts)
        feature_names = vec.get_feature_names_out()
        # Get top keywords for first document
        scores = X[0].toarray().flatten()
        top_indices = scores.argsort()[-5:][::-1]
        top_keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
        assert len(top_keywords) >= 1
