"""Tests for NLP Projecct 16.NLP for whatsapp chats"""
import pytest
import pandas as pd
import numpy as np
import re
from pathlib import Path


@pytest.fixture(scope="module")
def data_dir():
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "NLP Projecct 16.NLP for whatsapp chats"
    if not d.exists():
        pytest.skip(f"Data directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def df(data_dir):
    frames = []
    for fname in ['happy.csv', 'sad.csv', 'angry.csv']:
        frames.append(pd.read_csv(data_dir / fname))
    return pd.concat(frames, ignore_index=True)


class TestDataLoading:
    @pytest.mark.parametrize("fname", ['happy.csv', 'sad.csv', 'angry.csv'])
    def test_files_exist(self, data_dir, fname):
        assert (data_dir / fname).exists()

    def test_combined_not_empty(self, df):
        assert len(df) > 0

    def test_expected_columns(self, df):
        for col in ['content', 'sentiment']:
            assert col in df.columns


class TestPreprocessing:
    def test_text_column_type(self, df):
        assert pd.api.types.is_string_dtype(df["content"])

    def test_sentiment_column_values(self, df):
        assert df["sentiment"].nunique() >= 2

    def test_basic_cleaning(self, df):
        sample = str(df["content"].dropna().iloc[0])
        cleaned = re.sub(r"[^a-zA-Z\s]", "", sample).lower()
        assert isinstance(cleaned, str)


class TestModel:
    def test_vectorizer_fit(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = df["content"].dropna().astype(str).head(200)
        vec = TfidfVectorizer(max_features=100)
        X = vec.fit_transform(texts)
        assert X.shape[0] == len(texts)

    def test_classifier_fit(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        subset = df[["content", "sentiment"]].dropna().head(200)
        vec = TfidfVectorizer(max_features=100)
        X = vec.fit_transform(subset["content"].astype(str))
        y = subset["sentiment"]
        model = MultinomialNB()
        model.fit(X, y)
        assert hasattr(model, "classes_")


class TestPrediction:
    def test_prediction_output(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        subset = df[["content", "sentiment"]].dropna().head(200)
        vec = TfidfVectorizer(max_features=100)
        X = vec.fit_transform(subset["content"].astype(str))
        y = subset["sentiment"]
        model = MultinomialNB()
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert len(preds) == 10
