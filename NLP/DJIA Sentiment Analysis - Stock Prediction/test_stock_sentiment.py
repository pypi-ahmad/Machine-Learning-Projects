"""Tests for NLP Project 21. Sentiment Analysis - Dow Jones (DJIA) Stock using News Headlines"""
import pytest
import pandas as pd
import numpy as np
import re
from pathlib import Path


@pytest.fixture(scope="module")
def data_dir():
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "NLP Project 21. Sentiment Analysis - Dow Jones (DJIA) Stock using News Headlines"
    if not d.exists():
        pytest.skip(f"Data directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def df(data_dir):
    return pd.read_csv(data_dir / 'Stock Headlines.csv', encoding='ISO-8859-1')


class TestDataLoading:
    def test_file_exists(self, data_dir):
        assert (data_dir / "Stock Headlines.csv").exists()

    def test_loads_without_error(self, df):
        assert df is not None

    def test_not_empty(self, df):
        assert len(df) > 0

    def test_has_label_column(self, df):
        assert "Label" in df.columns

    def test_has_headline_columns(self, df):
        assert "Top1" in df.columns

    def test_label_binary(self, df):
        assert set(df["Label"].unique()).issubset({0, 1})


class TestPreprocessing:
    def test_headline_concatenation(self, df):
        headline_cols = [c for c in df.columns if c.startswith("Top")]
        assert len(headline_cols) >= 1
        row = df.iloc[0]
        combined = " ".join(str(row[c]) for c in headline_cols)
        assert len(combined) > 0

    def test_text_cleaning(self, df):
        sample = str(df["Top1"].iloc[0])
        cleaned = re.sub(r"[^a-zA-Z\s]", "", sample).lower().strip()
        assert isinstance(cleaned, str)


class TestModel:
    def test_vectorizer_on_headlines(self, df):
        from sklearn.feature_extraction.text import CountVectorizer
        headline_cols = [c for c in df.columns if c.startswith("Top")]
        texts = df[headline_cols].fillna("").astype(str).apply(lambda r: " ".join(r), axis=1).head(200)
        vec = CountVectorizer(max_features=100)
        X = vec.fit_transform(texts)
        assert X.shape[0] == len(texts)

    def test_model_fit(self, df):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.ensemble import RandomForestClassifier
        headline_cols = [c for c in df.columns if c.startswith("Top")]
        subset = df.head(200)
        texts = subset[headline_cols].astype(str).apply(lambda r: " ".join(r), axis=1)
        vec = CountVectorizer(max_features=100)
        X = vec.fit_transform(texts)
        y = subset["Label"]
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        assert hasattr(model, "classes_")


class TestPrediction:
    def test_prediction_output(self, df):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.ensemble import RandomForestClassifier
        headline_cols = [c for c in df.columns if c.startswith("Top")]
        subset = df.head(200)
        texts = subset[headline_cols].astype(str).apply(lambda r: " ".join(r), axis=1)
        vec = CountVectorizer(max_features=100)
        X = vec.fit_transform(texts)
        y = subset["Label"]
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert len(preds) == 10
        assert all(p in [0, 1] for p in preds)
