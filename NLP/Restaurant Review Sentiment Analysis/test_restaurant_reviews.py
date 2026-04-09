"""Tests for NLP Project 22. - Sentiment Analysis - Restaurant Reviews"""
import pytest
import pandas as pd
import numpy as np
import re
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def data_dir():
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "NLP Project 22. - Sentiment Analysis - Restaurant Reviews"
    if not d.exists():
        pytest.skip(f"Data directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def df(data_dir):
    """Load primary dataset."""
    return pd.read_csv(data_dir / 'Restaurant_Reviews.tsv', delimiter='\t', quoting=3)



# ── Data Loading Tests ────────────────────────────────────

class TestDataLoading:
    def test_file_exists(self, data_dir):
        assert (data_dir / "Restaurant_Reviews.tsv").exists()

    def test_loads_without_error(self, df):
        assert df is not None

    def test_not_empty(self, df):
        assert len(df) > 0

    def test_expected_columns(self, df):
        for col in ['Review', 'Liked']:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_fully_null_columns(self, df):
        key_cols = ['Review', 'Liked']
        for col in key_cols:
            if col in df.columns:
                assert not df[col].isna().all(), f"Column {col} is entirely null"


# ── Preprocessing Tests ───────────────────────────────────

class TestPreprocessing:
    def test_text_column_dtype(self, df):
        assert pd.api.types.is_string_dtype(df["Review"])

    def test_text_not_empty_strings(self, df):
        non_null = df["Review"].dropna()
        assert len(non_null) > 0
        assert non_null.str.len().mean() > 0

    def test_basic_text_cleaning(self, df):
        sample = str(df["Review"].dropna().iloc[0])
        cleaned = re.sub(r"[^a-zA-Z\s]", "", sample).lower().strip()
        assert len(cleaned) > 0
        assert cleaned == cleaned.lower()

    def test_label_has_multiple_classes(self, df):
        assert df["Liked"].nunique() >= 2


# ── Model Training Tests ─────────────────────────────────

class TestModel:
    def test_tfidf_vectorizer(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = df["Review"].astype(str).head(200)
        vec = TfidfVectorizer(max_features=100)
        X = vec.fit_transform(texts)
        assert X.shape[0] == len(texts)
        assert X.shape[1] <= 100

    def test_model_fit(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        subset = df[["Review", "Liked"]].dropna().head(200)
        vec = TfidfVectorizer(max_features=100)
        X = vec.fit_transform(subset["Review"].astype(str))
        y = subset["Liked"]
        model = MultinomialNB()
        model.fit(X, y)
        assert hasattr(model, "classes_")
        assert len(model.classes_) >= 2


# ── Prediction Tests ──────────────────────────────────────

class TestPrediction:
    def test_prediction_output(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        subset = df[["Review", "Liked"]].dropna().head(200)
        vec = TfidfVectorizer(max_features=100)
        X = vec.fit_transform(subset["Review"].astype(str))
        y = subset["Liked"]
        model = MultinomialNB()
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert len(preds) == 10
        assert all(p in model.classes_ for p in preds)

    def test_prediction_proba_shape(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        subset = df[["Review", "Liked"]].dropna().head(200)
        vec = TfidfVectorizer(max_features=100)
        X = vec.fit_transform(subset["Review"].astype(str))
        y = subset["Liked"]
        model = MultinomialNB()
        model.fit(X, y)
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, len(model.classes_))
        assert np.allclose(proba.sum(axis=1), 1.0)
