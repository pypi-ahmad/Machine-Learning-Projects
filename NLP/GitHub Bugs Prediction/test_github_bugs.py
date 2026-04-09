"""Tests for NLP Projects 33 - GitHub Bugs Prediction"""
import pytest
import pandas as pd
import numpy as np
import re
from pathlib import Path


@pytest.fixture(scope="module")
def data_dir():
    root = Path(__file__).resolve().parent.parent
    d = root / "data" / "NLP Projects 33 - GitHub Bugs Prediction"
    if not d.exists():
        pytest.skip(f"Data directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def df(data_dir):
    return pd.read_json(data_dir / 'embold_train.json')


class TestDataLoading:
    def test_file_exists(self, data_dir):
        assert (data_dir / "embold_train.json").exists()

    def test_loads_without_error(self, df):
        assert df is not None

    def test_not_empty(self, df):
        assert len(df) > 0

    def test_expected_columns(self, df):
        for col in ['title', 'body', 'label']:
            assert col in df.columns

    def test_extra_file_exists(self, data_dir):
        assert (data_dir / "embold_test.json").exists()

    def test_extra_file_loads(self, data_dir):
        import pandas as pd
        df2 = pd.read_json(data_dir / 'embold_test.json')
        assert len(df2) > 0


class TestPreprocessing:
    def test_text_column_type(self, df):
        assert pd.api.types.is_string_dtype(df["title"])

    def test_text_not_empty(self, df):
        non_null = df["title"].dropna()
        assert len(non_null) > 0

    def test_label_values(self, df):
        assert df["label"].nunique() >= 2


class TestModel:
    def test_tfidf_vectorizer(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = df["title"].astype(str).head(200)
        vec = TfidfVectorizer(max_features=100)
        X = vec.fit_transform(texts)
        assert X.shape[0] == len(texts)

    def test_model_fit(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        subset = df[["title", "label"]].dropna().head(200)
        vec = TfidfVectorizer(max_features=100)
        X = vec.fit_transform(subset["title"].astype(str))
        y = subset["label"]
        model = MultinomialNB()
        model.fit(X, y)
        assert hasattr(model, "classes_")


class TestPrediction:
    def test_prediction_output(self, df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        subset = df[["title", "label"]].dropna().head(200)
        vec = TfidfVectorizer(max_features=100)
        X = vec.fit_transform(subset["title"].astype(str))
        y = subset["label"]
        model = MultinomialNB()
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert len(preds) == 10
