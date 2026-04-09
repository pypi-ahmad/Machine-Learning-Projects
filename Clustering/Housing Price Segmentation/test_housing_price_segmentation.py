"""
Auto-generated tests for: Clustering/6 Housing price segmentation
Task type: clustering
"""
import pytest
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/housing_price_segmentation/House Price India.csv")

def load_data(nrows=None):
    """Load the primary dataset."""
    kwargs = dict(filepath_or_buffer=DATA_PATH)
    if nrows:
        kwargs["nrows"] = nrows
    try:
        return pd.read_csv(**kwargs)
    except UnicodeDecodeError:
        kwargs["encoding"] = "latin-1"
        return pd.read_csv(**kwargs)


class TestDataLoading:
    """Tests for data loading."""

    @pytest.mark.data_loading
    def test_data_file_exists(self):
        assert os.path.isfile(DATA_PATH), f"Data file not found: {DATA_PATH}"

    @pytest.mark.data_loading
    def test_data_loads_without_error(self):
        df = load_data(nrows=5)
        assert df is not None

    @pytest.mark.data_loading
    def test_data_has_rows(self):
        df = load_data(nrows=100)
        assert len(df) > 0, "Dataset has no rows"

    @pytest.mark.data_loading
    def test_data_has_columns(self):
        df = load_data(nrows=5)
        assert len(df.columns) > 0, "Dataset has no columns"

    @pytest.mark.data_loading
    def test_data_shape_is_valid(self):
        df = load_data(nrows=100)
        assert df.shape[0] > 0 and df.shape[1] > 0

    @pytest.mark.data_loading
    def test_column_dtypes_are_set(self):
        df = load_data(nrows=5)
        assert all(dtype is not None for dtype in df.dtypes)


class TestPreprocessing:
    """Tests for data preprocessing."""

    @pytest.mark.preprocessing
    def test_no_fully_null_columns(self):
        df = load_data(nrows=500)
        fully_null = df.columns[df.isnull().all()].tolist()
        assert len(fully_null) == 0, f"Fully null columns found: {fully_null}"

    @pytest.mark.preprocessing
    def test_no_duplicate_columns(self):
        df = load_data(nrows=5)
        assert len(df.columns) == len(set(df.columns)), "Duplicate column names found"

    @pytest.mark.preprocessing
    def test_numeric_columns_exist(self):
        df = load_data(nrows=100)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0, "No numeric columns found"

    @pytest.mark.preprocessing
    def test_fillna_does_not_crash(self):
        df = load_data(nrows=100)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        assert df[numeric_cols].isnull().sum().sum() == 0

    @pytest.mark.preprocessing
    def test_train_test_split(self):
        from sklearn.model_selection import train_test_split
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            X = df[numeric_cols[:-1]].fillna(0)
            y = df[numeric_cols[-1]].fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            assert len(X_train) > 0 and len(X_test) > 0
            assert len(X_train) + len(X_test) == len(X)


class TestModelTraining:
    """Tests for model training."""

    @pytest.mark.model_training
    def test_model_instantiation(self):
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=3)
        assert model is not None

    @pytest.mark.model_training
    def test_model_fit_on_sample(self):
        from sklearn.cluster import KMeans
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        model = KMeans(n_clusters=min(3, len(X)))
        labels = model.fit_predict(X)
        assert len(labels) == len(X)

    @pytest.mark.model_training
    def test_model_output_shape(self):
        from sklearn.cluster import KMeans
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        model = KMeans(n_clusters=min(3, len(X)))
        labels = model.fit_predict(X)
        assert labels.shape == (len(X),)


class TestPrediction:
    """Tests for model prediction."""

    @pytest.mark.prediction
    def test_prediction_output_not_null(self):
        from sklearn.cluster import KMeans
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        model = KMeans(n_clusters=min(3, len(X)))
        preds = model.fit_predict(X)
        assert preds is not None
        assert len(preds) > 0
        assert not np.isnan(preds).all()

    @pytest.mark.prediction
    def test_prediction_dtype(self):
        from sklearn.cluster import KMeans
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        model = KMeans(n_clusters=min(3, len(X)))
        preds = model.fit_predict(X)
        assert isinstance(preds, np.ndarray)
