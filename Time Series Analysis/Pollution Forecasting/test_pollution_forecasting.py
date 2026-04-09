"""
Auto-generated tests for: Time Series Analysis/Pollution Forecasting
Task type: timeseries
"""
import pytest
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/pollution_forecasting/data.csv")

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


class TestModelTraining:
    """Tests for model training."""

    @pytest.mark.model_training
    def test_timeseries_data_is_sortable(self):
        df = load_data(nrows=200)
        # Time series should have a date-like column
        date_cols = [c for c in df.columns if any(d in c.lower() for d in ["date", "time", "timestamp", "day", "month", "year"])]
        assert len(date_cols) > 0 or len(df.columns) > 1, "No temporal column found"

    @pytest.mark.model_training
    def test_linear_baseline_fit(self):
        from sklearn.linear_model import LinearRegression
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        y = df[numeric_cols[-1]].fillna(0).values
        model = LinearRegression()
        model.fit(X, y)
        assert hasattr(model, "predict")


class TestPrediction:
    """Tests for model prediction."""

    @pytest.mark.prediction
    def test_baseline_prediction(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        y = df[numeric_cols[-1]].fillna(0).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert np.issubdtype(preds.dtype, np.number)
