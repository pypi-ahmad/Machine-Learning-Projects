"""
Auto-generated tests for: Classification/Credit risk modeling using the German Credit Dataset
Task type: classification
"""
import pytest
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/credit_risk_german/german_credit_data.csv")

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
    def test_encoding_does_not_crash(self):
        df = load_data(nrows=100)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            encoded = pd.get_dummies(df, columns=cat_cols[:2], drop_first=True)
            assert len(encoded.columns) >= len(df.columns)

    @pytest.mark.preprocessing
    def test_scaling_does_not_crash(self):
        from sklearn.preprocessing import StandardScaler
        df = load_data(nrows=100)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df[numeric_cols].fillna(0))
            assert scaled.shape == (len(df), len(numeric_cols))

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
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        assert model is not None

    @pytest.mark.model_training
    def test_model_fit_on_sample(self):
        from sklearn.linear_model import LogisticRegression
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        y = df[numeric_cols[-1]].fillna(0).values
        # Bin target into classes if continuous
        if len(np.unique(y)) > 20:
            y = pd.qcut(y, q=3, labels=False, duplicates="drop")
        model = LogisticRegression()
        model.fit(X, y)
        assert hasattr(model, "predict")

    @pytest.mark.model_training
    def test_model_output_shape(self):
        from sklearn.linear_model import LogisticRegression
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        y = df[numeric_cols[-1]].fillna(0).values
        if len(np.unique(y)) > 20:
            y = pd.qcut(y, q=3, labels=False, duplicates="drop")
        model = LogisticRegression()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)


class TestPrediction:
    """Tests for model prediction."""

    @pytest.mark.prediction
    def test_prediction_output_not_null(self):
        from sklearn.linear_model import LogisticRegression
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        y = df[numeric_cols[-1]].fillna(0).values
        if len(np.unique(y)) > 20:
            y = pd.qcut(y, q=3, labels=False, duplicates="drop")
        model = LogisticRegression()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds is not None
        assert len(preds) > 0
        assert not np.isnan(preds).all()

    @pytest.mark.prediction
    def test_prediction_dtype(self):
        from sklearn.linear_model import LogisticRegression
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        y = df[numeric_cols[-1]].fillna(0).values
        if len(np.unique(y)) > 20:
            y = pd.qcut(y, q=3, labels=False, duplicates="drop")
        model = LogisticRegression()
        model.fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)

    @pytest.mark.prediction
    def test_prediction_classes_valid(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        df = load_data(nrows=200)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns")
        X = df[numeric_cols[:-1]].fillna(0).values
        y = df[numeric_cols[-1]].fillna(0).values
        if len(np.unique(y)) > 20:
            y = pd.qcut(y, q=3, labels=False, duplicates="drop")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert set(preds).issubset(set(y_train) | set(y_test))
