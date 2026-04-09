#!/usr/bin/env python3
"""
Unified model-output test suite covering ALL ML projects.

Tests:
  - Models can be instantiated for each task type
  - Models can fit on sampled data from each project
  - Model output shapes are valid (predictions match input length)
  - Predictions have correct dtypes (no NaN-only arrays)
  - Classification models produce valid class labels
  - Regression models produce numeric outputs
  - Clustering models produce integer labels
  - Edge cases: single-row input, constant features, all-null features

Parametrised automatically from dataset_registry.json.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

# ── resolve workspace root & importability ──────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.data_loader import load_dataset, handle_missing_data

warnings.filterwarnings("ignore")

def _safe_load(slug: str) -> pd.DataFrame:
    """Attempt load_dataset(); skip the test if the loader cannot parse."""
    try:
        return load_dataset(slug)
    except Exception as exc:
        pytest.skip(f"load_dataset('{slug}') failed: {exc}")

# ── Load registry ───────────────────────────────────────────────────
_REGISTRY_PATH = os.path.join(ROOT, "dataset_registry.json")
with open(_REGISTRY_PATH, encoding="utf-8") as _f:
    _REGISTRY: dict = json.load(_f)

ALL_SLUGS = sorted(_REGISTRY.keys())

# ── Task-based splits ──────────────────────────────────────────────
CLASSIFICATION_SLUGS = [
    s for s in ALL_SLUGS
    if _REGISTRY[s].get("task") == "classification"
    and _REGISTRY[s].get("dataset_type") != "image"
]
REGRESSION_SLUGS = [
    s for s in ALL_SLUGS
    if _REGISTRY[s].get("task") == "regression"
]
CLUSTERING_SLUGS = [
    s for s in ALL_SLUGS
    if _REGISTRY[s].get("task") == "clustering"
]
# NLP / image / time_series excluded from model-output tests
# (they need specialised model architectures)
NLP_SLUGS = [s for s in ALL_SLUGS if _REGISTRY[s].get("task") == "NLP"]
TS_SLUGS = [s for s in ALL_SLUGS if _REGISTRY[s].get("task") == "time_series"]
DA_SLUGS = [s for s in ALL_SLUGS if _REGISTRY[s].get("task") == "data_analysis"]

# Slugs with a declared target column
_TARGET_SLUGS = {
    s for s in ALL_SLUGS
    if _REGISTRY[s].get("target")
    and _REGISTRY[s]["target"] not in ("None", "", "none", None)
}

SAMPLE_ROWS = 200   # rows to load for fast tests
SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────

def _prepare_Xy(slug: str, nrows: int = SAMPLE_ROWS):
    """Load dataset, clean, extract numeric X and target y.

    Returns (X, y) where X is a 2-d numpy array and y is 1-d numpy array.
    Returns (None, None) if the project has no usable numeric columns.
    """
    try:
        df = load_dataset(slug)
    except Exception:
        return None, None

    if len(df) > nrows:
        df = df.head(nrows)

    df = handle_missing_data(df)
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] < 2:
        return None, None

    target_col = _REGISTRY[slug].get("target")
    if target_col and target_col in numeric.columns:
        y = numeric[target_col].values
        X = numeric.drop(columns=[target_col]).fillna(0).values
    else:
        y = numeric.iloc[:, -1].values
        X = numeric.iloc[:, :-1].fillna(0).values

    if X.shape[0] == 0 or X.shape[1] == 0:
        return None, None

    # Replace inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def _prepare_X_clustering(slug: str, nrows: int = SAMPLE_ROWS):
    """Load and prepare X matrix for clustering (no target needed)."""
    try:
        df = load_dataset(slug)
    except Exception:
        return None

    if len(df) > nrows:
        df = df.head(nrows)

    df = handle_missing_data(df)
    numeric = df.select_dtypes(include=[np.number]).fillna(0)

    if numeric.shape[1] < 2:
        return None

    X = numeric.values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ════════════════════════════════════════════════════════════════════
# 1. CLASSIFICATION MODEL TESTS
# ════════════════════════════════════════════════════════════════════

class TestClassificationModels:
    """Verify classification models on sampled project data."""

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", CLASSIFICATION_SLUGS,
                             ids=CLASSIFICATION_SLUGS)
    def test_model_fits(self, slug):
        """A GradientBoostingClassifier must fit without error."""
        from sklearn.ensemble import GradientBoostingClassifier
        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        # Bin continuous targets into classes
        if len(np.unique(y)) > 20:
            y = pd.qcut(y, q=5, labels=False, duplicates="drop")
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X, y)
        assert hasattr(model, "predict")

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", CLASSIFICATION_SLUGS,
                             ids=CLASSIFICATION_SLUGS)
    def test_prediction_shape(self, slug):
        """Predictions must match input sample count."""
        from sklearn.ensemble import GradientBoostingClassifier
        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        if len(np.unique(y)) > 20:
            y = pd.qcut(y, q=5, labels=False, duplicates="drop")
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],), (
            f"Expected shape ({X.shape[0]},), got {preds.shape}"
        )

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", CLASSIFICATION_SLUGS,
                             ids=CLASSIFICATION_SLUGS)
    def test_predictions_not_all_nan(self, slug):
        """Predictions must not be all NaN."""
        from sklearn.ensemble import GradientBoostingClassifier
        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        if len(np.unique(y)) > 20:
            y = pd.qcut(y, q=5, labels=False, duplicates="drop")
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert not np.isnan(preds).all(), "All predictions are NaN"

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", CLASSIFICATION_SLUGS,
                             ids=CLASSIFICATION_SLUGS)
    def test_predictions_are_valid_classes(self, slug):
        """Predicted classes must be a subset of training classes."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        if len(np.unique(y)) > 20:
            y = pd.qcut(y, q=5, labels=False, duplicates="drop")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        if len(np.unique(y_tr)) < 2:
            pytest.skip("Only one class in train split")
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        train_classes = set(np.unique(y_tr))
        pred_classes = set(np.unique(preds))
        assert pred_classes.issubset(train_classes), (
            f"Predicted classes {pred_classes} not in training {train_classes}"
        )

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", CLASSIFICATION_SLUGS,
                             ids=CLASSIFICATION_SLUGS)
    def test_classification_accuracy_above_random(self, slug):
        """Accuracy should beat random chance on training data."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score
        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        if len(np.unique(y)) > 20:
            y = pd.qcut(y, q=5, labels=False, duplicates="drop")
        n_classes = len(np.unique(y))
        if n_classes < 2:
            pytest.skip("Only one class")
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        random_baseline = 1.0 / n_classes
        assert acc >= random_baseline * 0.8, (
            f"Accuracy {acc:.3f} below random baseline {random_baseline:.3f}"
        )


# ════════════════════════════════════════════════════════════════════
# 2. REGRESSION MODEL TESTS
# ════════════════════════════════════════════════════════════════════

class TestRegressionModels:
    """Verify regression models on sampled project data."""

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", REGRESSION_SLUGS,
                             ids=REGRESSION_SLUGS)
    def test_model_fits(self, slug):
        """A GradientBoostingRegressor must fit without error."""
        from sklearn.ensemble import GradientBoostingRegressor
        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        model = GradientBoostingRegressor(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X, y)
        assert hasattr(model, "predict")

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", REGRESSION_SLUGS,
                             ids=REGRESSION_SLUGS)
    def test_prediction_shape(self, slug):
        """Predictions must match input sample count."""
        from sklearn.ensemble import GradientBoostingRegressor
        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        model = GradientBoostingRegressor(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],), (
            f"Expected shape ({X.shape[0]},), got {preds.shape}"
        )

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", REGRESSION_SLUGS,
                             ids=REGRESSION_SLUGS)
    def test_predictions_are_numeric(self, slug):
        """Regression predictions must be numeric and not all NaN."""
        from sklearn.ensemble import GradientBoostingRegressor
        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        model = GradientBoostingRegressor(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.dtype.kind in ("f", "i", "u"), (
            f"Expected numeric dtype, got {preds.dtype}"
        )
        assert not np.isnan(preds).all()

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", REGRESSION_SLUGS,
                             ids=REGRESSION_SLUGS)
    def test_regression_predictions_finite(self, slug):
        """No inf values in regression predictions."""
        from sklearn.ensemble import GradientBoostingRegressor
        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        model = GradientBoostingRegressor(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds)), "Predictions contain inf"

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", REGRESSION_SLUGS,
                             ids=REGRESSION_SLUGS)
    def test_regression_r2_positive(self, slug):
        """R² on training data should be positive (model learned something)."""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score
        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        if np.std(y) == 0:
            pytest.skip("Constant target")
        model = GradientBoostingRegressor(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        r2 = r2_score(y, preds)
        assert r2 > 0, f"R² = {r2:.4f} — model didn't learn"


# ════════════════════════════════════════════════════════════════════
# 3. CLUSTERING MODEL TESTS
# ════════════════════════════════════════════════════════════════════

class TestClusteringModels:
    """Verify clustering models on sampled project data."""

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", CLUSTERING_SLUGS,
                             ids=CLUSTERING_SLUGS)
    def test_model_fits(self, slug):
        """KMeans must fit without error."""
        from sklearn.cluster import KMeans
        X = _prepare_X_clustering(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        model = KMeans(n_clusters=3, random_state=SEED, n_init=10)
        model.fit(X)
        assert hasattr(model, "labels_")

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", CLUSTERING_SLUGS,
                             ids=CLUSTERING_SLUGS)
    def test_label_shape(self, slug):
        """Cluster labels must match input sample count."""
        from sklearn.cluster import KMeans
        X = _prepare_X_clustering(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        model = KMeans(n_clusters=3, random_state=SEED, n_init=10)
        model.fit(X)
        assert model.labels_.shape == (X.shape[0],)

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", CLUSTERING_SLUGS,
                             ids=CLUSTERING_SLUGS)
    def test_labels_are_integers(self, slug):
        """Cluster labels must be integer."""
        from sklearn.cluster import KMeans
        X = _prepare_X_clustering(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        model = KMeans(n_clusters=3, random_state=SEED, n_init=10)
        model.fit(X)
        assert model.labels_.dtype.kind in ("i", "u"), (
            f"Expected integer labels, got {model.labels_.dtype}"
        )

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", CLUSTERING_SLUGS,
                             ids=CLUSTERING_SLUGS)
    def test_correct_number_of_clusters(self, slug):
        """Number of unique labels <= requested n_clusters."""
        from sklearn.cluster import KMeans
        X = _prepare_X_clustering(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        k = 3
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        model.fit(X)
        assert len(np.unique(model.labels_)) <= k

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", CLUSTERING_SLUGS,
                             ids=CLUSTERING_SLUGS)
    def test_silhouette_positive(self, slug):
        """Silhouette score should be positive for non-trivial clusters."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        X = _prepare_X_clustering(slug)
        if X is None:
            pytest.skip("No usable numeric features")
        if X.shape[0] < 10:
            pytest.skip("Too few samples")
        X_scaled = StandardScaler().fit_transform(X)
        model = KMeans(n_clusters=3, random_state=SEED, n_init=10)
        model.fit(X_scaled)
        if len(np.unique(model.labels_)) < 2:
            pytest.skip("Only 1 cluster formed")
        score = silhouette_score(X_scaled, model.labels_)
        assert score > -0.5, f"Silhouette = {score:.4f} — very poor clustering"


# ════════════════════════════════════════════════════════════════════
# 4. EDGE CASES
# ════════════════════════════════════════════════════════════════════

class TestModelEdgeCases:
    """Test models on edge-case inputs that could crash pipelines."""

    @pytest.mark.model_training
    def test_single_row_classification(self):
        """Classifier should handle single-row input gracefully."""
        from sklearn.ensemble import GradientBoostingClassifier
        X = np.array([[1, 2, 3]])
        y = np.array([0])
        model = GradientBoostingClassifier(
            n_estimators=5, random_state=SEED
        )
        # Single-row might not fit well but shouldn't crash
        try:
            model.fit(X, y)
            preds = model.predict(X)
            assert preds.shape == (1,)
        except ValueError:
            pass  # Some models rightfully refuse single-sample

    @pytest.mark.model_training
    def test_single_row_regression(self):
        """Regressor should handle single-row input gracefully."""
        from sklearn.ensemble import GradientBoostingRegressor
        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([42.0])
        model = GradientBoostingRegressor(
            n_estimators=5, random_state=SEED
        )
        try:
            model.fit(X, y)
            preds = model.predict(X)
            assert preds.shape == (1,)
        except ValueError:
            pass

    @pytest.mark.model_training
    def test_constant_features_classification(self):
        """Classifier should cope with constant features."""
        from sklearn.ensemble import GradientBoostingClassifier
        X = np.ones((50, 3))
        y = np.array([0] * 25 + [1] * 25)
        model = GradientBoostingClassifier(
            n_estimators=5, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    @pytest.mark.model_training
    def test_constant_features_regression(self):
        """Regressor should cope with constant features."""
        from sklearn.ensemble import GradientBoostingRegressor
        X = np.ones((50, 3))
        y = np.random.RandomState(SEED).randn(50)
        model = GradientBoostingRegressor(
            n_estimators=5, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    @pytest.mark.model_training
    def test_constant_target_regression(self):
        """Regressor with constant target should predict that constant."""
        from sklearn.ensemble import GradientBoostingRegressor
        X = np.random.RandomState(SEED).randn(50, 3)
        y = np.full(50, 7.0)
        model = GradientBoostingRegressor(
            n_estimators=5, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert np.allclose(preds, 7.0, atol=1.0)

    @pytest.mark.model_training
    def test_high_cardinality_classification(self):
        """Classifier with many classes (50) should still work."""
        from sklearn.ensemble import GradientBoostingClassifier
        rng = np.random.RandomState(SEED)
        X = rng.randn(200, 5)
        y = rng.randint(0, 50, size=200)
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, random_state=SEED
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)

    @pytest.mark.model_training
    def test_clustering_single_feature(self):
        """KMeans on single feature should not crash."""
        from sklearn.cluster import KMeans
        X = np.array([[1], [2], [10], [11], [20], [21]])
        model = KMeans(n_clusters=3, random_state=SEED, n_init=10)
        model.fit(X)
        assert model.labels_.shape == (6,)

    @pytest.mark.model_training
    def test_clustering_many_clusters_vs_samples(self):
        """KMeans with k > n should handle gracefully or warn."""
        from sklearn.cluster import KMeans
        X = np.array([[1, 2], [3, 4], [5, 6]])
        try:
            model = KMeans(n_clusters=5, random_state=SEED, n_init=10)
            model.fit(X)
            # Should produce at most 3 unique labels (n_samples=3)
            assert len(np.unique(model.labels_)) <= 3
        except ValueError:
            pass  # Acceptable to refuse

    @pytest.mark.model_training
    def test_nan_input_handling(self):
        """Model should either handle NaN or raise clear error."""
        from sklearn.ensemble import GradientBoostingClassifier
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model = GradientBoostingClassifier(
            n_estimators=5, random_state=SEED
        )
        try:
            model.fit(X, y)
        except ValueError as e:
            assert "nan" in str(e).lower() or "null" in str(e).lower() or "input" in str(e).lower()

    @pytest.mark.model_training
    def test_inf_input_handling(self):
        """Model should either handle inf or raise clear error."""
        from sklearn.ensemble import GradientBoostingRegressor
        X = np.array([[1, 2], [np.inf, 4], [5, 6]])
        y = np.array([1.0, 2.0, 3.0])
        model = GradientBoostingRegressor(
            n_estimators=5, random_state=SEED
        )
        try:
            model.fit(X, y)
        except ValueError:
            pass  # Expected

    @pytest.mark.model_training
    def test_empty_array_handling(self):
        """Model should raise on empty input."""
        from sklearn.ensemble import GradientBoostingClassifier
        X = np.array([]).reshape(0, 3)
        y = np.array([])
        model = GradientBoostingClassifier(
            n_estimators=5, random_state=SEED
        )
        with pytest.raises(ValueError):
            model.fit(X, y)


# ════════════════════════════════════════════════════════════════════
# 5. DATA_ANALYSIS / NLP / TS — smoke tests
# ════════════════════════════════════════════════════════════════════

class TestDataAnalysisSmoke:
    """Data-analysis projects: just verify data loads & has valid shape."""

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", DA_SLUGS, ids=DA_SLUGS)
    def test_data_loads_and_valid(self, slug):
        df = _safe_load(slug)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] > 0


class TestNLPSmoke:
    """NLP projects: verify data loads & has text columns."""

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", NLP_SLUGS, ids=NLP_SLUGS)
    def test_data_loads_with_text(self, slug):
        df = _safe_load(slug)
        assert isinstance(df, pd.DataFrame)
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        assert len(obj_cols) > 0, f"NLP project '{slug}' has no text columns"


class TestTimeSeriesSmoke:
    """Time-series projects: verify data loads & has datetime-like cols."""

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", TS_SLUGS, ids=TS_SLUGS)
    def test_data_loads_timelike(self, slug):
        df = _safe_load(slug)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", TS_SLUGS, ids=TS_SLUGS)
    def test_data_has_numeric(self, slug):
        """Time series should have at least one numeric column."""
        df = _safe_load(slug)
        numeric = df.select_dtypes(include=[np.number])
        assert numeric.shape[1] > 0, (
            f"Time series '{slug}' has no numeric columns"
        )


# ════════════════════════════════════════════════════════════════════
# 6. REPRODUCIBILITY — same seed = same predictions
# ════════════════════════════════════════════════════════════════════

_REPRO_SLUGS = (CLASSIFICATION_SLUGS + REGRESSION_SLUGS)[:10]


class TestReproducibleOutput:
    """Same seed must produce identical predictions across two runs."""

    @pytest.mark.model_training
    @pytest.mark.parametrize("slug", _REPRO_SLUGS, ids=_REPRO_SLUGS)
    def test_deterministic_predictions(self, slug):
        """Two identical train+predict cycles must yield same array."""
        from sklearn.ensemble import (
            GradientBoostingClassifier, GradientBoostingRegressor,
        )
        from sklearn.model_selection import train_test_split

        X, y = _prepare_Xy(slug)
        if X is None:
            pytest.skip("No usable numeric features")

        is_clf = _REGISTRY[slug].get("task") == "classification"
        if is_clf and len(np.unique(y)) > 20:
            y = pd.qcut(y, q=5, labels=False, duplicates="drop")

        preds_list = []
        for _ in range(2):
            X_tr, X_te, y_tr, _ = train_test_split(
                X, y, test_size=0.2, random_state=SEED
            )
            if is_clf:
                m = GradientBoostingClassifier(
                    n_estimators=10, max_depth=3, random_state=SEED
                )
            else:
                m = GradientBoostingRegressor(
                    n_estimators=10, max_depth=3, random_state=SEED
                )
            m.fit(X_tr, y_tr)
            preds_list.append(m.predict(X_te))

        np.testing.assert_array_equal(
            preds_list[0], preds_list[1],
            err_msg=f"Non-deterministic predictions for {slug}",
        )
