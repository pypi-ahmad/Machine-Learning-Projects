"""
Shared tabular ML utilities — PyCaret AutoML with sklearn fallback.
When PyCaret is not available (e.g. Python 3.12+), falls back to
sklearn pipelines that produce the same output artifacts.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, save_metrics

_PYCARET_OK: bool | None = None


def _check_pycaret() -> bool:
    """Return True if PyCaret can be imported; cache result."""
    global _PYCARET_OK
    if _PYCARET_OK is None:
        try:
            import pycaret  # noqa: F401
            _PYCARET_OK = True
        except (ImportError, RuntimeError):
            _PYCARET_OK = False
    return _PYCARET_OK


# ═══════════════════════════════════════════════════════════════════════════════
# PyCaret — Classification (with sklearn fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pycaret_classification(df: pd.DataFrame, target: str,
                               output_dir: str | Path, session_id: int = 42):
    output_dir = ensure_dir(output_dir)
    if not _check_pycaret():
        return _sklearn_classification(df, target, output_dir, seed=session_id)
    from pycaret.classification import (
        setup, compare_models, pull, finalize_model,
        save_model, plot_model, predict_model,
    )

    print(f"\n  PyCaret Classification  |  target = '{target}'  |  rows = {len(df)}")
    setup(data=df, target=target, session_id=session_id, verbose=False)

    print("  Comparing models ...")
    best = compare_models(n_select=1)
    results = pull()
    print(results.to_string())
    results.to_csv(output_dir / "model_comparison.csv", index=False)

    final = finalize_model(best)
    save_model(final, str(output_dir / "best_model"))

    for plot_name in ("confusion_matrix", "auc", "feature"):
        try:
            plot_model(best, plot=plot_name, save=True)
            # PyCaret saves to cwd; move to output_dir
            for png in Path(".").glob("*.png"):
                shutil.move(str(png), str(output_dir / png.name))
        except Exception:
            pass

    metrics = results.iloc[0].to_dict() if len(results) else {}
    clean = {k: (round(v, 4) if isinstance(v, float) else str(v))
             for k, v in metrics.items()}
    save_metrics(clean, output_dir)
    print(f"\n  Best model saved -> {output_dir / 'best_model.pkl'}")
    return final


# ═══════════════════════════════════════════════════════════════════════════════
# PyCaret — Regression (with sklearn fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pycaret_regression(df: pd.DataFrame, target: str,
                            output_dir: str | Path, session_id: int = 42):
    output_dir = ensure_dir(output_dir)
    if not _check_pycaret():
        return _sklearn_regression(df, target, output_dir, seed=session_id)
    from pycaret.regression import (
        setup, compare_models, pull, finalize_model,
        save_model, plot_model,
    )

    print(f"\n  PyCaret Regression  |  target = '{target}'  |  rows = {len(df)}")
    setup(data=df, target=target, session_id=session_id, verbose=False)

    print("  Comparing models ...")
    best = compare_models(n_select=1)
    results = pull()
    print(results.to_string())
    results.to_csv(output_dir / "model_comparison.csv", index=False)

    final = finalize_model(best)
    save_model(final, str(output_dir / "best_model"))

    for plot_name in ("residuals", "error", "feature"):
        try:
            plot_model(best, plot=plot_name, save=True)
            for png in Path(".").glob("*.png"):
                shutil.move(str(png), str(output_dir / png.name))
        except Exception:
            pass

    metrics = results.iloc[0].to_dict() if len(results) else {}
    clean = {k: (round(v, 4) if isinstance(v, float) else str(v))
             for k, v in metrics.items()}
    save_metrics(clean, output_dir)
    print(f"\n  Best model saved -> {output_dir / 'best_model.pkl'}")
    return final


# ═══════════════════════════════════════════════════════════════════════════════
# PyCaret — Clustering (with sklearn fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pycaret_clustering(df: pd.DataFrame, output_dir: str | Path,
                            n_clusters: int = 5, session_id: int = 42):
    output_dir = ensure_dir(output_dir)
    if not _check_pycaret():
        return _sklearn_clustering(df, output_dir, n_clusters=n_clusters, seed=session_id)
    from pycaret.clustering import (
        setup, create_model, assign_model, save_model, plot_model,
    )

    print(f"\n  PyCaret Clustering  |  n_clusters = {n_clusters}  |  rows = {len(df)}")
    setup(data=df, session_id=session_id, verbose=False)

    model = create_model("kmeans", num_clusters=n_clusters)
    result = assign_model(model)
    result.to_csv(output_dir / "clustered_data.csv", index=False)

    for plot_name in ("elbow", "cluster", "distribution"):
        try:
            plot_model(model, plot=plot_name, save=True)
            for png in Path(".").glob("*.png"):
                shutil.move(str(png), str(output_dir / png.name))
        except Exception:
            pass

    save_model(model, str(output_dir / "cluster_model"))
    print(f"\n  Clustering results -> {output_dir}")
    return model, result


# ═══════════════════════════════════════════════════════════════════════════════
# LazyPredict fallback
# ═══════════════════════════════════════════════════════════════════════════════

def run_lazypredict_classification(X_train, X_test, y_train, y_test,
                                    output_dir: str | Path):
    output_dir = ensure_dir(output_dir)
    from lazypredict.Supervised import LazyClassifier

    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)
    models.to_csv(output_dir / "lazypredict_results.csv")
    return models


def run_lazypredict_regression(X_train, X_test, y_train, y_test,
                                output_dir: str | Path):
    output_dir = ensure_dir(output_dir)
    from lazypredict.Supervised import LazyRegressor

    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    print(models)
    models.to_csv(output_dir / "lazypredict_results.csv")
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# Sklearn fallbacks (used when PyCaret is unavailable, e.g. Python 3.12+)
# ═══════════════════════════════════════════════════════════════════════════════

def _sklearn_classification(df: pd.DataFrame, target: str,
                             output_dir: Path, *, seed: int = 42):
    """Pure-sklearn classification: tries several models, picks best by F1."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import (RandomForestClassifier,
                                   GradientBoostingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, f1_score,
                                  classification_report)
    import joblib

    print(f"\n  [sklearn fallback] Classification  |  target = '{target}'  |  rows = {len(df)}")

    y = df[target]
    X = df.drop(columns=[target])

    # Encode categorical columns
    le_map = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_map[col] = le

    le_target = None
    if y.dtype == "object" or y.dtype.name == "category":
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)))

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
    except ValueError:
        # Stratify fails when some classes have <2 members — fall back
        print("  [WARN] Stratified split failed (rare classes). Using plain split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "GBM": GradientBoostingClassifier(n_estimators=100, random_state=seed),
        "LogReg": LogisticRegression(max_iter=1000, random_state=seed),
    }

    best_name, best_model, best_f1 = None, None, -1
    rows = []
    for name, clf in candidates.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        rows.append({"Model": name, "Accuracy": round(acc, 4), "F1": round(f1, 4)})
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, pipe, f1

    comparison = pd.DataFrame(rows).sort_values("F1", ascending=False)
    print(comparison.to_string(index=False))
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    # Save best model
    joblib.dump(best_model, output_dir / "best_model.pkl")

    # Final evaluation
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_w = f1_score(y_test, preds, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, preds, average="macro", zero_division=0)
    report = classification_report(y_test, preds, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report)
    print(f"\n  Best: {best_name}  Acc={acc:.4f}  F1={f1_w:.4f}")
    print(report)

    metrics = {"accuracy": round(acc, 4), "macro_f1": round(f1_m, 4),
               "weighted_f1": round(f1_w, 4), "best_model": best_name}
    save_metrics(metrics, output_dir)
    return best_model


def _sklearn_regression(df: pd.DataFrame, target: str,
                         output_dir: Path, *, seed: int = 42):
    """Pure-sklearn regression: tries several models, picks best by R²."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import (RandomForestRegressor,
                                   GradientBoostingRegressor)
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib

    print(f"\n  [sklearn fallback] Regression  |  target = '{target}'  |  rows = {len(df)}")

    y = df[target]
    X = df.drop(columns=[target])

    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        "GBM": GradientBoostingRegressor(n_estimators=100, random_state=seed),
        "Ridge": Ridge(alpha=1.0),
    }

    best_name, best_model, best_r2 = None, None, -999
    rows = []
    for name, reg in candidates.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", reg),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        rows.append({"Model": name, "MAE": round(mae, 4),
                      "RMSE": round(rmse, 4), "R2": round(r2, 4)})
        if r2 > best_r2:
            best_name, best_model, best_r2 = name, pipe, r2

    comparison = pd.DataFrame(rows).sort_values("R2", ascending=False)
    print(comparison.to_string(index=False))
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    joblib.dump(best_model, output_dir / "best_model.pkl")

    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n  Best: {best_name}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4),
               "r2": round(r2, 4), "best_model": best_name}
    save_metrics(metrics, output_dir)
    return best_model


def _sklearn_clustering(df: pd.DataFrame, output_dir: Path, *,
                         n_clusters: int = 5, seed: int = 42):
    """Pure-sklearn clustering fallback."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    import joblib

    print(f"\n  [sklearn fallback] Clustering  |  k = {n_clusters}  |  rows = {len(df)}")

    X = df.copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_clean = scaler.fit_transform(imp.fit_transform(X))

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X_clean)

    sil = silhouette_score(X_clean, labels) if n_clusters > 1 else 0.0
    df_out = df.copy()
    df_out["Cluster"] = labels
    df_out.to_csv(output_dir / "clustered_data.csv", index=False)

    joblib.dump(km, output_dir / "cluster_model.pkl")

    metrics = {"n_clusters": n_clusters, "silhouette": round(sil, 4),
               "inertia": round(float(km.inertia_), 4)}
    save_metrics(metrics, output_dir)
    print(f"  Silhouette = {sil:.4f}  |  Inertia = {km.inertia_:.2f}")
    return km, df_out
