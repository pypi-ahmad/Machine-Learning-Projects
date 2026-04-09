"""
Modern Tabular Regression Pipeline (April 2026)
Models: CatBoost (GPU), LightGBM (GPU), XGBoost (CUDA), FLAML AutoML
Data: Auto-downloaded at runtime
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TARGET = "Close"


def load_data():
    import yfinance as yf
    df = yf.download("BTC-USD", period="10y", auto_adjust=True).reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    print(f"Dataset shape: {df.shape}")
    return df


def preprocess(df):
    df = df.copy()
    df.dropna(subset=[TARGET], inplace=True)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        X[c] = X[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}

    try:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=1000, lr=0.05, depth=8, task_type="GPU",
                              devices="0", early_stopping_rounds=50, verbose=100)
        m.fit(X_train, y_train, eval_set=(X_test, y_test))
        results["CatBoost"] = m.predict(X_test)
        print(f"✓ CatBoost RMSE: {mean_squared_error(y_test, results['CatBoost'], squared=False):.4f}")
    except Exception as e:
        print(f"✗ CatBoost: {e}")

    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=1000, lr=0.05, max_depth=8,
                              device="gpu", verbose=-1, n_jobs=-1)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        results["LightGBM"] = m.predict(X_test)
        print(f"✓ LightGBM RMSE: {mean_squared_error(y_test, results['LightGBM'], squared=False):.4f}")
    except Exception as e:
        print(f"✗ LightGBM: {e}")

    try:
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8,
                         device="cuda", tree_method="hist", early_stopping_rounds=50,
                         verbosity=1, n_jobs=-1)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
        results["XGBoost"] = m.predict(X_test)
        print(f"✓ XGBoost RMSE: {mean_squared_error(y_test, results['XGBoost'], squared=False):.4f}")
    except Exception as e:
        print(f"✗ XGBoost: {e}")

    try:
        from flaml import AutoML
        automl = AutoML()
        automl.fit(X_train, y_train, task="regression", time_budget=120, metric="rmse")
        results["FLAML"] = automl.predict(X_test)
        print(f"✓ FLAML Best: {automl.best_estimator} — RMSE: {mean_squared_error(y_test, results['FLAML'], squared=False):.4f}")
    except Exception as e:
        print(f"✗ FLAML: {e}")

    return results


def report(results, y_test, save_dir="."):
    print("\n" + "=" * 60)
    best_name, best_rmse = None, float("inf")
    for name, y_pred in results.items():
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"— {name} — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        if rmse < best_rmse:
            best_rmse, best_name = rmse, name
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.4, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_title(f"{name} — Predicted vs Actual")
        fig.savefig(os.path.join(save_dir, f"scatter_{name.lower()}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"\n🏆 Best: {best_name} (RMSE: {best_rmse:.4f})")


def main():
    print("=" * 60)
    print("MODERN TABULAR REGRESSION PIPELINE")
    print("CatBoost(GPU) | LightGBM(GPU) | XGBoost(CUDA) | FLAML")
    print("=" * 60)
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, y_test, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
