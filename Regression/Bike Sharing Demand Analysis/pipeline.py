"""
Modern Tabular Regression Pipeline (April 2026)
Models: CatBoost/LightGBM/XGBoost (GPU) + AutoGluon + RealTabPFN-v2 + TabM
Data: Auto-downloaded at runtime

Compute: GPU recommended (CatBoost/LightGBM/XGBoost use CUDA, TabM uses torch.cuda).
         CPU fallback is automatic. ~2-10 min per dataset on RTX 4060.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

TARGET = "cnt"


def load_data():
    import os, glob as _glob
    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(_data_dir, exist_ok=True)
    _fp = os.path.join(_data_dir, "day.csv")
    if not os.path.exists(_fp):
        from kaggle.api.kaggle_api_extended import KaggleApi
        _api = KaggleApi(); _api.authenticate()
        _api.dataset_download_files("lakshmi25npathi/bike-sharing-dataset", path=_data_dir, unzip=True)
        _matches = _glob.glob(os.path.join(_data_dir, "**", "day.csv"), recursive=True)
        if _matches: _fp = _matches[0]
        print(f"Downloaded lakshmi25npathi/bike-sharing-dataset from Kaggle")
    df = pd.read_csv(_fp)
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
        if hasattr(X[c], "cat"): X[c] = X[c].astype(str)
        X[c] = X[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])
    return train_test_split(X, y, test_size=0.2, random_state=42)


def run_eda(df, target, save_dir):
    """Exploratory Data Analysis."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Column types:\n{df.dtypes.value_counts().to_string()}")
    missing = df.isnull().sum()
    n_miss = missing[missing > 0]
    if len(n_miss):
        print(f"\nMissing values ({len(n_miss)} columns):")
        print(n_miss.sort_values(ascending=False).head(15).to_string())
    else:
        print("\nNo missing values")
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    print("Summary statistics saved to eda_summary.csv")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        n = len(num_cols)
        fig, ax = plt.subplots(figsize=(min(n + 2, 20), min(n, 16)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=n <= 15, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Heatmap")
        fig.savefig(os.path.join(save_dir, "eda_correlation.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        if target in num_cols:
            tc = corr[target].drop(target).abs().sort_values(ascending=False)
            print(f"\nTop correlations with '{target}':")
            print(tc.head(10).to_string())
    fig, ax = plt.subplots(figsize=(8, 5))
    if df[target].nunique() <= 30:
        df[target].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    else:
        df[target].hist(bins=50, ax=ax, color="steelblue", edgecolor="black")
    ax.set_title(f"Target Distribution: {target}")
    ax.set_xlabel(target)
    fig.savefig(os.path.join(save_dir, "eda_target.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    plot_cols = [c for c in num_cols if c != target][:20]
    if plot_cols:
        nr = max(1, (len(plot_cols) + 4) // 5)
        nc = min(5, len(plot_cols))
        fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
        for i, col in enumerate(plot_cols):
            ri, ci = divmod(i, nc)
            df[col].hist(bins=30, ax=axes[ri][ci], color="steelblue", edgecolor="black")
            axes[ri][ci].set_title(col, fontsize=9)
        for i in range(len(plot_cols), nr * nc):
            ri, ci = divmod(i, nc)
            axes[ri][ci].set_visible(False)
        fig.suptitle("Feature Distributions")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "eda_distributions.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print("EDA plots saved.")


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}      # name -> y_pred
    timings = {}      # name -> wall-clock seconds

    # ── CatBoost (GPU) ──
    try:
        from catboost import CatBoostRegressor
        t0 = time.perf_counter()
        m = CatBoostRegressor(iterations=1000, lr=0.05, depth=8, task_type="GPU",
                              devices="0", early_stopping_rounds=50, verbose=100)
        m.fit(X_train, y_train, eval_set=(X_test, y_test))
        timings["CatBoost"] = time.perf_counter() - t0
        results["CatBoost"] = m.predict(X_test)
        print(f"CatBoost RMSE: {mean_squared_error(y_test, results['CatBoost'], squared=False):.4f}  ({timings['CatBoost']:.1f}s)")
    except Exception as e:
        print(f"CatBoost: {e}")

    # ── LightGBM (GPU) ──
    try:
        import lightgbm as lgb
        t0 = time.perf_counter()
        m = lgb.LGBMRegressor(n_estimators=1000, lr=0.05, max_depth=8,
                              device="gpu", verbose=-1, n_jobs=-1)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        timings["LightGBM"] = time.perf_counter() - t0
        results["LightGBM"] = m.predict(X_test)
        print(f"LightGBM RMSE: {mean_squared_error(y_test, results['LightGBM'], squared=False):.4f}  ({timings['LightGBM']:.1f}s)")
    except Exception as e:
        print(f"LightGBM: {e}")

    # ── XGBoost (CUDA) ──
    try:
        from xgboost import XGBRegressor
        t0 = time.perf_counter()
        m = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8,
                         device="cuda", tree_method="hist", early_stopping_rounds=50,
                         verbosity=1, n_jobs=-1)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
        timings["XGBoost"] = time.perf_counter() - t0
        results["XGBoost"] = m.predict(X_test)
        print(f"XGBoost RMSE: {mean_squared_error(y_test, results['XGBoost'], squared=False):.4f}  ({timings['XGBoost']:.1f}s)")
    except Exception as e:
        print(f"XGBoost: {e}")

    # ── AutoGluon Tabular ──
    try:
        from autogluon.tabular import TabularPredictor
        import tempfile
        t0 = time.perf_counter()
        train_ag = X_train.copy(); train_ag["cnt"] = y_train.values
        with tempfile.TemporaryDirectory() as tmp:
            predictor = TabularPredictor(label="cnt", path=tmp, problem_type="regression", verbosity=0)
            predictor.fit(train_ag, time_limit=120, presets="medium_quality")
            results["AutoGluon"] = predictor.predict(X_test).values
            timings["AutoGluon"] = time.perf_counter() - t0
            print(f"AutoGluon RMSE: {mean_squared_error(y_test, results['AutoGluon'], squared=False):.4f}  ({timings['AutoGluon']:.1f}s)")
    except Exception as e:
        print(f"AutoGluon: {e}")

    # ── RealTabPFN-v2 (prior-fitted network — regression) ──
    try:
        from tabpfn import TabPFNRegressor
        if X_train.shape[0] <= 10000 and X_train.shape[1] <= 500:
            t0 = time.perf_counter()
            m = TabPFNRegressor(device="cuda", N_ensemble_configurations=32)
            m.fit(X_train.values, y_train.values)
            timings["TabPFN-v2"] = time.perf_counter() - t0
            results["TabPFN-v2"] = m.predict(X_test.values)
            print(f"TabPFN-v2 RMSE: {mean_squared_error(y_test, results['TabPFN-v2'], squared=False):.4f}  ({timings['TabPFN-v2']:.1f}s)")
        else:
            print("TabPFN-v2: dataset too large (>10k rows or >500 cols), skipped")
    except Exception as e:
        print(f"TabPFN-v2: {e}")

    # ── TabM (deep tabular) ──
    try:
        import torch, torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = StandardScaler()
        Xt = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
        Xv = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
        yt = torch.tensor(y_train.values, dtype=torch.float32).to(device)
        d_in = Xt.shape[1]
        class TabMBlock(nn.Module):
            def __init__(self, d, n_heads=4):
                super().__init__()
                self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)) for _ in range(n_heads)])
                self.norm = nn.LayerNorm(d)
            def forward(self, x): return self.norm(x + sum(h(x) for h in self.heads) / len(self.heads))
        class TabMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d_in, 256), nn.SiLU(), TabMBlock(256), TabMBlock(256), nn.Linear(256, 1))
            def forward(self, x): return self.net(x).squeeze(-1)
        t0 = time.perf_counter()
        model = TabMNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        for ep in range(100):
            model.train(); loss = nn.MSELoss()(model(Xt), yt); loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): results["TabM"] = model(Xv).cpu().numpy()
        timings["TabM"] = time.perf_counter() - t0
        print(f"TabM RMSE: {mean_squared_error(y_test, results['TabM'], squared=False):.4f}  ({timings['TabM']:.1f}s)")
    except Exception as e:
        print(f"TabM: {e}")

    # ── Baseline Comparison: FLAML AutoML ──
    try:
        from flaml import AutoML
        t0 = time.perf_counter()
        automl = AutoML()
        automl.fit(X_train, y_train, task="regression", time_budget=120, verbose=0)
        timings["FLAML"] = time.perf_counter() - t0
        results["FLAML"] = automl.predict(X_test)
        print(f"FLAML ({automl.best_estimator}) RMSE: {mean_squared_error(y_test, results['FLAML'], squared=False):.4f}  ({timings['FLAML']:.1f}s)")
    except Exception as e:
        print(f"FLAML: {e}")

    # ── Baseline Comparison: LazyPredict ──
    try:
        from lazypredict.Supervised import LazyRegressor
        t0 = time.perf_counter()
        lazy = LazyRegressor(verbose=0, ignore_warnings=True)
        lazy_models, _ = lazy.fit(X_train, X_test, y_train, y_test)
        timings["LazyPredict"] = time.perf_counter() - t0
        print(f"\nLazyPredict — Top 5 regressors:  ({timings['LazyPredict']:.1f}s)")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"LazyPredict: {e}")

    return results, timings


def report(results, timings, y_test, save_dir="."):
    metrics_out = {}

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    best_name, best_rmse = None, float("inf")
    for name, y_pred in results.items():
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        row = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}

        # MAPE — only meaningful when target has no zeros
        mape_str = ""
        try:
            if (y_test != 0).all():
                mape = mean_absolute_percentage_error(y_test, y_pred)
                row["mape"] = round(mape, 4)
                mape_str = f"  MAPE: {mape:.4f}"
        except Exception:
            pass

        if name in timings:
            row["time_s"] = round(timings[name], 1)

        print(f"\n— {name} — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}{mape_str}")
        if rmse < best_rmse:
            best_rmse, best_name = rmse, name
        metrics_out[name] = row

        # Predicted-vs-Actual scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.4, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_title(f"{name} — Predicted vs Actual")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        fig.savefig(os.path.join(save_dir, f"scatter_{name.lower().replace(' ', '_')}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ── Residual distribution for the best model ──
    if best_name:
        residuals = y_test.values - results[best_name]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(residuals, bins=40, edgecolor="black", alpha=0.7)
        axes[0].set_title(f"{best_name} — Residual Distribution")
        axes[0].set_xlabel("Residual (actual − predicted)")
        axes[1].scatter(results[best_name], residuals, alpha=0.4, s=10)
        axes[1].axhline(0, color="r", linestyle="--")
        axes[1].set_title(f"{best_name} — Residual vs Predicted")
        axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "residuals_best.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("\nResidual plots saved")

    print(f"\nBest: {best_name} (RMSE: {best_rmse:.4f})")

    # ── Save JSON metrics ──
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\nMetrics saved to {out_path}")


def cross_validate_best(X, y, save_dir):
    """5-fold cross-validation on gradient boosting models."""
    from sklearn.metrics import make_scorer
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-Fold)")
    print("=" * 60)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scorer = make_scorer(mean_squared_error, squared=False, greater_is_better=False)
    cv_results = {}
    for name, build_fn in [
        ("CatBoost", lambda: __import__("catboost").CatBoostRegressor(
            iterations=100, verbose=0, task_type="GPU", devices="0")),
        ("LightGBM", lambda: __import__("lightgbm").LGBMRegressor(
            n_estimators=100, device="gpu", verbose=-1, n_jobs=-1)),
        ("XGBoost", lambda: __import__("xgboost").XGBRegressor(
            n_estimators=100, device="cuda", tree_method="hist",
            verbosity=0, n_jobs=-1)),
    ]:
        try:
            model = build_fn()
            scores = -cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer, n_jobs=1)
            cv_results[name] = {"rmse_mean": round(float(scores.mean()), 4),
                                "rmse_std": round(float(scores.std()), 4),
                                "folds": [round(float(s), 4) for s in scores]}
            print(f"  {name}: RMSE {scores.mean():.4f} +/- {scores.std():.4f}")
        except Exception as e:
            print(f"  {name} CV skipped: {e}")
    if cv_results:
        out_path = os.path.join(save_dir, "cv_results.json")
        with open(out_path, "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"CV results saved to {out_path}")
    return cv_results


def main():
    print("=" * 60)
    print("MODERN TABULAR REGRESSION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | AutoGluon | TabPFN-v2 | TabM | FLAML | LazyPredict")
    print("=" * 60)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_data()
    run_eda(df, TARGET, save_dir)
    X_train, X_test, y_train, y_test = preprocess(df)
    results, timings = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, timings, y_test, save_dir)
    cross_validate_best(
        pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), save_dir)


if __name__ == "__main__":
    main()
