"""
Modern Tabular Regression Pipeline (April 2026)
Models: CatBoost/LightGBM/XGBoost (GPU) + AutoGluon + TabM
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

TARGET = "PRCP"


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("Zaherrr/Weather-Dataset", split="train").to_pandas()
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

    # ── AutoGluon Tabular ──
    try:
        from autogluon.tabular import TabularPredictor
        import tempfile
        train_ag = X_train.copy(); train_ag["PRCP"] = y_train.values
        with tempfile.TemporaryDirectory() as tmp:
            predictor = TabularPredictor(label="PRCP", path=tmp, problem_type="regression", verbosity=1)
            predictor.fit(train_ag, time_limit=180, presets="best_quality")
            results["AutoGluon"] = predictor.predict(X_test).values
            print(f"✓ AutoGluon RMSE: {mean_squared_error(y_test, results['AutoGluon'], squared=False):.4f}")
    except Exception as e:
        print(f"✗ AutoGluon: {e}")

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
        model = TabMNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        for ep in range(100):
            model.train(); loss = nn.MSELoss()(model(Xt), yt); loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): results["TabM"] = model(Xv).cpu().numpy()
        print(f"✓ TabM RMSE: {mean_squared_error(y_test, results['TabM'], squared=False):.4f}")
    except Exception as e:
        print(f"✗ TabM: {e}")

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
        fig.savefig(os.path.join(save_dir, f"scatter_{name.lower().replace(' ', '_')}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"\n🏆 Best: {best_name} (RMSE: {best_rmse:.4f})")


def main():
    print("=" * 60)
    print("MODERN TABULAR REGRESSION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | AutoGluon | TabM")
    print("=" * 60)
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, y_test, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
