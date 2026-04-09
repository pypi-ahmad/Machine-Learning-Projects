"""
Modern Tabular Classification Pipeline (April 2026)
Models: CatBoost/LightGBM/XGBoost (GPU) + AutoGluon + RealTabPFN-v2 + TabM
Data: Auto-downloaded at runtime — no local files needed
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

TARGET = "Type"


def load_data():
    """Download dataset from the internet."""
    from sklearn.datasets import fetch_openml
    _d = fetch_openml(data_id=41, as_frame=True, parser="auto")
    df = _d.frame
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df[TARGET].value_counts()}")
    return df


def preprocess(df):
    df = df.copy()
    df.dropna(subset=[TARGET], inplace=True)

    le_target = None
    if df[TARGET].dtype == "object" or df[TARGET].dtype.name == "category":
        le_target = LabelEncoder()
        df[TARGET] = le_target.fit_transform(df[TARGET])

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "unknown")

    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() < 50 else None
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, le_target


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    n_classes = y_train.nunique()
    is_binary = n_classes == 2

    # ── CatBoost (GPU) ──
    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=8,
            task_type="GPU", devices="0",
            eval_metric="AUC" if is_binary else "MultiClass",
            early_stopping_rounds=50, verbose=100,
            auto_class_weights="Balanced",
        )
        cb.fit(X_train, y_train, eval_set=(X_test, y_test))
        results["CatBoost"] = cb.predict(X_test).flatten()
        print(f"\n✓ CatBoost Accuracy: {accuracy_score(y_test, results['CatBoost']):.4f}")
    except Exception as e:
        print(f"✗ CatBoost: {e}")

    # ── LightGBM (GPU) ──
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8,
            device="gpu", class_weight="balanced", verbose=-1, n_jobs=-1,
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        results["LightGBM"] = m.predict(X_test)
        print(f"\n✓ LightGBM Accuracy: {accuracy_score(y_test, results['LightGBM']):.4f}")
    except Exception as e:
        print(f"✗ LightGBM: {e}")

    # ── XGBoost (CUDA) ──
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8,
            device="cuda", tree_method="hist",
            eval_metric="auc" if is_binary else "mlogloss",
            early_stopping_rounds=50, verbosity=1, n_jobs=-1,
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
        results["XGBoost"] = m.predict(X_test)
        print(f"\n✓ XGBoost Accuracy: {accuracy_score(y_test, results['XGBoost']):.4f}")
    except Exception as e:
        print(f"✗ XGBoost: {e}")

    # ── AutoGluon Tabular ──
    try:
        from autogluon.tabular import TabularPredictor
        import tempfile
        train_ag = X_train.copy(); train_ag["Type"] = y_train.values
        test_ag = X_test.copy(); test_ag["Type"] = y_test.values
        with tempfile.TemporaryDirectory() as tmp:
            predictor = TabularPredictor(label="Type", path=tmp, verbosity=1)
            predictor.fit(train_ag, time_limit=180, presets="best_quality")
            results["AutoGluon"] = predictor.predict(test_ag.drop(columns=["Type"])).values
            print(f"\n✓ AutoGluon Accuracy: {accuracy_score(y_test, results['AutoGluon']):.4f}")
    except Exception as e:
        print(f"✗ AutoGluon: {e}")

    # ── RealTabPFN-v2 (prior-fitted network) ──
    try:
        from tabpfn import TabPFNClassifier
        if X_train.shape[0] <= 10000 and X_train.shape[1] <= 500:
            m = TabPFNClassifier(device="cuda", N_ensemble_configurations=32)
            m.fit(X_train.values, y_train.values)
            results["TabPFN-v2"] = m.predict(X_test.values)
            print(f"\n✓ TabPFN-v2 Accuracy: {accuracy_score(y_test, results['TabPFN-v2']):.4f}")
        else:
            print("⚠ TabPFN-v2: dataset too large (>10k rows or >500 cols), skipped")
    except Exception as e:
        print(f"✗ TabPFN-v2: {e}")

    # ── TabM (parameter-efficient tabular ensembling) ──
    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = StandardScaler()
        Xt = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
        Xv = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
        yt = torch.tensor(y_train.values, dtype=torch.long).to(device)
        d_in = Xt.shape[1]; d_out = n_classes

        class TabMBlock(nn.Module):
            def __init__(self, d, n_heads=4):
                super().__init__()
                self.heads = nn.ModuleList([nn.Sequential(
                    nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
                ) for _ in range(n_heads)])
                self.norm = nn.LayerNorm(d)
            def forward(self, x):
                return self.norm(x + sum(h(x) for h in self.heads) / len(self.heads))

        class TabMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Sequential(nn.Linear(d_in, 256), nn.SiLU())
                self.blocks = nn.Sequential(TabMBlock(256), TabMBlock(256), TabMBlock(256))
                self.head = nn.Linear(256, d_out)
            def forward(self, x): return self.head(self.blocks(self.embed(x)))

        model = TabMNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        for ep in range(100):
            model.train(); loss = loss_fn(model(Xt), yt); loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            results["TabM"] = torch.argmax(model(Xv), dim=-1).cpu().numpy()
        print(f"\n✓ TabM Accuracy: {accuracy_score(y_test, results['TabM']):.4f}")
    except Exception as e:
        print(f"✗ TabM: {e}")

    return results


def report(results, y_test, save_dir="."):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    best_name, best_acc = None, 0
    for name, y_pred in results.items():
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"\n— {name} —  Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        if acc > best_acc:
            best_acc, best_name = acc, name
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        fig.savefig(os.path.join(save_dir, f"cm_{name.lower().replace(' ', '_')}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"\n🏆 Best: {best_name} ({best_acc:.4f})")


def main():
    print("=" * 60)
    print("MODERN TABULAR CLASSIFICATION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | AutoGluon | TabPFN-v2 | TabM")
    print("=" * 60)
    df = load_data()
    X_train, X_test, y_train, y_test, le = preprocess(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, y_test, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
