"""
Fraud / Imbalanced Classification Pipeline (April 2026)
Models: CatBoost, LightGBM, XGBoost + PyOD — GPU + threshold tuning
Data: Auto-downloaded at runtime
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    classification_report, f1_score,
    precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

TARGET = "Class"


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("imodels/credit-card", split="train").to_pandas()
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df[TARGET].mean():.4%}")
    return df


def preprocess(df):
    df = df.copy()
    df.dropna(subset=[TARGET], inplace=True)
    y = df[TARGET]; X = df.drop(columns=[TARGET])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols: X[c] = X[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def find_best_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    idx = np.argmax(f1s)
    return thresholds[idx] if idx < len(thresholds) else 0.5


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    for name, builder in [
        ("CatBoost", lambda: __import__("catboost").CatBoostClassifier(
            iterations=1000, lr=0.03, depth=8, task_type="GPU", devices="0",
            scale_pos_weight=scale, eval_metric="F1", early_stopping_rounds=50, verbose=100)),
        ("LightGBM", lambda: __import__("lightgbm").LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, max_depth=8,
            device="gpu", scale_pos_weight=scale, verbose=-1, n_jobs=-1)),
        ("XGBoost", lambda: __import__("xgboost").XGBClassifier(
            n_estimators=1000, learning_rate=0.03, max_depth=8,
            device="cuda", tree_method="hist", scale_pos_weight=scale,
            eval_metric="aucpr", early_stopping_rounds=50, verbosity=1, n_jobs=-1)),
    ]:
        try:
            m = builder()
            if name == "LightGBM":
                import lightgbm as lgb
                m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
            else:
                m.fit(X_train, y_train, eval_set=[(X_test, y_test)] if name == "XGBoost"
                      else (X_test, y_test), verbose=100 if name == "XGBoost" else None)
            proba = m.predict_proba(X_test)[:, 1]
            thresh = find_best_threshold(y_test, proba)
            preds = (proba >= thresh).astype(int)
            results[name] = {"preds": preds, "proba": proba, "thresh": thresh}
            print(f"✓ {name} F1: {f1_score(y_test, preds):.4f} (t={thresh:.3f})")
        except Exception as e:
            print(f"✗ {name}: {e}")

    # ── PyOD Anomaly Scoring (unsupervised cross-check) ──
    try:
        from pyod.models.ecod import ECOD
        ecod = ECOD(contamination=0.05)
        ecod.fit(X_train)
        anomaly_scores = ecod.decision_function(X_test)
        n_anom = (ecod.predict(X_test) == 1).sum()
        print(f"✓ PyOD ECOD: {n_anom} anomalies flagged in test set ({n_anom/len(X_test):.2%})")
    except Exception as e:
        print(f"✗ PyOD: {e}")

    return results


def report(results, y_test, save_dir="."):
    for name, r in results.items():
        print(f"\n— {name} (threshold={r['thresh']:.3f}) —")
        print(classification_report(y_test, r["preds"], target_names=["Legit", "Fraud"]))
        print(f"  AUPRC: {average_precision_score(y_test, r['proba']):.4f}  ROC-AUC: {roc_auc_score(y_test, r['proba']):.4f}")


def main():
    print("=" * 60)
    print("FRAUD / IMBALANCED CLASSIFICATION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | PyOD")
    print("=" * 60)
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, y_test, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
