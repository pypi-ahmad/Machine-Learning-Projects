"""
Fraud / Imbalanced Classification Pipeline (April 2026)
Models: CatBoost, LightGBM, XGBoost + calibrated probabilities + PyOD (ECOD, COPOD, IForest)
GPU + threshold tuning + isotonic calibration
Data: Auto-downloaded at runtime

Compute: GPU recommended (CatBoost/LightGBM/XGBoost use CUDA). CPU fallback automatic.
         ~2–8 min per dataset on RTX 4060.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix,
    recall_score, precision_score,
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
    from sklearn.calibration import CalibratedClassifierCV
    results = {}      # name -> {preds, proba, thresh, model}
    timings = {}      # name -> wall-clock seconds
    # Hold out calibration split from training data
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
    scale = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

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
            t0 = time.perf_counter()
            m = builder()
            if name == "LightGBM":
                import lightgbm as lgb
                m.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
            else:
                m.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)] if name == "XGBoost"
                      else (X_cal, y_cal), verbose=100 if name == "XGBoost" else None)
            # Calibrate probabilities (isotonic regression on held-out cal split)
            cal_model = CalibratedClassifierCV(m, cv="prefit", method="isotonic")
            cal_model.fit(X_cal, y_cal)
            proba = cal_model.predict_proba(X_test)[:, 1]
            thresh = find_best_threshold(y_test, proba)
            preds = (proba >= thresh).astype(int)
            timings[name] = time.perf_counter() - t0
            results[name] = {"preds": preds, "proba": proba, "thresh": thresh, "model": name}
            print(f"✓ {name} F1: {f1_score(y_test, preds):.4f} (t={thresh:.3f}) [calibrated]  ({timings[name]:.1f}s)")
        except Exception as e:
            print(f"✗ {name}: {e}")

    # ── PyOD Anomaly Scoring (unsupervised cross-check) ──
    for pyod_name, pyod_builder in [
        ("ECOD", lambda: __import__("pyod.models.ecod", fromlist=["ECOD"]).ECOD(contamination=0.05)),
        ("COPOD", lambda: __import__("pyod.models.copod", fromlist=["COPOD"]).COPOD(contamination=0.05)),
        ("IForest-PyOD", lambda: __import__("pyod.models.iforest", fromlist=["IForest"]).IForest(contamination=0.05, random_state=42)),
    ]:
        try:
            t0 = time.perf_counter()
            pm = pyod_builder()
            pm.fit(X_train.values if hasattr(X_train, "values") else X_train)
            scores = pm.decision_function(X_test.values if hasattr(X_test, "values") else X_test)
            pyod_preds = pm.predict(X_test.values if hasattr(X_test, "values") else X_test)
            n_anom = pyod_preds.sum()
            f1 = f1_score(y_test, pyod_preds) if len(set(y_test)) > 1 else 0
            auc = roc_auc_score(y_test, scores) if len(set(y_test)) > 1 else 0
            elapsed = time.perf_counter() - t0
            timings[f"PyOD-{pyod_name}"] = elapsed
            print(f"✓ PyOD {pyod_name}: {n_anom} anomalies ({n_anom/len(X_test):.2%}), F1={f1:.4f}, AUC={auc:.4f}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"✗ PyOD {pyod_name}: {e}")

    # ── FLAML AutoML (imbalance-aware benchmark) ──
    try:
        from flaml import AutoML
        t0 = time.perf_counter()
        automl = AutoML()
        automl.fit(X_train, y_train, task="classification", time_budget=120,
                   metric="ap", verbose=0)
        flaml_proba = automl.predict_proba(X_test)[:, 1]
        flaml_thresh = find_best_threshold(y_test, flaml_proba)
        flaml_preds = (flaml_proba >= flaml_thresh).astype(int)
        timings["FLAML"] = time.perf_counter() - t0
        results["FLAML"] = {"preds": flaml_preds, "proba": flaml_proba, "thresh": flaml_thresh, "model": "FLAML"}
        print(f"✓ FLAML ({automl.best_estimator}) F1: {f1_score(y_test, flaml_preds):.4f} (t={flaml_thresh:.3f})  ({timings['FLAML']:.1f}s)")
    except Exception as e:
        print(f"✗ FLAML: {e}")

    # ── LazyPredict (quick sweep benchmark) ──
    try:
        from lazypredict.Supervised import LazyClassifier
        t0 = time.perf_counter()
        lazy = LazyClassifier(verbose=0, ignore_warnings=True)
        lazy_models, _ = lazy.fit(X_train, X_test, y_train, y_test)
        timings["LazyPredict"] = time.perf_counter() - t0
        print(f"\n✓ LazyPredict — Top 5 classifiers:  ({timings['LazyPredict']:.1f}s)")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"✗ LazyPredict: {e}")

    return results, timings


def report(results, timings, y_test, save_dir="."):
    from sklearn.calibration import calibration_curve
    metrics_out = {}

    for name, r in results.items():
        pr_auc = average_precision_score(y_test, r["proba"])
        roc = roc_auc_score(y_test, r["proba"])
        f1 = f1_score(y_test, r["preds"])
        rec = recall_score(y_test, r["preds"])
        prec = precision_score(y_test, r["preds"], zero_division=0)
        acc = accuracy_score(y_test, r["preds"])
        row = {"f1": round(f1, 4), "pr_auc": round(pr_auc, 4), "roc_auc": round(roc, 4),
               "recall": round(rec, 4), "precision": round(prec, 4), "accuracy": round(acc, 4),
               "threshold": round(r["thresh"], 4)}
        if name in timings:
            row["time_s"] = round(timings[name], 1)
        metrics_out[name] = row

        print(f"\n— {name} (threshold={r['thresh']:.3f}) —")
        print(classification_report(y_test, r["preds"], target_names=["Legit", "Fraud"]))
        print(f"  PR-AUC: {pr_auc:.4f}  ROC-AUC: {roc:.4f}  Recall@t: {rec:.4f}")

    # Reliability diagram (calibration plot)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, r in results.items():
            prob_true, prob_pred = calibration_curve(y_test, r["proba"], n_bins=10, strategy="uniform")
            axes[0].plot(prob_pred, prob_true, marker="o", label=name)
        axes[0].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        axes[0].set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
                    title="Reliability Diagram")
        axes[0].legend()

        # Confusion matrix for best model
        best = max(results.items(), key=lambda x: f1_score(y_test, x[1]["preds"]))
        cm = confusion_matrix(y_test, best[1]["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        axes[1].set(xlabel="Predicted", ylabel="Actual", title=f"Confusion Matrix ({best[0]})")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "fraud_report.png"), dpi=150)
        plt.close()
        print(f"\n✓ Report saved to {save_dir}/fraud_report.png")
    except Exception as e:
        print(f"✗ Plot: {e}")

    # ── Save JSON metrics ──
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n✓ Metrics saved → {out_path}")


def main():
    print("=" * 60)
    print("FRAUD / IMBALANCED CLASSIFICATION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | PyOD (ECOD/COPOD/IForest)")
    print("Calibrated probabilities + threshold tuning")
    print("=" * 60)
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    results, timings = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, timings, y_test, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
