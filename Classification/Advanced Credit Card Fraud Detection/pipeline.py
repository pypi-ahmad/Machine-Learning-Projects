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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
    import os, glob as _glob
    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(_data_dir, exist_ok=True)
    _fp = os.path.join(_data_dir, "creditcard.csv")
    if not os.path.exists(_fp):
        from kaggle.api.kaggle_api_extended import KaggleApi
        _api = KaggleApi(); _api.authenticate()
        _api.dataset_download_files("mlg-ulb/creditcardfraud", path=_data_dir, unzip=True)
        _matches = _glob.glob(os.path.join(_data_dir, "**", "creditcard.csv"), recursive=True)
        if _matches: _fp = _matches[0]
        print(f"Downloaded mlg-ulb/creditcardfraud from Kaggle")
    df = pd.read_csv(_fp)
    # Cap rows to prevent timeout on very large datasets
    MAX_ROWS = 50000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        print(f"Sampled to {MAX_ROWS} rows")
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df[TARGET].mean():.4%}")
    return df


def preprocess(df):
    df = df.copy()
    df.dropna(subset=[TARGET], inplace=True)
    y = df[TARGET]; X = df.drop(columns=[TARGET])
    # Sanitize column names for LightGBM/XGBoost compatibility
    X.columns = [str(c).replace(' ', '_').replace('[', '_').replace(']', '_')
                 .replace(',', '_').replace(':', '_').replace('{', '_')
                 .replace('}', '_').replace('<', '_').replace('>', '_')
                 .replace('"', '_') for c in X.columns]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        if hasattr(X[c], "cat"): X[c] = X[c].astype(str)
        X[c] = X[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def run_eda(df, target, save_dir):
    """Exploratory Data Analysis for fraud/imbalanced datasets."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Column types:\n{df.dtypes.value_counts().to_string()}")
    fraud_rate = df[target].mean()
    print(f"\nClass balance: {1 - fraud_rate:.2%} legit / {fraud_rate:.2%} fraud (ratio {int(1/max(fraud_rate,1e-9))}:1)")
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
    df[target].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "salmon"], edgecolor="black")
    ax.set_title(f"Target Distribution: {target} (Fraud rate: {fraud_rate:.2%})")
    ax.set_xlabel(target)
    fig.savefig(os.path.join(save_dir, "eda_target.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("EDA plots saved.")


def find_best_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    idx = np.argmax(f1s)
    return thresholds[idx] if idx < len(thresholds) else 0.5


def train_and_evaluate(X_train, X_test, y_train, y_test):
    from sklearn.calibration import CalibratedClassifierCV
    import gc
    def _gpu_cleanup():
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    results = {}      # name -> {preds, proba, thresh, model}
    timings = {}      # name -> wall-clock seconds
    # Hold out calibration split from training data
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
    scale = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    for name, builder in [
        ("CatBoost", lambda: __import__("catboost").CatBoostClassifier(
            iterations=300, learning_rate=0.03, depth=8, task_type="GPU", devices="0",
            scale_pos_weight=scale, eval_metric="F1", early_stopping_rounds=30, verbose=100)),
        ("LightGBM", lambda: __import__("lightgbm").LGBMClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=8,
            device="gpu", scale_pos_weight=scale, verbose=-1, n_jobs=-1)),
        ("XGBoost", lambda: __import__("xgboost").XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=8,
            device="cuda", tree_method="hist", scale_pos_weight=scale,
            eval_metric="aucpr", early_stopping_rounds=30, verbosity=1, n_jobs=-1)),
    ]:
        try:
            t0 = time.perf_counter()
            m = builder()
            if name == "LightGBM":
                import lightgbm as lgb
                m.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])
            else:
                m.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)] if name == "XGBoost"
                      else (X_cal, y_cal), verbose=100 if name == "XGBoost" else None)
            # Calibrate probabilities (isotonic regression on held-out cal split)
            import sklearn; _skv = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
            if _skv >= (1, 6):
                cal_model = CalibratedClassifierCV(m, method="isotonic")
                cal_model.fit(X_cal, y_cal)
            else:
                cal_model = CalibratedClassifierCV(m, cv="prefit", method="isotonic")
                cal_model.fit(X_cal, y_cal)
            proba = cal_model.predict_proba(X_test)[:, 1]
            thresh = find_best_threshold(y_test, proba)
            preds = (proba >= thresh).astype(int)
            timings[name] = time.perf_counter() - t0
            results[name] = {"preds": preds, "proba": proba, "thresh": thresh, "model": name}
            print(f"{name} F1: {f1_score(y_test, preds):.4f} (t={thresh:.3f}) [calibrated]  ({timings[name]:.1f}s)")
        except Exception as e:
            print(f"{name}: {e}")
        _gpu_cleanup()

    # ── PyOD Anomaly Scoring (unsupervised cross-check) ──
    for pyod_name, pyod_builder in [
        ("ECOD", lambda: __import__("pyod.models.ecod", fromlist=["ECOD"]).ECOD(contamination=0.05)),
        ("COPOD", lambda: __import__("pyod.models.copod", fromlist=["COPOD"]).COPOD(contamination=0.05)),
        ("IForest-PyOD", lambda: __import__("pyod.models.iforest", fromlist=["IForest"]).IForest(contamination=0.05, random_state=42)),
    ]:
        try:
            t0 = time.perf_counter()
            pm = pyod_builder()
            # Subsample large datasets for PyOD (CPU-bound, slow > 50k rows)
            MAX_PYOD = 50_000
            if len(X_train) > MAX_PYOD:
                _idx = np.random.RandomState(42).choice(len(X_train), MAX_PYOD, replace=False)
                _Xp = X_train.iloc[_idx] if hasattr(X_train, "iloc") else X_train[_idx]
            else:
                _Xp = X_train
            pm.fit(_Xp.values if hasattr(_Xp, "values") else _Xp)
            scores = pm.decision_function(X_test.values if hasattr(X_test, "values") else X_test)
            pyod_preds = pm.predict(X_test.values if hasattr(X_test, "values") else X_test)
            n_anom = pyod_preds.sum()
            f1 = f1_score(y_test, pyod_preds) if len(set(y_test)) > 1 else 0
            auc = roc_auc_score(y_test, scores) if len(set(y_test)) > 1 else 0
            elapsed = time.perf_counter() - t0
            timings[f"PyOD-{pyod_name}"] = elapsed
            print(f"PyOD {pyod_name}: {n_anom} anomalies ({n_anom/len(X_test):.2%}), F1={f1:.4f}, AUC={auc:.4f}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"PyOD {pyod_name}: {e}")

    # ── FLAML AutoML (imbalance-aware benchmark) ──
    try:
        from flaml import AutoML
        t0 = time.perf_counter()
        automl = AutoML()
        # Scale FLAML budget with dataset size
        _flaml_budget = 30 if len(X_train) > 100_000 else 60
        automl.fit(X_train, y_train, task="classification", time_budget=_flaml_budget,
                   metric="ap", verbose=0)
        flaml_proba = automl.predict_proba(X_test)[:, 1]
        flaml_thresh = find_best_threshold(y_test, flaml_proba)
        flaml_preds = (flaml_proba >= flaml_thresh).astype(int)
        timings["FLAML"] = time.perf_counter() - t0
        results["FLAML"] = {"preds": flaml_preds, "proba": flaml_proba, "thresh": flaml_thresh, "model": "FLAML"}
        print(f"FLAML ({automl.best_estimator}) F1: {f1_score(y_test, flaml_preds):.4f} (t={flaml_thresh:.3f})  ({timings['FLAML']:.1f}s)")
    except Exception as e:
        print(f"FLAML: {e}")

    # ── LazyPredict (quick sweep benchmark) ──
    try:
        from lazypredict.Supervised import LazyClassifier
        import mlflow
        # Disable MLflow sklearn autologging during LazyPredict to avoid logging 25+ models
        mlflow.sklearn.autolog(disable=True)
        t0 = time.perf_counter()
        lazy = LazyClassifier(verbose=0, ignore_warnings=True)
        # Use a small sample to keep runtime under 60s
        _lp_max = 5000
        _lp_xt = X_train.iloc[:_lp_max] if len(X_train) > _lp_max else X_train
        _lp_xe = X_test.iloc[:_lp_max] if len(X_test) > _lp_max else X_test
        _lp_yt = y_train.iloc[:_lp_max] if len(y_train) > _lp_max else y_train
        _lp_ye = y_test.iloc[:_lp_max] if len(y_test) > _lp_max else y_test
        lazy_models, _ = lazy.fit(_lp_xt, _lp_xe, _lp_yt, _lp_ye)
        mlflow.sklearn.autolog(disable=False)
        timings["LazyPredict"] = time.perf_counter() - t0
        print(f"\nLazyPredict — Top 5 classifiers:  ({timings['LazyPredict']:.1f}s)")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"LazyPredict: {e}")

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
        print(f"\nReport saved to {save_dir}/fraud_report.png")
    except Exception as e:
        print(f"Plot: {e}")

    # ── Save JSON metrics ──
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\nMetrics saved to {out_path}")


def cross_validate_best(X, y, save_dir):
    """5-fold stratified cross-validation on gradient boosting models."""
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-Fold Stratified)")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, build_fn in [
        ("CatBoost", lambda: __import__("catboost").CatBoostClassifier(
            iterations=100, verbose=0, task_type="GPU", devices="0")),
        ("LightGBM", lambda: __import__("lightgbm").LGBMClassifier(
            n_estimators=100, device="gpu", verbose=-1, n_jobs=-1)),
        ("XGBoost", lambda: __import__("xgboost").XGBClassifier(
            n_estimators=100, device="cuda", tree_method="hist",
            verbosity=0, n_jobs=-1)),
    ]:
        try:
            model = build_fn()
            try:
                import mlflow as _mlflow_cv; _mlflow_cv.sklearn.autolog(disable=True)
            except Exception:
                pass
            scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=1)
            cv_results[name] = {"f1_mean": round(float(scores.mean()), 4),
                                "f1_std": round(float(scores.std()), 4),
                                "folds": [round(float(s), 4) for s in scores]}
            print(f"  {name}: F1 {scores.mean():.4f} +/- {scores.std():.4f}")
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
    print("FRAUD / IMBALANCED CLASSIFICATION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | PyOD (ECOD/COPOD/IForest)")
    print("Calibrated probabilities + threshold tuning")
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
