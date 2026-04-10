"""
Modern Tabular Classification Pipeline (April 2026)
Models: CatBoost/LightGBM/XGBoost (GPU) + AutoGluon + RealTabPFN-v2 + TabM
Data: Auto-downloaded at runtime — no local files needed

Compute: GPU recommended (CatBoost/LightGBM/XGBoost use CUDA, TabM uses torch.cuda).
         CPU fallback is automatic. ~2-10 min per dataset on RTX 4060.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

TARGET = "Survived"


def load_data():
    """Download dataset from the internet."""
    import os, glob as _glob
    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(_data_dir, exist_ok=True)
    _fp = os.path.join(_data_dir, "train_and_test2.csv")
    if not os.path.exists(_fp):
        from kaggle.api.kaggle_api_extended import KaggleApi
        _api = KaggleApi(); _api.authenticate()
        _api.dataset_download_files("heptapod/titanic", path=_data_dir, unzip=True)
        _matches = _glob.glob(os.path.join(_data_dir, "**", "train_and_test2.csv"), recursive=True)
        if _matches: _fp = _matches[0]
        print(f"Downloaded heptapod/titanic from Kaggle")
    df = pd.read_csv(_fp)
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
        if hasattr(X[c], "cat"): X[c] = X[c].astype(str)
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


def run_eda(df, target, save_dir):
    """Exploratory Data Analysis — summary stats, distributions, correlations."""
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
    probas  = {}      # name -> probability array (for ROC-AUC / PR-AUC / calibration)
    timings = {}      # name -> wall-clock seconds
    n_classes = y_train.nunique()
    is_binary = n_classes == 2

    # ── CatBoost (GPU) ──
    try:
        from catboost import CatBoostClassifier
        t0 = time.perf_counter()
        cb = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=8,
            task_type="GPU", devices="0",
            eval_metric="AUC" if is_binary else "MultiClass",
            early_stopping_rounds=50, verbose=100,
            auto_class_weights="Balanced",
        )
        cb.fit(X_train, y_train, eval_set=(X_test, y_test))
        timings["CatBoost"] = time.perf_counter() - t0
        results["CatBoost"] = cb.predict(X_test).flatten()
        probas["CatBoost"] = cb.predict_proba(X_test)
        print(f"\nCatBoost Accuracy: {accuracy_score(y_test, results['CatBoost']):.4f}  ({timings['CatBoost']:.1f}s)")
    except Exception as e:
        print(f"CatBoost: {e}")

    # ── LightGBM (GPU) ──
    try:
        import lightgbm as lgb
        t0 = time.perf_counter()
        m = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8,
            device="gpu", class_weight="balanced", verbose=-1, n_jobs=-1,
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        timings["LightGBM"] = time.perf_counter() - t0
        results["LightGBM"] = m.predict(X_test)
        probas["LightGBM"] = m.predict_proba(X_test)
        print(f"\nLightGBM Accuracy: {accuracy_score(y_test, results['LightGBM']):.4f}  ({timings['LightGBM']:.1f}s)")
    except Exception as e:
        print(f"LightGBM: {e}")

    # ── XGBoost (CUDA) ──
    try:
        from xgboost import XGBClassifier
        t0 = time.perf_counter()
        m = XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8,
            device="cuda", tree_method="hist",
            eval_metric="auc" if is_binary else "mlogloss",
            early_stopping_rounds=50, verbosity=1, n_jobs=-1,
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
        timings["XGBoost"] = time.perf_counter() - t0
        results["XGBoost"] = m.predict(X_test)
        probas["XGBoost"] = m.predict_proba(X_test)
        print(f"\nXGBoost Accuracy: {accuracy_score(y_test, results['XGBoost']):.4f}  ({timings['XGBoost']:.1f}s)")
    except Exception as e:
        print(f"XGBoost: {e}")

    # ── AutoGluon Tabular ──
    try:
        from autogluon.tabular import TabularPredictor
        import tempfile
        t0 = time.perf_counter()
        train_ag = X_train.copy(); train_ag["Survived"] = y_train.values
        test_ag = X_test.copy(); test_ag["Survived"] = y_test.values
        with tempfile.TemporaryDirectory() as tmp:
            predictor = TabularPredictor(label="Survived", path=tmp, verbosity=0)
            predictor.fit(train_ag, time_limit=120, presets="medium_quality")
            results["AutoGluon"] = predictor.predict(test_ag.drop(columns=["Survived"])).values
            try:
                probas["AutoGluon"] = predictor.predict_proba(test_ag.drop(columns=["Survived"])).values
            except Exception:
                pass
            timings["AutoGluon"] = time.perf_counter() - t0
            print(f"\nAutoGluon Accuracy: {accuracy_score(y_test, results['AutoGluon']):.4f}  ({timings['AutoGluon']:.1f}s)")
    except Exception as e:
        print(f"AutoGluon: {e}")

    # ── RealTabPFN-v2 (prior-fitted network) ──
    try:
        from tabpfn import TabPFNClassifier
        if X_train.shape[0] <= 10000 and X_train.shape[1] <= 500:
            t0 = time.perf_counter()
            m = TabPFNClassifier(device="cuda", N_ensemble_configurations=32)
            m.fit(X_train.values, y_train.values)
            timings["TabPFN-v2"] = time.perf_counter() - t0
            results["TabPFN-v2"] = m.predict(X_test.values)
            try:
                probas["TabPFN-v2"] = m.predict_proba(X_test.values)
            except Exception:
                pass
            print(f"\nTabPFN-v2 Accuracy: {accuracy_score(y_test, results['TabPFN-v2']):.4f}  ({timings['TabPFN-v2']:.1f}s)")
        else:
            print("TabPFN-v2: dataset too large (>10k rows or >500 cols), skipped")
    except Exception as e:
        print(f"TabPFN-v2: {e}")

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

        t0 = time.perf_counter()
        model = TabMNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        for ep in range(100):
            model.train(); loss = loss_fn(model(Xt), yt); loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            results["TabM"] = torch.argmax(logits, dim=-1).cpu().numpy()
            probas["TabM"] = torch.softmax(logits, dim=-1).cpu().numpy()
        timings["TabM"] = time.perf_counter() - t0
        print(f"\nTabM Accuracy: {accuracy_score(y_test, results['TabM']):.4f}  ({timings['TabM']:.1f}s)")
    except Exception as e:
        print(f"TabM: {e}")

    # ── Baseline Comparison: FLAML AutoML ──
    try:
        from flaml import AutoML
        t0 = time.perf_counter()
        automl = AutoML()
        automl.fit(X_train, y_train, task="classification", time_budget=120, verbose=0)
        timings["FLAML"] = time.perf_counter() - t0
        results["FLAML"] = automl.predict(X_test)
        try:
            probas["FLAML"] = automl.predict_proba(X_test)
        except Exception:
            pass
        print(f"\nFLAML ({automl.best_estimator}) Accuracy: {accuracy_score(y_test, results['FLAML']):.4f}  ({timings['FLAML']:.1f}s)")
    except Exception as e:
        print(f"FLAML: {e}")

    # ── Baseline Comparison: LazyPredict ──
    try:
        from lazypredict.Supervised import LazyClassifier
        t0 = time.perf_counter()
        lazy = LazyClassifier(verbose=0, ignore_warnings=True)
        lazy_models, _ = lazy.fit(X_train, X_test, y_train, y_test)
        timings["LazyPredict"] = time.perf_counter() - t0
        print(f"\nLazyPredict — Top 5 classifiers:  ({timings['LazyPredict']:.1f}s)")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"LazyPredict: {e}")

    return results, probas, timings


def report(results, probas, timings, y_test, save_dir="."):
    n_classes = len(set(y_test))
    is_binary = n_classes == 2
    metrics_out = {}

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    best_name, best_acc = None, 0
    for name, y_pred in results.items():
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        row = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}

        extra = ""
        if name in probas:
            p = probas[name]
            try:
                if is_binary:
                    p1 = p[:, 1] if p.ndim == 2 else p
                    row["roc_auc"] = round(roc_auc_score(y_test, p1), 4)
                    row["pr_auc"] = round(average_precision_score(y_test, p1), 4)
                    row["brier"] = round(brier_score_loss(y_test, p1), 4)
                    extra = f"  ROC-AUC: {row['roc_auc']:.4f}  PR-AUC: {row['pr_auc']:.4f}"
                else:
                    row["roc_auc_ovr"] = round(roc_auc_score(
                        y_test, p, multi_class="ovr", average="weighted"
                    ), 4)
                    extra = f"  ROC-AUC(OVR): {row['roc_auc_ovr']:.4f}"
            except Exception:
                pass

        if name in timings:
            row["time_s"] = round(timings[name], 1)

        print(f"\n— {name} —  Accuracy: {acc:.4f}  |  F1: {f1:.4f}{extra}")
        print(classification_report(y_test, y_pred, zero_division=0))
        if acc > best_acc:
            best_acc, best_name = acc, name
        metrics_out[name] = row

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        fig.savefig(os.path.join(save_dir, f"cm_{name.lower().replace(' ', '_')}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ── Calibration plot (binary only) ──
    if is_binary and probas:
        try:
            from sklearn.calibration import calibration_curve
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            for name, p in probas.items():
                p1 = p[:, 1] if p.ndim == 2 else p
                prob_true, prob_pred = calibration_curve(y_test, p1, n_bins=10, strategy="uniform")
                ax.plot(prob_pred, prob_true, "s-", label=name)
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title("Calibration Plot (Reliability Diagram)")
            ax.legend(loc="lower right", fontsize=8)
            fig.savefig(os.path.join(save_dir, "calibration_plot.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)
            print("\nCalibration plot saved")
        except Exception:
            pass

    print(f"\nBest: {best_name} ({best_acc:.4f})")

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
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
            cv_results[name] = {"mean": round(float(scores.mean()), 4),
                                "std": round(float(scores.std()), 4),
                                "folds": [round(float(s), 4) for s in scores]}
            print(f"  {name}: {scores.mean():.4f} +/- {scores.std():.4f}")
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
    print("MODERN TABULAR CLASSIFICATION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | AutoGluon | TabPFN-v2 | TabM | FLAML | LazyPredict")
    print("=" * 60)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_data()
    run_eda(df, TARGET, save_dir)
    X_train, X_test, y_train, y_test, le = preprocess(df)
    results, probas, timings = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, probas, timings, y_test, save_dir)
    cross_validate_best(
        pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), save_dir)


if __name__ == "__main__":
    main()
