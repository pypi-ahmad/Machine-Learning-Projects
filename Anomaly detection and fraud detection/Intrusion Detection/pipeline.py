"""
Modern Unsupervised Anomaly Detection Pipeline (April 2026)

Approach: Purely unsupervised — no labeled fraud targets.
          If ground-truth labels exist in the dataset they are used ONLY for
          post-hoc evaluation, never for training.

Models (PyOD 2):
  - ECOD  — Empirical CDF-based, parameter-free, fast
  - COPOD — Copula-based, parameter-free, fast
  - IForest — Isolation Forest (ensemble baseline)
  - LOF   — Local Outlier Factor (density-based, good for local anomalies)

Visual anomaly detection:
  - anomalib PatchCore (wide_resnet50_2 backbone, MVTec benchmark)

Compute: CPU-only for PyOD models (<30s each). GPU for anomalib PatchCore.
Data: Auto-downloaded at runtime
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def load_data():
    from sklearn.datasets import fetch_kddcup99
    _d = fetch_kddcup99(as_frame=True)
    df = _d.frame
    print(f"Dataset shape: {df.shape}")
    return df


def preprocess(df):
    """Prepare data for unsupervised detection.

    Returns (X_scaled, y_true_or_None).  Labels are auto-detected but NEVER
    used for training — only for optional post-hoc evaluation.
    """
    df = df.copy()
    label_col = next((c for c in df.columns if str(c).lower() in ("label","class","target","anomaly","outlier")), None)
    y = None
    if label_col:
        y = df[label_col].values; df.drop(columns=[label_col], inplace=True)
        print(f" Ground-truth column '{label_col}' detected — used for evaluation only (not training)")
    for c in df.columns:
        if str(c).lower() in ("id","timestamp","date","time"): df.drop(columns=[c], inplace=True, errors="ignore")
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        if hasattr(df[c], "cat"): df[c] = df[c].astype(str)
        df[c] = df[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = oe.fit_transform(df[cat_cols])
    return StandardScaler().fit_transform(df.select_dtypes(include=["number"])), y


def detect(X, y=None):
    results = {}      # name -> {labels, scores}
    timings = {}      # name -> wall-clock seconds
    metrics_out = {}  # name -> dict of metrics
    has_labels = y is not None and len(set(y)) > 1

    for name, Builder in [
        ("ECOD", lambda: __import__("pyod.models.ecod", fromlist=["ECOD"]).ECOD(contamination=0.05)),
        ("COPOD", lambda: __import__("pyod.models.copod", fromlist=["COPOD"]).COPOD(contamination=0.05)),
        ("IForest", lambda: __import__("pyod.models.iforest", fromlist=["IForest"]).IForest(contamination=0.05, random_state=42)),
        ("LOF", lambda: __import__("pyod.models.lof", fromlist=["LOF"]).LOF(contamination=0.05, n_neighbors=20)),
    ]:
        try:
            t0 = time.perf_counter()
            m = Builder()
            m.fit(X)
            elapsed = time.perf_counter() - t0
            labels = m.labels_
            scores = m.decision_scores_ if hasattr(m, "decision_scores_") else m.decision_function(X)
            timings[name] = elapsed
            results[name] = {"labels": labels, "scores": scores}
            n_anom = int(labels.sum())
            row = {"anomalies": n_anom, "anomaly_pct": round(n_anom / len(X), 4), "time_s": round(elapsed, 1)}

            # Score distribution summary
            p50, p90, p95, p99 = np.percentile(scores, [50, 90, 95, 99])
            row["score_p50"] = round(float(p50), 4)
            row["score_p95"] = round(float(p95), 4)
            row["score_p99"] = round(float(p99), 4)

            extra = ""
            if has_labels:
                f1 = f1_score(y, labels)
                auc = roc_auc_score(y, scores)
                row["f1"] = round(f1, 4)
                row["roc_auc"] = round(auc, 4)
                extra = f"  F1: {f1:.4f}  ROC-AUC: {auc:.4f}"

            metrics_out[name] = row
            print(f"{name}: {n_anom} anomalies ({n_anom/len(X):.2%})  ({elapsed:.1f}s){extra}")
        except Exception as e:
            print(f"{name}: {e}")

    # ── Score distribution plot ──
    save_dir = os.path.dirname(os.path.abspath(__file__))
    if results:
        try:
            n_models = len(results)
            fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
            if n_models == 1: axes = [axes]
            for ax, (name, r) in zip(axes, results.items()):
                ax.hist(r["scores"][r["labels"] == 0], bins=50, alpha=0.6, label="Normal", density=True)
                ax.hist(r["scores"][r["labels"] == 1], bins=50, alpha=0.6, label="Anomaly", density=True)
                ax.set_title(name); ax.set_xlabel("Anomaly score"); ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "anomaly_scores.png"), dpi=100, bbox_inches="tight")
            plt.close()
            print("Saved anomaly_scores.png")
        except Exception as e:
            print(f"Score plot: {e}")

    # ── Agreement matrix (how many detectors flag each point) ──
    if len(results) > 1:
        try:
            all_labels = np.column_stack([r["labels"] for r in results.values()])
            votes = all_labels.sum(axis=1)
            for k in range(1, len(results) + 1):
                n = int((votes >= k).sum())
                print(f"  Flagged by ≥{k} detectors: {n} ({n/len(X):.2%})")
        except Exception:
            pass

    # ── anomalib PatchCore (image-based anomaly detection) ──
    try:
        t0 = time.perf_counter()
        from anomalib.models import Patchcore
        from anomalib.data import MVTec
        from anomalib.engine import Engine
        datamodule = MVTec(category="bottle", image_size=(256, 256), train_batch_size=8, eval_batch_size=8)
        model = Patchcore(backbone="wide_resnet50_2", layers_to_extract=["layer2", "layer3"],
                          coreset_sampling_ratio=0.1, num_neighbors=9)
        engine = Engine(max_epochs=1, devices=1, accelerator="auto")
        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)
        elapsed = time.perf_counter() - t0
        timings["PatchCore"] = elapsed
        print(f"PatchCore (anomalib): {test_results}  ({elapsed:.1f}s)")
    except Exception as e:
        print(f"PatchCore: {e}")

    # ── Save JSON metrics ──
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\nMetrics saved to {out_path}")


def run_eda(df, save_dir):
    """Exploratory Data Analysis for anomaly detection."""
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
        import seaborn as _sns
        _sns.heatmap(corr, mask=mask, annot=n <= 15, fmt=".2f",
                     cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Heatmap")
        fig.savefig(os.path.join(save_dir, "eda_correlation.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    plot_cols = num_cols[:20]
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


def main():
    print("=" * 60)
    print("UNSUPERVISED ANOMALY DETECTION")
    print("PyOD 2 (ECOD / COPOD / IForest / LOF) + anomalib PatchCore")
    print("Labels used for evaluation only — never for training")
    print("=" * 60)
    df = load_data()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    run_eda(df, save_dir)
    X, y = preprocess(df)
    detect(X, y)


if __name__ == "__main__":
    main()
