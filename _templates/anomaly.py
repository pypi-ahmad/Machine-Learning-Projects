"""Anomaly Detection pipeline template: PyOD 2 (ECOD, COPOD, IForest) — April 2026"""
import textwrap


def generate(project_path, config):
    return textwrap.dedent('''\
        """
        Modern Anomaly Detection Pipeline (April 2026)
        Models: PyOD 2 — ECOD, COPOD, Isolation Forest, SUOD ensemble
        """
        import os, warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.preprocessing import StandardScaler, OrdinalEncoder
        from sklearn.metrics import (
            classification_report, roc_auc_score, precision_recall_curve,
            average_precision_score, f1_score
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        warnings.filterwarnings("ignore")


        def load_data():
            data_dir = Path(os.path.dirname(__file__))
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
            else:
                raise FileNotFoundError("No CSV data found in project folder.")
            print(f"Dataset shape: {df.shape}")
            return df


        def preprocess(df):
            df = df.copy()

            # Try to find label column
            label_col = None
            for c in df.columns:
                if c.lower() in ("label", "class", "is_anomaly", "anomaly", "target", "outlier"):
                    label_col = c
                    break

            y = None
            if label_col:
                y = df[label_col].values
                df.drop(columns=[label_col], inplace=True)
                print(f"Label column: {label_col}, anomaly rate: {y.mean():.4%}")

            # Drop ID-like columns
            for c in df.columns:
                if c.lower() in ("id", "timestamp", "date", "time"):
                    df.drop(columns=[c], inplace=True, errors="ignore")

            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()

            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            for c in cat_cols:
                df[c] = df[c].fillna("unknown")

            if cat_cols:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                df[cat_cols] = oe.fit_transform(df[cat_cols])

            scaler = StandardScaler()
            X = scaler.fit_transform(df.select_dtypes(include=["number"]))
            return X, y


        def detect_anomalies(X, y=None):
            results = {}

            # ── 1. ECOD (Empirical Cumulative Distribution) ──
            try:
                from pyod.models.ecod import ECOD
                ecod = ECOD(contamination=0.05)
                ecod.fit(X)
                labels = ecod.labels_
                scores = ecod.decision_scores_
                results["ECOD"] = {"labels": labels, "scores": scores}
                print(f"✓ ECOD: {labels.sum()} anomalies detected ({labels.mean():.2%})")
            except Exception as e:
                print(f"✗ ECOD: {e}")

            # ── 2. COPOD (Copula-based) ──
            try:
                from pyod.models.copod import COPOD
                copod = COPOD(contamination=0.05)
                copod.fit(X)
                labels = copod.labels_
                scores = copod.decision_scores_
                results["COPOD"] = {"labels": labels, "scores": scores}
                print(f"✓ COPOD: {labels.sum()} anomalies detected ({labels.mean():.2%})")
            except Exception as e:
                print(f"✗ COPOD: {e}")

            # ── 3. Isolation Forest (scikit-learn) ──
            try:
                from sklearn.ensemble import IsolationForest
                ifo = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
                preds = ifo.fit_predict(X)
                labels = (preds == -1).astype(int)
                scores = -ifo.decision_function(X)
                results["IsolationForest"] = {"labels": labels, "scores": scores}
                print(f"✓ IsolationForest: {labels.sum()} anomalies detected ({labels.mean():.2%})")
            except Exception as e:
                print(f"✗ IsolationForest: {e}")

            # ── 4. SUOD (Scalable Unsupervised Outlier Detection) ──
            try:
                from pyod.models.suod import SUOD
                from pyod.models.ecod import ECOD as ECOD2
                from pyod.models.copod import COPOD as COPOD2
                from pyod.models.iforest import IForest

                base = [ECOD2(), COPOD2(), IForest()]
                suod = SUOD(base_estimators=base, contamination=0.05, n_jobs=-1)
                suod.fit(X)
                labels = suod.labels_
                scores = suod.decision_scores_
                results["SUOD"] = {"labels": labels, "scores": scores}
                print(f"✓ SUOD ensemble: {labels.sum()} anomalies detected ({labels.mean():.2%})")
            except Exception as e:
                print(f"✗ SUOD: {e}")

            return results


        def report(results, y=None, save_dir="."):
            print("\\n" + "=" * 60)
            print("ANOMALY DETECTION COMPARISON")
            print("=" * 60)

            for name, res in results.items():
                labels, scores = res["labels"], res["scores"]
                print(f"\\n── {name} ──")
                print(f"  Anomalies: {labels.sum()} / {len(labels)} ({labels.mean():.2%})")

                if y is not None:
                    y_binary = (y > 0).astype(int) if y.dtype != 'int' else y
                    if len(set(y_binary)) > 1:
                        auc = roc_auc_score(y_binary, scores)
                        ap = average_precision_score(y_binary, scores)
                        f1 = f1_score(y_binary, labels)
                        print(f"  ROC-AUC: {auc:.4f}  |  AUPRC: {ap:.4f}  |  F1: {f1:.4f}")
                        print(classification_report(y_binary, labels,
                                                    target_names=["Normal", "Anomaly"], zero_division=0))

                # Score distribution plot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(scores[labels == 0], bins=50, alpha=0.6, label="Normal", color="steelblue")
                ax.hist(scores[labels == 1], bins=50, alpha=0.6, label="Anomaly", color="crimson")
                ax.set_title(f"{name} — Anomaly Score Distribution")
                ax.set_xlabel("Anomaly Score")
                ax.legend()
                fig.savefig(os.path.join(save_dir, f"scores_{name.lower()}.png"),
                            dpi=100, bbox_inches="tight")
                plt.close(fig)


        def main():
            print("=" * 60)
            print("MODERN ANOMALY DETECTION PIPELINE")
            print("PyOD 2: ECOD | COPOD | IsolationForest | SUOD")
            print("=" * 60)
            df = load_data()
            X, y = preprocess(df)
            results = detect_anomalies(X, y)
            if results:
                report(results, y, os.path.dirname(os.path.abspath(__file__)))


        if __name__ == "__main__":
            main()
    ''')
