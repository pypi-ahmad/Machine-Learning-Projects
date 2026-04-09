"""
Modern Anomaly Detection Pipeline (April 2026)
Models: PyOD 2 (ECOD, COPOD, IForest) + anomalib PatchCore
Data: Auto-downloaded at runtime
"""
import os, warnings
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
    df = df.copy()
    label_col = next((c for c in df.columns if c.lower() in ("label","class","target","anomaly","outlier")), None)
    y = None
    if label_col:
        y = df[label_col].values; df.drop(columns=[label_col], inplace=True)
    for c in df.columns:
        if c.lower() in ("id","timestamp","date","time"): df.drop(columns=[c], inplace=True, errors="ignore")
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols: df[c] = df[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = oe.fit_transform(df[cat_cols])
    return StandardScaler().fit_transform(df.select_dtypes(include=["number"])), y


def detect(X, y=None):
    for name, Builder in [
        ("ECOD", lambda: __import__("pyod.models.ecod", fromlist=["ECOD"]).ECOD(contamination=0.05)),
        ("COPOD", lambda: __import__("pyod.models.copod", fromlist=["COPOD"]).COPOD(contamination=0.05)),
        ("IForest", lambda: __import__("sklearn.ensemble", fromlist=["IsolationForest"]).IsolationForest(contamination=0.05, random_state=42)),
    ]:
        try:
            m = Builder()
            if "IForest" in name:
                preds = m.fit_predict(X); labels = (preds == -1).astype(int)
            else:
                m.fit(X); labels = m.labels_
            print(f"✓ {name}: {labels.sum()} anomalies ({labels.mean():.2%})")
            if y is not None and len(set(y)) > 1:
                print(f"  F1: {f1_score(y, labels):.4f}")
        except Exception as e:
            print(f"✗ {name}: {e}")

    # anomalib PatchCore (image-based anomaly detection)
    try:
        from anomalib.models import Patchcore
        from anomalib.data import MVTec
        from anomalib.engine import Engine
        datamodule = MVTec(category="bottle", image_size=(256, 256), train_batch_size=8, eval_batch_size=8)
        model = Patchcore(backbone="wide_resnet50_2", layers_to_extract=["layer2", "layer3"],
                          coreset_sampling_ratio=0.1, num_neighbors=9)
        engine = Engine(max_epochs=1, devices=1, accelerator="auto")
        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)
        print(f"✓ PatchCore (anomalib): {test_results}")
    except Exception as e:
        print(f"✗ PatchCore: {e}")


def main():
    print("=" * 60)
    print("ANOMALY DETECTION — PyOD 2 + anomalib PatchCore")
    print("=" * 60)
    df = load_data()
    X, y = preprocess(df)
    detect(X, y)


if __name__ == "__main__":
    main()
