"""
Modern Clustering Pipeline (April 2026)
Models: UMAP + HDBSCAN + GMM
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def load_data():
    from sklearn.datasets import fetch_california_housing
    _d = fetch_california_housing(as_frame=True)
    df = _d.frame
    # Drop ID-like columns
    for c in df.columns:
        if c.lower() in ("id", "customerid", "customer_id"): df.drop(columns=[c], inplace=True, errors="ignore")
    print(f"Dataset shape: {df.shape}")
    return df


def preprocess(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols: df[c] = df[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = oe.fit_transform(df[cat_cols])
    return StandardScaler().fit_transform(df.select_dtypes(include=["number"]))


def cluster(X):
    results = {}
    try:
        import umap, hdbscan
        X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
        labels = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(X_umap)
        n = len(set(labels)) - (1 if -1 in labels else 0)
        results["HDBSCAN"] = {"labels": labels, "embedding": X_umap, "n": n}
        mask = labels >= 0
        sil = silhouette_score(X_umap[mask], labels[mask]) if mask.sum() > 1 and n > 1 else 0
        print(f"✓ HDBSCAN: {n} clusters, silhouette={sil:.4f}")
    except Exception as e: print(f"✗ HDBSCAN: {e}")

    try:
        from sklearn.mixture import GaussianMixture
        bics = [GaussianMixture(n_components=k, random_state=42).fit(X).bic(X) for k in range(2, 11)]
        best_k = range(2, 11)[np.argmin(bics)]
        labels = GaussianMixture(n_components=best_k, random_state=42).fit_predict(X)
        sil = silhouette_score(X, labels) if best_k > 1 else 0
        results["GMM"] = {"labels": labels, "n": best_k}
        print(f"✓ GMM: k={best_k}, silhouette={sil:.4f}")
    except Exception as e: print(f"✗ GMM: {e}")

    return results


def main():
    print("=" * 60)
    print("CLUSTERING: UMAP + HDBSCAN + GMM")
    print("=" * 60)
    df = load_data()
    X = preprocess(df)
    cluster(X)


if __name__ == "__main__":
    main()
