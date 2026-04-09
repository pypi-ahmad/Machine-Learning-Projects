"""
Modern Clustering Pipeline (April 2026)
Models: UMAP + HDBSCAN (primary) + GMM (soft assignments) + K-Means (baseline)
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
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


def eval_clustering(X, labels, name):
    mask = labels >= 0
    n = len(set(labels[mask]))
    noise = (labels == -1).sum()
    if n > 1 and mask.sum() > n:
        sil = silhouette_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
        db = davies_bouldin_score(X[mask], labels[mask])
        print(f"  {name}: {n} clusters, noise={noise}, silhouette={sil:.4f}, CH={ch:.1f}, DB={db:.4f}")
        return sil
    else:
        print(f"  {name}: {n} clusters, noise={noise} — insufficient for metrics")
        return 0


def cluster(X):
    results = {}

    # ═══ PRIMARY: UMAP + HDBSCAN ═══
    try:
        import umap, hdbscan
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X)

        # Auto-tune min_cluster_size
        best_sil, best_mcs, best_labels = -1, 15, None
        for mcs in [5, 10, 15, 25, 50]:
            lbls = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5).fit_predict(X_umap)
            mask = lbls >= 0
            n = len(set(lbls[mask]))
            if n > 1 and mask.sum() > n:
                s = silhouette_score(X_umap[mask], lbls[mask])
                if s > best_sil:
                    best_sil, best_mcs, best_labels = s, mcs, lbls
        if best_labels is None:
            best_labels = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(X_umap)
        print(f"✓ UMAP + HDBSCAN (min_cluster_size={best_mcs}):")
        eval_clustering(X_umap, best_labels, "HDBSCAN")
        results["HDBSCAN"] = {"labels": best_labels, "embedding": X_umap}
    except Exception as e:
        print(f"✗ UMAP + HDBSCAN: {e}")
        # Fallback: PCA for embedding
        from sklearn.decomposition import PCA
        X_umap = PCA(n_components=2).fit_transform(X)

    # ═══ SOFT ASSIGNMENTS: Gaussian Mixture Model ═══
    try:
        from sklearn.mixture import GaussianMixture
        bics = [GaussianMixture(n_components=k, random_state=42).fit(X).bic(X) for k in range(2, 11)]
        best_k = range(2, 11)[np.argmin(bics)]
        gmm = GaussianMixture(n_components=best_k, random_state=42).fit(X)
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)
        print(f"✓ GMM (BIC-optimal k={best_k}):")
        eval_clustering(X, labels, "GMM")
        avg_confidence = probs.max(axis=1).mean()
        print(f"  Avg assignment confidence: {avg_confidence:.4f}")
        results["GMM"] = {"labels": labels, "n": best_k, "probs": probs}
    except Exception as e:
        print(f"✗ GMM: {e}")

    # ═══ BASELINE: K-Means (Elbow + Silhouette) ═══
    try:
        from sklearn.cluster import KMeans
        inertias, sils = [], []
        K_range = range(2, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbls = km.fit_predict(X)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X, lbls))
        best_k = K_range[np.argmax(sils)]
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
        print(f"✓ K-Means baseline (best k={best_k}, silhouette={max(sils):.4f}):")
        eval_clustering(X, labels, "K-Means")
        results["KMeans"] = {"labels": labels, "n": best_k, "inertias": inertias, "sils": sils}
    except Exception as e:
        print(f"✗ K-Means: {e}")

    # ═══ VISUALIZATION ═══
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        embed = results.get("HDBSCAN", {}).get("embedding", X[:, :2] if X.shape[1] >= 2 else X)
        for ax, (name, data) in zip(axes, [
            ("HDBSCAN", results.get("HDBSCAN", {}).get("labels")),
            ("GMM", results.get("GMM", {}).get("labels")),
            ("K-Means", results.get("KMeans", {}).get("labels")),
        ]):
            if data is not None:
                scatter = ax.scatter(embed[:, 0], embed[:, 1], c=data, cmap="tab10", s=10, alpha=0.6)
                ax.set_title(name); ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
            else:
                ax.set_title(f"{name} (N/A)")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "clustering_results.png"), dpi=100)
        print("Saved clustering_results.png")
    except Exception as e:
        print(f"⚠ Plot: {e}")

    # ═══ SUMMARY ═══
    print("\n" + "=" * 40)
    print("CLUSTERING COMPARISON:")
    for name in ["HDBSCAN", "GMM", "KMeans"]:
        if name in results:
            n = len(set(results[name]["labels"])) - (1 if -1 in results[name]["labels"] else 0)
            print(f"  {name}: {n} clusters")
    print("=" * 40)


def main():
    print("=" * 60)
    print("CLUSTERING: UMAP + HDBSCAN (primary) | GMM | K-Means baseline")
    print("=" * 60)
    df = load_data()
    X = preprocess(df)
    cluster(X)


if __name__ == "__main__":
    main()
