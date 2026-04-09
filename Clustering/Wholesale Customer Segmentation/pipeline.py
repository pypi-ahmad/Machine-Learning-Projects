"""
Modern Clustering Pipeline (April 2026)
Models: UMAP + HDBSCAN (primary) + GMM (soft assignments) + K-Means (baseline)
Data: Auto-downloaded at runtime

Compute: CPU-only for K-Means/GMM (<10s). UMAP + HDBSCAN ~10-60s depending
         on dataset size. No GPU required.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def load_data():
    from sklearn.datasets import fetch_openml
    _d = fetch_openml(data_id=1511, as_frame=True, parser="auto")
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
        return {"n_clusters": n, "noise": noise, "silhouette": round(sil, 4),
                "calinski_harabasz": round(ch, 1), "davies_bouldin": round(db, 4)}
    else:
        print(f"  {name}: {n} clusters, noise={noise} — insufficient for metrics")
        return {"n_clusters": n, "noise": noise}


def cluster(X):
    results = {}
    timings = {}
    metrics_out = {}
    save_dir = os.path.dirname(os.path.abspath(__file__))

    # ═══ PRIMARY: UMAP + HDBSCAN ═══
    try:
        import umap, hdbscan
        t0 = time.perf_counter()
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
        timings["HDBSCAN"] = time.perf_counter() - t0
        print(f"✓ UMAP + HDBSCAN (min_cluster_size={best_mcs})  ({timings['HDBSCAN']:.1f}s):")
        m = eval_clustering(X_umap, best_labels, "HDBSCAN")
        m["time_s"] = round(timings["HDBSCAN"], 1)
        m["min_cluster_size"] = best_mcs
        metrics_out["HDBSCAN"] = m
        results["HDBSCAN"] = {"labels": best_labels, "embedding": X_umap}
    except Exception as e:
        print(f"✗ UMAP + HDBSCAN: {e}")
        # Fallback: PCA for embedding
        from sklearn.decomposition import PCA
        X_umap = PCA(n_components=2).fit_transform(X)

    # ═══ SOFT ASSIGNMENTS: Gaussian Mixture Model ═══
    try:
        from sklearn.mixture import GaussianMixture
        t0 = time.perf_counter()
        bics = [GaussianMixture(n_components=k, random_state=42).fit(X).bic(X) for k in range(2, 11)]
        best_k = range(2, 11)[np.argmin(bics)]
        gmm = GaussianMixture(n_components=best_k, random_state=42).fit(X)
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)
        timings["GMM"] = time.perf_counter() - t0
        print(f"✓ GMM (BIC-optimal k={best_k})  ({timings['GMM']:.1f}s):")
        m = eval_clustering(X, labels, "GMM")
        m["time_s"] = round(timings["GMM"], 1)
        m["best_k"] = best_k
        avg_confidence = float(probs.max(axis=1).mean())
        m["avg_confidence"] = round(avg_confidence, 4)
        metrics_out["GMM"] = m
        print(f"  Avg assignment confidence: {avg_confidence:.4f}")
        results["GMM"] = {"labels": labels, "n": best_k, "probs": probs}
    except Exception as e:
        print(f"✗ GMM: {e}")

    # ═══ BASELINE: K-Means (Elbow + Silhouette) ═══
    try:
        from sklearn.cluster import KMeans
        t0 = time.perf_counter()
        inertias, sils = [], []
        K_range = range(2, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbls = km.fit_predict(X)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X, lbls))
        best_k = K_range[np.argmax(sils)]
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
        timings["KMeans"] = time.perf_counter() - t0
        print(f"✓ K-Means baseline (best k={best_k}, silhouette={max(sils):.4f})  ({timings['KMeans']:.1f}s):")
        m = eval_clustering(X, labels, "K-Means")
        m["time_s"] = round(timings["KMeans"], 1)
        m["best_k"] = best_k
        metrics_out["KMeans"] = m
        results["KMeans"] = {"labels": labels, "n": best_k, "inertias": inertias, "sils": sils}

        # Elbow + Silhouette plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(list(K_range), inertias, "bo-")
        axes[0].set_title("Elbow Method"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
        axes[1].plot(list(K_range), sils, "rs-")
        axes[1].set_title("Silhouette Scores"); axes[1].set_xlabel("k"); axes[1].set_ylabel("Score")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "kmeans_elbow_silhouette.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("✓ Saved kmeans_elbow_silhouette.png")
    except Exception as e:
        print(f"✗ K-Means: {e}")

    # ═══ VISUALIZATION ═══
    try:
        embed = results.get("HDBSCAN", {}).get("embedding", X[:, :2] if X.shape[1] >= 2 else X)
        active = [(n, results[n]["labels"]) for n in ["HDBSCAN", "GMM", "KMeans"] if n in results]
        n_plots = len(active)
        fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(6 * max(n_plots, 1), 5))
        if n_plots == 1: axes = [axes]
        for ax, (name, lbls) in zip(axes, active):
            scatter = ax.scatter(embed[:, 0], embed[:, 1], c=lbls, cmap="tab10", s=10, alpha=0.6)
            ax.set_title(name); ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "clustering_results.png"), dpi=100, bbox_inches="tight")
        plt.close()
        print("✓ Saved clustering_results.png")
    except Exception as e:
        print(f"⚠ Plot: {e}")

    # ═══ SUMMARY ═══
    print("\n" + "=" * 40)
    print("CLUSTERING COMPARISON:")
    for name in ["HDBSCAN", "GMM", "KMeans"]:
        if name in results:
            n = len(set(results[name]["labels"])) - (1 if -1 in results[name]["labels"] else 0)
            t = f"  ({timings[name]:.1f}s)" if name in timings else ""
            print(f"  {name}: {n} clusters{t}")
    print("=" * 40)

    # ── Save JSON metrics ──
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n✓ Metrics saved → {out_path}")


def main():
    print("=" * 60)
    print("CLUSTERING: UMAP + HDBSCAN (primary) | GMM | K-Means baseline")
    print("=" * 60)
    df = load_data()
    X = preprocess(df)
    cluster(X)


if __name__ == "__main__":
    main()
