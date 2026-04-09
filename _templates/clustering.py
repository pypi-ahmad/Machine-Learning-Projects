"""Clustering pipeline template: UMAP + HDBSCAN + GMM (April 2026)"""
import textwrap


def generate(project_path, config):
    return textwrap.dedent('''\
        """
        Modern Clustering Pipeline (April 2026)
        Models: UMAP dimensionality reduction + HDBSCAN + Gaussian Mixture Model
        """
        import os, warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.preprocessing import StandardScaler, OrdinalEncoder
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

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

            # Drop ID-like columns
            for c in df.columns:
                if c.lower() in ("id", "customerid", "customer_id"):
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
            X_scaled = scaler.fit_transform(df.select_dtypes(include=["number"]))
            return X_scaled, df


        def cluster_and_evaluate(X):
            results = {}

            # ── 1. UMAP + HDBSCAN ──
            try:
                import umap
                import hdbscan

                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                X_umap = reducer.fit_transform(X)

                clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)
                labels = clusterer.fit_predict(X_umap)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                results["HDBSCAN"] = {"labels": labels, "embedding": X_umap, "n_clusters": n_clusters}
                mask = labels >= 0
                if mask.sum() > 1 and n_clusters > 1:
                    sil = silhouette_score(X_umap[mask], labels[mask])
                    print(f"✓ HDBSCAN: {n_clusters} clusters, silhouette={sil:.4f}, noise={(~mask).sum()}")
                else:
                    print(f"✓ HDBSCAN: {n_clusters} clusters, noise={(~mask).sum()}")
            except Exception as e:
                print(f"✗ HDBSCAN: {e}")

            # ── 2. Gaussian Mixture Model ──
            try:
                from sklearn.mixture import GaussianMixture

                # Use BIC to find optimal k
                bics = []
                K_range = range(2, min(11, len(X)))
                for k in K_range:
                    gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
                    gmm.fit(X)
                    bics.append(gmm.bic(X))

                best_k = list(K_range)[np.argmin(bics)]
                gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=5)
                labels = gmm.fit_predict(X)
                sil = silhouette_score(X, labels) if best_k > 1 else 0

                results["GMM"] = {"labels": labels, "n_clusters": best_k, "bics": bics, "K_range": list(K_range)}
                print(f"✓ GMM: best_k={best_k}, silhouette={sil:.4f}")
            except Exception as e:
                print(f"✗ GMM: {e}")

            # ── 3. UMAP + KMeans (baseline comparison) ──
            try:
                from sklearn.cluster import KMeans
                import umap as umap_lib

                if "HDBSCAN" not in results:
                    reducer = umap_lib.UMAP(n_components=2, random_state=42)
                    X_umap = reducer.fit_transform(X)
                else:
                    X_umap = results["HDBSCAN"]["embedding"]

                inertias = []
                K_range = range(2, min(11, len(X)))
                for k in K_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_umap)
                    inertias.append(km.inertia_)

                # Simple elbow: biggest drop
                diffs = np.diff(inertias)
                best_k = list(K_range)[np.argmin(diffs) + 1] if len(diffs) > 0 else 3
                km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                labels = km.fit_predict(X_umap)
                sil = silhouette_score(X_umap, labels)

                results["UMAP+KMeans"] = {"labels": labels, "embedding": X_umap, "n_clusters": best_k}
                print(f"✓ UMAP+KMeans: k={best_k}, silhouette={sil:.4f}")
            except Exception as e:
                print(f"✗ UMAP+KMeans: {e}")

            return results


        def report(results, X, save_dir="."):
            print("\\n" + "=" * 60)
            print("CLUSTERING COMPARISON")
            print("=" * 60)

            for name, res in results.items():
                labels = res["labels"]
                n = res["n_clusters"]
                mask = labels >= 0
                if mask.sum() > 1 and n > 1:
                    sil = silhouette_score(X[mask] if X.shape[0] == len(labels) else X[:len(labels)][mask], labels[mask])
                    ch = calinski_harabasz_score(X[mask] if X.shape[0] == len(labels) else X[:len(labels)][mask], labels[mask])
                    db = davies_bouldin_score(X[mask] if X.shape[0] == len(labels) else X[:len(labels)][mask], labels[mask])
                    print(f"\\n── {name} ──")
                    print(f"  Clusters: {n}  |  Silhouette: {sil:.4f}  |  CH: {ch:.1f}  |  DB: {db:.4f}")

                # Plot if embedding available
                embedding = res.get("embedding")
                if embedding is not None:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                                         c=labels, cmap="tab20", s=5, alpha=0.6)
                    ax.set_title(f"{name} ({n} clusters)")
                    ax.set_xlabel("UMAP 1")
                    ax.set_ylabel("UMAP 2")
                    plt.colorbar(scatter, ax=ax)
                    fig.savefig(os.path.join(save_dir, f"clusters_{name.lower().replace('+', '_').replace(' ', '_')}.png"),
                                dpi=100, bbox_inches="tight")
                    plt.close(fig)


        def main():
            print("=" * 60)
            print("MODERN CLUSTERING PIPELINE")
            print("UMAP + HDBSCAN | GMM (BIC) | UMAP + KMeans")
            print("=" * 60)
            df = load_data()
            X, df_clean = preprocess(df)
            results = cluster_and_evaluate(X)
            if results:
                report(results, X, os.path.dirname(os.path.abspath(__file__)))


        if __name__ == "__main__":
            main()
    ''')
