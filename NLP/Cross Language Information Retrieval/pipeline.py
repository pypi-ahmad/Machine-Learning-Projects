"""
Modern NLP Similarity / Retrieval Pipeline (April 2026)
Models: BGE-M3 + Qwen3-Embedding + Sentence Transformers
        TF-IDF cosine similarity as baseline
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("wmt16", "de-en", split="train[:1000]").to_pandas()
    print(f"Dataset shape: {df.shape}")
    return df


def get_texts(df, n=500):
    """Extract text column, return up to n samples."""
    for c in df.columns:
        if df[c].dtype == "object" and df[c].str.len().mean() > 20:
            return df[c].dropna().head(n).tolist()
    text_cols = df.select_dtypes("object").columns
    if len(text_cols) > 0:
        return df[text_cols[0]].dropna().head(n).tolist()
    return df.iloc[:, 0].astype(str).head(n).tolist()


# ═══════════════════════════════════════════════════════════════
# BASELINE: TF-IDF Cosine Similarity
# ═══════════════════════════════════════════════════════════════
def run_tfidf_baseline(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    vecs = tfidf.fit_transform(texts)
    sim = cosine_similarity(vecs)
    avg_sim = (sim.sum() - len(texts)) / (len(texts) * (len(texts) - 1))
    print(f"  [Baseline] TF-IDF cosine: avg pairwise similarity = {avg_sim:.4f}")
    # Show top-3 pairs
    np.fill_diagonal(sim, 0)
    for i in range(min(3, len(texts))):
        top = np.argsort(sim[i])[-3:][::-1]
        scores = [str(j) + f"({sim[i,j]:.3f})" for j in top]
        joined = ", ".join(scores)
        print(f"    Text {i} most similar to: {joined}")
    return sim


# ═══════════════════════════════════════════════════════════════
# PRIMARY: BGE-M3 Embedding Similarity
# ═══════════════════════════════════════════════════════════════
def run_bge_m3(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        sim = cosine_similarity(embs)
        avg_sim = (sim.sum() - len(texts)) / (len(texts) * (len(texts) - 1))
        print("")
        print(f"✓ BGE-M3: {len(texts)} texts embedded (dim={embs.shape[1]})")
        print(f"  Avg pairwise semantic similarity = {avg_sim:.4f}")
        np.fill_diagonal(sim, 0)
        for i in range(min(3, len(texts))):
            top = np.argsort(sim[i])[-3:][::-1]
            scores = [str(j) + f"({sim[i,j]:.3f})" for j in top]
            joined = ", ".join(scores)
            print(f"    Text {i} most similar to: {joined}")
        return embs, sim
    except Exception as e:
        print(f"✗ BGE-M3: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════
# PRIMARY: Qwen3-Embedding
# ═══════════════════════════════════════════════════════════════
def run_qwen3_embedding(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embs = model.encode(texts[:200], batch_size=16, show_progress_bar=True, normalize_embeddings=True)
        sim = cosine_similarity(embs)
        avg_sim = (sim.sum() - len(embs)) / (len(embs) * (len(embs) - 1))
        print("")
        print(f"✓ Qwen3-Embedding: {len(embs)} texts embedded (dim={embs.shape[1]})")
        print(f"  Avg pairwise semantic similarity = {avg_sim:.4f}")
        return embs, sim
    except Exception as e:
        print(f"✗ Qwen3-Embedding: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════
# CLUSTERING: Embedding-based topic discovery
# ═══════════════════════════════════════════════════════════════
def run_embedding_clustering(embs, texts):
    if embs is None:
        print("⚠ Skipping embedding clustering (no embeddings)")
        return
    try:
        import umap, hdbscan
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        X_2d = reducer.fit_transform(embs)
        labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print("")
        print(f"✓ UMAP + HDBSCAN on embeddings: {n_clusters} topics/clusters")

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=15, alpha=0.6)
        ax.set_title("Embedding Space — UMAP + HDBSCAN"); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "embedding_clusters.png"), dpi=100)
        print("✓ Saved embedding_clusters.png")
    except Exception as e:
        print(f"✗ Embedding clustering: {e}")


def main():
    print("=" * 60)
    print("NLP SIMILARITY / RETRIEVAL — BGE-M3 + Qwen3-Embedding")
    print("TF-IDF baseline | Embedding clustering")
    print("=" * 60)
    df = load_data()
    texts = get_texts(df)
    print(f"Using {len(texts)} text samples")

    print(""); print("— TF-IDF Baseline —")
    run_tfidf_baseline(texts)

    print(""); print("— BGE-M3 Embeddings —")
    embs, sim = run_bge_m3(texts)

    print(""); print("— Qwen3-Embedding —")
    run_qwen3_embedding(texts)

    print(""); print("— Embedding Clustering —")
    run_embedding_clustering(embs, texts)


if __name__ == "__main__":
    main()
