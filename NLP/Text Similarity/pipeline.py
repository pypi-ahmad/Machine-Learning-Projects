"""
Modern NLP Similarity / Retrieval Pipeline (April 2026)

Primary   : Qwen3-Embedding-0.6B  — state-of-the-art dense embeddings.
Secondary : BGE-M3                 — multilingual dense embeddings.
Baseline  : TF-IDF cosine          — sparse bag-of-words comparison.

Evaluation: If the dataset contains sentence pairs with gold similarity
            scores (e.g. STS-B), Spearman and Pearson correlations are
            computed. Otherwise, average pairwise cosine similarity and
            top-k retrieval examples are reported.

Exploration: UMAP + HDBSCAN embedding cluster visualisation.
Timing    : Wall-clock per model.
Export    : metrics.json with all scores and timings.
Compute   : GPU recommended for Qwen3; BGE-M3 and TF-IDF run on CPU.
Data      : Auto-downloaded at runtime.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("mteb/stsbenchmark-sts", split="train").to_pandas()
    print(f"Dataset shape: {df.shape}")
    return df


# ── helpers ──────────────────────────────────────────────────
def get_texts(df, n=500):
    """Extract the best text column, return up to *n* samples."""
    for c in df.columns:
        if df[c].dtype == "object" and df[c].str.len().mean() > 20:
            return df[c].dropna().head(n).tolist()
    text_cols = df.select_dtypes("object").columns
    if len(text_cols) > 0:
        return df[text_cols[0]].dropna().head(n).tolist()
    return df.iloc[:, 0].astype(str).head(n).tolist()


def detect_sts_pairs(df):
    """Return (sent1, sent2, gold_scores) if the dataset is a sentence-pair
    benchmark (STS-B style), else (None, None, None)."""
    # STS-B columns: sentence1, sentence2, score  (or label)
    s1 = s2 = scores = None
    for a, b in [("sentence1", "sentence2"), ("text1", "text2"),
                  ("premise", "hypothesis"), ("text_a", "text_b")]:
        if a in df.columns and b in df.columns:
            s1, s2 = df[a].astype(str).tolist(), df[b].astype(str).tolist()
            break
    if s1 is None:
        return None, None, None
    for sc in ["score", "label", "similarity", "relatedness"]:
        if sc in df.columns:
            vals = pd.to_numeric(df[sc], errors="coerce")
            if vals.notna().sum() > 10:
                scores = vals.tolist()
                break
    return s1, s2, scores


def show_top_pairs(sim, texts, n=3):
    """Print the top-k most similar pairs for the first *n* texts."""
    tmp = sim.copy(); np.fill_diagonal(tmp, 0)
    for i in range(min(n, len(texts))):
        top = np.argsort(tmp[i])[-3:][::-1]
        parts = [f"{j}({tmp[i,j]:.3f})" for j in top]
        print(f"    Text {i} most similar to: {', '.join(parts)}")


# ═══════════════════════════════════════════════════════════════
# BASELINE: TF-IDF Cosine Similarity
# ═══════════════════════════════════════════════════════════════
def run_tfidf(texts, pairs=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    t0 = time.perf_counter()
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    if pairs:
        s1, s2, gold = pairs
        all_texts = s1 + s2
        vecs = tfidf.fit_transform(all_texts)
        v1, v2 = vecs[:len(s1)], vecs[len(s1):]
        pred = np.array([cosine_similarity(v1[i], v2[i])[0, 0] for i in range(len(s1))])
        elapsed = time.perf_counter() - t0
        return {"pred_scores": pred, "time_s": round(elapsed, 1)}
    vecs = tfidf.fit_transform(texts)
    sim = cosine_similarity(vecs)
    avg = (sim.sum() - len(texts)) / max(len(texts) * (len(texts) - 1), 1)
    elapsed = time.perf_counter() - t0
    print(f"  TF-IDF avg pairwise similarity = {avg:.4f}")
    show_top_pairs(sim, texts)
    return {"avg_cosine": round(float(avg), 4), "time_s": round(elapsed, 1)}


# ═══════════════════════════════════════════════════════════════
# PRIMARY: Qwen3-Embedding
# ═══════════════════════════════════════════════════════════════
def run_qwen3(texts, pairs=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    t0 = time.perf_counter()
    if pairs:
        s1, s2, gold = pairs
        e1 = model.encode(s1, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        e2 = model.encode(s2, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        pred = np.array([float(cosine_similarity(e1[i:i+1], e2[i:i+1])[0, 0]) for i in range(len(s1))])
        elapsed = time.perf_counter() - t0
        return {"pred_scores": pred, "dim": int(e1.shape[1]), "time_s": round(elapsed, 1)}
    embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    sim = cosine_similarity(embs)
    avg = (sim.sum() - len(texts)) / max(len(texts) * (len(texts) - 1), 1)
    elapsed = time.perf_counter() - t0
    print(f"  Qwen3: {len(texts)} texts embedded (dim={embs.shape[1]})")
    print(f"  Avg pairwise semantic similarity = {avg:.4f}")
    show_top_pairs(sim, texts)
    return {"embs": embs, "avg_cosine": round(float(avg), 4),
             "dim": int(embs.shape[1]), "time_s": round(elapsed, 1)}


# ═══════════════════════════════════════════════════════════════
# SECONDARY: BGE-M3 Embedding Similarity
# ═══════════════════════════════════════════════════════════════
def run_bge_m3(texts, pairs=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3")
    t0 = time.perf_counter()
    if pairs:
        s1, s2, gold = pairs
        e1 = model.encode(s1, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        e2 = model.encode(s2, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        pred = np.array([float(cosine_similarity(e1[i:i+1], e2[i:i+1])[0, 0]) for i in range(len(s1))])
        elapsed = time.perf_counter() - t0
        return {"pred_scores": pred, "dim": int(e1.shape[1]), "time_s": round(elapsed, 1)}
    embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    sim = cosine_similarity(embs)
    avg = (sim.sum() - len(texts)) / max(len(texts) * (len(texts) - 1), 1)
    elapsed = time.perf_counter() - t0
    print(f"  BGE-M3: {len(texts)} texts embedded (dim={embs.shape[1]})")
    print(f"  Avg pairwise semantic similarity = {avg:.4f}")
    show_top_pairs(sim, texts)
    return {"embs": embs, "avg_cosine": round(float(avg), 4),
             "dim": int(embs.shape[1]), "time_s": round(elapsed, 1)}


# ═══════════════════════════════════════════════════════════════
# EVALUATION: STS correlation (when gold scores available)
# ═══════════════════════════════════════════════════════════════
def eval_sts(pred, gold, model_name):
    """Spearman + Pearson correlation between predicted and gold similarity."""
    from scipy.stats import spearmanr, pearsonr
    mask = ~np.isnan(gold)
    pred, gold = np.asarray(pred)[mask], np.asarray(gold)[mask]
    sp, _ = spearmanr(pred, gold)
    pr, _ = pearsonr(pred, gold)
    print(f"  [{model_name}] Spearman: {sp:.4f}  |  Pearson: {pr:.4f}")
    return {"spearman": round(float(sp), 4), "pearson": round(float(pr), 4)}


# ═══════════════════════════════════════════════════════════════
# VISUALISATION: UMAP + HDBSCAN embedding clusters
# ═══════════════════════════════════════════════════════════════
def plot_clusters(embs, save_name="embedding_clusters.png"):
    if embs is None:
        return
    try:
        import umap, hdbscan
    except ImportError:
        print("  (umap/hdbscan not installed — skipping cluster plot)")
        return
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d = reducer.fit_transform(embs)
    labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  UMAP + HDBSCAN: {n_clusters} clusters")
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=15, alpha=0.6)
    ax.set_title("Embedding Space (UMAP + HDBSCAN)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, save_name), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_name}")


# ═══════════════════════════════════════════════════════════════
# VISUALISATION: Similarity heatmap
# ═══════════════════════════════════════════════════════════════
def plot_similarity_heatmap(sim, title, save_name):
    n = min(30, sim.shape[0])
    fig, ax = plt.subplots(figsize=(8, 7))
    import seaborn as sns
    sns.heatmap(sim[:n, :n], ax=ax, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, save_name), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_name}")


def main():
    print("=" * 60)
    print("NLP SIMILARITY / RETRIEVAL")
    print("Qwen3-Embedding | BGE-M3 | TF-IDF baseline")
    print("=" * 60)
    df = load_data()
    metrics = {}

    # Check for sentence-pair benchmark structure (STS-B style)
    s1, s2, gold = detect_sts_pairs(df)
    is_paired = s1 is not None and gold is not None

    if is_paired:
        n = min(len(s1), len(gold))
        s1, s2, gold = s1[:n], s2[:n], gold[:n]
        gold_arr = np.array([float(g) if g is not None else float("nan") for g in gold])
        pairs = (s1, s2, gold_arr)
        print(f"Detected sentence-pair benchmark: {n} pairs")
        print()

        print("-- TF-IDF Baseline --")
        try:
            r = run_tfidf(None, pairs=pairs)
            m = eval_sts(r["pred_scores"], gold_arr, "TF-IDF")
            m["time_s"] = r["time_s"]
            metrics["TF-IDF"] = m
        except Exception as e:
            print(f"  TF-IDF failed: {e}")
        print()

        print("-- Qwen3-Embedding (primary) --")
        try:
            r = run_qwen3(None, pairs=pairs)
            m = eval_sts(r["pred_scores"], gold_arr, "Qwen3-Embedding")
            m["time_s"] = r["time_s"]; m["dim"] = r.get("dim")
            metrics["Qwen3-Embedding"] = m
        except Exception as e:
            print(f"  Qwen3-Embedding failed: {e}")
        print()

        print("-- BGE-M3 (secondary) --")
        try:
            r = run_bge_m3(None, pairs=pairs)
            m = eval_sts(r["pred_scores"], gold_arr, "BGE-M3")
            m["time_s"] = r["time_s"]; m["dim"] = r.get("dim")
            metrics["BGE-M3"] = m
        except Exception as e:
            print(f"  BGE-M3 failed: {e}")
    else:
        texts = get_texts(df)
        print(f"Using {len(texts)} text samples (unpaired mode)")
        print()

        print("-- TF-IDF Baseline --")
        try:
            r = run_tfidf(texts)
            metrics["TF-IDF"] = r
        except Exception as e:
            print(f"  TF-IDF failed: {e}")
        print()

        print("-- Qwen3-Embedding (primary) --")
        try:
            r = run_qwen3(texts)
            embs_q = r.pop("embs", None)
            metrics["Qwen3-Embedding"] = r
        except Exception as e:
            embs_q = None
            print(f"  Qwen3-Embedding failed: {e}")
        print()

        print("-- BGE-M3 (secondary) --")
        try:
            r = run_bge_m3(texts)
            embs_b = r.pop("embs", None)
            metrics["BGE-M3"] = r
        except Exception as e:
            embs_b = None
            print(f"  BGE-M3 failed: {e}")
        print()

        # Use best available embeddings for clustering + heatmap
        best_embs = embs_q if embs_q is not None else embs_b
        print("-- Embedding Clustering --")
        plot_clusters(best_embs)
        if best_embs is not None:
            sim_mat = cosine_similarity(best_embs)
            plot_similarity_heatmap(sim_mat, "Cosine Similarity Heatmap",
                                    "similarity_heatmap.png")

    # ── Summary ──
    if metrics:
        # Pick best by spearman (paired) or avg_cosine (unpaired)
        key = "spearman" if is_paired else "avg_cosine"
        scored = {k: v.get(key, 0) for k, v in metrics.items()}
        best = max(scored, key=scored.get)
        print()
        print(f"Best model: {best} ({key}={scored[best]:.4f})")

    # Save metrics
    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
