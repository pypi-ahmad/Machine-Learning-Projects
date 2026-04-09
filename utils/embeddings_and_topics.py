"""Sentence-transformer embeddings, clustering, and topic modelling.

Used for the five clustering / topic-modelling projects and the
analysis-only projects that lack classification targets.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.training_common import (
    SEED, cleanup_gpu, ensure_output_dirs, save_json, seed_everything,
)

logger = get_logger(__name__)


# ======================================================================
# Embedding
# ======================================================================

def compute_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with sentence-transformers and return (N, D) array."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    del model
    cleanup_gpu()
    return embs


# ======================================================================
# Clustering
# ======================================================================

def run_clustering(
    embeddings: np.ndarray,
    *,
    n_clusters: int = 5,
    method: str = "auto",
    seed: int = SEED,
) -> dict:
    """Cluster embeddings and return labels + metrics."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        calinski_harabasz_score, davies_bouldin_score, silhouette_score,
    )

    # Reduce dims for faster clustering
    n_components = min(50, embeddings.shape[1], embeddings.shape[0])
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(embeddings)

    labels = None
    used_method = "kmeans"

    if method in ("auto", "hdbscan"):
        try:
            from hdbscan import HDBSCAN
            clusterer = HDBSCAN(min_cluster_size=max(5, len(embeddings) // 100), min_samples=3, metric="euclidean")
            labels = clusterer.fit_predict(reduced)
            n_found = len(set(labels) - {-1})
            noise_frac = (labels == -1).mean()
            if n_found < 2 or noise_frac > 0.6:
                logger.info("HDBSCAN gave %d clusters, %.0f%% noise -- falling back to KMeans", n_found, noise_frac * 100)
                labels = None
            else:
                used_method = "hdbscan"
        except ImportError:
            if method == "hdbscan":
                logger.warning("hdbscan not installed; using KMeans")

    if labels is None:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = km.fit_predict(reduced)
        used_method = "kmeans"

    # Metrics (on non-noise points)
    mask = labels >= 0
    n_valid = mask.sum()
    n_clusters_found = len(set(labels[mask]))
    metrics: dict = {"method": used_method, "n_clusters": n_clusters_found, "n_noise": int((~mask).sum())}

    if n_valid > 10 and n_clusters_found > 1:
        metrics["silhouette"] = float(silhouette_score(reduced[mask], labels[mask], sample_size=min(10_000, n_valid)))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(reduced[mask], labels[mask]))
        metrics["davies_bouldin"] = float(davies_bouldin_score(reduced[mask], labels[mask]))
    else:
        metrics["silhouette"] = 0.0

    return {"labels": labels.tolist(), "metrics": metrics, "reduced": reduced}


# ======================================================================
# Topic modelling
# ======================================================================

def run_topics(
    texts: list[str],
    embeddings: np.ndarray | None = None,
    *,
    n_topics: int = 10,
    method: str = "auto",
    seed: int = SEED,
) -> dict:
    """Extract topics via BERTopic or LDA."""
    # Try BERTopic first
    if method in ("auto", "bertopic"):
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(stop_words="english", max_features=5000)
            topic_model = BERTopic(
                nr_topics=n_topics,
                vectorizer_model=vectorizer,
                verbose=False,
            )
            topics, probs = topic_model.fit_transform(texts, embeddings)
            info = topic_model.get_topic_info()
            top_topics = []
            for _, row in info.head(min(n_topics + 1, len(info))).iterrows():
                tid = row.get("Topic", -1)
                if tid == -1:
                    continue
                words = topic_model.get_topic(tid)
                top_topics.append({
                    "topic_id": int(tid),
                    "count": int(row.get("Count", 0)),
                    "top_words": [w for w, _ in words[:10]] if words else [],
                })
            return {
                "method": "bertopic",
                "n_topics": len(top_topics),
                "topics": top_topics,
            }
        except ImportError:
            if method == "bertopic":
                logger.warning("BERTopic not installed; falling back to LDA")
        except Exception as exc:
            logger.warning("BERTopic failed (%s); falling back to LDA", exc)

    # LDA fallback
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(max_features=5000, stop_words="english")
    X = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=seed, max_iter=20)
    lda.fit(X)
    feature_names = vec.get_feature_names_out()

    top_topics = []
    for idx, topic_vec in enumerate(lda.components_):
        top_idx = topic_vec.argsort()[-10:][::-1]
        top_topics.append({
            "topic_id": idx,
            "top_words": [feature_names[i] for i in top_idx],
        })

    return {"method": "lda", "n_topics": n_topics, "topics": top_topics}


# ======================================================================
# Visualization
# ======================================================================

def _plot_clusters(reduced: np.ndarray, labels, slug: str, save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    vis = PCA(n_components=2).fit_transform(reduced) if reduced.shape[1] > 2 else reduced

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    for i, lab in enumerate(unique_labels):
        mask = np.array(labels) == lab
        ax.scatter(vis[mask, 0], vis[mask, 1], s=5, alpha=0.5,
                   color=cmap(i), label=f"C{lab}" if lab >= 0 else "noise")
    ax.set_title(f"{slug}: Cluster visualization (PCA-2D)")
    if len(unique_labels) <= 15:
        ax.legend(markerscale=3, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_topics_bar(topics: list[dict], slug: str, save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not topics:
        return
    n_show = min(8, len(topics))
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 3, 4), sharey=False)
    if n_show == 1:
        axes = [axes]
    for ax, t in zip(axes, topics[:n_show]):
        words = t.get("top_words", [])[:8]
        ax.barh(range(len(words)), range(len(words), 0, -1), tick_label=words, color="steelblue")
        ax.set_title(f"Topic {t.get('topic_id', '?')}", fontsize=9)
        ax.invert_yaxis()
    fig.suptitle(f"{slug}: Top Topics", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# MAIN pipeline
# ======================================================================

def run_embedding_pipeline(
    slug: str,
    texts: list[str],
    *,
    embedding_model: str = "all-MiniLM-L6-v2",
    n_clusters: int = 5,
    n_topics: int = 10,
    max_texts: int = 15_000,
    seed: int = SEED,
    force: bool = False,
) -> dict:
    """Full pipeline: encode -> cluster -> topic model -> visualize."""
    seed_everything(seed)
    dirs = ensure_output_dirs(slug)
    metrics_file = dirs["metrics"] / "phase2_metrics.json"
    if metrics_file.exists() and not force:
        logger.info("Phase 2 already done for %s", slug)
        return json.loads(metrics_file.read_text())

    # Clean texts
    texts = [str(t).strip() for t in texts if str(t).strip()]
    if not texts:
        return {"slug": slug, "status": "ERROR", "error": "No texts provided"}

    if len(texts) > max_texts:
        random.seed(seed)
        texts = random.sample(texts, max_texts)

    logger.info("[%s] Embedding %d texts with %s", slug, len(texts), embedding_model)

    # 1. Embed
    embeddings = compute_embeddings(texts, model_name=embedding_model)

    # 2. Cluster
    cluster_result = run_clustering(embeddings, n_clusters=n_clusters, seed=seed)
    labels = cluster_result["labels"]
    reduced = cluster_result["reduced"]

    # 3. Topics
    topic_result = run_topics(texts, embeddings, n_topics=n_topics, seed=seed)

    # 4. Visualize
    try:
        _plot_clusters(reduced, labels, slug, dirs["figures"] / "clusters_2d.png")
    except Exception as exc:
        logger.warning("Cluster plot failed: %s", exc)

    try:
        _plot_topics_bar(topic_result.get("topics", []), slug, dirs["figures"] / "topics_bar.png")
    except Exception as exc:
        logger.warning("Topic plot failed: %s", exc)

    # 5. Assemble results
    result = {
        "slug": slug,
        "task": "embedding_cluster_topic",
        "n_texts": len(texts),
        "embedding_model": embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "clustering": cluster_result["metrics"],
        "topics": {
            "method": topic_result.get("method", "unknown"),
            "n_topics": topic_result.get("n_topics", 0),
            "top_topics": topic_result.get("topics", [])[:10],
        },
        "status": "OK",
    }

    save_json(result, metrics_file)

    # Save embeddings
    np.save(dirs["artifacts"] / "embeddings.npy", embeddings)
    np.save(dirs["artifacts"] / "cluster_labels.npy", np.array(labels))

    del embeddings
    cleanup_gpu()
    return result


def run_project(project_slug, project_dir, raw_paths, processed_dir, outputs_dir, config, force=False):
    """Unified entry point called by the Phase 2.1 orchestrator."""
    texts = config["texts"]
    _KEYS = {"embedding_model","n_clusters","n_topics","max_texts","seed"}
    kw = {k: v for k, v in config.items() if k in _KEYS}
    r = run_embedding_pipeline(project_slug, texts, force=force, **kw)
    return {
        "status": r.get("status", "UNKNOWN"),
        "model_name": kw.get("embedding_model", "all-MiniLM-L6-v2"),
        "dataset_size": r.get("n_texts", 0),
        "main_metrics": r.get("clustering", {}),
        "val_metrics": {},
        "training_mode": "inference",
        "train_runtime_sec": 0,
        "notes": "embedding + clustering + topics",
        "full_result": r,
    }
