"""
Modern Recommendation Pipeline (April 2026)

CF task :  implicit ALS + BPR (primary), Surprise SVD/KNN (baseline).
Hybrid  :  LightFM WARP/BPR (metadata-aware, cold-start capable).
Content :  Sentence Transformers / BGE-M3 / Qwen3-Embedding.
Timing  :  Wall-clock per model stage (CF and baseline).
Export  :  metrics.json with per-model evaluation + timing.
Data    :  Auto-downloaded at runtime.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib; matplotlib.use("Agg")

warnings.filterwarnings("ignore")

TASK = "hybrid"  # cf | hybrid | content
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("Yelp/yelp_review_full", split="train").to_pandas()
    print(f"Dataset shape: {df.shape}")
    return df


def detect_columns(df):
    cols = {c.lower(): c for c in df.columns}
    user = next((cols[k] for k in cols if "user" in k), df.columns[0])
    item = next((cols[k] for k in cols if any(x in k for x in ["item","movie","book","product","title","name"])), df.columns[1])
    rating = next((cols[k] for k in cols if any(x in k for x in ["rating","score","stars"])), None)
    content = next((cols[k] for k in cols if any(x in k for x in ["title","name","description","text","headline","category"])), None)
    return user, item, rating, content


def build_interaction_matrix(df, user_col, item_col, rating_col):
    ue, ie = LabelEncoder(), LabelEncoder()
    df["u"] = ue.fit_transform(df[user_col])
    df["i"] = ie.fit_transform(df[item_col])
    vals = df[rating_col].values.astype(np.float32) if rating_col else np.ones(len(df), dtype=np.float32)
    mat = csr_matrix((vals, (df["u"].values, df["i"].values)),
                     shape=(df["u"].nunique(), df["i"].nunique()))
    return mat, ue, ie


# -- PRIMARY: implicit ALS + BPR (collaborative filtering) --
def run_implicit_cf(df, user_col, item_col, rating_col):
    mat, ue, ie = build_interaction_matrix(df, user_col, item_col, rating_col)
    n_users, n_items = mat.shape[0], mat.shape[1]
    results = {}

    # implicit ALS
    try:
        from implicit.als import AlternatingLeastSquares
        from implicit.evaluation import precision_at_k, train_test_split
        t0 = time.perf_counter()
        train_m, test_m = train_test_split(mat, train_percentage=0.8)
        als = AlternatingLeastSquares(factors=128, iterations=30, use_gpu=True)
        als.fit(train_m)
        p_at_10 = precision_at_k(als, train_m, test_m, K=10)
        als_elapsed = round(time.perf_counter() - t0, 1)
        print(f"  implicit ALS -- {n_users} users, {n_items} items, P@10={p_at_10:.4f} ({als_elapsed}s)")
        results["implicit_ALS"] = {"users": n_users, "items": n_items,
                                    "P@10": round(float(p_at_10), 4), "time_s": als_elapsed}
    except Exception as e:
        print(f"  implicit ALS failed: {e}")

    # implicit BPR
    try:
        from implicit.bpr import BayesianPersonalizedRanking
        from implicit.evaluation import precision_at_k, train_test_split
        t1 = time.perf_counter()
        train_m, test_m = train_test_split(mat, train_percentage=0.8)
        bpr = BayesianPersonalizedRanking(factors=128, iterations=100, use_gpu=True)
        bpr.fit(train_m)
        bpr_p10 = precision_at_k(bpr, train_m, test_m, K=10)
        bpr_elapsed = round(time.perf_counter() - t1, 1)
        print(f"  implicit BPR -- P@10={bpr_p10:.4f} ({bpr_elapsed}s)")
        results["implicit_BPR"] = {"P@10": round(float(bpr_p10), 4), "time_s": bpr_elapsed}
    except Exception as e:
        print(f"  implicit BPR failed: {e}")

    return results


# ═══════════════════════════════════════════════════════════════
# -- HYBRID: LightFM (user/item metadata, cold-start capable) --
def run_lightfm_hybrid(df, user_col, item_col, rating_col, content_col):
    results = {}
    try:
        from lightfm import LightFM
        from lightfm.evaluation import precision_at_k, auc_score
        from lightfm.data import Dataset as LFDataset

        lfds = LFDataset()
        lfds.fit(df[user_col].unique(), df[item_col].unique())

        (interactions, weights) = lfds.build_interactions(
            ((r[user_col], r[item_col], r[rating_col] if rating_col else 1.0)
             for _, r in df.iterrows()))

        # Build item features from content column if available
        item_features = None
        n_item_features = 0
        if content_col and content_col in df.columns:
            unique_items = df[[item_col, content_col]].drop_duplicates(subset=[item_col])
            all_features = set()
            for text in unique_items[content_col].fillna("").astype(str):
                for w in text.lower().split()[:10]:
                    all_features.add(w)
            if all_features:
                n_item_features = len(all_features)
                lfds.fit_partial(item_features=all_features)
                item_feat_list = []
                for _, row in unique_items.iterrows():
                    words = row[content_col] if isinstance(row[content_col], str) else ""
                    feats = [w for w in words.lower().split()[:10] if w in all_features]
                    if feats:
                        item_feat_list.append((row[item_col], feats))
                if item_feat_list:
                    item_features = lfds.build_item_features(item_feat_list)
        print(f"  Item features: {n_item_features} unique tokens" if n_item_features else "  No item features (pure interactions)")

        # Train WARP model (for implicit/ranking) and BPR model
        for loss in ["warp", "bpr"]:
            t0 = time.perf_counter()
            model = LightFM(loss=loss, no_components=64, learning_rate=0.05)
            model.fit(interactions, item_features=item_features, epochs=30, num_threads=4)
            p_at_k = precision_at_k(model, interactions, item_features=item_features, k=10).mean()
            auc = auc_score(model, interactions, item_features=item_features).mean()
            elapsed = round(time.perf_counter() - t0, 1)
            print(f"  LightFM ({loss}) -- P@10={p_at_k:.4f}, AUC={auc:.4f} ({elapsed}s)")
            results[f"LightFM_{loss}"] = {"P@10": round(float(p_at_k), 4),
                                            "AUC": round(float(auc), 4),
                                            "item_features": n_item_features,
                                            "time_s": elapsed}
    except Exception as e:
        print(f"  LightFM failed: {e}")
    return results


# -- CONTENT-BASED: BGE-M3 / Qwen3-Embedding + TF-IDF baseline --
def run_content_embeddings(df, item_col, content_col):
    if not content_col or content_col not in df.columns:
        print("  No content column found -- skipping content-based embeddings")
        return {}
    results = {}
    items = df[[item_col, content_col]].drop_duplicates(subset=[item_col]).head(1000)
    texts = items[content_col].fillna("").astype(str).tolist()
    print(f"  {len(items)} unique items with content column '{content_col}'")

    # PRIMARY: BGE-M3
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        t0 = time.perf_counter()
        model = SentenceTransformer("BAAI/bge-m3")
        embs = model.encode(texts, batch_size=32, show_progress_bar=True)
        sim = cosine_similarity(embs)
        elapsed = round(time.perf_counter() - t0, 1)
        for i in range(min(3, len(items))):
            top_idx = np.argsort(sim[i])[-4:-1][::-1]
            top_items = items.iloc[top_idx][item_col].tolist()
            print(f"  BGE-M3 '{items.iloc[i][item_col]}' -> {top_items}")
        print(f"  BGE-M3: {len(items)} items, dim={embs.shape[1]} ({elapsed}s)")
        results["BGE-M3"] = {"items": len(items), "dim": int(embs.shape[1]), "time_s": elapsed}
    except Exception as e:
        print(f"  BGE-M3 failed: {e}")

    # ALTERNATIVE: Qwen3-Embedding
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        t1 = time.perf_counter()
        qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        qwen_embs = qwen.encode(texts, batch_size=16, show_progress_bar=True)
        qwen_sim = cosine_similarity(qwen_embs)
        qwen_elapsed = round(time.perf_counter() - t1, 1)
        for i in range(min(3, len(items))):
            top_idx = np.argsort(qwen_sim[i])[-4:-1][::-1]
            top_items = items.iloc[top_idx][item_col].tolist()
            print(f"  Qwen3 '{items.iloc[i][item_col]}' -> {top_items}")
        print(f"  Qwen3-Embedding: {len(qwen_embs)} items, dim={qwen_embs.shape[1]} ({qwen_elapsed}s)")
        results["Qwen3-Embedding"] = {"items": len(qwen_embs), "dim": int(qwen_embs.shape[1]), "time_s": qwen_elapsed}
    except Exception as e:
        print(f"  Qwen3-Embedding failed: {e}")

    # BASELINE: TF-IDF cosine similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        t2 = time.perf_counter()
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf_mat = tfidf.fit_transform(texts)
        tfidf_sim = cos_sim(tfidf_mat)
        tfidf_elapsed = round(time.perf_counter() - t2, 1)
        for i in range(min(3, len(items))):
            top_idx = np.argsort(tfidf_sim[i])[-4:-1][::-1]
            top_items = items.iloc[top_idx][item_col].tolist()
            print(f"  TF-IDF '{items.iloc[i][item_col]}' -> {top_items}")
        print(f"  TF-IDF baseline: {tfidf_mat.shape[0]} items, {tfidf_mat.shape[1]} features ({tfidf_elapsed}s)")
        results["TF-IDF"] = {"items": int(tfidf_mat.shape[0]), "features": int(tfidf_mat.shape[1]), "time_s": tfidf_elapsed}
    except Exception as e:
        print(f"  TF-IDF baseline failed: {e}")

    return results


# -- BASELINE: Surprise SVD + KNN --
def run_surprise_baseline(df, user_col, item_col, rating_col):
    if not rating_col:
        print("  No explicit ratings -- skipping Surprise baseline")
        return {}
    results = {}
    try:
        from surprise import Dataset as SDataset, Reader, SVD, KNNBasic, accuracy
        from surprise.model_selection import cross_validate

        reader = Reader(rating_scale=(df[rating_col].min(), df[rating_col].max()))
        data = SDataset.load_from_df(df[[user_col, item_col, rating_col]].dropna(), reader)

        for algo_cls, name in [(SVD, "SVD"), (KNNBasic, "KNN")]:
            t0 = time.perf_counter()
            algo = algo_cls()
            cv = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=3, verbose=False)
            elapsed = round(time.perf_counter() - t0, 1)
            rmse = round(float(cv["test_rmse"].mean()), 4)
            mae = round(float(cv["test_mae"].mean()), 4)
            print(f"  Surprise {name} -- RMSE={rmse}, MAE={mae} ({elapsed}s)")
            results[f"Surprise_{name}"] = {"RMSE": rmse, "MAE": mae, "time_s": elapsed}
        print("  Surprise baseline complete")
    except Exception as e:
        print(f"  Surprise baseline failed: {e}")
    return results


def train(df):
    user_col, item_col, rating_col, content_col = detect_columns(df)
    print(f"Columns -- user: {user_col}, item: {item_col}, rating: {rating_col}, content: {content_col}")
    metrics = {"task": TASK}

    if TASK == "cf":
        # Primary: implicit ALS/BPR -> Baseline: Surprise SVD/KNN
        print()
        print("-- implicit ALS + BPR (primary) --")
        metrics.update(run_implicit_cf(df, user_col, item_col, rating_col))
        print()
        print("-- Surprise SVD + KNN (baseline) --")
        metrics.update(run_surprise_baseline(df, user_col, item_col, rating_col))
    elif TASK == "hybrid":
        # Primary: LightFM -> Baseline: implicit ALS/BPR
        print()
        print("-- LightFM hybrid (primary) --")
        metrics.update(run_lightfm_hybrid(df, user_col, item_col, rating_col, content_col))
        print()
        print("-- implicit ALS + BPR (baseline) --")
        metrics.update(run_implicit_cf(df, user_col, item_col, rating_col))
    elif TASK == "content":
        # Primary: embedding-based content similarity -> TF-IDF baseline
        print()
        print("-- Content embeddings (primary) --")
        metrics.update(run_content_embeddings(df, item_col, content_col))
        print()
        print("-- implicit ALS/BPR (optional baseline) --")
        try:
            metrics.update(run_implicit_cf(df, user_col, item_col, rating_col))
        except Exception:
            print("  Skipped (no interaction data)")
    else:
        run_implicit_cf(df, user_col, item_col, rating_col)
        run_content_embeddings(df, item_col, content_col)

    return metrics


def run_eda(df, save_dir):
    """Exploratory Data Analysis for recommendation data."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Column types:\n{df.dtypes.value_counts().to_string()}")
    # Detect user/item columns
    for col in df.columns:
        nuniq = df[col].nunique()
        if nuniq < len(df) * 0.5 and nuniq > 1:
            print(f"  {col}: {nuniq} unique values")
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    missing = df.isnull().sum()
    n_miss = missing[missing > 0]
    if len(n_miss):
        print(f"\nMissing values ({len(n_miss)} columns):")
        print(n_miss.sort_values(ascending=False).head(10).to_string())
    else:
        print("\nNo missing values")
    # Rating distribution if numeric column exists
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        df[num_cols[0]].hist(bins=30, ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"Distribution: {num_cols[0]}")
        fig.savefig(os.path.join(save_dir, "eda_rating_dist.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print("Summary statistics saved to eda_summary.csv")
    print("EDA complete.")


def main():
    print("=" * 60)
    print(f"RECOMMENDATION ({TASK}) | implicit + LightFM + SentenceTransformers")
    print("=" * 60)
    df = load_data()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    run_eda(df, save_dir)
    metrics = train(df)

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
