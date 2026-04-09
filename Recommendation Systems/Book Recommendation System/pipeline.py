"""
Modern Recommendation Pipeline (April 2026)
Models: implicit ALS/BPR (CF) + LightFM (hybrid) + Sentence Transformers/BGE-M3/Qwen3-Embedding (content)
        Surprise SVD/KNN as baseline
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib; matplotlib.use("Agg")

warnings.filterwarnings("ignore")

TASK = "content"  # cf | hybrid | content


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("zhengyun21/Book-Crossing", split="train").to_pandas()
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


# ═══════════════════════════════════════════════════════════════
# PRIMARY: implicit ALS + BPR (collaborative filtering)
# ═══════════════════════════════════════════════════════════════
def run_implicit_cf(df, user_col, item_col, rating_col):
    mat, ue, ie = build_interaction_matrix(df, user_col, item_col, rating_col)
    n_users, n_items = mat.shape[0], mat.shape[1]

    # implicit ALS
    try:
        from implicit.als import AlternatingLeastSquares
        als = AlternatingLeastSquares(factors=128, iterations=30, use_gpu=True)
        als.fit(mat)
        # Evaluate: precision@10 on last interaction
        from implicit.evaluation import precision_at_k, train_test_split
        train_m, test_m = train_test_split(mat, train_percentage=0.8)
        als2 = AlternatingLeastSquares(factors=128, iterations=30, use_gpu=True)
        als2.fit(train_m)
        p_at_10 = precision_at_k(als2, train_m, test_m, K=10)
        print(f"✓ implicit ALS — {n_users} users, {n_items} items, P@10={p_at_10:.4f}")
    except Exception as e:
        print(f"✗ implicit ALS: {e}")

    # implicit BPR
    try:
        from implicit.bpr import BayesianPersonalizedRanking
        bpr = BayesianPersonalizedRanking(factors=128, iterations=100, use_gpu=True)
        bpr.fit(mat)
        print(f"✓ implicit BPR trained")
    except Exception as e:
        print(f"✗ implicit BPR: {e}")


# ═══════════════════════════════════════════════════════════════
# HYBRID: LightFM (user/item metadata, cold-start capable)
# ═══════════════════════════════════════════════════════════════
def run_lightfm_hybrid(df, user_col, item_col, rating_col, content_col):
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
        if content_col and content_col in df.columns:
            unique_items = df[[item_col, content_col]].drop_duplicates(subset=[item_col])
            # Simple: use content words as features
            all_features = set()
            for text in unique_items[content_col].fillna("").astype(str):
                for w in text.lower().split()[:10]:
                    all_features.add(w)
            if all_features:
                lfds.fit_partial(item_features=all_features)
                item_feat_list = []
                for _, row in unique_items.iterrows():
                    words = row[content_col] if isinstance(row[content_col], str) else ""
                    feats = [w for w in words.lower().split()[:10] if w in all_features]
                    if feats:
                        item_feat_list.append((row[item_col], feats))
                if item_feat_list:
                    item_features = lfds.build_item_features(item_feat_list)

        # Train WARP model (for implicit/ranking) and BPR model
        for loss in ["warp", "bpr"]:
            model = LightFM(loss=loss, no_components=64, learning_rate=0.05)
            model.fit(interactions, item_features=item_features, epochs=30, num_threads=4)
            p_at_k = precision_at_k(model, interactions, item_features=item_features, k=10).mean()
            auc = auc_score(model, interactions, item_features=item_features).mean()
            print(f"✓ LightFM ({loss}) — P@10={p_at_k:.4f}, AUC={auc:.4f}")
    except Exception as e:
        print(f"✗ LightFM: {e}")


# ═══════════════════════════════════════════════════════════════
# CONTENT-BASED: Sentence Transformers / BGE-M3 / Qwen3-Embedding
# ═══════════════════════════════════════════════════════════════
def run_content_embeddings(df, item_col, content_col):
    if not content_col or content_col not in df.columns:
        print("⚠ No content column found — skipping content-based embeddings")
        return

    items = df[[item_col, content_col]].drop_duplicates(subset=[item_col]).head(1000)
    texts = items[content_col].fillna("").astype(str).tolist()

    # BGE-M3
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        model = SentenceTransformer("BAAI/bge-m3")
        embs = model.encode(texts, batch_size=32, show_progress_bar=True)
        sim = cosine_similarity(embs)
        # Show top-3 similar items for first 3 items
        for i in range(min(3, len(items))):
            top_idx = np.argsort(sim[i])[-4:-1][::-1]  # top 3 excluding self
            top_items = items.iloc[top_idx][item_col].tolist()
            print(f"  Item '{items.iloc[i][item_col]}' → similar: {top_items}")
        print(f"✓ BGE-M3 content-based: {len(items)} items embedded (dim={embs.shape[1]})")
    except Exception as e:
        print(f"✗ BGE-M3: {e}")

    # Qwen3-Embedding
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        qwen_embs = qwen.encode(texts[:200], batch_size=16, show_progress_bar=True)
        qwen_sim = cosine_similarity(qwen_embs)
        print(f"✓ Qwen3-Embedding: {len(qwen_embs)} items embedded (dim={qwen_embs.shape[1]})")
    except Exception as e:
        print(f"✗ Qwen3-Embedding: {e}")


# ═══════════════════════════════════════════════════════════════
# BASELINE: Surprise SVD + KNN
# ═══════════════════════════════════════════════════════════════
def run_surprise_baseline(df, user_col, item_col, rating_col):
    if not rating_col:
        print("⚠ No explicit ratings — skipping Surprise baseline")
        return
    try:
        from surprise import Dataset as SDataset, Reader, SVD, KNNBasic, accuracy
        from surprise.model_selection import cross_validate

        reader = Reader(rating_scale=(df[rating_col].min(), df[rating_col].max()))
        data = SDataset.load_from_df(df[[user_col, item_col, rating_col]].dropna(), reader)

        for algo_cls, name in [(SVD, "SVD"), (KNNBasic, "KNN")]:
            algo = algo_cls()
            results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=3, verbose=False)
            rmse = results["test_rmse"].mean()
            mae = results["test_mae"].mean()
            print(f"  Surprise {name} — RMSE={rmse:.4f}, MAE={mae:.4f}")
        print("✓ Surprise baseline complete")
    except Exception as e:
        print(f"✗ Surprise baseline: {e}")


def train(df):
    user_col, item_col, rating_col, content_col = detect_columns(df)
    print(f"Columns — user: {user_col}, item: {item_col}, rating: {rating_col}, content: {content_col}")

    if TASK == "cf":
        # Primary: implicit ALS/BPR → Baseline: Surprise SVD/KNN
        run_implicit_cf(df, user_col, item_col, rating_col)
        run_surprise_baseline(df, user_col, item_col, rating_col)
    elif TASK == "hybrid":
        # Primary: LightFM → also run implicit CF
        run_lightfm_hybrid(df, user_col, item_col, rating_col, content_col)
        run_implicit_cf(df, user_col, item_col, rating_col)
    elif TASK == "content":
        # Primary: embedding-based content similarity → also run implicit if possible
        run_content_embeddings(df, item_col, content_col)
        try:
            run_implicit_cf(df, user_col, item_col, rating_col)
        except Exception:
            pass
    else:
        run_implicit_cf(df, user_col, item_col, rating_col)
        run_content_embeddings(df, item_col, content_col)


def main():
    print("=" * 60)
    print(f"RECOMMENDATION ({TASK}) — implicit + LightFM + SentenceTransformers | Surprise baseline")
    print("=" * 60)
    df = load_data()
    train(df)


if __name__ == "__main__":
    main()
