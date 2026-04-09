"""
Modern Recommendation Pipeline (April 2026)
Models: implicit (ALS/BPR), LightFM, Sentence Transformers
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib; matplotlib.use("Agg")

warnings.filterwarnings("ignore")


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("Yelp/yelp_review_full", split="train").to_pandas()
    print(f"Dataset shape: {df.shape}")
    return df


def detect_columns(df):
    cols = {c.lower(): c for c in df.columns}
    user = next((cols[k] for k in cols if "user" in k), df.columns[0])
    item = next((cols[k] for k in cols if any(x in k for x in ["item","movie","book","product","title"])), df.columns[1])
    rating = next((cols[k] for k in cols if any(x in k for x in ["rating","score"])), None)
    content = next((cols[k] for k in cols if any(x in k for x in ["title","name","description"])), None)
    return user, item, rating, content


def train(df):
    user_col, item_col, rating_col, content_col = detect_columns(df)
    print(f"Columns — user: {user_col}, item: {item_col}, rating: {rating_col}")

    ue, ie = LabelEncoder(), LabelEncoder()
    df["u"] = ue.fit_transform(df[user_col]); df["i"] = ie.fit_transform(df[item_col])
    vals = df[rating_col].values.astype(np.float32) if rating_col else np.ones(len(df), dtype=np.float32)
    mat = csr_matrix((vals, (df["u"].values, df["i"].values)), shape=(df["u"].nunique(), df["i"].nunique()))

    try:
        from implicit.als import AlternatingLeastSquares
        als = AlternatingLeastSquares(factors=128, iterations=30, use_gpu=True)
        als.fit(mat)
        print(f"✓ ALS trained ({df['u'].nunique()} users, {df['i'].nunique()} items)")
    except Exception as e: print(f"✗ ALS: {e}")

    try:
        from implicit.bpr import BayesianPersonalizedRanking
        bpr = BayesianPersonalizedRanking(factors=128, iterations=100, use_gpu=True)
        bpr.fit(mat)
        print("✓ BPR trained")
    except Exception as e: print(f"✗ BPR: {e}")

    if content_col:
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            items = df[[item_col, content_col]].drop_duplicates(subset=[item_col]).head(1000)
            model = SentenceTransformer("BAAI/bge-m3")
            embs = model.encode(items[content_col].fillna("").tolist(), batch_size=32)
            sim = cosine_similarity(embs)
            print(f"✓ Content-based: {len(items)} items embedded")
        except Exception as e: print(f"✗ Content-based: {e}")


def main():
    print("=" * 60)
    print("RECOMMENDATION: implicit + LightFM + SentenceTransformers")
    print("=" * 60)
    df = load_data()
    train(df)


if __name__ == "__main__":
    main()
