"""Recommendation Systems template: implicit, LightFM, SentenceTransformers — April 2026"""
import textwrap


def generate(project_path, config):
    return textwrap.dedent('''\
        """
        Modern Recommendation Pipeline (April 2026)
        Models: implicit (ALS, BPR), LightFM (hybrid), Sentence Transformers (content-based)
        """
        import os, warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from scipy.sparse import csr_matrix
        from sklearn.model_selection import train_test_split
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        warnings.filterwarnings("ignore")


        def load_data():
            data_dir = Path(os.path.dirname(__file__))
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
            else:
                raise FileNotFoundError("No CSV data found.")
            print(f"Dataset shape: {df.shape}")
            return df


        def detect_columns(df):
            """Auto-detect user, item, rating, and content columns."""
            cols = {c.lower(): c for c in df.columns}
            user_col = item_col = rating_col = content_col = None

            for key, orig in cols.items():
                if "user" in key and "id" in key:
                    user_col = orig
                elif "user" in key and not user_col:
                    user_col = orig
                elif "item" in key or "movie" in key or "book" in key or "product" in key:
                    if "id" in key:
                        item_col = orig
                    elif not item_col:
                        item_col = orig
                elif "rating" in key or "score" in key:
                    rating_col = orig
                elif "title" in key or "name" in key or "description" in key:
                    content_col = orig

            if not user_col:
                user_col = df.columns[0]
            if not item_col:
                item_col = df.columns[1]
            if not rating_col and df.select_dtypes("number").columns.any():
                rating_col = df.select_dtypes("number").columns[0]

            print(f"Detected — user: {user_col}, item: {item_col}, rating: {rating_col}, content: {content_col}")
            return user_col, item_col, rating_col, content_col


        def train_implicit_models(df, user_col, item_col, rating_col):
            """Train ALS and BPR using implicit library."""
            results = {}

            from sklearn.preprocessing import LabelEncoder
            user_enc = LabelEncoder()
            item_enc = LabelEncoder()
            df["user_idx"] = user_enc.fit_transform(df[user_col])
            df["item_idx"] = item_enc.fit_transform(df[item_col])

            n_users = df["user_idx"].nunique()
            n_items = df["item_idx"].nunique()

            if rating_col:
                values = df[rating_col].values.astype(np.float32)
            else:
                values = np.ones(len(df), dtype=np.float32)

            interaction = csr_matrix(
                (values, (df["user_idx"].values, df["item_idx"].values)),
                shape=(n_users, n_items),
            )

            # ── ALS ──
            try:
                from implicit.als import AlternatingLeastSquares
                als = AlternatingLeastSquares(
                    factors=128, regularization=0.01, iterations=30,
                    use_gpu=True,
                )
                als.fit(interaction)
                results["ALS"] = {"model": als}
                print(f"✓ ALS trained ({n_users} users, {n_items} items)")
            except Exception as e:
                print(f"✗ ALS: {e}")

            # ── BPR ──
            try:
                from implicit.bpr import BayesianPersonalizedRanking
                bpr = BayesianPersonalizedRanking(
                    factors=128, learning_rate=0.05, iterations=100,
                    use_gpu=True,
                )
                bpr.fit(interaction)
                results["BPR"] = {"model": bpr}
                print(f"✓ BPR trained")
            except Exception as e:
                print(f"✗ BPR: {e}")

            return results, interaction, user_enc, item_enc


        def train_lightfm(df, user_col, item_col, rating_col):
            """Train LightFM hybrid model."""
            try:
                from lightfm import LightFM
                from lightfm.evaluation import precision_at_k, auc_score
                from lightfm.data import Dataset as LFDataset

                ds = LFDataset()
                ds.fit(df[user_col].unique(), df[item_col].unique())
                interactions, _ = ds.build_interactions(
                    zip(df[user_col], df[item_col],
                        df[rating_col] if rating_col else [1.0] * len(df))
                )

                train_int, test_int = train_test_split(
                    np.arange(interactions.nnz), test_size=0.2, random_state=42
                )

                model = LightFM(loss="warp", no_components=64)
                model.fit(interactions, epochs=30, num_threads=4)

                p_at_k = precision_at_k(model, interactions, k=10).mean()
                print(f"✓ LightFM — Precision@10: {p_at_k:.4f}")
                return {"model": model, "precision_at_10": p_at_k}
            except Exception as e:
                print(f"✗ LightFM: {e}")
                return None


        def train_content_based(df, item_col, content_col):
            """Content-based recommendations using Sentence Transformers."""
            if content_col is None:
                print("✗ Content-based: no text column found")
                return None

            try:
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity

                model = SentenceTransformer("BAAI/bge-m3")
                items = df[[item_col, content_col]].drop_duplicates(subset=[item_col]).head(1000)
                texts = items[content_col].fillna("").tolist()

                embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
                sim_matrix = cosine_similarity(embeddings)

                # Show sample recommendations
                print(f"\\n✓ Content-based embeddings computed ({len(texts)} items)")
                for i in range(min(3, len(texts))):
                    similar = np.argsort(sim_matrix[i])[::-1][1:6]
                    print(f"  Similar to '{texts[i][:50]}...':")
                    for j in similar:
                        print(f"    → '{texts[j][:50]}...' (sim={sim_matrix[i][j]:.3f})")

                return {"embeddings": embeddings, "sim_matrix": sim_matrix}
            except Exception as e:
                print(f"✗ Content-based: {e}")
                return None


        def main():
            print("=" * 60)
            print("MODERN RECOMMENDATION PIPELINE")
            print("implicit (ALS/BPR) | LightFM | Sentence Transformers")
            print("=" * 60)

            df = load_data()
            user_col, item_col, rating_col, content_col = detect_columns(df)

            # Collaborative filtering
            cf_results, interaction, user_enc, item_enc = train_implicit_models(
                df, user_col, item_col, rating_col
            )

            # Hybrid
            lfm_result = train_lightfm(df, user_col, item_col, rating_col)

            # Content-based
            cb_result = train_content_based(df, item_col, content_col)

            print("\\n" + "=" * 60)
            print("RECOMMENDATION PIPELINE COMPLETE")
            print(f"  CF models: {list(cf_results.keys())}")
            print(f"  LightFM: {'✓' if lfm_result else '✗'}")
            print(f"  Content-based: {'✓' if cb_result else '✗'}")
            print("=" * 60)


        if __name__ == "__main__":
            main()
    ''')
