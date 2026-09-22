"""Synthetic collaborative-filtering demo using cosine similarity."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def gen_data() -> pd.DataFrame:
    """Create a deterministic, synthetic user-product rating matrix."""
    rng = np.random.default_rng(42)
    users = [f"User_{i}" for i in range(1, 21)]
    products = ["Laptop", "Phone", "Headphones", "Tablet", "Watch", "Camera", "Speaker", "Monitor", "Keyboard", "Mouse"]
    ratings = rng.integers(0, 6, (20, 10)).astype(float)
    ratings[ratings == 0] = np.nan
    return pd.DataFrame(ratings, index=users, columns=products)


def recommend(df: pd.DataFrame, user: str, n: int = 5) -> list[tuple[str, float]]:
    """Recommend unrated products using positive-similarity neighbor ratings."""
    filled = df.fillna(0)
    sim = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim, index=df.index, columns=df.index)
    user_sims = sim_df[user].drop(user).sort_values(ascending=False)
    similar_users = user_sims.head(5).index
    user_ratings = df.loc[user]
    unrated = user_ratings[user_ratings.isna()].index
    scores = {}
    for product in unrated:
        rated_by = df[product].dropna()
        common = rated_by.index.intersection(similar_users)
        weights = user_sims[common]
        if len(common) > 0 and weights.sum() > 0:
            scores[product] = (rated_by[common] * weights).sum() / weights.sum()
    return sorted(scores.items(), key=lambda x: -x[1])[:n]


def main() -> None:
    """Render the synthetic recommendation demo."""
    st.set_page_config(page_title="Recommendation demo", page_icon=":material/recommend:")
    st.title("E-commerce recommendation demo")
    st.caption("Synthetic ratings with user-based cosine-similarity retrieval.")
    df = gen_data()
    st.header("User-product rating matrix")
    st.dataframe(df, hide_index=False)
    user = st.selectbox("Select user", df.index)
    recs = recommend(df, user)
    st.header(f"Recommendations for {user}")
    if recs:
        for product, score in recs:
            st.write(f"**{product}** — predicted rating: {score:.1f}")
    else:
        st.info("No recommendations are available for this user.")

if __name__ == "__main__":
    main()

