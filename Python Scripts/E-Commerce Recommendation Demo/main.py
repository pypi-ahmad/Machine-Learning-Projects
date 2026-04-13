"""E-Commerce Recommendation Demo — Streamlit app.
Simple collaborative filtering demo using cosine similarity.
Dependencies: pip install streamlit pandas numpy scikit-learn
Usage: streamlit run main.py
"""
import numpy as np, pandas as pd, streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

def gen_data():
    np.random.seed(42)
    users = [f"User_{i}" for i in range(1, 21)]
    products = ["Laptop", "Phone", "Headphones", "Tablet", "Watch", "Camera", "Speaker", "Monitor", "Keyboard", "Mouse"]
    ratings = np.random.randint(0, 6, (20, 10)).astype(float)
    ratings[ratings == 0] = np.nan
    return pd.DataFrame(ratings, index=users, columns=products)

def recommend(df, user, n=5):
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
        if len(common) > 0:
            scores[product] = (rated_by[common] * user_sims[common]).sum() / user_sims[common].sum()
    return sorted(scores.items(), key=lambda x: -x[1])[:n]

def main():
    st.set_page_config(page_title="Recommendation Demo")
    st.title("E-Commerce Recommendation Demo")
    df = gen_data()
    st.header("User-Product Rating Matrix")
    st.dataframe(df, use_container_width=True)
    user = st.selectbox("Select user", df.index)
    recs = recommend(df, user)
    st.header(f"Recommendations for {user}")
    if recs:
        for product, score in recs:
            st.write(f"**{product}** — predicted rating: {score:.1f}")
    else:
        st.write("No recommendations available.")

if __name__ == "__main__":
    main()

