"""E-commerce Product Recommender — Streamlit ML demo.

Recommend products to users based on collaborative filtering
(user-item cosine similarity) and content-based TF-IDF similarity.

Usage:
    streamlit run main.py
"""

import math
import random
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Product Recommender", layout="wide")
st.title("🛒 E-commerce Product Recommender")


# ── Math helpers ──────────────────────────────────────────────────────────────

def cosine_sim(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    dot  = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na   = math.sqrt(sum(v ** 2 for v in a.values()))
    nb   = math.sqrt(sum(v ** 2 for v in b.values()))
    return dot / (na * nb + 1e-9)


def tfidf(docs: list[list[str]]) -> list[dict]:
    N = len(docs)
    df_count = Counter(tok for doc in docs for tok in set(doc))
    result = []
    for doc in docs:
        tf  = Counter(doc)
        total = len(doc) or 1
        vec = {tok: (tf[tok] / total) * math.log(N / (df_count[tok] + 1) + 1)
               for tok in tf}
        result.append(vec)
    return result


# ── Product catalog ───────────────────────────────────────────────────────────

PRODUCTS = [
    {"id": 1,  "name": "Wireless Headphones",    "category": "Electronics",   "price": 79,  "tags": ["audio", "wireless", "headphones", "music", "bluetooth"]},
    {"id": 2,  "name": "Bluetooth Speaker",       "category": "Electronics",   "price": 49,  "tags": ["audio", "wireless", "bluetooth", "portable", "music"]},
    {"id": 3,  "name": "USB-C Hub",               "category": "Electronics",   "price": 35,  "tags": ["usb", "laptop", "accessories", "hub", "productivity"]},
    {"id": 4,  "name": "Mechanical Keyboard",     "category": "Electronics",   "price": 120, "tags": ["keyboard", "gaming", "typing", "mechanical", "rgb"]},
    {"id": 5,  "name": "Gaming Mouse",            "category": "Electronics",   "price": 65,  "tags": ["mouse", "gaming", "dpi", "rgb", "precision"]},
    {"id": 6,  "name": "Running Shoes",           "category": "Sports",        "price": 95,  "tags": ["shoes", "running", "fitness", "sports", "outdoor"]},
    {"id": 7,  "name": "Yoga Mat",                "category": "Sports",        "price": 30,  "tags": ["yoga", "fitness", "mat", "exercise", "flexibility"]},
    {"id": 8,  "name": "Resistance Bands",        "category": "Sports",        "price": 20,  "tags": ["fitness", "exercise", "bands", "strength", "portable"]},
    {"id": 9,  "name": "Water Bottle (Insulated)","category": "Sports",        "price": 25,  "tags": ["hydration", "sports", "bottle", "insulated", "outdoor"]},
    {"id": 10, "name": "Protein Powder",          "category": "Health",        "price": 45,  "tags": ["protein", "nutrition", "fitness", "health", "supplement"]},
    {"id": 11, "name": "Vitamin C Supplements",   "category": "Health",        "price": 18,  "tags": ["vitamins", "health", "supplement", "immune", "wellness"]},
    {"id": 12, "name": "Standing Desk",           "category": "Furniture",     "price": 350, "tags": ["desk", "ergonomic", "productivity", "office", "standing"]},
    {"id": 13, "name": "Ergonomic Chair",         "category": "Furniture",     "price": 280, "tags": ["chair", "ergonomic", "office", "posture", "comfort"]},
    {"id": 14, "name": "Desk Lamp LED",           "category": "Furniture",     "price": 40,  "tags": ["lamp", "desk", "led", "office", "lighting"]},
    {"id": 15, "name": "Python Programming Book", "category": "Books",         "price": 40,  "tags": ["python", "programming", "coding", "learning", "tech"]},
    {"id": 16, "name": "Data Science Handbook",   "category": "Books",         "price": 55,  "tags": ["data", "science", "ml", "programming", "learning"]},
    {"id": 17, "name": "Novel: The Alchemist",    "category": "Books",         "price": 14,  "tags": ["fiction", "novel", "reading", "story", "bestseller"]},
    {"id": 18, "name": "Backpack 30L",            "category": "Travel",        "price": 70,  "tags": ["backpack", "travel", "outdoor", "hiking", "bag"]},
    {"id": 19, "name": "Travel Neck Pillow",      "category": "Travel",        "price": 22,  "tags": ["travel", "pillow", "comfort", "flight", "portable"]},
    {"id": 20, "name": "Noise-Cancelling Earbuds","category": "Electronics",   "price": 130, "tags": ["audio", "earbuds", "wireless", "noise-cancel", "music"]},
]

PROD_BY_ID = {p["id"]: p for p in PRODUCTS}

# ── Synthetic user-item ratings ───────────────────────────────────────────────

def make_ratings(seed: int = 42) -> dict[str, dict[int, float]]:
    rng    = random.Random(seed)
    users  = [f"user_{i:03d}" for i in range(1, 61)]
    categories = list({p["category"] for p in PRODUCTS})
    ratings: dict[str, dict[int, float]] = {}
    for u in users:
        # Each user has 1–2 preferred categories
        pref = rng.sample(categories, k=rng.randint(1, 2))
        r    = {}
        for p in PRODUCTS:
            if rng.random() < 0.3:          # ~30% items rated per user
                base = 4.0 if p["category"] in pref else 2.5
                r[p["id"]] = max(1, min(5, round(base + rng.gauss(0, 0.8), 1)))
        if r:
            ratings[u] = r
    return ratings


@st.cache_resource
def build_models():
    ratings   = make_ratings()
    prod_docs = [p["tags"] for p in PRODUCTS]
    tfidf_vecs = tfidf(prod_docs)
    content_matrix = {
        PRODUCTS[i]["id"]: tfidf_vecs[i] for i in range(len(PRODUCTS))
    }
    return ratings, content_matrix


ratings, content_matrix = build_models()
USERS = sorted(ratings.keys())


def collab_recommend(user_id: str, top_n: int = 5) -> list[tuple[int, float]]:
    """User-based collaborative filtering via cosine similarity."""
    u_ratings = ratings.get(user_id, {})
    if not u_ratings:
        return []
    sims = {}
    for other, o_ratings in ratings.items():
        if other == user_id:
            continue
        sims[other] = cosine_sim(u_ratings, o_ratings)

    # Weighted average of other users' ratings for unseen items
    scores: dict[int, float] = {}
    weight_sum: dict[int, float] = {}
    for other, sim in sorted(sims.items(), key=lambda x: -x[1])[:20]:
        if sim <= 0:
            continue
        for pid, r in ratings[other].items():
            if pid not in u_ratings:
                scores[pid]     = scores.get(pid, 0) + sim * r
                weight_sum[pid] = weight_sum.get(pid, 0) + sim

    results = [(pid, scores[pid] / weight_sum[pid])
               for pid in scores if weight_sum[pid] > 0]
    results.sort(key=lambda x: -x[1])
    return results[:top_n]


def content_recommend(prod_id: int, top_n: int = 5) -> list[tuple[int, float]]:
    """Content-based recommendation via TF-IDF tag cosine similarity."""
    base = content_matrix[prod_id]
    sims = [(p["id"], cosine_sim(base, content_matrix[p["id"]]))
            for p in PRODUCTS if p["id"] != prod_id]
    sims.sort(key=lambda x: -x[1])
    return sims[:top_n]


# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["For You", "Similar Products", "Catalog & Ratings"])

with tab1:
    st.subheader("Personalized Recommendations")
    user = st.selectbox("Select User", USERS)
    top_n = st.slider("Number of recommendations", 3, 10, 5)

    user_rated = ratings.get(user, {})
    if user_rated:
        st.markdown(f"**Previously rated {len(user_rated)} products**")
        rated_df = pd.DataFrame([
            {"Product": PROD_BY_ID[pid]["name"], "Category": PROD_BY_ID[pid]["category"],
             "Your Rating": r, "Price": f"${PROD_BY_ID[pid]['price']}"}
            for pid, r in sorted(user_rated.items(), key=lambda x: -x[1])[:5]
        ])
        st.dataframe(rated_df, use_container_width=True, hide_index=True)

    recs = collab_recommend(user, top_n)
    if recs:
        st.subheader("Recommended for You")
        rec_df = pd.DataFrame([
            {"Product": PROD_BY_ID[pid]["name"],
             "Category": PROD_BY_ID[pid]["category"],
             "Predicted Rating": f"{score:.2f}⭐",
             "Price": f"${PROD_BY_ID[pid]['price']}",
             "Tags": ", ".join(PROD_BY_ID[pid]["tags"][:3])}
            for pid, score in recs
        ])
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough rating history to generate recommendations.")

with tab2:
    st.subheader("Find Similar Products")
    prod_name = st.selectbox("Choose a product", [p["name"] for p in PRODUCTS])
    prod_id   = next(p["id"] for p in PRODUCTS if p["name"] == prod_name)
    top_m     = st.slider("Number of similar products", 3, 8, 4)

    base_prod = PROD_BY_ID[prod_id]
    st.markdown(f"**{base_prod['name']}** — {base_prod['category']} — ${base_prod['price']}")
    st.caption("Tags: " + ", ".join(base_prod["tags"]))

    sims = content_recommend(prod_id, top_m)
    sim_df = pd.DataFrame([
        {"Product": PROD_BY_ID[pid]["name"],
         "Category": PROD_BY_ID[pid]["category"],
         "Similarity": f"{s:.3f}",
         "Price": f"${PROD_BY_ID[pid]['price']}",
         "Tags": ", ".join(PROD_BY_ID[pid]["tags"][:3])}
        for pid, s in sims
    ])
    st.dataframe(sim_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Product Catalog")
    cat_filter = st.multiselect("Filter by category",
                                 sorted({p["category"] for p in PRODUCTS}))
    filtered = [p for p in PRODUCTS if not cat_filter or p["category"] in cat_filter]
    cat_df = pd.DataFrame([
        {"ID": p["id"], "Name": p["name"], "Category": p["category"],
         "Price": f"${p['price']}", "Tags": ", ".join(p["tags"])}
        for p in filtered
    ])
    st.dataframe(cat_df, use_container_width=True, hide_index=True)

    st.subheader("Rating Statistics")
    all_ratings = [(pid, r) for u_r in ratings.values() for pid, r in u_r.items()]
    r_df = pd.DataFrame(all_ratings, columns=["product_id", "rating"])
    avg_r = r_df.groupby("product_id")["rating"].mean()
    top_prods = avg_r.sort_values(ascending=False).head(10)
    top_df = pd.DataFrame({
        "Product": [PROD_BY_ID[pid]["name"] for pid in top_prods.index],
        "Avg Rating": top_prods.values.round(2),
    })
    st.dataframe(top_df, use_container_width=True, hide_index=True)
    st.caption(f"Total ratings in dataset: {len(all_ratings)}")
