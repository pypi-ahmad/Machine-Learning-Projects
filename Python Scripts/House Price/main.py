"""House Price Estimator — Streamlit ML demo.

Estimate house prices using a linear regression model.
Feature importance, market comparison, and price distribution.

Usage:
    streamlit run main.py
"""

import math
import random
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Estimator", layout="wide")
st.title("🏠 House Price Estimator")


# ── Simple multiple linear regression (OLS) ──────────────────────────────────

def ols_fit(X: list[list[float]], y: list[float]) -> list[float]:
    """Normal equations solution: w = (X'X)^{-1} X'y"""
    n, m = len(X), len(X[0])
    # Add bias column
    Xb = [[1.0] + row for row in X]
    m1 = m + 1
    # X'X
    XtX = [[sum(Xb[i][a]*Xb[i][b] for i in range(n)) for b in range(m1)] for a in range(m1)]
    Xty = [sum(Xb[i][a]*y[i] for i in range(n)) for a in range(m1)]
    # Gaussian elimination
    A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    for col in range(m1):
        pivot = max(range(col, m1), key=lambda r: abs(A[r][col]))
        A[col], A[pivot] = A[pivot], A[col]
        if abs(A[col][col]) < 1e-12: continue
        for row in range(m1):
            if row == col: continue
            factor = A[row][col] / A[col][col]
            A[row] = [A[row][j] - factor*A[col][j] for j in range(m1+1)]
    w = [A[i][m1] / A[i][i] if abs(A[i][i]) > 1e-12 else 0 for i in range(m1)]
    return w


def ols_predict(w: list[float], X: list[list[float]]) -> list[float]:
    return [w[0] + sum(w[j+1]*row[j] for j in range(len(row))) for row in X]


def r2(y: list[float], yp: list[float]) -> float:
    mean_y  = sum(y) / len(y)
    ss_res  = sum((yi-ypi)**2 for yi, ypi in zip(y, yp))
    ss_tot  = sum((yi-mean_y)**2 for yi in y)
    return 1 - ss_res / ss_tot if ss_tot else 1.0


def rmse(y: list[float], yp: list[float]) -> float:
    return math.sqrt(sum((a-b)**2 for a,b in zip(y, yp)) / len(y))


# ── Synthetic housing dataset ─────────────────────────────────────────────────

def make_housing_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        sqft      = rng.randint(600, 4500)
        bedrooms  = rng.randint(1, 6)
        bathrooms = rng.randint(1, 4)
        garage    = rng.randint(0, 3)
        year      = rng.randint(1950, 2023)
        lot_size  = rng.randint(2000, 20000)
        condition = rng.randint(1, 5)     # 1=poor, 5=excellent
        school    = rng.uniform(3.0, 10.0)

        price = (
            sqft * 120
            + bedrooms * 8000
            + bathrooms * 12000
            + garage * 15000
            + (year - 1950) * 800
            + lot_size * 2
            + condition * 20000
            + school * 10000
            + rng.gauss(0, 30000)
        )
        price = max(50000, round(price, -2))
        rows.append({
            "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "garage": garage, "year_built": year, "lot_size": lot_size,
            "condition": condition, "school_rating": round(school, 1),
            "price": price,
        })
    return pd.DataFrame(rows)


@st.cache_resource
def train():
    df      = make_housing_data(400)
    feat    = ["sqft","bedrooms","bathrooms","garage","year_built",
               "lot_size","condition","school_rating"]
    X       = df[feat].values.tolist()
    y       = df["price"].tolist()
    w       = ols_fit(X, y)
    y_pred  = ols_predict(w, X)
    return w, feat, r2(y, y_pred), rmse(y, y_pred), df


model_w, feats, model_r2, model_rmse, df_all = train()

# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Estimate Price", "Model Info", "Market Data"])

with tab1:
    st.subheader("Property Details")
    c1, c2 = st.columns(2)
    sqft         = c1.number_input("Living Area (sqft)", 400, 8000, 1800, step=50)
    lot_size     = c2.number_input("Lot Size (sqft)", 1000, 50000, 6000, step=500)
    c1, c2, c3   = st.columns(3)
    bedrooms     = c1.slider("Bedrooms", 1, 8, 3)
    bathrooms    = c2.slider("Bathrooms", 1, 5, 2)
    garage       = c3.slider("Garage Spaces", 0, 3, 1)
    c1, c2       = st.columns(2)
    year_built   = c1.slider("Year Built", 1900, 2023, 1995)
    condition    = c2.slider("Condition (1–5)", 1, 5, 3)
    school_rating = st.slider("School Rating (1–10)", 1.0, 10.0, 7.0, step=0.5)

    xi     = [sqft, bedrooms, bathrooms, garage, year_built,
              lot_size, condition, school_rating]
    est    = ols_predict(model_w, [xi])[0]
    est    = max(50000, est)

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated Price", f"${est:,.0f}")
    col2.metric("Price per sqft", f"${est/sqft:,.0f}")
    col3.metric("Model R²", f"{model_r2:.3f}")

    # Confidence range ±15%
    low, high = est * 0.85, est * 1.15
    st.info(f"Estimated range: ${low:,.0f} — ${high:,.0f}")

with tab2:
    st.subheader("Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{model_r2:.4f}")
    c2.metric("RMSE", f"${model_rmse:,.0f}")
    st.caption("Trained on 400 synthetic property records via OLS regression.")

    st.subheader("Feature Coefficients")
    coef_df = pd.DataFrame({
        "Feature":     feats,
        "Coefficient": [round(model_w[i+1], 2) for i in range(len(feats))],
    }).sort_values("Coefficient", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

with tab3:
    st.dataframe(df_all.head(50), use_container_width=True, hide_index=True)
    st.subheader("Price Distribution")
    price_bins = pd.cut(df_all["price"], bins=20).value_counts().sort_index()
    st.bar_chart(price_bins)
    st.metric("Median Price", f"${df_all['price'].median():,.0f}")
    st.metric("Average Price", f"${df_all['price'].mean():,.0f}")
