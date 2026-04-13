"""Car Price Predictor — Streamlit ML demo.

Predict used car prices based on vehicle attributes using
a linear regression model trained on synthetic data.

Usage:
    streamlit run main.py
"""

import math
import random

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Used Car Price Predictor")


def ols_fit(X, y):
    n, m = len(X), len(X[0])
    Xb  = [[1.0] + row for row in X]
    m1  = m + 1
    XtX = [[sum(Xb[i][a] * Xb[i][b] for i in range(n)) for b in range(m1)] for a in range(m1)]
    Xty = [sum(Xb[i][a] * y[i] for i in range(n)) for a in range(m1)]
    A   = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    for col in range(m1):
        piv = max(range(col, m1), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        if abs(A[col][col]) < 1e-12:
            continue
        for row in range(m1):
            if row == col:
                continue
            f = A[row][col] / A[col][col]
            A[row] = [A[row][j] - f * A[col][j] for j in range(m1 + 1)]
    return [A[i][m1] / A[i][i] if abs(A[i][i]) > 1e-12 else 0 for i in range(m1)]


def ols_predict(w, X):
    return [w[0] + sum(w[j + 1] * row[j] for j in range(len(row))) for row in X]


def r2(y, yp):
    my = sum(y) / len(y)
    return 1 - sum((a - b) ** 2 for a, b in zip(y, yp)) / (sum((a - my) ** 2 for a in y) or 1)


def rmse(y, yp):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y, yp)) / len(y))


# ── Synthetic dataset ─────────────────────────────────────────────────────────

MAKES = ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Chevrolet", "Hyundai", "Nissan"]
FUELS = ["Petrol", "Diesel", "Hybrid", "Electric"]
TRANS = ["Manual", "Automatic"]

def make_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng  = random.Random(seed)
    rows = []
    for _ in range(n):
        year      = rng.randint(2005, 2023)
        mileage   = rng.randint(0, 200000)
        engine_cc = rng.choice([1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000])
        horsepower= rng.randint(70, 400)
        doors     = rng.choice([2, 4, 5])
        fuel_idx  = rng.randint(0, 3)
        trans_idx = rng.randint(0, 1)
        make_idx  = rng.randint(0, 7)

        age       = 2024 - year
        price = (
            25000
            - age * 1200
            - mileage * 0.05
            + horsepower * 40
            + engine_cc * 2
            + fuel_idx * 2000        # hybrid/electric premium
            + trans_idx * 1500       # automatic premium
            + make_idx * 500
            + rng.gauss(0, 2000)
        )
        price = max(1000, round(price, 0))
        rows.append({
            "year": year, "mileage": mileage, "engine_cc": engine_cc,
            "horsepower": horsepower, "doors": doors,
            "fuel_idx": fuel_idx, "trans_idx": trans_idx, "make_idx": make_idx,
            "make": MAKES[make_idx], "fuel": FUELS[fuel_idx],
            "transmission": TRANS[trans_idx], "price": price,
        })
    return pd.DataFrame(rows)


@st.cache_resource
def train():
    df   = make_data(600)
    feat = ["year", "mileage", "engine_cc", "horsepower", "doors",
            "fuel_idx", "trans_idx", "make_idx"]
    X    = df[feat].values.tolist()
    y    = df["price"].tolist()
    w    = ols_fit(X, y)
    yp   = ols_predict(w, X)
    return w, feat, r2(y, yp), rmse(y, yp), df


model_w, feats, model_r2, model_rmse, df_all = train()

# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Predict Price", "Market Analysis", "Dataset"])

with tab1:
    st.subheader("Enter Vehicle Details")
    c1, c2 = st.columns(2)
    make     = c1.selectbox("Make", MAKES)
    year     = c2.slider("Year", 2005, 2023, 2018)
    c1, c2   = st.columns(2)
    mileage  = c1.number_input("Mileage (km)", 0, 300000, 50000, step=5000)
    engine   = c2.selectbox("Engine (cc)", [1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], index=4)
    c1, c2   = st.columns(2)
    hp       = c1.slider("Horsepower", 70, 400, 150)
    doors    = c2.selectbox("Doors", [2, 4, 5], index=1)
    c1, c2   = st.columns(2)
    fuel     = c1.selectbox("Fuel Type", FUELS)
    trans    = c2.selectbox("Transmission", TRANS)

    make_idx  = MAKES.index(make)
    fuel_idx  = FUELS.index(fuel)
    trans_idx = TRANS.index(trans)

    xi    = [year, mileage, engine, hp, doors, fuel_idx, trans_idx, make_idx]
    price = max(500, ols_predict(model_w, [xi])[0])

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Price",   f"${price:,.0f}")
    c2.metric("Model R²",          f"{model_r2:.3f}")
    c3.metric("RMSE",              f"${model_rmse:,.0f}")

    age = 2024 - year
    if age >= 10:
        st.info(f"Vehicle is {age} years old — depreciation is a major factor.")
    if mileage > 150000:
        st.warning("High mileage may reduce buyer interest and actual resale value.")
    if fuel in ("Hybrid", "Electric"):
        st.success("Hybrid/Electric vehicles typically command a premium in the current market.")

with tab2:
    st.subheader("Price by Make")
    avg_by_make = df_all.groupby("make")["price"].mean().sort_values(ascending=False)
    st.bar_chart(avg_by_make)

    st.subheader("Price vs Mileage (sample)")
    sample = df_all.sample(100, random_state=1)[["mileage", "price"]].set_index("mileage").sort_index()
    st.line_chart(sample)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Price",    f"${df_all['price'].mean():,.0f}")
    c2.metric("Min Price",    f"${df_all['price'].min():,.0f}")
    c3.metric("Max Price",    f"${df_all['price'].max():,.0f}")

with tab3:
    st.dataframe(
        df_all[["make", "year", "mileage", "engine_cc", "horsepower",
                "fuel", "transmission", "price"]].head(50),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Training set: {len(df_all)} vehicles")
