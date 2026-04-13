"""Insurance Premium Estimator — Streamlit ML demo.

Estimate health and auto insurance premiums based on risk factors.
Uses a regression model trained on synthetic data.

Usage:
    streamlit run main.py
"""

import math
import random

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Insurance Estimator", layout="wide")
st.title("🛡️ Insurance Premium Estimator")


def ols_fit(X, y):
    n, m = len(X), len(X[0])
    Xb   = [[1.0]+row for row in X]
    m1   = m+1
    XtX  = [[sum(Xb[i][a]*Xb[i][b] for i in range(n)) for b in range(m1)] for a in range(m1)]
    Xty  = [sum(Xb[i][a]*y[i] for i in range(n)) for a in range(m1)]
    A    = [row[:]+[Xty[i]] for i,row in enumerate(XtX)]
    for col in range(m1):
        piv = max(range(col,m1), key=lambda r: abs(A[r][col]))
        A[col],A[piv] = A[piv],A[col]
        if abs(A[col][col])<1e-12: continue
        for row in range(m1):
            if row==col: continue
            f = A[row][col]/A[col][col]
            A[row] = [A[row][j]-f*A[col][j] for j in range(m1+1)]
    return [A[i][m1]/A[i][i] if abs(A[i][i])>1e-12 else 0 for i in range(m1)]


def ols_predict(w, X):
    return [w[0]+sum(w[j+1]*row[j] for j in range(len(row))) for row in X]


def r2(y, yp):
    my = sum(y)/len(y)
    return 1 - sum((a-b)**2 for a,b in zip(y,yp))/(sum((a-my)**2 for a in y) or 1)


# ── Synthetic datasets ────────────────────────────────────────────────────────

def make_health_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        age     = rng.randint(18, 70)
        bmi     = round(rng.uniform(16, 45), 1)
        smoker  = rng.random() < 0.2
        children= rng.randint(0, 5)
        region  = rng.choice([0,1,2,3])   # NE,NW,SE,SW
        chronic = rng.random() < 0.15
        premium = (
            age * 200
            + bmi * 100
            + smoker * 15000
            + children * 800
            + region * 300
            + chronic * 8000
            + rng.gauss(0, 1500)
        )
        premium = max(1000, round(premium, 2))
        rows.append({"age":age,"bmi":bmi,"smoker":int(smoker),"children":children,
                     "region":region,"chronic":int(chronic),"premium":premium})
    return pd.DataFrame(rows)


def make_auto_data(n: int = 300, seed: int = 43) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        age       = rng.randint(16, 75)
        accidents = rng.randint(0, 5)
        violations= rng.randint(0, 4)
        car_age   = rng.randint(0, 20)
        car_value = rng.randint(5000, 80000)
        coverage  = rng.choice([0, 1, 2])   # basic/standard/full
        premium = (
            max(0, 40 - age) * 80
            + accidents * 600
            + violations * 400
            + car_value * 0.03
            + car_age * (-50)
            + coverage * 300
            + rng.gauss(0, 200)
        )
        premium = max(300, round(premium, 2))
        rows.append({"age":age,"accidents":accidents,"violations":violations,
                     "car_age":car_age,"car_value":car_value,"coverage":coverage,
                     "premium":premium})
    return pd.DataFrame(rows)


@st.cache_resource
def train_models():
    dh   = make_health_data(400)
    fh   = ["age","bmi","smoker","children","region","chronic"]
    Xh   = dh[fh].values.tolist()
    yh   = dh["premium"].tolist()
    wh   = ols_fit(Xh, yh)
    r2h  = r2(yh, ols_predict(wh, Xh))

    da   = make_auto_data(400)
    fa   = ["age","accidents","violations","car_age","car_value","coverage"]
    Xa   = da[fa].values.tolist()
    ya   = da["premium"].tolist()
    wa   = ols_fit(Xa, ya)
    r2a  = r2(ya, ols_predict(wa, Xa))
    return wh, fh, r2h, wa, fa, r2a, dh, da


wh, fh, r2h, wa, fa, r2a, df_health, df_auto = train_models()

# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Health Insurance", "Auto Insurance", "Market Data"])

with tab1:
    st.subheader("Health Insurance Estimate")
    c1, c2 = st.columns(2)
    age      = c1.slider("Age", 18, 70, 35)
    bmi      = c2.slider("BMI", 16.0, 45.0, 24.0, step=0.5)
    c1, c2   = st.columns(2)
    smoker   = c1.selectbox("Smoker", [0,1], format_func=lambda x: "Yes" if x else "No")
    children = c2.slider("Dependents", 0, 5, 1)
    c1, c2   = st.columns(2)
    region   = c1.selectbox("Region", [0,1,2,3],
                              format_func=lambda x: ["Northeast","Northwest","Southeast","Southwest"][x])
    chronic  = c2.selectbox("Chronic Condition", [0,1], format_func=lambda x: "Yes" if x else "No")

    xi     = [age, bmi, smoker, children, region, chronic]
    h_prem = max(1000, ols_predict(wh, [xi])[0])
    st.metric("Estimated Annual Premium", f"${h_prem:,.0f}")
    st.metric("Monthly Premium", f"${h_prem/12:,.0f}")
    st.metric("Model R²", f"{r2h:.3f}")

    bmi_cat = "Underweight" if bmi<18.5 else "Normal" if bmi<25 else "Overweight" if bmi<30 else "Obese"
    st.caption(f"BMI Category: {bmi_cat}")
    if smoker:
        st.error("Smoking significantly increases premiums (typically +$10,000–$20,000/year).")

with tab2:
    st.subheader("Auto Insurance Estimate")
    c1, c2   = st.columns(2)
    a_age    = c1.slider("Driver Age", 16, 75, 35)
    car_val  = c2.number_input("Car Value ($)", 5000, 100000, 25000, step=1000)
    c1, c2   = st.columns(2)
    accid    = c1.slider("Accidents (last 5 yr)", 0, 5, 0)
    viol     = c2.slider("Violations (last 5 yr)", 0, 4, 0)
    c1, c2   = st.columns(2)
    car_age  = c1.slider("Vehicle Age (years)", 0, 20, 3)
    coverage = c2.selectbox("Coverage Level", [0,1,2],
                              format_func=lambda x: ["Basic","Standard","Full"][x])

    xi2      = [a_age, accid, viol, car_age, car_val, coverage]
    a_prem   = max(300, ols_predict(wa, [xi2])[0])
    st.metric("Estimated Annual Premium", f"${a_prem:,.0f}")
    st.metric("Monthly Premium", f"${a_prem/12:,.0f}")
    st.metric("Model R²", f"{r2a:.3f}")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Health Premium Distribution")
        st.bar_chart(pd.cut(df_health["premium"],bins=15).value_counts().sort_index())
    with col2:
        st.subheader("Auto Premium Distribution")
        st.bar_chart(pd.cut(df_auto["premium"],bins=15).value_counts().sort_index())
