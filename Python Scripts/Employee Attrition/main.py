"""Employee Attrition Predictor — Streamlit ML demo.

Predict the likelihood of an employee leaving the company
using logistic regression trained on synthetic HR data.

Usage:
    streamlit run main.py
"""

import math
import random

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Attrition", layout="wide")
st.title("👔 Employee Attrition Predictor")


# ── Logistic regression (pure Python) ────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-500, min(500, z))))


def normalize(X):
    m = len(X[0])
    mins  = [min(row[j] for row in X) for j in range(m)]
    maxes = [max(row[j] for row in X) for j in range(m)]
    nX = []
    for row in X:
        nX.append([(row[j] - mins[j]) / (maxes[j] - mins[j] + 1e-9) for j in range(m)])
    return nX, mins, maxes


def train_lr(X, y, lr=0.1, epochs=500):
    n, m = len(X), len(X[0])
    w = [0.0] * m
    b = 0.0
    for _ in range(epochs):
        dw = [0.0] * m
        db = 0.0
        for i in range(n):
            z    = b + sum(w[j] * X[i][j] for j in range(m))
            pred = sigmoid(z)
            err  = pred - y[i]
            for j in range(m):
                dw[j] += err * X[i][j]
            db += err
        w = [w[j] - lr * dw[j] / n for j in range(m)]
        b = b - lr * db / n
    return w, b


def predict_lr(w, b, mins, maxes, row):
    m    = len(row)
    xn   = [(row[j] - mins[j]) / (maxes[j] - mins[j] + 1e-9) for j in range(m)]
    z    = b + sum(w[j] * xn[j] for j in range(m))
    return sigmoid(z)


def metrics(y, probs, threshold=0.5):
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for a, p in zip(y, preds) if a == 1 and p == 1)
    fp = sum(1 for a, p in zip(y, preds) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(y, preds) if a == 1 and p == 0)
    tn = sum(1 for a, p in zip(y, preds) if a == 0 and p == 0)
    acc  = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, f1


# ── Synthetic HR dataset ──────────────────────────────────────────────────────

DEPTS      = ["Engineering", "Sales", "HR", "Finance", "Operations", "Marketing"]
EDU_LEVELS = ["High School", "Bachelor's", "Master's", "PhD"]

def make_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng  = random.Random(seed)
    rows = []
    for _ in range(n):
        age            = rng.randint(22, 60)
        tenure_years   = rng.randint(0, 20)
        salary_k       = rng.randint(30, 150)
        satisfaction   = rng.randint(1, 5)
        last_promo_yrs = rng.randint(0, 10)
        overtime       = rng.random() < 0.35
        dept_idx       = rng.randint(0, 5)
        edu_idx        = rng.randint(0, 3)
        monthly_hours  = rng.randint(140, 310)
        num_projects   = rng.randint(1, 7)

        # Attrition signal
        score = (
            -satisfaction * 0.6
            + (last_promo_yrs > 3) * 0.5
            + (overtime) * 0.7
            + (monthly_hours > 220) * 0.4
            + (salary_k < 50) * 0.5
            + (tenure_years < 2) * 0.3
            + rng.gauss(0, 0.5)
        )
        attrition = 1 if score > 0.3 else 0
        rows.append({
            "age": age, "tenure_years": tenure_years, "salary_k": salary_k,
            "satisfaction": satisfaction, "last_promo_yrs": last_promo_yrs,
            "overtime": int(overtime), "dept_idx": dept_idx, "edu_idx": edu_idx,
            "monthly_hours": monthly_hours, "num_projects": num_projects,
            "dept": DEPTS[dept_idx], "education": EDU_LEVELS[edu_idx],
            "attrition": attrition,
        })
    return pd.DataFrame(rows)


@st.cache_resource
def train():
    df   = make_data(600)
    feat = ["age", "tenure_years", "salary_k", "satisfaction", "last_promo_yrs",
            "overtime", "dept_idx", "edu_idx", "monthly_hours", "num_projects"]
    X    = df[feat].values.tolist()
    y    = df["attrition"].tolist()
    Xn, mins, maxes = normalize(X)
    w, b = train_lr(Xn, y)
    probs = [sigmoid(b + sum(w[j] * Xn[i][j] for j in range(len(w)))) for i in range(len(Xn))]
    acc, prec, rec, f1 = metrics(y, probs)
    return w, b, mins, maxes, feat, acc, prec, rec, f1, df


w, b, mins, maxes, feats, acc, prec, rec, f1, df_all = train()

# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Predict Attrition", "Workforce Analysis", "Dataset"])

with tab1:
    st.subheader("Employee Profile")
    c1, c2 = st.columns(2)
    age          = c1.slider("Age", 22, 60, 35)
    tenure       = c2.slider("Tenure (years)", 0, 20, 3)
    c1, c2       = st.columns(2)
    salary       = c1.slider("Annual Salary ($K)", 30, 150, 65)
    satisfaction = c2.slider("Job Satisfaction (1–5)", 1, 5, 3)
    c1, c2       = st.columns(2)
    promo_yrs    = c1.slider("Years Since Last Promotion", 0, 10, 2)
    monthly_hrs  = c2.slider("Monthly Work Hours", 140, 310, 170)
    c1, c2       = st.columns(2)
    overtime     = c1.selectbox("Works Overtime", [0, 1], format_func=lambda x: "Yes" if x else "No")
    num_projects = c2.slider("Number of Projects", 1, 7, 3)
    c1, c2       = st.columns(2)
    dept         = c1.selectbox("Department", DEPTS)
    edu          = c2.selectbox("Education", EDU_LEVELS)

    dept_idx = DEPTS.index(dept)
    edu_idx  = EDU_LEVELS.index(edu)
    xi       = [age, tenure, salary, satisfaction, promo_yrs, overtime,
                dept_idx, edu_idx, monthly_hrs, num_projects]
    prob     = predict_lr(w, b, mins, maxes, xi)

    st.divider()
    risk = "HIGH" if prob >= 0.6 else "MEDIUM" if prob >= 0.35 else "LOW"
    color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[risk]
    c1, c2, c3 = st.columns(3)
    c1.metric("Attrition Probability", f"{prob:.1%}")
    c2.metric("Risk Level",            risk)
    c3.metric("Retention Likelihood",  f"{1 - prob:.1%}")
    st.progress(prob)

    if risk == "HIGH":
        st.error("High attrition risk. Immediate intervention recommended.")
        if satisfaction < 3:
            st.info("Low job satisfaction is a major driver — consider engagement programs.")
        if overtime:
            st.info("Overtime load may be causing burnout.")
        if promo_yrs > 4:
            st.info("Long time without promotion — consider career development opportunities.")
    elif risk == "MEDIUM":
        st.warning("Moderate risk. Monitor and address key concerns proactively.")

    st.subheader("Model Performance")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy",  f"{acc:.2%}")
    mc2.metric("Precision", f"{prec:.2%}")
    mc3.metric("Recall",    f"{rec:.2%}")
    mc4.metric("F1 Score",  f"{f1:.2%}")

with tab2:
    st.subheader("Attrition by Department")
    dept_attr = df_all.groupby("dept")["attrition"].mean().sort_values(ascending=False)
    st.bar_chart(dept_attr)

    st.subheader("Satisfaction Distribution")
    sat_counts = df_all["satisfaction"].value_counts().sort_index()
    st.bar_chart(sat_counts)

    total      = len(df_all)
    attr_count = df_all["attrition"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Employees", total)
    c2.metric("Attrition Count", attr_count)
    c3.metric("Attrition Rate",  f"{attr_count / total:.1%}")

with tab3:
    st.dataframe(
        df_all[["age", "tenure_years", "salary_k", "satisfaction", "dept",
                "education", "overtime", "monthly_hours", "attrition"]].head(50),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Training set: {len(df_all)} employees")
