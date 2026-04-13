"""Customer Churn Predictor — Streamlit ML demo.

Predict whether a customer will churn using a trained classifier.
Interactive prediction form + model performance metrics.

Usage:
    streamlit run main.py
"""

import random
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📉 Customer Churn Predictor")


# ── Pure-Python logistic regression ──────────────────────────────────────────

import math


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(-500, min(500, z))))


def normalize(X: list[list[float]]) -> tuple:
    """Min-max normalization. Returns (X_norm, mins, ranges)."""
    n_feat = len(X[0])
    mins   = [min(row[j] for row in X) for j in range(n_feat)]
    maxs   = [max(row[j] for row in X) for j in range(n_feat)]
    rngs   = [mx - mn or 1 for mn, mx in zip(mins, maxs)]
    X_norm = [[(row[j] - mins[j]) / rngs[j] for j in range(n_feat)] for row in X]
    return X_norm, mins, rngs


def train_lr(X: list[list[float]], y: list[int],
             lr: float = 0.1, epochs: int = 300) -> list[float]:
    """Train logistic regression with gradient descent."""
    n, m = len(X), len(X[0])
    w    = [0.0] * (m + 1)    # w[0] = bias
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z    = w[0] + sum(w[j+1] * xi[j] for j in range(m))
            pred = sigmoid(z)
            err  = pred - yi
            w[0] -= lr * err
            for j in range(m):
                w[j+1] -= lr * err * xi[j]
    return w


def predict_lr(w: list[float], X_norm: list[list[float]], threshold: float = 0.5):
    preds = []
    probs = []
    for xi in X_norm:
        z    = w[0] + sum(w[j+1] * xi[j] for j in range(len(xi)))
        p    = sigmoid(z)
        probs.append(p)
        preds.append(1 if p >= threshold else 0)
    return preds, probs


# ── Generate synthetic churn dataset ─────────────────────────────────────────

def make_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        tenure        = rng.randint(1, 72)
        monthly_chg   = round(rng.uniform(20, 120), 2)
        support_calls = rng.randint(0, 10)
        contract      = rng.choice([0, 1, 2])   # 0=M2M, 1=1yr, 2=2yr
        age           = rng.randint(18, 75)
        num_products  = rng.randint(1, 4)

        # Churn probability depends on features
        churn_score = (
            (72 - tenure) / 72 * 0.3
            + (monthly_chg - 20) / 100 * 0.25
            + support_calls / 10 * 0.25
            + (2 - contract) / 2 * 0.15
            + (5 - num_products) / 4 * 0.05
        )
        churn = 1 if rng.random() < churn_score else 0
        rows.append({
            "tenure":        tenure,
            "monthly_charge": monthly_chg,
            "support_calls": support_calls,
            "contract":      contract,
            "age":           age,
            "num_products":  num_products,
            "churn":         churn,
        })
    return pd.DataFrame(rows)


# ── Train on startup ──────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    df   = make_dataset(600)
    feat = ["tenure","monthly_charge","support_calls","contract","age","num_products"]
    X    = df[feat].values.tolist()
    y    = df["churn"].tolist()
    X_norm, mins, rngs = normalize(X)
    w    = train_lr(X_norm, y, lr=0.05, epochs=500)
    # Eval
    preds, probs = predict_lr(w, X_norm)
    acc  = sum(p == yi for p, yi in zip(preds, y)) / len(y)
    tp   = sum(p == 1 and yi == 1 for p, yi in zip(preds, y))
    fp   = sum(p == 1 and yi == 0 for p, yi in zip(preds, y))
    fn   = sum(p == 0 and yi == 1 for p, yi in zip(preds, y))
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return w, mins, rngs, feat, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


w, mins, rngs, feat_names, metrics = load_model()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Predict", "Model Performance", "Sample Data"])

with tab1:
    st.subheader("Predict Churn for a Customer")
    c1, c2 = st.columns(2)
    tenure       = c1.slider("Tenure (months)", 1, 72, 24)
    monthly_chg  = c2.slider("Monthly Charge ($)", 20.0, 120.0, 65.0, step=0.5)
    support_calls= c1.slider("Support Calls (last year)", 0, 10, 2)
    contract     = c2.selectbox("Contract Type", [0, 1, 2],
                                 format_func=lambda x: ["Month-to-Month","1-Year","2-Year"][x])
    age          = c1.slider("Customer Age", 18, 75, 35)
    num_products = c2.slider("Number of Products", 1, 4, 2)

    xi  = [tenure, monthly_chg, support_calls, contract, age, num_products]
    xi_norm = [(xi[j] - mins[j]) / rngs[j] for j in range(len(xi))]
    z   = w[0] + sum(w[j+1] * xi_norm[j] for j in range(len(xi_norm)))
    prob = sigmoid(z)
    pred = "Will Churn" if prob >= 0.5 else "Will Stay"

    st.subheader("Prediction")
    col1, col2 = st.columns(2)
    col1.metric("Prediction", pred)
    col2.metric("Churn Probability", f"{prob:.1%}")
    st.progress(prob)

    risk = "🔴 High Risk" if prob >= 0.7 else ("🟡 Medium Risk" if prob >= 0.4 else "🟢 Low Risk")
    st.info(f"Risk Level: {risk}")

with tab2:
    st.subheader("Model Performance (Logistic Regression)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall",    f"{metrics['recall']:.2%}")
    c4.metric("F1 Score",  f"{metrics['f1']:.2%}")
    st.caption("Trained on 600 synthetic customer records.")

with tab3:
    df = make_dataset(100)
    st.dataframe(df, use_container_width=True, hide_index=True)
    churn_rate = df["churn"].mean()
    st.metric("Sample Churn Rate", f"{churn_rate:.1%}")
