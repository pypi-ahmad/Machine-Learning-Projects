"""Telecom Churn Predictor — Streamlit ML demo.

Predict whether a telecom customer will churn using
logistic regression trained on synthetic subscriber data.

Usage:
    streamlit run main.py
"""

import math
import random

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Churn", layout="wide")
st.title("📡 Telecom Customer Churn Predictor")


# ── Logistic regression (pure Python) ────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-500, min(500, z))))


def normalize(X):
    m = len(X[0])
    mins  = [min(row[j] for row in X) for j in range(m)]
    maxes = [max(row[j] for row in X) for j in range(m)]
    Xn    = [[(row[j] - mins[j]) / (maxes[j] - mins[j] + 1e-9) for j in range(m)] for row in X]
    return Xn, mins, maxes


def train_lr(Xn, y, lr=0.1, epochs=600):
    n, m = len(Xn), len(Xn[0])
    w = [0.0] * m
    b = 0.0
    for _ in range(epochs):
        dw = [0.0] * m
        db = 0.0
        for i in range(n):
            z   = b + sum(w[j] * Xn[i][j] for j in range(m))
            err = sigmoid(z) - y[i]
            for j in range(m):
                dw[j] += err * Xn[i][j]
            db += err
        w = [w[j] - lr * dw[j] / n for j in range(m)]
        b = b - lr * db / n
    return w, b


def predict(w, b, mins, maxes, row):
    m  = len(row)
    xn = [(row[j] - mins[j]) / (maxes[j] - mins[j] + 1e-9) for j in range(m)]
    return sigmoid(b + sum(w[j] * xn[j] for j in range(m)))


def classification_metrics(y, probs, t=0.5):
    p  = [1 if x >= t else 0 for x in probs]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, p))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, p))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, p))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, p))
    acc  = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, f1


# ── Synthetic dataset ─────────────────────────────────────────────────────────

PLANS      = ["Basic", "Standard", "Premium"]
CONTRACTS  = ["Month-to-Month", "One Year", "Two Year"]
PAYMENT    = ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"]

def make_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng  = random.Random(seed)
    rows = []
    for _ in range(n):
        tenure      = rng.randint(0, 72)        # months
        monthly_fee = rng.uniform(20, 120)
        total_spend = monthly_fee * tenure + rng.gauss(0, 50)
        plan_idx    = rng.randint(0, 2)
        contract    = rng.randint(0, 2)         # 0=month-month, 1=1yr, 2=2yr
        payment_idx = rng.randint(0, 3)
        tech_support= rng.random() < 0.5
        online_sec  = rng.random() < 0.4
        num_services= rng.randint(1, 6)
        complaints  = rng.randint(0, 5)
        data_gb     = rng.uniform(0, 50)

        score = (
            (1 - tenure / 72) * 0.8
            + (contract == 0) * 0.7
            + (monthly_fee > 80) * 0.4
            - num_services * 0.15
            - tech_support * 0.3
            + complaints * 0.4
            + (payment_idx == 3) * 0.3      # mailed check → less engaged
            + rng.gauss(0, 0.5)
        )
        churn = 1 if score > 0.5 else 0
        rows.append({
            "tenure": tenure, "monthly_fee": round(monthly_fee, 2),
            "total_spend": round(max(0, total_spend), 2),
            "plan_idx": plan_idx, "contract": contract,
            "payment_idx": payment_idx, "tech_support": int(tech_support),
            "online_sec": int(online_sec), "num_services": num_services,
            "complaints": complaints, "data_gb": round(data_gb, 1),
            "plan": PLANS[plan_idx], "contract_type": CONTRACTS[contract],
            "payment": PAYMENT[payment_idx], "churn": churn,
        })
    return pd.DataFrame(rows)


@st.cache_resource
def train():
    df   = make_data(600)
    feat = ["tenure", "monthly_fee", "plan_idx", "contract", "payment_idx",
            "tech_support", "online_sec", "num_services", "complaints", "data_gb"]
    X    = df[feat].values.tolist()
    y    = df["churn"].tolist()
    Xn, mins, maxes = normalize(X)
    w, b  = train_lr(Xn, y)
    probs = [sigmoid(b + sum(w[j] * Xn[i][j] for j in range(len(w)))) for i in range(len(Xn))]
    mets  = classification_metrics(y, probs)
    return w, b, mins, maxes, feat, mets, df


w, b, mins, maxes, feats, mets, df_all = train()
acc, prec, rec, f1 = mets

# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Predict Churn", "Churn Analysis", "Dataset"])

with tab1:
    st.subheader("Customer Profile")
    c1, c2    = st.columns(2)
    tenure    = c1.slider("Tenure (months)", 0, 72, 12)
    monthly   = c2.slider("Monthly Fee ($)", 20, 120, 55)
    c1, c2    = st.columns(2)
    plan      = c1.selectbox("Plan", PLANS)
    contract  = c2.selectbox("Contract Type", CONTRACTS)
    c1, c2    = st.columns(2)
    payment   = c1.selectbox("Payment Method", PAYMENT)
    num_svc   = c2.slider("Number of Services", 1, 6, 3)
    c1, c2    = st.columns(2)
    tech_sup  = c1.selectbox("Tech Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
    online_s  = c2.selectbox("Online Security", [0, 1], format_func=lambda x: "Yes" if x else "No")
    c1, c2    = st.columns(2)
    complaints= c1.slider("Complaints Filed", 0, 5, 0)
    data_gb   = c2.slider("Monthly Data Usage (GB)", 0.0, 50.0, 15.0, step=0.5)

    plan_idx  = PLANS.index(plan)
    cont_idx  = CONTRACTS.index(contract)
    pay_idx   = PAYMENT.index(payment)

    xi   = [tenure, monthly, plan_idx, cont_idx, pay_idx,
            tech_sup, online_s, num_svc, complaints, data_gb]
    prob = predict(w, b, mins, maxes, xi)

    st.divider()
    risk = "HIGH" if prob >= 0.6 else "MEDIUM" if prob >= 0.35 else "LOW"
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Probability",  f"{prob:.1%}")
    c2.metric("Churn Risk",         risk)
    c3.metric("Retention Score",    f"{(1 - prob) * 100:.0f}/100")
    st.progress(prob)

    if risk == "HIGH":
        st.error("High churn risk — proactive retention action recommended.")
        if contract == "Month-to-Month":
            st.info("Offering a discounted annual contract may reduce churn risk.")
        if complaints > 2:
            st.info("Multiple complaints — resolve issues quickly to restore satisfaction.")
        if tenure < 6:
            st.info("Early tenure — onboarding support reduces early churn significantly.")
    elif risk == "MEDIUM":
        st.warning("Moderate churn risk — monitor and engage proactively.")

    st.subheader("Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{acc:.2%}")
    m2.metric("Precision", f"{prec:.2%}")
    m3.metric("Recall",    f"{rec:.2%}")
    m4.metric("F1 Score",  f"{f1:.2%}")

with tab2:
    st.subheader("Churn by Contract Type")
    cc = df_all.groupby("contract_type")["churn"].mean().sort_values(ascending=False)
    st.bar_chart(cc)

    st.subheader("Churn by Tenure Bucket")
    df_all["tenure_bucket"] = pd.cut(df_all["tenure"], bins=[0, 12, 24, 48, 72],
                                      labels=["0–12m", "12–24m", "24–48m", "48–72m"])
    tb = df_all.groupby("tenure_bucket", observed=True)["churn"].mean()
    st.bar_chart(tb)

    total = len(df_all)
    c_cnt = df_all["churn"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", total)
    c2.metric("Churned",         c_cnt)
    c3.metric("Churn Rate",      f"{c_cnt / total:.1%}")

with tab3:
    st.dataframe(
        df_all[["tenure", "monthly_fee", "plan", "contract_type",
                "num_services", "complaints", "churn"]].head(50),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Training set: {len(df_all)} customers")
