"""Loan Eligibility Predictor — Streamlit ML demo.

Predict loan approval based on applicant details.
Uses a rule-based scoring model with threshold classification.

Usage:
    streamlit run main.py
"""

import math
import random
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Eligibility", layout="wide")
st.title("🏦 Loan Eligibility Predictor")


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(-500, min(500, z))))


def compute_score(income: float, loan_amount: float, loan_term: int,
                  credit_score: int, employment: str, dependents: int,
                  property_area: str, education: str) -> dict:
    """Heuristic scoring model for loan eligibility."""

    # Debt-to-income ratio
    monthly_income  = income / 12
    monthly_payment = loan_amount / (loan_term * 12) * 1.1   # approx with interest
    dti = monthly_payment / monthly_income if monthly_income > 0 else 1.0

    # Credit score contribution (300–850 range)
    credit_factor = (credit_score - 300) / 550

    # Employment factor
    emp_map = {"Salaried": 1.0, "Self-Employed": 0.75, "Unemployed": 0.1, "Part-time": 0.6}
    emp_factor = emp_map.get(employment, 0.5)

    # Education factor
    edu_factor = 1.05 if education == "Graduate" else 0.95

    # Property factor
    prop_map = {"Urban": 1.05, "Semiurban": 1.0, "Rural": 0.90}
    prop_factor = prop_map.get(property_area, 1.0)

    # Dependents penalty
    dep_factor = max(0.7, 1.0 - dependents * 0.05)

    # Loan-to-income ratio
    lti = loan_amount / income if income > 0 else 10
    lti_factor = max(0, 1 - (lti / 10))

    # Combined score
    base = (
        credit_factor * 0.35
        + emp_factor   * 0.25
        + (1 - dti)    * 0.20
        + lti_factor   * 0.10
        + dep_factor   * 0.05
        + edu_factor   * 0.03
        + prop_factor  * 0.02
    )
    prob = sigmoid((base - 0.5) * 8)    # scale to [0,1]
    approved = prob >= 0.55

    return {
        "prob":       prob,
        "approved":   approved,
        "dti":        dti,
        "lti":        lti,
        "credit_factor": credit_factor,
        "emp_factor":    emp_factor,
        "base_score":    base,
    }


def make_sample_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    employment_opts = ["Salaried","Self-Employed","Part-time","Unemployed"]
    property_opts   = ["Urban","Semiurban","Rural"]
    edu_opts        = ["Graduate","Not Graduate"]
    rows = []
    for _ in range(n):
        income       = rng.randint(20000, 150000)
        loan_amount  = rng.randint(5000, min(income * 5, 500000))
        loan_term    = rng.choice([5, 10, 15, 20, 30])
        credit_score = rng.randint(350, 820)
        employment   = rng.choice(employment_opts)
        dependents   = rng.randint(0, 4)
        prop_area    = rng.choice(property_opts)
        education    = rng.choice(edu_opts)
        result = compute_score(income, loan_amount, loan_term, credit_score,
                               employment, dependents, prop_area, education)
        rows.append({
            "income": income, "loan_amount": loan_amount, "loan_term": loan_term,
            "credit_score": credit_score, "employment": employment,
            "dependents": dependents, "property_area": prop_area,
            "education": education,
            "approved": "Yes" if result["approved"] else "No",
            "probability": round(result["prob"], 3),
        })
    return pd.DataFrame(rows)


# ── UI ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Check Eligibility", "Approval Factors", "Sample Dataset"])

with tab1:
    st.subheader("Loan Application")
    c1, c2 = st.columns(2)
    income      = c1.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
    loan_amount = c2.number_input("Loan Amount ($)", 1000, 1000000, 150000, step=1000)
    c1, c2      = st.columns(2)
    loan_term   = c1.selectbox("Loan Term (years)", [5, 10, 15, 20, 30], index=2)
    credit_score= c2.slider("Credit Score", 300, 850, 680)
    c1, c2      = st.columns(2)
    employment  = c1.selectbox("Employment", ["Salaried","Self-Employed","Part-time","Unemployed"])
    dependents  = c2.slider("Dependents", 0, 5, 1)
    c1, c2      = st.columns(2)
    prop_area   = c1.selectbox("Property Area", ["Urban","Semiurban","Rural"])
    education   = c2.selectbox("Education", ["Graduate","Not Graduate"])

    result = compute_score(income, loan_amount, loan_term, credit_score,
                           employment, dependents, prop_area, education)

    st.divider()
    verdict = "✅ APPROVED" if result["approved"] else "❌ NOT APPROVED"
    col1, col2, col3 = st.columns(3)
    col1.metric("Decision", verdict)
    col2.metric("Approval Probability", f"{result['prob']:.1%}")
    col3.metric("Credit Factor", f"{result['credit_factor']:.2f}")
    st.progress(result["prob"])

    c1, c2 = st.columns(2)
    c1.metric("Debt-to-Income Ratio", f"{result['dti']:.2%}",
              delta="good" if result["dti"] < 0.36 else "high",
              delta_color="normal" if result["dti"] < 0.36 else "inverse")
    c2.metric("Loan-to-Income Ratio", f"{result['lti']:.2f}x",
              delta="good" if result["lti"] < 4 else "high",
              delta_color="normal" if result["lti"] < 4 else "inverse")

with tab2:
    st.subheader("What affects your score?")
    st.markdown("""
    | Factor | Weight | Your Situation |
    |---|---|---|
    | Credit Score | 35% | {credit} |
    | Employment | 25% | {emp} |
    | Debt-to-Income | 20% | {dti:.1%} |
    | Loan-to-Income | 10% | {lti:.2f}x |
    | Dependents | 5% | {dep} |
    | Education | 3% | {edu} |
    | Property Area | 2% | {prop} |
    """.format(
        credit=f"{credit_score} ({result['credit_factor']:.2f})",
        emp=employment,
        dti=result["dti"],
        lti=result["lti"],
        dep=dependents,
        edu=education,
        prop=prop_area,
    ))

    st.subheader("Tips to improve eligibility")
    tips = []
    if result["dti"] > 0.36:
        tips.append("• Reduce the loan amount or extend the loan term to lower monthly payments.")
    if credit_score < 650:
        tips.append("• Improve your credit score by paying bills on time and reducing debt.")
    if result["lti"] > 5:
        tips.append("• Request a smaller loan relative to your income.")
    if employment == "Unemployed":
        tips.append("• Secure stable employment before applying.")
    if not tips:
        tips.append("• Your profile looks strong. Maintain your credit score.")
    for tip in tips:
        st.info(tip)

with tab3:
    df = make_sample_data()
    st.dataframe(df, use_container_width=True, hide_index=True)
    approval_rate = (df["approved"] == "Yes").mean()
    st.metric("Sample Approval Rate", f"{approval_rate:.1%}")
    st.bar_chart(df["approved"].value_counts())
