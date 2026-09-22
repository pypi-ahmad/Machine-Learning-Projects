"""Student Performance — Streamlit regression demo using synthetic data.

Explore how a simple linear regression responds to illustrative study-related
inputs. It is not a student assessment or decision-making tool.

Usage:
    uv run streamlit run main.py
"""

import math
import random

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance", layout="wide")
st.title("🎓 Student Performance Demo")
st.caption("Synthetic data only. This educational example is not for real student assessment or decisions.")


# ── Simple linear regression helpers ─────────────────────────────────────────

def ols_fit(X, y):
    n, m = len(X), len(X[0])
    Xb = [[1.0]+row for row in X]
    m1 = m+1
    XtX = [[sum(Xb[i][a]*Xb[i][b] for i in range(n)) for b in range(m1)] for a in range(m1)]
    Xty = [sum(Xb[i][a]*y[i] for i in range(n)) for a in range(m1)]
    A = [row[:]+[Xty[i]] for i,row in enumerate(XtX)]
    for col in range(m1):
        piv = max(range(col,m1), key=lambda r: abs(A[r][col]))
        A[col],A[piv] = A[piv],A[col]
        if abs(A[col][col])<1e-12: continue
        for row in range(m1):
            if row==col: continue
            f = A[row][col]/A[col][col]
            A[row] = [A[row][j]-f*A[col][j] for j in range(m1+1)]
    return [A[i][m1]/A[i][i] if abs(A[i][i])>1e-12 else 0 for i in range(m1)]


def predict(w, X):
    return [w[0]+sum(w[j+1]*row[j] for j in range(len(row))) for row in X]


def r2(y, yp):
    my = sum(y)/len(y)
    return 1 - sum((a-b)**2 for a,b in zip(y,yp)) / (sum((a-my)**2 for a in y) or 1)


def rmse(y, yp):
    return math.sqrt(sum((a-b)**2 for a,b in zip(y,yp))/len(y))


# ── Synthetic student dataset ─────────────────────────────────────────────────

def make_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        study_hours   = round(rng.uniform(0, 12), 1)
        attendance    = rng.randint(40, 100)
        prev_score    = rng.randint(40, 100)
        sleep_hours   = round(rng.uniform(4, 10), 1)
        extra_classes = rng.randint(0, 10)

        score = (
            study_hours * 3.2
            + attendance * 0.3
            + prev_score * 0.35
            + sleep_hours * 1.5
            + extra_classes * 1.8
            + rng.gauss(0, 5)
        )
        score = max(0, min(100, round(score)))
        grade = "A" if score>=90 else "B" if score>=75 else "C" if score>=60 else "D" if score>=50 else "F"
        rows.append({
            "study_hours": study_hours, "attendance": attendance,
            "prev_score": prev_score, "sleep_hours": sleep_hours,
            "extra_classes": extra_classes,
            "score": score, "grade": grade,
        })
    return pd.DataFrame(rows)


@st.cache_data
def train():
    df = make_data(500)
    feat = ["study_hours", "attendance", "prev_score", "sleep_hours", "extra_classes"]
    X    = df[feat].values.tolist()
    y    = df["score"].tolist()
    w    = ols_fit(X, y)
    yp   = predict(w, X)
    return w, feat, r2(y, yp), rmse(y, yp), df


model_w, feats, model_r2, model_rmse, df_all = train()

# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Predict Score", "Class Analysis", "Sample Data"])

with tab1:
    st.subheader("Explore an Illustrative Score")
    c1, c2 = st.columns(2)
    study_hours   = c1.slider("Daily Study Hours", 0.0, 12.0, 4.0, step=0.5)
    attendance    = c2.slider("Attendance (%)", 40, 100, 80)
    c1, c2        = st.columns(2)
    prev_score    = c1.slider("Previous Exam Score", 40, 100, 70)
    sleep_hours   = c2.slider("Daily Sleep Hours", 4.0, 10.0, 7.0, step=0.5)
    extra_classes = st.slider("Extra Classes Attended", 0, 10, 3)

    xi   = [study_hours, attendance, prev_score, sleep_hours, extra_classes]
    pred = predict(model_w, [xi])[0]
    pred = max(0, min(100, pred))
    grade = "A" if pred>=90 else "B" if pred>=75 else "C" if pred>=60 else "D" if pred>=50 else "F"

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Illustrative score", f"{pred:.1f}/100")
    col2.metric("Illustrative band",  grade)
    col3.metric("Threshold",         "Below 60" if pred < 60 else "60 or above")
    st.progress(pred / 100)

    if pred < 60:
        st.warning("This synthetic example falls below its illustrative threshold. It is not a student-risk assessment or a recommendation.")
        tips = []
        if study_hours < 3: tips.append("• The demo input has fewer than 3 daily study hours.")
        if attendance  < 75: tips.append("• The demo input has attendance below 75%.")
        if sleep_hours < 6:  tips.append("• The demo input has fewer than 6 daily sleep hours.")
        for tip in tips: st.info(tip)

with tab2:
    st.subheader("Synthetic Class Analysis")
    grade_counts = df_all["grade"].value_counts().sort_index()
    st.bar_chart(grade_counts)
    below_threshold = (df_all["score"] < 60).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Score",  f"{df_all['score'].mean():.1f}")
    c2.metric("Pass Rate",      f"{(df_all['score']>=50).mean():.1%}")
    c3.metric("Below illustrative threshold", f"{below_threshold} ({below_threshold/len(df_all):.1%})")
    st.metric("Model R²",   f"{model_r2:.3f}")
    st.metric("Model RMSE", f"{model_rmse:.2f} points")

with tab3:
    st.dataframe(df_all.head(50), hide_index=True)
    st.caption(f"Synthetic training rows: {len(df_all)}")
