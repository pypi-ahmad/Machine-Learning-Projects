"""Student Grade Tracker — Streamlit app.

Track student grades per subject, compute GPA, visualise performance
trends, and export reports.  Data stored locally as CSV.

Usage:
    streamlit run main.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Grade Tracker", layout="wide")
st.title("🎓 Student Grade Tracker")

DATA_FILE = Path("grades.csv")
GRADE_SCALE = {
    (90, 100): ("A+", 4.0), (85, 90): ("A",  4.0), (80, 85): ("A-", 3.7),
    (75, 80):  ("B+", 3.3), (70, 75): ("B",  3.0), (65, 70): ("B-", 2.7),
    (60, 65):  ("C+", 2.3), (55, 60): ("C",  2.0), (50, 55): ("C-", 1.7),
    (45, 50):  ("D+", 1.3), (40, 45): ("D",  1.0), (0,  40): ("F",  0.0),
}


def score_to_grade(score: float) -> tuple[str, float]:
    for (lo, hi), (letter, gpa) in GRADE_SCALE.items():
        if lo <= score <= hi:
            return letter, gpa
    return "F", 0.0


def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        try:
            return pd.read_csv(DATA_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["Student", "Subject", "Score", "Date", "Notes"])


def save_data(df: pd.DataFrame) -> None:
    df.to_csv(DATA_FILE, index=False)


if "df" not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# ---------------------------------------------------------------------------
# Sidebar — add grade
# ---------------------------------------------------------------------------
st.sidebar.header("Add Grade")
with st.sidebar.form("add_grade"):
    student = st.text_input("Student name")
    subject = st.text_input("Subject")
    score   = st.slider("Score", 0, 100, 75)
    date    = st.date_input("Date")
    notes   = st.text_input("Notes (optional)")
    submit  = st.form_submit_button("Add")

if submit and student.strip() and subject.strip():
    letter, gpa = score_to_grade(score)
    new_row = pd.DataFrame([{
        "Student": student.strip(),
        "Subject": subject.strip(),
        "Score":   score,
        "Letter":  letter,
        "GPA":     gpa,
        "Date":    str(date),
        "Notes":   notes,
    }])
    st.session_state.df = pd.concat([df, new_row], ignore_index=True)
    save_data(st.session_state.df)
    st.sidebar.success("Grade added!")
    df = st.session_state.df

if df.empty:
    st.info("No grades yet. Add some using the sidebar.")
    st.stop()

# Compute letter/GPA if not present
if "Letter" not in df.columns:
    df["Letter"] = df["Score"].apply(lambda s: score_to_grade(s)[0])
    df["GPA"]    = df["Score"].apply(lambda s: score_to_grade(s)[1])

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["All Grades", "By Student", "By Subject", "Reports"])

with tab1:
    st.subheader("All Grades")
    students = ["All"] + sorted(df["Student"].unique().tolist())
    sel = st.selectbox("Filter by student", students)
    view = df if sel == "All" else df[df["Student"] == sel]
    st.dataframe(view.sort_values("Date", ascending=False), use_container_width=True)

with tab2:
    st.subheader("Student Summary")
    student_sel = st.selectbox("Student", sorted(df["Student"].unique()))
    sdf = df[df["Student"] == student_sel]
    avg_score = sdf["Score"].mean()
    avg_gpa   = sdf["GPA"].mean() if "GPA" in sdf else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Score", f"{avg_score:.1f}%")
    c2.metric("GPA",       f"{avg_gpa:.2f}")
    c3.metric("Subjects",  sdf["Subject"].nunique())

    st.bar_chart(sdf.groupby("Subject")["Score"].mean().sort_values())
    st.dataframe(sdf, use_container_width=True)

with tab3:
    st.subheader("Subject Summary")
    sub_sel = st.selectbox("Subject", sorted(df["Subject"].unique()))
    subdf = df[df["Subject"] == sub_sel]
    st.bar_chart(subdf.set_index("Student")["Score"].sort_values())
    avg = subdf["Score"].mean()
    top = subdf.loc[subdf["Score"].idxmax()]
    st.metric("Class average", f"{avg:.1f}%")
    st.write(f"**Top performer:** {top['Student']} — {top['Score']}%")
    st.dataframe(subdf.sort_values("Score", ascending=False), use_container_width=True)

with tab4:
    st.subheader("GPA Report — All Students")
    gpa_report = df.groupby("Student").agg(
        avg_score=("Score", "mean"),
        avg_gpa=("GPA",   "mean") if "GPA" in df else ("Score", "count"),
        total_grades=("Score", "count"),
    ).round(2).sort_values("avg_gpa", ascending=False)
    st.dataframe(gpa_report, use_container_width=True)

    st.divider()
    if st.button("📥 Export all grades as CSV"):
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv_bytes, "grades_export.csv", "text/csv")
