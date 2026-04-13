"""Resume Screener — Streamlit ML demo.

Upload resumes (text or PDF) and a job description.
Score and rank candidates using TF-IDF cosine similarity.

Usage:
    streamlit run main.py
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("📄 Resume Screener")

# ── Helpers ───────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","by","it","this","that","from",
    "as","we","you","they","he","she","i","our","your","their","its",
    "will","can","may","have","has","had","do","does","did","not","all",
}


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'\b[a-zA-Z][a-zA-Z0-9]+\b', text)
            if w.lower() not in STOPWORDS and len(w) > 2]


def tfidf_vector(tokens: list[str], vocab: list[str]) -> list[float]:
    tf   = Counter(tokens)
    n    = len(tokens) or 1
    return [tf.get(w, 0) / n for w in vocab]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot  = sum(x*y for x, y in zip(a, b))
    na   = sum(x*x for x in a) ** 0.5
    nb   = sum(x*x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def extract_keywords(tokens: list[str], n: int = 10) -> list[str]:
    return [w for w, _ in Counter(tokens).most_common(n)]


def read_text(uploaded) -> str:
    content = uploaded.read()
    try:
        if uploaded.name.endswith(".pdf"):
            # Basic PDF text extraction without dependencies
            text = content.decode("latin-1", errors="ignore")
            # Extract readable ASCII sequences
            readable = re.findall(r'[A-Za-z][A-Za-z0-9 ,.\-:;/()&+@]{3,}', text)
            return " ".join(readable)
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_JD = """
We are looking for a Senior Python Developer with 5+ years of experience.
Required skills: Python, Django, REST API, PostgreSQL, Docker, AWS, microservices.
Experience with machine learning, scikit-learn, pandas, and data pipelines is a plus.
Must have strong communication skills, ability to work in agile teams.
Bachelor's degree in Computer Science or related field.
"""

SAMPLE_RESUMES = {
    "Alice Johnson": """
    Senior Software Engineer with 7 years of experience in Python and Django.
    Proficient in REST API design, PostgreSQL, Docker, and AWS deployment.
    Led microservices migration project. Experience with pandas and scikit-learn.
    BS Computer Science, Stanford. Excellent communicator, agile practitioner.
    """,
    "Bob Smith": """
    Junior Web Developer with 2 years experience. Knows HTML, CSS, JavaScript.
    Some Python scripting. Familiar with MySQL. Looking to grow in backend development.
    Self-taught programmer. Good team player.
    """,
    "Carol White": """
    Data Scientist with 4 years of experience in Python, scikit-learn, pandas, NumPy.
    Built ML pipelines for fraud detection. AWS SageMaker, Docker experience.
    PostgreSQL and REST API experience. Master's in Data Science. Agile teams.
    """,
    "David Chen": """
    Backend Engineer, 6 years experience. Python, FastAPI, PostgreSQL, microservices.
    Docker, Kubernetes, AWS. Worked on high-traffic APIs. Machine learning basics.
    BS Computer Engineering. Strong communication and leadership skills.
    """,
}

# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Screen Resumes", "About"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Job Description")
        jd_text = st.text_area("Paste job description", value=SAMPLE_JD, height=220)
        use_sample = st.checkbox("Use sample resumes", value=True)

    with col2:
        st.subheader("Resumes")
        if not use_sample:
            uploaded = st.file_uploader("Upload resumes (.txt or .pdf)",
                                        accept_multiple_files=True,
                                        type=["txt","pdf"])
            resumes = {f.name: read_text(f) for f in uploaded}
        else:
            resumes = SAMPLE_RESUMES
            st.info(f"Using {len(resumes)} sample resumes.")

    if st.button("🔍 Screen Candidates", type="primary") and jd_text and resumes:
        jd_tokens = tokenize(jd_text)
        jd_keys   = extract_keywords(jd_tokens, 20)
        st.caption(f"Top JD keywords: {', '.join(jd_keys)}")

        # Build vocab from JD + all resumes
        all_tokens = jd_tokens[:]
        for r in resumes.values():
            all_tokens += tokenize(r)
        vocab = list(set(all_tokens))

        jd_vec = tfidf_vector(jd_tokens, vocab)

        rows = []
        for name, text in resumes.items():
            rtoks   = tokenize(text)
            rvec    = tfidf_vector(rtoks, vocab)
            score   = cosine_sim(jd_vec, rvec)
            r_keys  = extract_keywords(rtoks, 15)
            overlap = [k for k in jd_keys if k in r_keys]
            rows.append({
                "Candidate":    name,
                "Match Score":  round(score, 4),
                "Match %":      f"{score*100:.1f}%",
                "Keyword Hits": len(overlap),
                "Keywords":     ", ".join(overlap[:8]),
            })

        df = pd.DataFrame(rows).sort_values("Match Score", ascending=False).reset_index(drop=True)
        df.index += 1

        st.subheader("Ranking Results")
        st.dataframe(df, use_container_width=True)

        # Highlight top candidate
        top = df.iloc[0]
        st.success(f"🏆 Top candidate: **{top['Candidate']}** — {top['Match %']} match")

        # Score chart
        st.subheader("Match Scores")
        chart_df = df.set_index("Candidate")["Match Score"]
        st.bar_chart(chart_df)

with tab2:
    st.markdown("""
    ### How it works
    1. Tokenizes the job description and each resume
    2. Builds a shared vocabulary
    3. Computes TF-IDF vectors for each document
    4. Ranks candidates by cosine similarity to the job description

    **Limitations:** This is a keyword-based approach. Real-world ATS systems
    use more sophisticated NLP including semantic similarity, skills ontologies,
    and experience parsing.
    """)
