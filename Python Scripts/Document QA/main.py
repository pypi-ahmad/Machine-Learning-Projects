"""Document Q&A — Streamlit ML demo.

Answer questions about uploaded or pasted text documents
using TF-IDF sentence retrieval (no external ML libraries).

Usage:
    streamlit run main.py
"""

import math
import re
from collections import Counter

import streamlit as st

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Document Q&A")
st.caption("Ask questions about any text document using keyword-based sentence retrieval.")


# ── NLP helpers ───────────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and", "or",
    "but", "for", "with", "this", "that", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "from", "by", "as", "into", "about",
    "which", "who", "whom", "what", "when", "where", "how", "why", "its", "their",
    "they", "we", "you", "i", "he", "she", "his", "her", "our", "your", "my",
    "not", "no", "nor", "so", "yet", "both", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "s", "t",
}


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 20]


def tfidf_vectors(docs: list[list[str]]) -> list[dict]:
    N = len(docs)
    df_cnt = Counter(tok for doc in docs for tok in set(doc))
    result = []
    for doc in docs:
        tf  = Counter(doc)
        tot = len(doc) or 1
        vec = {t: (tf[t] / tot) * math.log((N + 1) / (df_cnt[t] + 1))
               for t in tf}
        result.append(vec)
    return result


def cosine(a: dict, b: dict) -> float:
    keys = set(a) & set(b)
    dot  = sum(a[k] * b[k] for k in keys)
    na   = math.sqrt(sum(v ** 2 for v in a.values()))
    nb   = math.sqrt(sum(v ** 2 for v in b.values()))
    return dot / (na * nb + 1e-9)


def answer_question(question: str, sentences: list[str],
                    sent_vecs: list[dict], top_k: int = 3) -> list[tuple[str, float]]:
    q_vec = tfidf_vectors([tokenize(question)])[0]
    scores = [(sentences[i], cosine(q_vec, sent_vecs[i])) for i in range(len(sentences))]
    scores.sort(key=lambda x: -x[1])
    return [(s, sc) for s, sc in scores[:top_k] if sc > 0]


# ── Sample documents ──────────────────────────────────────────────────────────

SAMPLE_DOCS = {
    "Machine Learning Overview": """
Machine learning is a subset of artificial intelligence that enables systems to learn from data
without being explicitly programmed. It focuses on developing algorithms that can access data
and use it to learn for themselves.

There are three main types of machine learning: supervised learning, unsupervised learning, and
reinforcement learning. In supervised learning, the algorithm is trained on labeled data, meaning
each training example is paired with an output label. Common supervised learning tasks include
classification and regression.

Unsupervised learning deals with unlabeled data. The algorithm tries to find hidden patterns or
intrinsic structures in the input data. Clustering is a well-known unsupervised learning task.

Reinforcement learning is a type of machine learning where an agent learns to make decisions
by performing actions and receiving rewards or penalties. It is widely used in robotics and game playing.

Deep learning is a subset of machine learning that uses neural networks with many layers.
Convolutional neural networks (CNNs) are commonly used for image recognition tasks.
Recurrent neural networks (RNNs) and transformers are used for natural language processing.

Popular machine learning frameworks include TensorFlow, PyTorch, and scikit-learn.
Python is the most widely used programming language for machine learning development.
""",
    "Python Programming": """
Python is a high-level, interpreted programming language known for its simplicity and readability.
It was created by Guido van Rossum and first released in 1991.

Python supports multiple programming paradigms including procedural, object-oriented, and
functional programming. Its design philosophy emphasizes code readability and simplicity.

Python has a large standard library that provides modules and packages for many common tasks.
The Package Index (PyPI) contains hundreds of thousands of third-party packages.

Key Python features include dynamic typing, automatic memory management, and a comprehensive
standard library. Python uses indentation to define code blocks, making it visually clean.

Python is widely used in web development (Django, Flask), data science (pandas, NumPy),
machine learning (scikit-learn, TensorFlow), automation, and scripting.

List comprehensions are a concise way to create lists in Python. Dictionary and set comprehensions
are also supported. Generators allow lazy evaluation of sequences.

Python 3 is the current major version. Python 2 reached end-of-life in January 2020.
Virtual environments help isolate project dependencies using tools like venv and conda.
""",
    "Climate Change": """
Climate change refers to long-term shifts in global temperatures and weather patterns.
While natural processes have always influenced climate, since the 1800s human activities have
become the primary driver of climate change, mainly due to burning fossil fuels.

Carbon dioxide and methane are greenhouse gases that trap heat in the atmosphere.
The burning of coal, oil, and natural gas releases large amounts of carbon dioxide.
Deforestation also contributes because trees that absorbed carbon dioxide are removed.

The effects of climate change include rising sea levels, more frequent extreme weather events,
melting ice caps, and shifts in ecosystems and wildlife habitats. Global average temperatures
have increased by approximately 1.1 degrees Celsius since the pre-industrial period.

The Paris Agreement, adopted in 2015, aims to limit global warming to well below 2 degrees
Celsius above pre-industrial levels. Countries committed to nationally determined contributions
to reduce greenhouse gas emissions.

Renewable energy sources like solar and wind power produce no direct carbon emissions.
Energy efficiency improvements, electric vehicles, and carbon capture technologies
are among the solutions being pursued to mitigate climate change.
""",
}


# ── Session state ─────────────────────────────────────────────────────────────

if "sentences" not in st.session_state:
    st.session_state.sentences = []
    st.session_state.sent_vecs = []
    st.session_state.doc_loaded = False
    st.session_state.qa_history = []

tab1, tab2, tab3 = st.tabs(["Load Document", "Ask Questions", "Q&A History"])

with tab1:
    st.subheader("Load a Document")
    source = st.radio("Source", ["Sample Document", "Paste Text", "Upload .txt File"])

    if source == "Sample Document":
        chosen = st.selectbox("Choose sample", list(SAMPLE_DOCS.keys()))
        doc_text = SAMPLE_DOCS[chosen].strip()
        st.text_area("Preview", doc_text[:500] + "...", height=150, disabled=True)
    elif source == "Paste Text":
        doc_text = st.text_area("Paste your document here", height=250)
    else:
        uploaded = st.file_uploader("Upload .txt file", type="txt")
        doc_text = uploaded.read().decode("utf-8") if uploaded else ""

    if st.button("📥 Load Document", type="primary") and doc_text.strip():
        sents = split_sentences(doc_text)
        if len(sents) < 3:
            st.error("Document too short — needs at least 3 sentences.")
        else:
            tok_sents = [tokenize(s) for s in sents]
            vecs      = tfidf_vectors(tok_sents)
            st.session_state.sentences  = sents
            st.session_state.sent_vecs  = vecs
            st.session_state.doc_loaded = True
            st.session_state.qa_history = []
            st.success(f"Document loaded: {len(sents)} sentences indexed.")

with tab2:
    if not st.session_state.doc_loaded:
        st.info("Load a document first in the 'Load Document' tab.")
    else:
        st.subheader("Ask a Question")
        st.caption(f"Document has {len(st.session_state.sentences)} indexed sentences.")
        question = st.text_input("Your question", placeholder="e.g. What is supervised learning?")
        top_k    = st.slider("Number of answers to retrieve", 1, 5, 3)

        if st.button("🔍 Find Answer", type="primary") and question.strip():
            results = answer_question(
                question,
                st.session_state.sentences,
                st.session_state.sent_vecs,
                top_k,
            )
            if results:
                st.divider()
                st.subheader("Best Matching Passages")
                for i, (sent, score) in enumerate(results, 1):
                    with st.expander(f"#{i} — Relevance: {score:.4f}", expanded=(i == 1)):
                        st.write(sent)
                st.session_state.qa_history.append({
                    "question": question,
                    "answers": [s for s, _ in results],
                })
            else:
                st.warning("No relevant passages found. Try rephrasing your question.")

with tab3:
    if st.session_state.qa_history:
        st.subheader("Previous Questions & Answers")
        for item in reversed(st.session_state.qa_history):
            st.markdown(f"**Q:** {item['question']}")
            for i, ans in enumerate(item["answers"], 1):
                st.markdown(f"> {i}. {ans}")
            st.divider()
    else:
        st.info("No Q&A history yet.")
