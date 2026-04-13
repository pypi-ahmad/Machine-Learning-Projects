"""PDF / Text Summarizer — Streamlit ML demo.

Summarize long text documents using extractive summarization
(TF-IDF sentence scoring). Supports pasted text and .txt uploads.

Usage:
    streamlit run main.py
"""

import math
import re
from collections import Counter

import streamlit as st

st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("📝 Text Summarizer")
st.caption("Extractive summarization using TF-IDF sentence scoring — no external ML libraries.")


# ── NLP core ──────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and", "or",
    "but", "for", "with", "this", "that", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "from", "by", "as", "into", "about", "which", "who", "what", "when", "where",
    "how", "not", "no", "its", "their", "they", "we", "you", "i", "he", "she",
    "his", "her", "our", "your", "my", "very", "so", "too", "also", "just",
    "more", "than", "some", "such", "can", "may", "both", "each", "s", "t",
}


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z]+", text.lower())
            if w not in STOPWORDS and len(w) > 2]


def split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 25]


def score_sentences(sentences: list[str]) -> list[float]:
    """TF-IDF sentence scoring: sum of word TF-IDF scores."""
    docs   = [tokenize(s) for s in sentences]
    N      = len(docs)
    df_cnt = Counter(tok for doc in docs for tok in set(doc))
    scores = []
    for doc in docs:
        tf  = Counter(doc)
        tot = len(doc) or 1
        sc  = sum((tf[t] / tot) * math.log((N + 1) / (df_cnt[t] + 1) + 1)
                  for t in tf)
        scores.append(sc)
    return scores


def summarize(text: str, ratio: float = 0.3, max_sentences: int = 10) -> dict:
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return {"error": "Text too short to summarize (need ≥ 3 sentences)."}

    raw_scores   = score_sentences(sentences)
    n_select     = max(1, min(max_sentences, round(len(sentences) * ratio)))
    indexed      = sorted(enumerate(raw_scores), key=lambda x: -x[1])
    top_indices  = sorted([i for i, _ in indexed[:n_select]])
    summary_sents= [sentences[i] for i in top_indices]
    summary      = " ".join(summary_sents)

    word_count_orig = len(text.split())
    word_count_summ = len(summary.split())

    # Keyword extraction (top TF-IDF terms from full doc)
    all_tokens = tokenize(text)
    freq       = Counter(all_tokens)
    keywords   = [w for w, _ in freq.most_common(10)]

    return {
        "summary":        summary,
        "sentences":      summary_sents,
        "orig_sentences": len(sentences),
        "kept_sentences": n_select,
        "orig_words":     word_count_orig,
        "summ_words":     word_count_summ,
        "compression":    1 - word_count_summ / max(word_count_orig, 1),
        "keywords":       keywords,
    }


# ── Sample texts ──────────────────────────────────────────────────────────────

SAMPLES = {
    "Artificial Intelligence": """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural
intelligence displayed by animals including humans. AI research has been defined as the field of
study of intelligent agents, which refers to any system that perceives its environment and takes
actions that maximize its chances of achieving its goals.

The term artificial intelligence was first used by John McCarthy in 1956 at the Dartmouth Conference.
Early AI research explored topics like problem solving, symbolic methods, and language understanding.
By the 1980s, machine learning became central to AI progress.

Modern AI includes deep learning, which uses neural networks with many layers trained on large
datasets. Breakthroughs in deep learning led to major advances in computer vision, speech recognition,
and natural language processing.

Large language models such as GPT and BERT have transformed natural language processing tasks.
These models are trained on billions of words of text and can generate coherent, contextually
appropriate text in response to prompts.

AI is now applied in healthcare for disease diagnosis, in finance for fraud detection, in transportation
for autonomous vehicles, and in countless other domains. Concerns around AI include job displacement,
algorithmic bias, privacy, and the long-term safety of highly capable systems.

Research organizations like DeepMind, OpenAI, and major university labs continue to push the
boundaries of what AI systems can do. The field remains one of the fastest-growing areas of
technology and scientific research worldwide.
""",
    "The Internet": """
The internet is a global network of interconnected computers and servers that communicate using
standardized protocols. It began as ARPANET, a research project funded by the United States
Department of Defense in the 1960s.

The World Wide Web, invented by Tim Berners-Lee in 1989, made the internet accessible to the
general public. Web browsers allowed users to navigate hyperlinked documents hosted on web servers.

The internet has transformed communication, commerce, entertainment, and access to information.
Email replaced much of postal correspondence for business and personal communication.
Social media platforms changed how people interact and share information globally.

E-commerce grew rapidly in the late 1990s and early 2000s. Amazon, eBay, and later Alibaba
became dominant online marketplaces. Mobile internet access via smartphones dramatically
expanded the reach of online services.

The internet enables cloud computing, where businesses and individuals store data and run
applications on remote servers rather than local hardware. Streaming services like Netflix
and YouTube deliver video content over the internet to millions of simultaneous users.

Cybersecurity has become critical as more systems depend on internet connectivity.
Threats include malware, phishing attacks, ransomware, and state-sponsored hacking.
Encryption protocols like HTTPS protect data in transit across the web.

Internet governance involves organizations like ICANN, which manages domain names, and the
Internet Engineering Task Force, which develops technical standards. Net neutrality remains
a debated policy issue in many countries.
""",
}

# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Summarize", "About"])

with tab1:
    source = st.radio("Input source", ["Sample Text", "Paste Text", "Upload .txt"])

    if source == "Sample Text":
        chosen   = st.selectbox("Choose sample", list(SAMPLES.keys()))
        raw_text = SAMPLES[chosen].strip()
    elif source == "Paste Text":
        raw_text = st.text_area("Paste text here", height=250)
    else:
        upl = st.file_uploader("Upload .txt file", type="txt")
        raw_text = upl.read().decode("utf-8") if upl else ""

    if source == "Sample Text" or raw_text.strip():
        st.markdown(f"**Input length:** {len(raw_text.split())} words")

    c1, c2 = st.columns(2)
    ratio  = c1.slider("Summary length (% of original)", 10, 60, 30, step=5) / 100
    max_s  = c2.slider("Max sentences in summary", 2, 15, 8)

    if st.button("✂️ Summarize", type="primary"):
        if not raw_text.strip():
            st.error("Please provide some text first.")
        else:
            result = summarize(raw_text, ratio=ratio, max_sentences=max_s)
            if "error" in result:
                st.error(result["error"])
            else:
                st.divider()
                st.subheader("Summary")
                st.write(result["summary"])

                st.subheader("Key Topics")
                st.write(" · ".join(f"**{k}**" for k in result["keywords"]))

                st.subheader("Statistics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Original Sentences", result["orig_sentences"])
                m2.metric("Summary Sentences",  result["kept_sentences"])
                m3.metric("Compression",         f"{result['compression']:.1%}")
                w1, w2 = st.columns(2)
                w1.metric("Original Words", result["orig_words"])
                w2.metric("Summary Words",  result["summ_words"])

                with st.expander("Show selected sentences"):
                    for i, s in enumerate(result["sentences"], 1):
                        st.markdown(f"**{i}.** {s}")

with tab2:
    st.markdown("""
    ### How Extractive Summarization Works
    1. **Split** text into sentences.
    2. **Tokenize** each sentence (remove stop words, lowercase).
    3. **Score** each sentence using TF-IDF:
       - High-scoring sentences contain terms that are frequent in the document but not ubiquitous.
    4. **Select** the top-N sentences by score (preserving original order).
    5. **Join** selected sentences to form the summary.

    ### Limitations
    - Purely extractive — does not rephrase or combine ideas.
    - Quality depends on sentence boundaries being clear.
    - Does not understand meaning — just keyword frequency.

    ### When to Use
    - Quick summarization of reports, articles, or research papers.
    - Keyword extraction for document tagging.
    - Pre-processing before manual review.
    """)
