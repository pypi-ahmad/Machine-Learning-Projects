"""Local Streamlit demo for keyword-based document passage retrieval."""

import re

import streamlit as st


def tokenize(text: str) -> set[str]:
    """Return lowercase word tokens without punctuation."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def search_document(text: str, query: str) -> list[str]:
    """Return up to five sentences ranked by shared keyword count."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    query_words = tokenize(query)
    results = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        overlap = len(query_words & tokenize(sentence))
        if overlap > 0:
            results.append((overlap, sentence))
    results.sort(key=lambda result: -result[0])
    return [result[1] for result in results[:5]]

SAMPLE_DOC = """Python is a high-level programming language. It was created by Guido van Rossum and released in 1991. Python emphasizes code readability with significant indentation. It supports multiple paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, machine learning, and automation. The Python Package Index hosts thousands of third-party modules. Python 3 is the current major version, with Python 2 reaching end of life in 2020."""

def main() -> None:
    """Render the document and question form."""
    st.set_page_config(page_title="Document Q&A Demo", page_icon=":material/article:")
    st.title("Document Q&A demo")
    st.caption("Paste text and retrieve passages with overlapping keywords.")
    with st.form("document_question"):
        document = st.text_area("Document", SAMPLE_DOC, height=200)
        query = st.text_input("Ask a question")
        submitted = st.form_submit_button("Find passages", icon=":material/search:")
    if submitted:
        results = search_document(document, query) if document and query else []
        st.header("Relevant passages")
        if results:
            for index, result in enumerate(results, 1):
                st.write(f"**{index}.** {result}")
        else:
            st.info("No relevant passages found. Enter both a document and question, then try different keywords.")

if __name__ == "__main__":
    main()

