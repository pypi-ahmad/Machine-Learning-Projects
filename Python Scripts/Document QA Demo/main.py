"""Document Q&A Demo — Streamlit app. Simple keyword-based document search.
Dependencies: pip install streamlit
Usage: streamlit run main.py
"""
import re, streamlit as st

def search_document(text, query):
    sentences = re.split(r'[.!?]+', text)
    query_words = set(query.lower().split())
    results = []
    for s in sentences:
        s = s.strip()
        if not s: continue
        s_words = set(s.lower().split())
        overlap = len(query_words & s_words)
        if overlap > 0:
            results.append((overlap, s))
    results.sort(key=lambda x: -x[0])
    return [r[1] for r in results[:5]]

SAMPLE_DOC = """Python is a high-level programming language. It was created by Guido van Rossum and released in 1991. Python emphasizes code readability with significant indentation. It supports multiple paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, machine learning, and automation. The Python Package Index hosts thousands of third-party modules. Python 3 is the current major version, with Python 2 reaching end of life in 2020."""

def main():
    st.set_page_config(page_title="Document Q&A Demo")
    st.title("Document Q&A Demo")
    st.markdown("Paste a document and ask questions about it.")
    doc = st.text_area("Document", SAMPLE_DOC, height=200)
    query = st.text_input("Ask a question")
    if query and doc:
        results = search_document(doc, query)
        st.header("Relevant Passages")
        if results:
            for i, r in enumerate(results, 1):
                st.write(f"**{i}.** {r}")
        else:
            st.write("No relevant passages found.")

if __name__ == "__main__":
    main()

