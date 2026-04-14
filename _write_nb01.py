"""Temporary script to write Project 01 notebook. Delete after use."""
import json

nb_path = r"E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\Beginner_Local_LLM_Apps\01_Local_PDF_QA_Tutor\notebook.ipynb"

cells = []

def md(src):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

def code(src):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source})

# ============================================================
# Cell 0: Title + Overview
# ============================================================
md("""# Project 1 — Local PDF Q&A Tutor

## Chat with PDFs Using Local Embeddings and Ollama

**Goal:** Build a complete local RAG pipeline that loads a PDF, chunks it, embeds it,
stores vectors in ChromaDB, and answers questions with source citations — all running locally.

**Stack:** Ollama · LangChain · ChromaDB · PyPDF · fpdf2 · Jupyter

```
PDF ──► PyPDFLoader ──► pages ──► TextSplitter ──► chunks
                                                      │
                                          Ollama Embeddings
                                                      │
                                                      ▼
                                                  ChromaDB
                                                      │
User question ──► embed ──► similarity search ──► top chunks ──► LLM ──► Answer
```

### What You'll Learn

1. Load and parse PDFs with **PyPDF**
2. Split documents into overlapping chunks for better retrieval
3. Create embeddings with a **local Ollama** model
4. Store and search vectors in **ChromaDB**
5. Build a retrieval-augmented QA chain with **LangChain**
6. Inspect retrieved chunks and analyze **failure cases**

### Prerequisites

- **Ollama** installed and running (`ollama serve`)
- Models pulled: `ollama pull nomic-embed-text` and `ollama pull qwen3:8b`
- Python 3.9+""")

# ============================================================
# Cell 1: Install
# ============================================================
code("""# Install dependencies (uncomment and run once)
# !pip install -q langchain langchain-ollama langchain-community chromadb pypdf fpdf2""")

# ============================================================
# Cell 2: Step 1 markdown
# ============================================================
md("""## Step 1 — Verify Ollama Is Running

Before we do anything, let's make sure Ollama is reachable and the required models are available.
If this cell fails, start Ollama with `ollama serve` and pull the models.""")

# ============================================================
# Cell 3: Verify Ollama code
# ============================================================
code("""import requests, sys

OLLAMA_BASE = "http://localhost:11434"

try:
    r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"✅ Ollama is running — {len(models)} model(s) available")
    for m in models:
        print(f"   • {m}")
except Exception as e:
    print(f"❌ Cannot reach Ollama at {OLLAMA_BASE}: {e}")
    print("\\nFix: start Ollama and run:")
    print("  ollama pull qwen3:8b")
    print("  ollama pull nomic-embed-text")""")

# ============================================================
# Cell 4: Step 2 markdown
# ============================================================
md("""## Step 2 — Configure LLM and Embeddings

We use two Ollama models:
- **nomic-embed-text** — converts text into dense vectors for similarity search
- **qwen3:8b** — the instruction-following LLM that generates answers

Both run entirely on your local machine. No API keys, no cloud calls.""")

# ============================================================
# Cell 5: Configure code
# ============================================================
code("""from langchain_ollama import ChatOllama, OllamaEmbeddings

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"

llm = ChatOllama(model=LLM_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# Quick smoke test — verify both models respond
vec = embeddings.embed_query("test")
print(f"✅ Embedding model ready — dimension: {len(vec)}")

resp = llm.invoke("Say 'hello' in one word.")
print(f"✅ LLM ready — response: {resp.content[:120]}")""")

# ============================================================
# Cell 6: Step 3 markdown
# ============================================================
md("""## Step 3 — Create a Sample PDF

To keep this notebook **fully self-contained**, we generate a small 4-page PDF
about machine learning topics using `fpdf2`. In a real project you would
point `PyPDFLoader` at your own PDF files.

The PDF has four chapters so we can later test whether retrieval finds the
correct page for each question.""")

# ============================================================
# Cell 7: Create PDF code
# ============================================================
code("""from fpdf import FPDF
from pathlib import Path

SAMPLE_DIR = Path("sample_data")
SAMPLE_DIR.mkdir(exist_ok=True)

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

chapters = {
    "Chapter 1: Machine Learning Fundamentals": (
        "Machine learning is a branch of artificial intelligence that enables "
        "computers to learn patterns from data without being explicitly programmed "
        "for every task.\\n\\n"
        "Supervised Learning uses labeled training data. Each example has input "
        "features and a known output label. The model learns to map inputs to "
        "outputs. Common algorithms include Linear Regression for continuous "
        "targets, Logistic Regression for binary classification, Decision Trees "
        "that split data on feature thresholds, and Random Forests that combine "
        "many trees to reduce overfitting.\\n\\n"
        "Unsupervised Learning works with unlabeled data. The algorithm discovers "
        "hidden patterns on its own. K-Means clustering groups similar data points. "
        "Principal Component Analysis reduces dimensionality while preserving "
        "variance. Anomaly detection identifies unusual data points.\\n\\n"
        "Key Concepts: Features are measurable input properties. Labels are target "
        "values the model predicts. Overfitting happens when a model memorizes "
        "training data but fails on new data. Cross-validation estimates how well "
        "a model generalizes."
    ),
    "Chapter 2: Deep Learning and Neural Networks": (
        "Neural networks are computing systems inspired by biological brains. They "
        "consist of layers of artificial neurons that process information.\\n\\n"
        "Architecture: A basic neural network has an input layer, hidden layers, "
        "and an output layer. Each neuron computes a weighted sum of inputs, adds "
        "a bias, and applies an activation function such as ReLU, sigmoid, or tanh.\\n\\n"
        "Convolutional Neural Networks (CNNs) are specialized for grid-like data "
        "such as images. They use convolutional layers with learnable filters to "
        "detect edges, textures, and shapes. Pooling layers reduce spatial dimensions. "
        "CNNs excel at image classification, object detection, and segmentation.\\n\\n"
        "Recurrent Neural Networks (RNNs) handle sequential data where order matters. "
        "They maintain a hidden state carrying information across time steps. Standard "
        "RNNs suffer from the vanishing gradient problem with long sequences. Long "
        "Short-Term Memory networks (LSTMs) solve this with gating mechanisms that "
        "control what to remember and forget.\\n\\n"
        "Training uses backpropagation to compute gradients of the loss function "
        "with respect to each weight. Gradient descent adjusts weights to minimize "
        "loss. Learning rate, batch size, and epochs are key hyperparameters."
    ),
    "Chapter 3: Transformers and Large Language Models": (
        "The Transformer architecture, introduced in the 2017 paper Attention Is "
        "All You Need, revolutionized NLP by replacing recurrence with self-attention.\\n\\n"
        "Self-Attention allows each token in a sequence to attend to every other token, "
        "computing relevance scores. This captures long-range dependencies without "
        "the sequential bottleneck of RNNs. Multi-head attention runs multiple attention "
        "operations in parallel for different relationship types.\\n\\n"
        "BERT (Bidirectional Encoder Representations from Transformers) uses masked "
        "language modeling to learn deep bidirectional representations. It reads text "
        "in both directions simultaneously and is fine-tuned for question answering, "
        "sentiment analysis, and named entity recognition.\\n\\n"
        "GPT (Generative Pre-trained Transformer) uses autoregressive modeling, "
        "predicting the next token given all previous tokens. GPT models generate "
        "coherent text and each version increased in size and capability.\\n\\n"
        "Scaling Laws show that model performance improves predictably with more "
        "parameters, data, and compute. This drove development of billion-parameter "
        "models, though they require significant memory, compute, and energy."
    ),
    "Chapter 4: Retrieval-Augmented Generation (RAG)": (
        "RAG is a technique that combines information retrieval with text generation "
        "to produce more accurate and grounded responses.\\n\\n"
        "The Problem: Large language models sometimes generate plausible but incorrect "
        "information, known as hallucination. They cannot access real-time or private "
        "data beyond their training set. Fine-tuning is expensive and must be repeated "
        "when data changes.\\n\\n"
        "The RAG Solution: RAG retrieves relevant information from an external "
        "knowledge base before generating a response. The retrieved context is "
        "included in the prompt, grounding the model in actual source material. "
        "This reduces hallucination and enables access to current or private data.\\n\\n"
        "Key Components: Document chunking splits large documents into smaller, "
        "overlapping pieces for retrieval. Embedding models convert text into dense "
        "vectors capturing semantic meaning. Vector databases store embeddings and "
        "enable fast similarity search. The retriever finds relevant chunks for a "
        "query. The generator LLM produces a final answer from retrieved context.\\n\\n"
        "Applications include document QA, customer support, research assistance, "
        "and enterprise search. Quality depends on chunk size, embedding quality, "
        "retrieval strategy, and prompt design."
    ),
}

for title, body in chapters.items():
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title)
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 11)
    for para in body.split("\\n\\n"):
        pdf.multi_cell(0, 6, para.strip())
        pdf.ln(4)

pdf_path = SAMPLE_DIR / "ml_textbook.pdf"
pdf.output(str(pdf_path))
print(f"✅ PDF created: {pdf_path} ({pdf_path.stat().st_size:,} bytes, {len(chapters)} pages)")""")

# ============================================================
# Cell 8: Step 4 markdown
# ============================================================
md("""## Step 4 — Load and Parse the PDF

`PyPDFLoader` extracts text from each page and preserves **page-level metadata**.
This metadata lets us cite which page an answer came from.""")

# ============================================================
# Cell 9: Load PDF code
# ============================================================
code("""from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(str(pdf_path))
pages = loader.load()

print(f"Loaded {len(pages)} pages\\n")
for i, page in enumerate(pages):
    print(f"  Page {i+1}: {len(page.page_content)} chars")
    print(f"    metadata: {page.metadata}")
    print(f"    preview:  {page.page_content[:80]}...\\n")""")

# ============================================================
# Cell 10: Step 5 markdown
# ============================================================
md("""## Step 5 — Split Pages into Chunks

**Why chunking matters:** Embedding models have limited context windows, and
retrieval works better with focused, smaller pieces of text.

`RecursiveCharacterTextSplitter` tries to split on natural boundaries:
paragraph breaks first, then lines, then sentences, then words.

**Overlap** keeps some text from the previous chunk so context around
split boundaries is not lost.""")

# ============================================================
# Cell 11: Chunk code
# ============================================================
code("""from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\\n\\n", "\\n", ". ", " "],
)
chunks = splitter.split_documents(pages)

total_chars = sum(len(p.page_content) for p in pages)
print(f"Split {len(pages)} pages ({total_chars:,} chars) -> {len(chunks)} chunks\\n")
for i, c in enumerate(chunks):
    pg = int(c.metadata.get('page', 0)) + 1
    print(f"  Chunk {i:2d} | Page {pg} | {len(c.page_content):3d} chars | {c.page_content[:60]}...")""")

# ============================================================
# Cell 12: Step 6 markdown
# ============================================================
md("""## Step 6 — Create the Vector Store

**ChromaDB** stores each chunk as a vector (embedding) and enables fast
similarity search. When we ask a question later, the question is also
embedded and the closest chunk-vectors are returned.

`persist_directory` saves the database to disk so you don't have to
re-embed everything each time.""")

# ============================================================
# Cell 13: Vector store code
# ============================================================
code("""from langchain_community.vectorstores import Chroma
import shutil

CHROMA_DIR = str(SAMPLE_DIR / "chroma_db")

# Start fresh each run to avoid stale data
if Path(CHROMA_DIR).exists():
    shutil.rmtree(CHROMA_DIR)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name="pdf_qa_tutor",
)

print(f"✅ Vector store created — {vectorstore._collection.count()} vectors stored")""")

# ============================================================
# Cell 14: Step 7 markdown
# ============================================================
md("""## Step 7 — Test Similarity Search

Before building the full QA chain, let's verify that retrieval works.
We embed a question and find the closest chunks. The **score** is the
distance — lower means more similar.""")

# ============================================================
# Cell 15: Similarity search code
# ============================================================
code("""test_queries = [
    "What is RAG?",
    "How do CNNs work?",
    "What is self-attention?",
]

for query in test_queries:
    print(f"\\nQuery: '{query}'")
    results = vectorstore.similarity_search_with_score(query, k=2)
    for i, (doc, score) in enumerate(results):
        pg = int(doc.metadata.get('page', 0)) + 1
        print(f"   [{i+1}] score={score:.4f} | Page {pg} | {doc.page_content[:90]}...")""")

# ============================================================
# Cell 16: Step 8 markdown
# ============================================================
md("""## Step 8 — Build the RAG Chain

We combine the retriever and the LLM into a single **retrieval chain**.
The chain:
1. Takes the user's question
2. Retrieves the top-k most relevant chunks
3. Stuffs them into the system prompt as `{context}`
4. Asks the LLM to answer using **only** that context

This prevents the LLM from making things up — if the answer isn't in
the retrieved context, it should say so.""")

# ============================================================
# Cell 17: RAG chain code
# ============================================================
code("""from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are a helpful tutor. Answer the question using ONLY the provided context. "
    "If the answer is not in the context, say: "
    "'I don't have enough information in the provided material.'\\n\\n"
    "Context:\\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("✅ RAG chain ready!")""")

# ============================================================
# Cell 18: Step 9 markdown
# ============================================================
md("""## Step 9 — Ask Questions with Citations

Now let's test the full pipeline end-to-end. For each question we show:
- The **answer** generated by the LLM
- The **source chunks** that were retrieved and used as context""")

# ============================================================
# Cell 19: Q&A demo code
# ============================================================
code("""questions = [
    "What is the difference between CNNs and RNNs?",
    "How does RAG reduce hallucination?",
    "What is self-attention in transformers?",
    "What are the types of machine learning?",
]

for q in questions:
    print(f"\\n{'='*70}")
    print(f"Q: {q}")
    result = rag_chain.invoke({"input": q})
    print(f"\\nA: {result['answer']}")
    print(f"\\nSources ({len(result['context'])} chunks retrieved):")
    for i, doc in enumerate(result["context"]):
        pg = int(doc.metadata.get('page', 0)) + 1
        print(f"   [{i+1}] Page {pg}: {doc.page_content[:80]}...")""")

# ============================================================
# Cell 20: Step 10 markdown
# ============================================================
md("""## Step 10 — Explore Failure Cases

A well-designed RAG system should **refuse to answer** when the retrieved
context doesn't contain the information. Let's test with questions that
are clearly outside our PDF's content.

This is important because it shows the LLM is **grounded** — it won't
hallucinate answers from its own training data.""")

# ============================================================
# Cell 21: Failure cases code
# ============================================================
code("""out_of_scope = [
    "What is the capital of France?",
    "Compare PyTorch and TensorFlow.",
    "Who invented the internet?",
]

print("=== Out-of-Scope Questions (should decline to answer) ===\\n")
for q in out_of_scope:
    result = rag_chain.invoke({"input": q})
    answer = result["answer"]
    print(f"Q: {q}")
    print(f"A: {answer}")
    grounded = any(phrase in answer.lower() for phrase in [
        "don't have enough", "not in the context", "no information",
        "not mentioned", "does not contain", "cannot find",
    ])
    print(f"   Grounded refusal: {'Yes' if grounded else 'No (model may have hallucinated)'}\\n")""")

# ============================================================
# Cell 22: Step 11 markdown
# ============================================================
md("""## Step 11 — Interactive Q&A Helper

A reusable `ask()` function you can call with any question.
Try your own questions about the ML textbook!""")

# ============================================================
# Cell 23: Interactive helper code
# ============================================================
code("""def ask(question: str) -> str:
    \"\"\"Ask the PDF Q&A tutor a question and display the answer with sources.\"\"\"
    result = rag_chain.invoke({"input": question})
    answer = result["answer"]
    sources = result["context"]

    print(f"Answer:\\n{answer}\\n")
    print(f"Sources ({len(sources)} chunks):")
    for i, doc in enumerate(sources):
        pg = int(doc.metadata.get('page', 0)) + 1
        print(f"   [{i+1}] Page {pg}: {doc.page_content[:80]}...")
    return answer

# Try it!
_ = ask("What are the key components of a RAG system?")""")

# ============================================================
# Cell 24: Limitations markdown
# ============================================================
md("""## Limitations & Tradeoffs

| Aspect | What happens | How to improve |
|--------|-------------|----------------|
| **Chunk size** | Too small: fragments lose context. Too large: dilutes relevance | Experiment with 300-1000 char chunks |
| **Embedding quality** | Generic embeddings may miss domain nuance | Use domain-tuned or fine-tuned embeddings |
| **Top-k** | Too few: misses relevant info. Too many: adds noise | Try k=2 to k=5 and compare answers |
| **PDF parsing** | Complex layouts (tables, columns) may extract poorly | Use specialized parsers for complex PDFs |
| **Hallucination** | The LLM may still guess if the prompt is not strict enough | Strengthen the system prompt constraints |
| **Thinking tags** | qwen3 may include reasoning in output | Use post-processing or a non-thinking model |

### What this project does NOT cover
- Multi-PDF collections (see Project 13)
- Reranking retrieved results (see Project 23)
- Hybrid search combining keywords and vectors (see Project 21)
- Production deployment (this is a learning notebook)""")

# ============================================================
# Cell 25: What You Learned markdown
# ============================================================
md("""## What You Learned

1. **PDF loading** — `PyPDFLoader` extracts text page-by-page with metadata
2. **Chunking** — `RecursiveCharacterTextSplitter` creates overlapping pieces for retrieval
3. **Local embeddings** — Ollama `nomic-embed-text` converts text to vectors locally
4. **Vector store** — ChromaDB stores and searches embeddings by similarity
5. **RAG chain** — LangChain's `create_retrieval_chain` wires retrieval to LLM in one call
6. **Grounding** — the system prompt constrains the LLM to answer from context only
7. **Failure analysis** — testing out-of-scope questions reveals grounding quality

## Exercises

1. **Swap the PDF** — replace the sample with your own PDF and run the pipeline
2. **Change chunk size** — try 300 and 1000 and compare retrieval quality
3. **Change k** — set `search_kwargs={"k": 5}` and see if answers improve or get noisier
4. **Try a different model** — swap `qwen3:8b` for `llama3.1:8b` or `mistral:7b`
5. **Add a quiz mode** — use the LLM to generate questions from the PDF, then answer them

---

*Next project: **02 — Local Markdown Knowledge Bot** (indexing markdown notes with LlamaIndex)*""")

# ============================================================
# Write the notebook
# ============================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Written {len(cells)} cells to {nb_path}")

