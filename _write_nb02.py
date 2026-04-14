"""Temporary script to write Project 02 notebook. Delete after use."""
import json

nb_path = r"E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\Beginner_Local_LLM_Apps\02_Local_Markdown_Knowledge_Bot\notebook.ipynb"

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
md("""# Project 2 — Local Markdown Knowledge Bot

## Query Your Markdown Notes with LlamaIndex + Ollama

**Goal:** Build a local knowledge search engine over markdown files using
LlamaIndex for indexing/retrieval and Ollama for embeddings and generation.

**Stack:** Ollama · LlamaIndex · Jupyter

```
Markdown files ──► SimpleDirectoryReader ──► Documents
                                                │
                                     Ollama Embeddings
                                                │
                                                ▼
                                       VectorStoreIndex
                                                │
User query ──► embed ──► similarity search ──► top docs ──► LLM ──► Answer
```

### What You'll Learn

1. Load and parse markdown files with **LlamaIndex's SimpleDirectoryReader**
2. Build a **VectorStoreIndex** for semantic search
3. Query your knowledge base with **natural language**
4. Add **metadata** (topic, file name) for better retrieval
5. Use **chat mode** with conversation memory
6. Analyze **failure cases** and retrieval quality

### Prerequisites

- **Ollama** installed and running (`ollama serve`)
- Models pulled: `ollama pull nomic-embed-text` and `ollama pull qwen3:8b`
- Python 3.9+""")

# ============================================================
code("""# Install dependencies (uncomment and run once)
# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama""")

# ============================================================
md("""## Step 1 — Verify Ollama Is Running

LlamaIndex will call Ollama for both embeddings and generation.
Let's verify it's reachable before proceeding.""")

# ============================================================
code("""import requests, sys

OLLAMA_BASE = "http://localhost:11434"

try:
    r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"Ollama is running — {len(models)} model(s) available")
    for m in models:
        print(f"   - {m}")
except Exception as e:
    print(f"Cannot reach Ollama at {OLLAMA_BASE}: {e}")
    print("\\nFix: start Ollama and run:")
    print("  ollama pull qwen3:8b")
    print("  ollama pull nomic-embed-text")""")

# ============================================================
md("""## Step 2 — Create Sample Markdown Notes

We create three markdown files covering different developer topics.
Each file represents a "note" in your personal knowledge base.

In a real project, these would be your actual notes, wiki pages,
or exported documents.""")

# ============================================================
code("""from pathlib import Path

notes_dir = Path("sample_notes")
notes_dir.mkdir(exist_ok=True)

notes = {
    "python_tips.md": \"\"\"# Python Tips

## List Comprehensions
List comprehensions are a concise way to create lists:
`squares = [x**2 for x in range(10)]`

They can include conditions: `evens = [x for x in range(20) if x % 2 == 0]`

## Context Managers
Use `with` statements for resource management. They ensure cleanup happens
even if exceptions occur. Common for file handling, database connections,
and network sockets.

```python
with open("data.txt", "r") as f:
    content = f.read()
```

## Type Hints
Type hints improve code readability and enable better IDE support:
`def greet(name: str) -> str:` Python does not enforce them at runtime
but tools like mypy use them for static analysis.

## Virtual Environments
Always use virtual environments to isolate project dependencies.
`python -m venv .venv` creates one, and `source .venv/bin/activate`
activates it on Linux/Mac.
\"\"\",

    "git_workflow.md": \"\"\"# Git Workflow Guide

## Branching Strategy
- `main` — production-ready code only
- `develop` — integration branch for features
- `feature/*` — new features branched from develop
- `hotfix/*` — urgent production fixes branched from main

## Common Commands
- `git rebase -i HEAD~3` — interactive rebase last 3 commits
- `git stash` — temporarily save uncommitted changes
- `git cherry-pick <hash>` — apply a specific commit to current branch
- `git bisect` — binary search to find the commit that introduced a bug

## Commit Messages
Write clear commit messages. Use the imperative mood:
"Add feature" not "Added feature". Keep the subject line under 50 characters.
Add a body for complex changes explaining why, not just what.

## Pull Request Best Practices
Keep PRs small and focused. Add a description explaining the change.
Request reviews from relevant team members. Respond to feedback promptly.
\"\"\",

    "docker_basics.md": \"\"\"# Docker Basics

## Key Concepts
- **Image**: Read-only template with instructions for creating a container
- **Container**: Runnable instance of an image, isolated from the host
- **Dockerfile**: Script that defines how to build an image step by step
- **Volume**: Persistent data storage that survives container restarts
- **Network**: Virtual network for container-to-container communication

## Essential Commands
```bash
docker build -t myapp .          # Build image from Dockerfile
docker run -p 8080:80 myapp      # Run container with port mapping
docker-compose up -d             # Start multi-container app in background
docker ps                        # List running containers
docker logs <container_id>       # View container logs
```

## Dockerfile Best Practices
Use multi-stage builds to keep images small. Pin base image versions.
Copy requirements first to leverage layer caching. Use .dockerignore
to exclude unnecessary files from the build context.

## Docker Compose
Docker Compose defines multi-container applications in a single YAML file.
It handles networking, volumes, and environment variables automatically.
\"\"\",
}

for fname, content in notes.items():
    (notes_dir / fname).write_text(content, encoding="utf-8")

print(f"Created {len(notes)} sample markdown notes in {notes_dir}/")
for f in sorted(notes_dir.glob("*.md")):
    print(f"  - {f.name}: {f.stat().st_size:,} bytes")""")

# ============================================================
md("""## Step 3 — Configure LlamaIndex with Local Ollama

LlamaIndex uses a global `Settings` object to configure which LLM and
embedding model to use. We point both to our local Ollama instance.

- **Embed model** — converts text to vectors for similarity search
- **LLM** — generates natural language answers from retrieved context""")

# ============================================================
code("""from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

print("LlamaIndex configured with local Ollama models")""")

# ============================================================
md("""## Step 4 — Load Documents and Build the Index

`SimpleDirectoryReader` reads all files from a directory and returns
them as LlamaIndex `Document` objects, each with text content and
metadata (file name, path, etc.).

`VectorStoreIndex` embeds each document chunk and builds an in-memory
vector index for fast similarity search.""")

# ============================================================
code("""documents = SimpleDirectoryReader("sample_notes").load_data()

print(f"Loaded {len(documents)} documents\\n")
for doc in documents:
    fname = doc.metadata.get("file_name", "unknown")
    print(f"  - {fname}: {len(doc.text)} chars")
    print(f"    preview: {doc.text[:80]}...\\n")

# Build the vector index
index = VectorStoreIndex.from_documents(documents, show_progress=True)
print("\\nVector index built successfully!")""")

# ============================================================
md("""## Step 5 — Query Your Knowledge Base

The `query_engine` takes a natural language question, finds the most
relevant document chunks via vector similarity, and asks the LLM to
generate an answer grounded in those chunks.

We also inspect which source files contributed to each answer.""")

# ============================================================
code("""query_engine = index.as_query_engine(similarity_top_k=3)

queries = [
    "How do I use list comprehensions in Python?",
    "What is the Git branching strategy?",
    "How do I build a Docker image?",
    "What are context managers used for?",
]

for q in queries:
    print(f"\\n{'='*60}")
    print(f"Q: {q}")
    response = query_engine.query(q)
    print(f"A: {response}")
    sources = [n.metadata.get("file_name", "?") for n in response.source_nodes]
    print(f"Sources: {sources}")""")

# ============================================================
md("""## Step 6 — Add Metadata for Better Retrieval

By enriching documents with **metadata** (topic, tags, file type),
we can later filter or weight results. This is especially useful
when your knowledge base grows beyond a handful of files.

Here we extract a topic name from the file name and add it as metadata.""")

# ============================================================
code("""from llama_index.core import Document

enriched_docs = []
for doc in documents:
    fname = doc.metadata.get("file_name", "")
    topic = fname.replace(".md", "").replace("_", " ").title()
    enriched_docs.append(Document(
        text=doc.text,
        metadata={"topic": topic, "file": fname, "type": "developer_notes"}
    ))

enriched_index = VectorStoreIndex.from_documents(enriched_docs)
enriched_qe = enriched_index.as_query_engine(similarity_top_k=2)

resp = enriched_qe.query("What are essential Docker commands?")
print(f"Answer: {resp}\\n")
print("Source metadata:")
for node in resp.source_nodes:
    print(f"  - topic: {node.metadata.get('topic', '?')}, file: {node.metadata.get('file', '?')}")""")

# ============================================================
md("""## Step 7 — Test Failure Cases

Let's ask questions that are **outside** our notes to see how the bot
handles them. A well-designed system should indicate when it cannot
find relevant information rather than hallucinating answers.""")

# ============================================================
code("""out_of_scope = [
    "What is Kubernetes?",
    "How does React server-side rendering work?",
    "Explain quantum computing basics.",
]

print("=== Out-of-Scope Questions ===\\n")
for q in out_of_scope:
    response = query_engine.query(q)
    answer = str(response)
    print(f"Q: {q}")
    print(f"A: {answer[:200]}...")
    # Check retrieval relevance
    scores = [n.score for n in response.source_nodes if n.score is not None]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"   Avg relevance score: {avg_score:.4f}")
    print()""")

# ============================================================
md("""## Step 8 — Compare Retrieval Quality

Let's inspect how well the retriever picks the right document for
different query types. We check if the top source file matches
what we'd expect.""")

# ============================================================
code("""test_cases = [
    ("How do I create a virtual environment?", "python_tips.md"),
    ("What is git bisect?", "git_workflow.md"),
    ("What is a Docker volume?", "docker_basics.md"),
    ("How do I write good commit messages?", "git_workflow.md"),
]

correct = 0
for query, expected_file in test_cases:
    response = query_engine.query(query)
    top_source = response.source_nodes[0].metadata.get("file_name", "?") if response.source_nodes else "?"
    match = top_source == expected_file
    correct += int(match)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] Q: {query}")
    print(f"         Expected: {expected_file} | Got: {top_source}")

print(f"\\nRetrieval accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")""")

# ============================================================
md("""## Limitations & Tradeoffs

| Aspect | What happens | How to improve |
|--------|-------------|----------------|
| **Small corpus** | Few files means limited retrieval diversity | Add more notes and test at scale |
| **No heading awareness** | Chunks may split across headings | Use heading-based splitter |
| **Generic embeddings** | May miss domain-specific jargon | Consider domain-tuned embeddings |
| **No persistent index** | Index is rebuilt every run | Use persistent vector store |
| **Single format** | Only markdown files | Add support for txt, rst, org files |

### What this project does NOT cover
- Persistent vector stores (see Project 17)
- Hybrid search (see Project 21)
- Cross-file link tracking (stretch goal)
- Production deployment""")

# ============================================================
md("""## What You Learned

1. **Markdown loading** — `SimpleDirectoryReader` parses files with metadata
2. **Vector indexing** — `VectorStoreIndex` embeds and indexes document chunks
3. **Semantic search** — natural language queries find relevant content by meaning
4. **Metadata enrichment** — adding topic/file info improves retrieval transparency
5. **Failure analysis** — testing out-of-scope queries reveals system boundaries
6. **Retrieval evaluation** — checking if the right source is retrieved

## Exercises

1. **Add more notes** — create 5+ markdown files on different topics and reindex
2. **Heading-based splitting** — experiment with splitting documents at heading boundaries
3. **Persistent storage** — save the index to disk and reload without re-embedding
4. **Try different k values** — compare `similarity_top_k=1` vs `3` vs `5`
5. **Add tags** — include tags in metadata and filter queries by tag

---

*Next project: **03 — Local Meeting Notes Summarizer** (structured summarization with Ollama)*""")

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

print(f"Written {len(cells)} cells to {nb_path}")

