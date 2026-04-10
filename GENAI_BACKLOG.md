# GenAI & Agentic AI — 100 Project Backlog

> **Scope**: 100 local-first learning projects covering GenAI, RAG, agents, multi-agent systems, evaluation, fine-tuning prep, multimodal, and developer tooling.
>
> **Repo context**: Continues from existing projects GenAI/01–10 and RAG/11–16. New projects will be numbered **17–116** in the repo directory structure.
>
> **Created**: April 2026

---

## Environment Defaults

| Component | Default |
|---|---|
| **LLM runtime** | Ollama @ `localhost:11434` |
| **Chat model** | `qwen3.5:9b` (or latest local-compatible Qwen) |
| **Embedding model** | `nomic-embed-text-v2-moe` via `OllamaEmbeddings` |
| **VLM** | `llava:13b` or `minicpm-v` via Ollama |
| **Vector store** | ChromaDB (default) · FAISS (alternative) |
| **Python** | 3.11 |
| **GPU** | CUDA (local GPU) |
| **Format** | Jupyter notebooks (unless stated otherwise) |
| **Cloud APIs** | Never required · marked *optional extension* when mentioned |

---

## Naming Convention

```
{category}/
  {NUM:03d}_{Project_Name}/
    {project_name_snake_case}.ipynb
```

Categories for the new projects:

| Repo Folder | Backlog Group |
|---|---|
| `GenAI/` | Groups A–B (beginner LLM apps, local RAG) |
| `RAG/` | Group C (advanced RAG & retrieval engineering) |
| `GenAI/` | Groups D–J (LangGraph, CrewAI, agents, eval, fine-tune, multimodal, coding, advanced) |

---

## Group A — Beginner Local LLM Apps (Projects 1–10)

These build foundational skills: calling local LLMs, prompt engineering, output parsing, and chain composition.

---

### A-01 · Local PDF Q&A Tutor

| Field | Detail |
|---|---|
| **Backlog #** | 1 |
| **Repo path** | `GenAI/017_Local_PDF_QA_Tutor/` |
| **One-line idea** | Chat with any PDF using local embeddings and Ollama — with a teaching focus on explaining how each RAG step works |
| **Learning outcome** | Understand the full RAG pipeline: load → chunk → embed → store → retrieve → generate |
| **Stack** | LangChain, Ollama, ChromaDB or FAISS, PyPDF, Jupyter |
| **Local model** | `qwen3.5:9b` (chat) · `nomic-embed-text-v2-moe` (embeddings) |
| **Difficulty** | Beginner |
| **Why useful** | The fundamental building block of every RAG system; every GenAI practitioner must understand this end-to-end |
| **Notebook path** | 1) Install/import deps → 2) Load PDF with PyPDFLoader → 3) Chunk with RecursiveCharacterTextSplitter → 4) Embed and store in Chroma → 5) Build retriever → 6) Create RetrievalQA chain → 7) Ask 3–5 questions and display answers with source pages → 8) Key Concepts Recap |
| **Stretch goal** | Add a follow-up chain that generates quiz questions from retrieved context |

---

### A-02 · Local Markdown Knowledge Bot

| Field | Detail |
|---|---|
| **Backlog #** | 2 |
| **Repo path** | `GenAI/018_Local_Markdown_Knowledge_Bot/` |
| **One-line idea** | Query your own markdown notes and docs with a local LLM |
| **Learning outcome** | Learn to use LlamaIndex's SimpleDirectoryReader and local Ollama integration for document QA |
| **Stack** | LlamaIndex, Ollama, local vector store, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Beginner |
| **Why useful** | Teaches an alternative to LangChain (LlamaIndex) and works with the files developers already have |
| **Notebook path** | 1) Install LlamaIndex + Ollama integration → 2) Create sample markdown files → 3) Load with SimpleDirectoryReader → 4) Build VectorStoreIndex → 5) Query engine with local LLM → 6) Compare answers across markdown files → 7) Key Concepts Recap |
| **Stretch goal** | Add metadata filters to restrict search to specific folders or tags |

---

### A-03 · Local Meeting Notes Summarizer

| Field | Detail |
|---|---|
| **Backlog #** | 3 |
| **Repo path** | `GenAI/019_Local_Meeting_Notes_Summarizer/` |
| **One-line idea** | Summarize meeting transcripts into action items, decisions, and blockers |
| **Learning outcome** | Structured output extraction from unstructured text using prompt templates and output parsers |
| **Stack** | Ollama, LangChain, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Beginner |
| **Why useful** | Practical daily-use skill; teaches structured prompting and chain composition |
| **Notebook path** | 1) Create sample meeting transcripts (2–3 examples) → 2) Build prompt template for structured extraction → 3) Use StrOutputParser for plain summary → 4) Use JsonOutputParser for structured output (actions, decisions, blockers) → 5) Compare outputs across transcript lengths → 6) Key Concepts Recap |
| **Stretch goal** | Add a follow-up chain that generates a Slack-style update from the structured summary |

---

### A-04 · Local Resume Rewriter

| Field | Detail |
|---|---|
| **Backlog #** | 4 |
| **Repo path** | `GenAI/020_Local_Resume_Rewriter/` |
| **One-line idea** | Improve resume bullet points and tailor them to a target role |
| **Learning outcome** | Multi-step chain composition: analyze → rewrite → refine |
| **Stack** | Ollama, LangChain, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Beginner |
| **Why useful** | Immediate personal utility; teaches iterative prompt refinement and multi-chain pipelines |
| **Notebook path** | 1) Define sample resume bullets and target JD → 2) Build analysis chain (identify weak points) → 3) Build rewrite chain (improve bullets) → 4) Build tailoring chain (align with JD keywords) → 5) Run full pipeline end-to-end → 6) Compare before/after → 7) Key Concepts Recap |
| **Stretch goal** | Add scoring of bullet impact using a second LLM-as-judge call |

---

### A-05 · Local Cover Letter Generator

| Field | Detail |
|---|---|
| **Backlog #** | 5 |
| **Repo path** | `GenAI/021_Local_Cover_Letter_Generator/` |
| **One-line idea** | Generate tailored cover letters from a job description and resume |
| **Learning outcome** | Multi-input prompt templates and chaining context from multiple sources |
| **Stack** | Ollama, LangChain, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Beginner |
| **Why useful** | Teaches combining multiple document inputs into a single generation prompt |
| **Notebook path** | 1) Define sample resume and JD → 2) Build JD analysis chain (extract key requirements) → 3) Build cover letter prompt with {resume}, {jd_analysis}, {company_name} variables → 4) Generate 2–3 variants with different tones → 5) Side-by-side comparison → 6) Key Concepts Recap |
| **Stretch goal** | Add a critique chain that scores each variant on relevance and specificity |

---

### A-06 · Local Email Reply Assistant

| Field | Detail |
|---|---|
| **Backlog #** | 6 |
| **Repo path** | `GenAI/022_Local_Email_Reply_Assistant/` |
| **One-line idea** | Classify email intent and draft appropriate replies with structured output |
| **Learning outcome** | Pydantic-structured LLM outputs and conditional chain routing |
| **Stack** | Ollama, LangChain, Pydantic, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Beginner |
| **Why useful** | Teaches structured output parsing — a critical production skill |
| **Notebook path** | 1) Create 5+ sample emails of different types → 2) Build classification chain with Pydantic model (intent, urgency, sentiment) → 3) Build reply drafting chain conditioned on classification → 4) Test across all email types → 5) Display classification + draft side by side → 6) Key Concepts Recap |
| **Stretch goal** | Add tone selection (formal / friendly / brief) as an additional parameter |

---

### A-07 · Local Research Paper Explainer

| Field | Detail |
|---|---|
| **Backlog #** | 7 |
| **Repo path** | `GenAI/023_Local_Research_Paper_Explainer/` |
| **One-line idea** | Explain research papers in plain English using local PDF parsing and LLM |
| **Learning outcome** | Long-document handling: section extraction, map-reduce summarization |
| **Stack** | Ollama, LlamaIndex, PyPDF, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Beginner |
| **Why useful** | Teaches strategies for handling documents that exceed context windows |
| **Notebook path** | 1) Load a research paper PDF → 2) Extract sections (abstract, methods, results, conclusion) → 3) Summarize each section individually → 4) Build plain-English explanation from section summaries → 5) Generate "key takeaways" list → 6) Key Concepts Recap |
| **Stretch goal** | Add a "ELI5" mode that explains at different complexity levels |

---

### A-08 · Local Blog-to-Thread Converter

| Field | Detail |
|---|---|
| **Backlog #** | 8 |
| **Repo path** | `GenAI/024_Local_Blog_To_Thread_Converter/` |
| **One-line idea** | Repurpose a blog article into a Twitter/X thread, LinkedIn post, and email newsletter |
| **Learning outcome** | Content transformation chains with format-specific constraints |
| **Stack** | Ollama, LangChain, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Beginner |
| **Why useful** | Practical content repurposing; teaches per-format prompt engineering |
| **Notebook path** | 1) Paste/load a sample blog post → 2) Build thread chain (numbered tweets, max 280 chars each) → 3) Build LinkedIn chain (professional tone, 1300 char limit) → 4) Build email newsletter chain (subject + body) → 5) Display all 3 outputs → 6) Key Concepts Recap |
| **Stretch goal** | Add a hook-quality scorer that rates the opening of each format |

---

### A-09 · Local Study Notes Generator

| Field | Detail |
|---|---|
| **Backlog #** | 9 |
| **Repo path** | `GenAI/025_Local_Study_Notes_Generator/` |
| **One-line idea** | Turn raw text or lecture content into organized study notes and quiz questions |
| **Learning outcome** | Multi-output generation: notes, flashcards, and quizzes from a single source |
| **Stack** | Ollama, LangChain, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Beginner |
| **Why useful** | Teaches generating multiple structured outputs from one input with different prompt templates |
| **Notebook path** | 1) Provide raw lecture text → 2) Build notes extraction chain (key concepts, definitions) → 3) Build flashcard chain (Q&A pairs as JSON) → 4) Build quiz chain (multiple choice with answer key) → 5) Display all outputs formatted → 6) Key Concepts Recap |
| **Stretch goal** | Add spaced-repetition scheduling metadata to flashcards |

---

### A-10 · Local Code Explainer

| Field | Detail |
|---|---|
| **Backlog #** | 10 |
| **Repo path** | `GenAI/026_Local_Code_Explainer/` |
| **One-line idea** | Explain code snippets, detect issues, and suggest improvements locally |
| **Learning outcome** | Code-aware prompting and structured analysis output |
| **Stack** | Ollama, LangChain, Pydantic, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Beginner |
| **Why useful** | Teaches prompting for technical/code domains and structured bug reports |
| **Notebook path** | 1) Define 3–5 code snippets (Python, JS, SQL) → 2) Build explanation chain → 3) Build bug detection chain with Pydantic output (severity, line, description) → 4) Build improvement suggestion chain → 5) Run full pipeline on each snippet → 6) Key Concepts Recap |
| **Stretch goal** | Add a translation chain that converts the snippet to another language |

---

## Group B — Local RAG Applications (Projects 11–20)

Build real-world RAG applications across domains: policy, finance, legal, medical, course material, and customer support.

---

### B-01 · Local Website FAQ Bot

| Field | Detail |
|---|---|
| **Backlog #** | 11 |
| **Repo path** | `GenAI/027_Local_Website_FAQ_Bot/` |
| **One-line idea** | Ingest one website and answer questions grounded in its content |
| **Learning outcome** | Web scraping → chunking → embedding → RAG with source citations |
| **Stack** | Ollama, LangChain, BeautifulSoup or Trafilatura, ChromaDB, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Beginner |
| **Why useful** | Extends RAG beyond PDFs to live web content; teaches HTML-aware text extraction |
| **Notebook path** | 1) Pick a public docs site → 2) Scrape and clean HTML → 3) Chunk text → 4) Embed and store in Chroma → 5) Build retriever + QA chain → 6) Ask 5+ questions and show answers with source URLs → 7) Key Concepts Recap |
| **Stretch goal** | Crawl multiple sub-pages and track source URL per chunk |

---

### B-02 · Local Policy Assistant

| Field | Detail |
|---|---|
| **Backlog #** | 12 |
| **Repo path** | `RAG/017_Local_Policy_Assistant/` |
| **One-line idea** | Search HR/IT/company policies and return answers with exact citations |
| **Learning outcome** | Citation-grounded RAG with metadata-enriched retrieval |
| **Stack** | Ollama, LangChain, ChromaDB, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Beginner |
| **Why useful** | Enterprise-relevant pattern; teaches metadata attachment and citation extraction |
| **Notebook path** | 1) Create 3–4 sample policy docs (HR, IT, Travel) → 2) Load and tag with doc_type metadata → 3) Chunk and embed → 4) Build QA chain that returns answer + exact source section → 5) Test cross-policy questions → 6) Key Concepts Recap |
| **Stretch goal** | Add a follow-up chain that flags conflicting policies across documents |

---

### B-03 · Local Multi-PDF Research Librarian

| Field | Detail |
|---|---|
| **Backlog #** | 13 |
| **Repo path** | `RAG/018_Local_Multi_PDF_Research_Librarian/` |
| **One-line idea** | Answer questions across multiple research papers with evidence from each |
| **Learning outcome** | Multi-document RAG with per-document attribution |
| **Stack** | Ollama, LlamaIndex, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches multi-source retrieval, deduplication, and evidence aggregation |
| **Notebook path** | 1) Load 3+ PDFs into LlamaIndex → 2) Build sub-indices per document → 3) Create a composable graph index → 4) Query across all papers → 5) Display answer with per-paper evidence snippets → 6) Key Concepts Recap |
| **Stretch goal** | Add a comparison chain that contrasts findings across papers on the same topic |

---

### B-04 · Local Financial Report Analyst

| Field | Detail |
|---|---|
| **Backlog #** | 14 |
| **Repo path** | `RAG/019_Local_Financial_Report_Analyst/` |
| **One-line idea** | QA over annual reports and financial filings mixing text and tables |
| **Learning outcome** | Hybrid text+table parsing for RAG; handling structured data in documents |
| **Stack** | Ollama, LangChain, tabular extraction (camelot or pdfplumber), Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Real-world financial docs contain both prose and tables; teaches multi-modal document parsing |
| **Notebook path** | 1) Load a sample annual report PDF → 2) Extract text sections and tables separately → 3) Chunk text and serialize tables → 4) Embed both into Chroma with type metadata → 5) Build QA chain → 6) Test with questions requiring table data → 7) Key Concepts Recap |
| **Stretch goal** | Add a calculation verification step that checks if LLM math matches table values |

---

### B-05 · Local Contract Clause Finder

| Field | Detail |
|---|---|
| **Backlog #** | 15 |
| **Repo path** | `RAG/020_Local_Contract_Clause_Finder/` |
| **One-line idea** | Retrieve risky or important clauses from contracts using semantic search |
| **Learning outcome** | Domain-specific retrieval with Haystack pipelines |
| **Stack** | Haystack, Ollama, local document store, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Introduces Haystack as an alternative RAG framework; teaches legal-domain retrieval patterns |
| **Notebook path** | 1) Install Haystack → 2) Create sample contracts → 3) Build Haystack document store → 4) Build retrieval pipeline with embedding retriever → 5) Query for specific clause types (indemnity, termination, IP) → 6) Rank by risk score → 7) Key Concepts Recap |
| **Stretch goal** | Add a clause comparison chain that diffs similar clauses across contracts |

---

### B-06 · Local Course Tutor

| Field | Detail |
|---|---|
| **Backlog #** | 16 |
| **Repo path** | `RAG/021_Local_Course_Tutor/` |
| **One-line idea** | QA over lecture notes, slides, and course material with topic-aware retrieval |
| **Learning outcome** | Multi-format document ingestion (PDF slides, markdown notes, text) into a unified index |
| **Stack** | Ollama, LangChain, ChromaDB, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Beginner |
| **Why useful** | Teaches handling mixed document formats in a single RAG pipeline |
| **Notebook path** | 1) Create sample course materials (lecture notes, slides summary, reading list) → 2) Load with format-specific loaders → 3) Tag metadata (week, topic, format) → 4) Embed and store → 5) Query with topic filtering → 6) Key Concepts Recap |
| **Stretch goal** | Generate a study guide that covers all topics from the indexed material |

---

### B-07 · Local Personal Wiki Copilot

| Field | Detail |
|---|---|
| **Backlog #** | 17 |
| **Repo path** | `GenAI/028_Local_Personal_Wiki_Copilot/` |
| **One-line idea** | Query exported Obsidian/Notion/wiki files with semantic search |
| **Learning outcome** | LlamaIndex document readers and knowledge graph–style retrieval |
| **Stack** | Ollama, LlamaIndex, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Beginner |
| **Why useful** | Personal productivity tool; teaches directory-based ingestion with link-aware parsing |
| **Notebook path** | 1) Create sample wiki-style markdown files with `[[wikilinks]]` → 2) Load with SimpleDirectoryReader → 3) Build index → 4) Query across notes → 5) Show answer with source notes → 6) Key Concepts Recap |
| **Stretch goal** | Parse `[[wikilinks]]` to build a simple knowledge graph overlay |

---

### B-08 · Local Customer Support Memory Bot

| Field | Detail |
|---|---|
| **Backlog #** | 18 |
| **Repo path** | `RAG/022_Local_Customer_Support_Memory_Bot/` |
| **One-line idea** | Retrieve similar past tickets and fixes to assist with new support requests |
| **Learning outcome** | Similarity search for case-based reasoning; embedding-driven ticket retrieval |
| **Stack** | Ollama, LangChain, ChromaDB, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Beginner |
| **Why useful** | Common enterprise pattern; teaches embedding similarity as a retrieval mechanism |
| **Notebook path** | 1) Create sample support ticket dataset (20+ tickets with resolutions) → 2) Embed tickets → 3) For a new ticket, retrieve top-k similar → 4) Generate suggested resolution from similar cases → 5) Display matches with similarity scores → 6) Key Concepts Recap |
| **Stretch goal** | Add a feedback loop that updates ticket embeddings when a resolution is confirmed |

---

### B-09 · Local Product Docs Copilot

| Field | Detail |
|---|---|
| **Backlog #** | 19 |
| **Repo path** | `RAG/023_Local_Product_Docs_Copilot/` |
| **One-line idea** | Chat over internal product or API documentation with doc-type-aware retrieval |
| **Learning outcome** | Metadata-filtered retrieval and doc-type routing |
| **Stack** | Ollama, LangChain, ChromaDB, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Beginner |
| **Why useful** | Builds on existing pattern (project 16) with more advanced metadata filtering |
| **Notebook path** | 1) Create sample docs (API ref, guides, changelog, FAQ) → 2) Load with doc_type metadata → 3) Embed and store → 4) Build retriever with metadata filter → 5) Route questions to appropriate doc types → 6) Key Concepts Recap |
| **Stretch goal** | Add version-aware retrieval to handle multiple doc versions |

---

### B-10 · Local Medical Literature Finder

| Field | Detail |
|---|---|
| **Backlog #** | 20 |
| **Repo path** | `RAG/024_Local_Medical_Literature_Finder/` |
| **One-line idea** | Search medical papers by topic, study type, and evidence level |
| **Learning outcome** | Advanced metadata filtering and faceted search in vector stores |
| **Stack** | Ollama, LlamaIndex, metadata filters, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches structured metadata schemas and filter-based retrieval for domain-specific applications |
| **Notebook path** | 1) Create sample abstracts with metadata (study_type, year, evidence_level) → 2) Build LlamaIndex with metadata schema → 3) Query with filters (e.g., "RCTs after 2020 on diabetes") → 4) Rank by evidence level → 5) Generate summary of findings → 6) Key Concepts Recap |
| **Stretch goal** | Add a contradicting-evidence detector across papers |

---

## Group C — Advanced RAG & Retrieval Engineering (Projects 21–30)

Deep-dive into retrieval quality: hybrid search, query rewriting, reranking, compression, multi-hop, evaluation.

---

### C-01 · Hybrid Retrieval Lab

| Field | Detail |
|---|---|
| **Backlog #** | 21 |
| **Repo path** | `RAG/025_Hybrid_Retrieval_Lab/` |
| **One-line idea** | Compare BM25, dense, and hybrid retrieval strategies side-by-side |
| **Learning outcome** | Understand sparse vs dense vs hybrid retrieval and when each wins |
| **Stack** | Haystack, Ollama, rank_bm25, ChromaDB, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Retrieval strategy selection is the highest-leverage decision in RAG systems |
| **Notebook path** | 1) Create a test corpus (50+ passages) with ground-truth Q&A pairs → 2) Implement BM25 retriever → 3) Implement dense retriever (embeddings) → 4) Implement hybrid (reciprocal rank fusion) → 5) Evaluate all 3 on recall@k → 6) Visualize results → 7) Key Concepts Recap |
| **Stretch goal** | Add a learned fusion weight tuning step |

---

### C-02 · Query Rewriting RAG Lab

| Field | Detail |
|---|---|
| **Backlog #** | 22 |
| **Repo path** | `RAG/026_Query_Rewriting_RAG_Lab/` |
| **One-line idea** | Rewrite vague user questions before retrieval to improve results |
| **Learning outcome** | Query expansion, HyDE (Hypothetical Document Embeddings), and multi-query retrieval |
| **Stack** | LangChain, DSPy, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Query quality is the #1 bottleneck in RAG; teaches rewriting techniques |
| **Notebook path** | 1) Create test queries (vague, ambiguous, complex) with ground-truth docs → 2) Implement basic query pass-through → 3) Implement LLM query rewriting → 4) Implement HyDE → 5) Implement multi-query expansion → 6) Compare retrieval recall for each → 7) Key Concepts Recap |
| **Stretch goal** | Use DSPy to optimize the rewriting prompt automatically |

---

### C-03 · Retrieval Reranking Lab

| Field | Detail |
|---|---|
| **Backlog #** | 23 |
| **Repo path** | `RAG/027_Retrieval_Reranking_Lab/` |
| **One-line idea** | Compare no-rerank vs cross-encoder reranking on retrieval quality |
| **Learning outcome** | Two-stage retrieval (retrieve → rerank) with local cross-encoder models |
| **Stack** | Local retriever, sentence-transformers cross-encoder, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` · `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| **Difficulty** | Intermediate |
| **Why useful** | Reranking is the easiest high-impact improvement to any RAG system |
| **Notebook path** | 1) Create corpus with ground-truth relevance labels → 2) Retrieve top-20 with dense retriever → 3) Rerank with local cross-encoder → 4) Compare recall@5 and MRR before/after → 5) Visualize rank changes → 6) Key Concepts Recap |
| **Stretch goal** | Compare multiple cross-encoder models on speed vs accuracy |

---

### C-04 · Context Compression RAG

| Field | Detail |
|---|---|
| **Backlog #** | 24 |
| **Repo path** | `RAG/028_Context_Compression_RAG/` |
| **One-line idea** | Compress large retrieved context before feeding to the answer-generation LLM |
| **Learning outcome** | Context window management: extractive compression, LLM-based summarization, and LongContextReorder |
| **Stack** | LangChain, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Essential for production RAG where retrieved content may exceed useful context size |
| **Notebook path** | 1) Build a RAG pipeline that retrieves 10+ chunks → 2) Baseline: stuff all chunks → 3) Implement extractive compression (keep relevant sentences only) → 4) Implement LLM summarization of context → 5) Compare answer quality and latency → 6) Key Concepts Recap |
| **Stretch goal** | Implement LongContextReorder (important chunks first and last) |

---

### C-05 · Multi-Hop RAG Research Agent

| Field | Detail |
|---|---|
| **Backlog #** | 25 |
| **Repo path** | `RAG/029_Multi_Hop_RAG_Research_Agent/` |
| **One-line idea** | Use multiple retrieval hops before answering complex questions |
| **Learning outcome** | Iterative retrieval-generation loops with LangGraph state management |
| **Stack** | LangGraph, Ollama, ChromaDB, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Advanced |
| **Why useful** | Many real questions require chaining evidence; teaches iterative RAG patterns |
| **Notebook path** | 1) Create a corpus requiring multi-hop reasoning → 2) Build single-hop baseline → 3) Build LangGraph graph with retrieve → assess → decide (more hops?) → answer → 4) Compare single-hop vs multi-hop answers → 5) Visualize the agent's hop trace → 6) Key Concepts Recap |
| **Stretch goal** | Add a "confident enough" threshold that stops retrieval early |

---

### C-06 · Table + Text Local RAG

| Field | Detail |
|---|---|
| **Backlog #** | 26 |
| **Repo path** | `RAG/030_Table_Text_Local_RAG/` |
| **One-line idea** | Combine CSV data and text documents in a unified RAG pipeline |
| **Learning outcome** | Multi-modal data indexing: natural language over structured and unstructured data |
| **Stack** | LlamaIndex, Ollama, pandas, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Real-world data is rarely just text; teaches unified indexing strategies |
| **Notebook path** | 1) Load sample CSV and markdown docs → 2) Serialize table rows as natural language → 3) Build unified LlamaIndex index → 4) Query with questions that span both sources → 5) Show which source type contributed to each answer → 6) Key Concepts Recap |
| **Stretch goal** | Add SQL querying as a fallback for precise numerical questions |

---

### C-07 · Freshness-Aware News RAG

| Field | Detail |
|---|---|
| **Backlog #** | 27 |
| **Repo path** | `RAG/031_Freshness_Aware_News_RAG/` |
| **One-line idea** | Prioritize recent documents in retrieval for time-sensitive queries |
| **Learning outcome** | Time-decay scoring, metadata-based filtering, and freshness-aware ranking |
| **Stack** | LangChain, metadata filters, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | News and event data requires recency awareness; teaches temporal retrieval patterns |
| **Notebook path** | 1) Create news articles with timestamps → 2) Embed with date metadata → 3) Implement time-decay scoring (recency bonus) → 4) Compare time-naive vs time-aware retrieval → 5) Test with "latest" vs "historical" queries → 6) Key Concepts Recap |
| **Stretch goal** | Add automatic time-sensitivity detection in queries |

---

### C-08 · Multilingual Local RAG

| Field | Detail |
|---|---|
| **Backlog #** | 28 |
| **Repo path** | `RAG/032_Multilingual_Local_RAG/` |
| **One-line idea** | Retrieve documents in one language and answer in another |
| **Learning outcome** | Cross-lingual embeddings and multilingual generation |
| **Stack** | Ollama, multilingual embeddings (e.g., `multilingual-e5-large` via sentence-transformers), Jupyter |
| **Local model** | `qwen3.5:9b` (multilingual capable) · multilingual embedding model |
| **Difficulty** | Intermediate |
| **Why useful** | Real-world knowledge bases often span languages; teaches cross-lingual retrieval |
| **Notebook path** | 1) Create docs in 2–3 languages → 2) Embed with multilingual model → 3) Query in English, retrieve from any language → 4) Generate answer in query language → 5) Compare cross-lingual vs monolingual retrieval → 6) Key Concepts Recap |
| **Stretch goal** | Add language detection to auto-select the answer language |

---

### C-09 · Citation Verifier for RAG

| Field | Detail |
|---|---|
| **Backlog #** | 29 |
| **Repo path** | `RAG/033_Citation_Verifier_RAG/` |
| **One-line idea** | Check whether a RAG answer is actually supported by the retrieved chunks |
| **Learning outcome** | Faithfulness evaluation: NLI-based and LLM-as-judge verification |
| **Stack** | LangChain, Ollama, local eval notebook, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Hallucination detection is critical for trustworthy RAG; teaches eval techniques |
| **Notebook path** | 1) Build a RAG pipeline → 2) Generate answers for 10+ questions → 3) Implement sentence-level citation checking (does chunk support claim?) → 4) Use LLM-as-judge to score faithfulness → 5) Flag unsupported claims → 6) Visualize faithfulness scores → 7) Key Concepts Recap |
| **Stretch goal** | Build an auto-correction chain that revises unfaithful answers |

---

### C-10 · RAG Evaluation Dashboard Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 30 |
| **Repo path** | `RAG/034_RAG_Evaluation_Dashboard/` |
| **One-line idea** | Compare chunking strategies, retrieval methods, and groundedness in one evaluation notebook |
| **Learning outcome** | End-to-end RAG evaluation: retrieval metrics, generation quality, and parameter sensitivity |
| **Stack** | LangChain, Ollama, matplotlib, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Advanced |
| **Why useful** | Teaches systematic RAG optimization — the skill that separates demos from production systems |
| **Notebook path** | 1) Create evaluation dataset (questions + ground-truth answers + relevant docs) → 2) Sweep chunk_size and chunk_overlap → 3) Sweep retriever k values → 4) Measure recall, precision, faithfulness, answer relevance → 5) Plot parameter sensitivity curves → 6) Summarize best config → 7) Key Concepts Recap |
| **Stretch goal** | Add automated prompt variant comparison to the sweep |

---

## Group D — LangGraph Workflows (Projects 31–40)

Stateful, multi-step agent workflows with human-in-the-loop, routing, checkpointing, and memory.

---

### D-01 · LangGraph Human Approval Workflow

| Field | Detail |
|---|---|
| **Backlog #** | 31 |
| **Repo path** | `GenAI/029_LangGraph_Human_Approval_Workflow/` |
| **One-line idea** | Build an agent that pauses execution and waits for human approval before proceeding |
| **Learning outcome** | LangGraph fundamentals: StateGraph, nodes, edges, conditional routing, and interrupt/resume |
| **Stack** | LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Human-in-the-loop is a core pattern for safe agentic systems |
| **Notebook path** | 1) Install LangGraph → 2) Define state schema → 3) Build nodes: draft → review → approve/reject → finalize → 4) Add conditional edge for human approval simulation → 5) Run graph with sample data → 6) Visualize graph execution → 7) Key Concepts Recap |
| **Stretch goal** | Add multi-level approval (manager → director) with escalation logic |

---

### D-02 · LangGraph Multi-Step Sales Research Flow

| Field | Detail |
|---|---|
| **Backlog #** | 32 |
| **Repo path** | `GenAI/030_LangGraph_Sales_Research_Flow/` |
| **One-line idea** | Company lookup → competitor analysis → pain points → outreach email draft |
| **Learning outcome** | Multi-node sequential workflows with state accumulation |
| **Stack** | LangGraph, Ollama, local tools (mock company data), Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches real-world multi-step workflows with dependent state |
| **Notebook path** | 1) Define state (company_info, competitors, pain_points, draft) → 2) Build nodes for each step → 3) Chain nodes in sequence → 4) Run for a sample company → 5) Display accumulated state at each step → 6) Key Concepts Recap |
| **Stretch goal** | Add a feedback loop that revises the outreach based on tone analysis |

---

### D-03 · LangGraph Incident Summary Flow

| Field | Detail |
|---|---|
| **Backlog #** | 33 |
| **Repo path** | `GenAI/031_LangGraph_Incident_Summary_Flow/` |
| **One-line idea** | Parse raw logs → extract incident details → generate summary → suggest next steps |
| **Learning outcome** | Error-handling in graphs, structured extraction, and conditional branching |
| **Stack** | LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Incident management is a high-value automation target; teaches structured log processing |
| **Notebook path** | 1) Create sample incident logs → 2) Build parsing node → 3) Build severity classification node (low/medium/high/critical) → 4) Add conditional branch by severity → 5) Build summary + next-steps generation node → 6) Run for multiple incidents → 7) Key Concepts Recap |
| **Stretch goal** | Add a root-cause hypothesis generator node |

---

### D-04 · LangGraph Data Cleaning Approval Flow

| Field | Detail |
|---|---|
| **Backlog #** | 34 |
| **Repo path** | `GenAI/032_LangGraph_Data_Cleaning_Flow/` |
| **One-line idea** | Agent suggests data cleaning transforms; human reviews and approves each |
| **Learning outcome** | Tool-calling agents with pandas and human-in-the-loop approval |
| **Stack** | LangGraph, pandas, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches safe automation: AI suggests, human decides |
| **Notebook path** | 1) Load dirty CSV dataset → 2) Build analysis node (detect nulls, types, outliers) → 3) Build suggestion node (propose transforms) → 4) Build approval node (simulate human review) → 5) Build apply node (execute approved transforms) → 6) Show before/after state → 7) Key Concepts Recap |
| **Stretch goal** | Add rollback capability if applied transform degrades data quality |

---

### D-05 · LangGraph Resume Tailoring Flow

| Field | Detail |
|---|---|
| **Backlog #** | 35 |
| **Repo path** | `GenAI/033_LangGraph_Resume_Tailoring_Flow/` |
| **One-line idea** | Parse JD → analyze requirements → tailor resume → draft cover letter in one graph |
| **Learning outcome** | Complex state graphs with parallel and sequential node execution |
| **Stack** | LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Practical personal utility; teaches graph composition for multi-output workflows |
| **Notebook path** | 1) Define state (jd, resume, analysis, tailored_resume, cover_letter) → 2) Build JD analysis node → 3) Build resume tailoring node → 4) Build cover letter node → 5) Add quality check node → 6) Run full graph → 7) Display all outputs → 8) Key Concepts Recap |
| **Stretch goal** | Add a scoring node that rates the tailored resume against the JD |

---

### D-06 · LangGraph Procurement Review Flow

| Field | Detail |
|---|---|
| **Backlog #** | 36 |
| **Repo path** | `GenAI/034_LangGraph_Procurement_Review_Flow/` |
| **One-line idea** | Compare vendor proposals, score them, and generate a recommendation summary |
| **Learning outcome** | Multi-input comparison workflows with scoring rubrics |
| **Stack** | LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches structured comparison and scoring patterns used in business decision support |
| **Notebook path** | 1) Create 3 sample vendor proposals → 2) Build extraction node (price, features, SLAs) → 3) Build scoring node (weighted rubric) → 4) Build comparison node → 5) Build recommendation summary node → 6) Run and display ranked results → 7) Key Concepts Recap |
| **Stretch goal** | Add sensitivity analysis (what if feature weights change?) |

---

### D-07 · LangGraph Travel Planner Flow

| Field | Detail |
|---|---|
| **Backlog #** | 37 |
| **Repo path** | `GenAI/035_LangGraph_Travel_Planner_Flow/` |
| **One-line idea** | Gather preferences → generate itinerary → revise with checkpoints → finalize |
| **Learning outcome** | LangGraph checkpointing, state persistence, and iterative refinement loops |
| **Stack** | LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches checkpointing and iterative agent loops — critical for long-running workflows |
| **Notebook path** | 1) Build preference gathering node → 2) Build itinerary generation node → 3) Build review node → 4) Add revision loop with checkpoint → 5) Build finalization node → 6) Run with 2 revision cycles → 7) Show state at each checkpoint → 8) Key Concepts Recap |
| **Stretch goal** | Add budget tracking across itinerary revisions |

---

### D-08 · LangGraph Research Workflow with Memory

| Field | Detail |
|---|---|
| **Backlog #** | 38 |
| **Repo path** | `GenAI/036_LangGraph_Research_Memory_Workflow/` |
| **One-line idea** | Accumulate research findings over multiple sessions using persistent memory |
| **Learning outcome** | Long-term memory patterns in LangGraph: MemorySaver and persistent state |
| **Stack** | LangGraph, persistence (SQLite checkpointer), Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Advanced |
| **Why useful** | Memory is what separates toy demos from useful agents; teaches state persistence |
| **Notebook path** | 1) Set up SQLite-based LangGraph checkpointer → 2) Build research node (generate finding) → 3) Build accumulation node (merge with existing findings) → 4) Build synthesis node → 5) Run across 3 "sessions" showing memory accumulation → 6) Query accumulated knowledge → 7) Key Concepts Recap |
| **Stretch goal** | Add a memory pruning strategy for old or low-relevance findings |

---

### D-09 · LangGraph Ticket Escalation Router

| Field | Detail |
|---|---|
| **Backlog #** | 39 |
| **Repo path** | `GenAI/037_LangGraph_Ticket_Escalation_Router/` |
| **One-line idea** | Auto-classify and resolve simple tickets; escalate complex ones |
| **Learning outcome** | Conditional routing and early-exit patterns in agent graphs |
| **Stack** | LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Routing patterns are essential for production agent systems |
| **Notebook path** | 1) Create sample tickets (easy, medium, hard) → 2) Build classification node → 3) Build auto-resolution node (for simple tickets) → 4) Build escalation node (for complex tickets) → 5) Add conditional routing by complexity → 6) Run batch of tickets → 7) Show resolution vs escalation stats → 8) Key Concepts Recap |
| **Stretch goal** | Add confidence thresholding to the classification decision |

---

### D-10 · LangGraph Compliance Checklist Flow

| Field | Detail |
|---|---|
| **Backlog #** | 40 |
| **Repo path** | `GenAI/038_LangGraph_Compliance_Checklist_Flow/` |
| **One-line idea** | Gather evidence, check against requirements, and generate a compliance checklist |
| **Learning outcome** | Multi-step evidence gathering with structured validation |
| **Stack** | LangGraph, local tools, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Advanced |
| **Why useful** | Compliance workflows are high-value enterprise automation targets |
| **Notebook path** | 1) Define compliance requirements (5–8 items) → 2) Build evidence gathering node → 3) Build verification node (check each requirement) → 4) Build gap analysis node → 5) Build checklist generation node → 6) Run end-to-end → 7) Display pass/fail/partial for each requirement → 8) Key Concepts Recap |
| **Stretch goal** | Add a remediation suggestion generator for failed checks |

---

## Group E — CrewAI Multi-Agent Systems (Projects 41–50)

Multi-agent collaboration with specialized roles, delegation, and structured crew workflows.

---

### E-01 · CrewAI Startup Validation Crew

| Field | Detail |
|---|---|
| **Backlog #** | 41 |
| **Repo path** | `GenAI/039_CrewAI_Startup_Validation_Crew/` |
| **One-line idea** | Market researcher, competitor analyst, pricing strategist, and critic agents validate a startup idea |
| **Learning outcome** | CrewAI fundamentals: agents, tasks, crews, sequential process, and role specialization |
| **Stack** | CrewAI, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches the foundational multi-agent pattern and how role specialization improves output quality |
| **Notebook path** | 1) Install CrewAI → 2) Define 4 agents with roles and backstories → 3) Define tasks for each agent → 4) Create crew with sequential process → 5) Run for a sample startup idea → 6) Display each agent's output → 7) Key Concepts Recap |
| **Stretch goal** | Add a "devil's advocate" agent that challenges the crew's conclusion |

---

### E-02 · CrewAI Content Studio

| Field | Detail |
|---|---|
| **Backlog #** | 42 |
| **Repo path** | `GenAI/040_CrewAI_Content_Studio/` |
| **One-line idea** | Researcher → writer → editor → repurposer agents create polished content |
| **Learning outcome** | Task chaining in CrewAI with output dependencies |
| **Stack** | CrewAI, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Content pipelines are one of the most practical multi-agent use cases |
| **Notebook path** | 1) Define 4 agents (researcher, writer, editor, repurposer) → 2) Chain tasks with context passing → 3) Run crew for a blog topic → 4) Display output at each stage → 5) Compare final vs first-draft quality → 6) Key Concepts Recap |
| **Stretch goal** | Add a fact-checker agent before publishing |

---

### E-03 · CrewAI Lead Gen Crew

| Field | Detail |
|---|---|
| **Backlog #** | 43 |
| **Repo path** | `GenAI/041_CrewAI_Lead_Gen_Crew/` |
| **One-line idea** | ICP definer, company researcher, personalization agent, and email drafter work together for outbound sales |
| **Learning outcome** | CrewAI with structured inputs/outputs and inter-task dependencies |
| **Stack** | CrewAI, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches parameterized multi-agent workflows with real business value |
| **Notebook path** | 1) Define ICP agent → 2) Define company research agent → 3) Define personalization agent → 4) Define email drafting agent → 5) Run crew for 2–3 target companies → 6) Display personalized emails → 7) Key Concepts Recap |
| **Stretch goal** | Add A/B variant generation for email subject lines |

---

### E-04 · CrewAI Job Hunt Crew

| Field | Detail |
|---|---|
| **Backlog #** | 44 |
| **Repo path** | `GenAI/042_CrewAI_Job_Hunt_Crew/` |
| **One-line idea** | JD analyzer, resume tailor, interview coach, and networking advisor collaborate |
| **Learning outcome** | Multi-agent with diverse output formats (structured analysis, prose, Q&A) |
| **Stack** | CrewAI, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Personally useful; teaches handling different output formats across agents |
| **Notebook path** | 1) Define 4 agents → 2) JD analysis task → 3) Resume tailoring task (depends on JD analysis) → 4) Interview prep task (mock Q&A) → 5) Networking advice task → 6) Run crew → 7) Display all outputs → 8) Key Concepts Recap |
| **Stretch goal** | Add salary negotiation strategy agent |

---

### E-05 · CrewAI Academic Research Crew

| Field | Detail |
|---|---|
| **Backlog #** | 45 |
| **Repo path** | `GenAI/043_CrewAI_Academic_Research_Crew/` |
| **One-line idea** | Literature searcher, summarizer, gap finder, and bibliography builder collaborate on a research topic |
| **Learning outcome** | CrewAI hierarchical process and tool-using agents |
| **Stack** | CrewAI, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches hierarchical crew processes where a manager coordinates specialists |
| **Notebook path** | 1) Define agents with hierarchical process → 2) Literature search task → 3) Summarization task → 4) Gap analysis task → 5) Bibliography task → 6) Run with manager agent → 7) Display structured research output → 8) Key Concepts Recap |
| **Stretch goal** | Add a research proposal generator as the final agent |

---

### E-06 · CrewAI Product Launch Crew

| Field | Detail |
|---|---|
| **Backlog #** | 46 |
| **Repo path** | `GenAI/044_CrewAI_Product_Launch_Crew/` |
| **One-line idea** | Product manager, marketer, data analyst, and QA agents plan a product launch |
| **Learning outcome** | CrewAI with custom tools and structured deliverables |
| **Stack** | CrewAI, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches creating custom tools for CrewAI agents |
| **Notebook path** | 1) Define 4 agents → 2) Create custom tools (market_size_estimator, competitor_checker) → 3) Assign tools to agents → 4) Define launch plan tasks → 5) Run crew → 6) Display launch plan with all deliverables → 7) Key Concepts Recap |
| **Stretch goal** | Add a risk assessment agent that reviews the full plan |

---

### E-07 · CrewAI Competitor Intelligence Crew

| Field | Detail |
|---|---|
| **Backlog #** | 47 |
| **Repo path** | `GenAI/045_CrewAI_Competitor_Intelligence_Crew/` |
| **One-line idea** | Feature analyst, pricing researcher, news watcher, and memo writer produce intelligence reports |
| **Learning outcome** | CrewAI with memory and knowledge sharing between agents |
| **Stack** | CrewAI, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches agent memory and cross-agent knowledge sharing |
| **Notebook path** | 1) Define 4 specialist agents → 2) Enable crew memory → 3) Define research tasks → 4) Build memo compilation task → 5) Run crew with shared memory → 6) Display intelligence memo → 7) Key Concepts Recap |
| **Stretch goal** | Add trend detection across multiple crew runs |

---

### E-08 · CrewAI Customer Success Crew

| Field | Detail |
|---|---|
| **Backlog #** | 48 |
| **Repo path** | `GenAI/046_CrewAI_Customer_Success_Crew/` |
| **One-line idea** | Complaint analyst, churn risk scorer, and response writer handle customer issues |
| **Learning outcome** | CrewAI with conditional task routing based on agent output |
| **Stack** | CrewAI, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches routing decisions within a crew based on intermediate results |
| **Notebook path** | 1) Define 3 agents → 2) Complaint analysis task → 3) Churn risk scoring task → 4) Route: high-risk → retention response vs low-risk → standard response → 5) Run for batch of customer scenarios → 6) Display routing decisions and responses → 7) Key Concepts Recap |
| **Stretch goal** | Add sentiment trend tracking across multiple complaints |

---

### E-09 · CrewAI Recruiting Crew

| Field | Detail |
|---|---|
| **Backlog #** | 49 |
| **Repo path** | `GenAI/047_CrewAI_Recruiting_Crew/` |
| **One-line idea** | Resume screener, interviewer question generator, and candidate summarizer automate recruiting |
| **Learning outcome** | CrewAI with file I/O tools and structured evaluation rubrics |
| **Stack** | CrewAI, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches structured evaluation criteria and rubric-based scoring in multi-agent systems |
| **Notebook path** | 1) Create sample resumes and JD → 2) Build screening agent with rubric → 3) Build interview question agent → 4) Build summary agent → 5) Run crew for 3–4 candidates → 6) Display ranked candidate summaries → 7) Key Concepts Recap |
| **Stretch goal** | Add a diversity-awareness check agent |

---

### E-10 · CrewAI Ops Review Crew

| Field | Detail |
|---|---|
| **Backlog #** | 50 |
| **Repo path** | `GenAI/048_CrewAI_Ops_Review_Crew/` |
| **One-line idea** | Operations analyst, risk reviewer, and summary agent produce weekly ops reviews |
| **Learning outcome** | CrewAI with templated outputs and repeatable workflows |
| **Stack** | CrewAI, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches repeatable workflow design and templated report generation |
| **Notebook path** | 1) Create sample ops data (metrics, incidents, changes) → 2) Build ops analyst agent → 3) Build risk reviewer agent → 4) Build report writer agent with template → 5) Run crew → 6) Display formatted weekly review → 7) Key Concepts Recap |
| **Stretch goal** | Add week-over-week comparison by persisting prior reports |

---

## Group F — Local Tool-Using Agents (Projects 51–60)

Agents that interact with the real world: files, databases, spreadsheets, browsers, and CLIs.

---

### F-01 · Local Web Research Agent

| Field | Detail |
|---|---|
| **Backlog #** | 51 |
| **Repo path** | `GenAI/049_Local_Web_Research_Agent/` |
| **One-line idea** | Search the web, compare sources, and write a cited answer — all locally |
| **Learning outcome** | Tool-calling agents with web search integration |
| **Stack** | LangChain or PydanticAI, Ollama, DuckDuckGo search, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches building agents that interact with external information sources |
| **Notebook path** | 1) Set up DuckDuckGo search tool → 2) Define agent with search + scrape tools → 3) Build research prompt → 4) Run agent on 3 research questions → 5) Display cited answers with sources → 6) Key Concepts Recap |
| **Stretch goal** | Add source credibility scoring |

---

### F-02 · Local Spreadsheet Analyst Agent

| Field | Detail |
|---|---|
| **Backlog #** | 52 |
| **Repo path** | `GenAI/050_Local_Spreadsheet_Analyst_Agent/` |
| **One-line idea** | Answer natural language questions over CSV/XLSX files and generate insights |
| **Learning outcome** | Agents with pandas tool calling and data analysis |
| **Stack** | LangChain, pandas tool, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Natural-language data analysis is one of the most requested GenAI capabilities |
| **Notebook path** | 1) Load sample CSV → 2) Create pandas agent with code execution tool → 3) Ask analytical questions (aggregations, trends, outliers) → 4) Display generated code + results → 5) Generate insight summary → 6) Key Concepts Recap |
| **Stretch goal** | Add chart generation from natural language queries |

---

### F-03 · Local SQL Analyst Agent

| Field | Detail |
|---|---|
| **Backlog #** | 53 |
| **Repo path** | `GenAI/051_Local_SQL_Analyst_Agent/` |
| **One-line idea** | Natural language to SQL query to human-readable summary |
| **Learning outcome** | Text-to-SQL generation with validation and result interpretation |
| **Stack** | LangChain, SQLite, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Text-to-SQL is a core enterprise use case; teaches safe code generation patterns |
| **Notebook path** | 1) Create sample SQLite database → 2) Build schema inspection tool → 3) Build NL-to-SQL chain → 4) Add SQL validation step → 5) Execute and interpret results → 6) Test with 5+ queries of varying complexity → 7) Key Concepts Recap |
| **Stretch goal** | Add query explanation chain that describes what the SQL does |

---

### F-04 · Local Filesystem Agent

| Field | Detail |
|---|---|
| **Backlog #** | 54 |
| **Repo path** | `GenAI/052_Local_Filesystem_Agent/` |
| **One-line idea** | Search, summarize, and organize files with human approval |
| **Learning outcome** | Filesystem tools with safety constraints and approval workflows |
| **Stack** | LangGraph, filesystem tools, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches building agents with real-world side effects and safety guardrails |
| **Notebook path** | 1) Create sandbox directory with sample files → 2) Build file search tool → 3) Build file read/summarize tool → 4) Build file organize tool (with approval gate) → 5) Run agent with organizational query → 6) Display actions taken → 7) Key Concepts Recap |
| **Stretch goal** | Add undo capability for file operations |

---

### F-05 · Local GitHub Repo Reader Agent

| Field | Detail |
|---|---|
| **Backlog #** | 55 |
| **Repo path** | `GenAI/053_Local_GitHub_Repo_Reader_Agent/` |
| **One-line idea** | Inspect a local codebase and answer repo-level questions |
| **Learning outcome** | Code-aware retrieval and repository-level QA |
| **Stack** | LangChain, local code search, tree-sitter or AST parsing, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches code-aware chunking and retrieval — different from prose RAG |
| **Notebook path** | 1) Clone or use a local repo → 2) Build code-aware chunker (by function/class) → 3) Embed code chunks → 4) Build QA chain with code context → 5) Ask architectural questions → 6) Display answers with file references → 7) Key Concepts Recap |
| **Stretch goal** | Add dependency graph extraction |

---

### F-06 · Local CLI Command Planner Agent

| Field | Detail |
|---|---|
| **Backlog #** | 56 |
| **Repo path** | `GenAI/054_Local_CLI_Command_Planner_Agent/` |
| **One-line idea** | Suggest CLI commands for a task with explanation and approval before execution |
| **Learning outcome** | Plan-then-execute agent pattern with safety constraints |
| **Stack** | PydanticAI or LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches the plan → approve → execute agent pattern critical for safety |
| **Notebook path** | 1) Define task descriptions → 2) Build planning node (generate command + explanation) → 3) Build safety check node (flag dangerous commands) → 4) Build approval gate → 5) Build execution node (simulated) → 6) Run for 5+ tasks → 7) Key Concepts Recap |
| **Stretch goal** | Add OS-aware command adaptation (Linux vs Windows vs macOS) |

---

### F-07 · Local Expense Processing Agent

| Field | Detail |
|---|---|
| **Backlog #** | 57 |
| **Repo path** | `GenAI/055_Local_Expense_Processing_Agent/` |
| **One-line idea** | OCR a receipt, categorize the expense, and add to a summary |
| **Learning outcome** | Multi-tool agent combining OCR, classification, and structured output |
| **Stack** | Ollama, PaddleOCR or Tesseract, LangGraph, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches combining vision/OCR tools with LLM reasoning in a practical workflow |
| **Notebook path** | 1) Set up local OCR (PaddleOCR) → 2) Create sample receipt images → 3) Build OCR extraction node → 4) Build categorization node → 5) Build summary aggregation node → 6) Run for multiple receipts → 7) Display expense report → 8) Key Concepts Recap |
| **Stretch goal** | Add duplicate receipt detection via embedding similarity |

---

### F-08 · Local Calendar Planner Agent

| Field | Detail |
|---|---|
| **Backlog #** | 58 |
| **Repo path** | `GenAI/056_Local_Calendar_Planner_Agent/` |
| **One-line idea** | Propose schedules using mocked calendar state and natural language preferences |
| **Learning outcome** | Agents with state management and constraint satisfaction |
| **Stack** | LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches constraint-based reasoning and state-aware tool use |
| **Notebook path** | 1) Create mock calendar state (existing events, preferences) → 2) Build availability checker tool → 3) Build scheduling suggestion node → 4) Build conflict detection node → 5) Run for scheduling requests → 6) Display proposed schedule → 7) Key Concepts Recap |
| **Stretch goal** | Add priority-based rescheduling when conflicts arise |

---

### F-09 · Local CRM Enrichment Agent

| Field | Detail |
|---|---|
| **Backlog #** | 59 |
| **Repo path** | `GenAI/057_Local_CRM_Enrichment_Agent/` |
| **One-line idea** | Summarize account information and suggest next actions from CRM-like data |
| **Learning outcome** | Multi-source data aggregation and action recommendation |
| **Stack** | CrewAI or LangChain, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches data enrichment patterns common in enterprise AI |
| **Notebook path** | 1) Create mock CRM dataset (accounts, interactions, deals) → 2) Build account summary agent → 3) Build interaction analysis agent → 4) Build next-action recommender → 5) Run for 3 accounts → 6) Display enriched account cards → 7) Key Concepts Recap |
| **Stretch goal** | Add churn risk scoring based on interaction patterns |

---

### F-10 · Local Browser Task Agent

| Field | Detail |
|---|---|
| **Backlog #** | 60 |
| **Repo path** | `GenAI/058_Local_Browser_Task_Agent/` |
| **One-line idea** | Navigate simple web tasks in a controlled notebook prototype |
| **Learning outcome** | Browser automation with LLM planning — the foundation of web agents |
| **Stack** | LangChain or AutoGen, Playwright (local), Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Advanced |
| **Why useful** | Web agents are a frontier of agentic AI; teaches page understanding and action planning |
| **Notebook path** | 1) Set up Playwright in notebook → 2) Build page snapshot tool (accessibility tree) → 3) Build action execution tool (click, type, navigate) → 4) Build LLM planner that reads page and decides actions → 5) Run on a simple local HTML page → 6) Display action trace → 7) Key Concepts Recap |
| **Stretch goal** | Add error recovery when an action fails |

---

## Group G — Local Eval & Observability (Projects 61–70)

Systematic evaluation, benchmarking, and debugging of LLM and agent systems.

---

### G-01 · Local Prompt Evaluation Lab

| Field | Detail |
|---|---|
| **Backlog #** | 61 |
| **Repo path** | `GenAI/059_Local_Prompt_Evaluation_Lab/` |
| **One-line idea** | Compare prompt variants systematically with local models |
| **Learning outcome** | Prompt engineering as a measurable discipline: A/B testing with metrics |
| **Stack** | LangChain, Ollama, matplotlib, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches data-driven prompt optimization instead of guesswork |
| **Notebook path** | 1) Define a task with ground-truth answers → 2) Create 3–4 prompt variants → 3) Run each variant across test cases → 4) Score with LLM-as-judge + exact match → 5) Statistical comparison (mean, std, CI) → 6) Visualize results → 7) Key Concepts Recap |
| **Stretch goal** | Automate prompt variant generation with meta-prompting |

---

### G-02 · Local Output Judge Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 62 |
| **Repo path** | `GenAI/060_Local_Output_Judge/` |
| **One-line idea** | Use one local model to critique and score another model's outputs |
| **Learning outcome** | LLM-as-judge evaluation pattern with rubric design |
| **Stack** | Ollama, LangChain, Jupyter |
| **Local model** | `qwen3.5:9b` (both generator and judge) |
| **Difficulty** | Intermediate |
| **Why useful** | LLM-as-judge is the most scalable evaluation method when human eval is expensive |
| **Notebook path** | 1) Generate outputs for 10+ prompts → 2) Design scoring rubric (relevance, coherence, helpfulness) → 3) Build judge chain with rubric → 4) Score all outputs → 5) Analyze score distribution → 6) Identify failure patterns → 7) Key Concepts Recap |
| **Stretch goal** | Compare judge consistency by running evaluation twice and measuring agreement |

---

### G-03 · Local RAG A/B Testing Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 63 |
| **Repo path** | `GenAI/061_Local_RAG_AB_Testing/` |
| **One-line idea** | Compare two RAG configurations (retrieval + prompt) side by side |
| **Learning outcome** | Controlled experiment design for RAG systems |
| **Stack** | Ollama, LangChain, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches the experimental method for iterating on RAG quality |
| **Notebook path** | 1) Build two RAG configurations (different chunking or prompts) → 2) Create evaluation set → 3) Run both configs on same questions → 4) Score with LLM-as-judge → 5) Statistical comparison → 6) Declare winner with context → 7) Key Concepts Recap |
| **Stretch goal** | Automate winner selection with significance testing |

---

### G-04 · Local Tool Selection Benchmark

| Field | Detail |
|---|---|
| **Backlog #** | 64 |
| **Repo path** | `GenAI/062_Local_Tool_Selection_Benchmark/` |
| **One-line idea** | Evaluate whether an agent picks the right tool for each query |
| **Learning outcome** | Tool-selection accuracy measurement and failure analysis |
| **Stack** | LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Tool selection is the make-or-break capability for tool-using agents |
| **Notebook path** | 1) Define 5+ tools with clear descriptions → 2) Create 20+ test queries with expected tool labels → 3) Run agent and record selected tools → 4) Calculate accuracy, confusion matrix → 5) Analyze failure patterns → 6) Improve tool descriptions → 7) Re-evaluate → 8) Key Concepts Recap |
| **Stretch goal** | Add multi-tool queries where the agent should use 2+ tools |

---

### G-05 · Local Hallucination Audit Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 65 |
| **Repo path** | `GenAI/063_Local_Hallucination_Audit/` |
| **One-line idea** | Inspect and flag unsupported claims in LLM-generated text |
| **Learning outcome** | Claim extraction and evidence verification pipeline |
| **Stack** | Ollama, local eval harness, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Hallucination detection is critical for any production LLM deployment |
| **Notebook path** | 1) Generate text on 5 factual topics → 2) Extract individual claims from generated text → 3) For each claim, search for supporting evidence → 4) Score support level (supported / partial / unsupported) → 5) Calculate hallucination rate → 6) Visualize by topic → 7) Key Concepts Recap |
| **Stretch goal** | Add automatic correction chain for unsupported claims |

---

### G-06 · Local Groundedness Checker

| Field | Detail |
|---|---|
| **Backlog #** | 66 |
| **Repo path** | `GenAI/064_Local_Groundedness_Checker/` |
| **One-line idea** | Score how well an answer is grounded in provided evidence |
| **Learning outcome** | NLI-inspired faithfulness scoring with local models |
| **Stack** | LangChain, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Groundedness is a core quality metric for RAG systems |
| **Notebook path** | 1) Create evidence-answer pairs (10+ examples) → 2) Build sentence-level decomposition → 3) Build entailment checking chain → 4) Score each sentence as entailed / neutral / contradicted → 5) Aggregate to overall score → 6) Visualize → 7) Key Concepts Recap |
| **Stretch goal** | Compare groundedness across different temperature settings |

---

### G-07 · Local Structured Output Reliability Test

| Field | Detail |
|---|---|
| **Backlog #** | 67 |
| **Repo path** | `GenAI/065_Local_Structured_Output_Reliability/` |
| **One-line idea** | Compare JSON adherence across different prompts, models, and output parsers |
| **Learning outcome** | Structured output strategies: native JSON mode vs Pydantic parsing vs retry logic |
| **Stack** | Ollama, PydanticAI, LangChain, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Reliable structured output is essential for any tool-using or data-extracting agent |
| **Notebook path** | 1) Define 3 output schemas (simple, nested, complex) → 2) Test with different prompting strategies → 3) Test with Pydantic parser vs raw JSON → 4) Measure parse success rate → 5) Measure schema compliance → 6) Rank strategies by reliability → 7) Key Concepts Recap |
| **Stretch goal** | Add retry-with-error-feedback and measure improvement |

---

### G-08 · Local Cost/Latency Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 68 |
| **Repo path** | `GenAI/066_Local_Cost_Latency_Benchmark/` |
| **One-line idea** | Compare speed, throughput, and output quality across local Ollama models |
| **Learning outcome** | Practical model selection: the speed–quality tradeoff |
| **Stack** | Ollama, time measurement, Jupyter |
| **Local model** | Multiple Ollama models (small, medium, large) |
| **Difficulty** | Beginner |
| **Why useful** | Model selection is a key engineering decision; teaches measurement-driven choices |
| **Notebook path** | 1) Select 3+ Ollama models of different sizes → 2) Create benchmark prompts (summarization, reasoning, coding) → 3) Measure time-to-first-token and total latency → 4) Score output quality with rubric → 5) Plot speed vs quality tradeoff → 6) Recommend model per task type → 7) Key Concepts Recap |
| **Stretch goal** | Add throughput testing (concurrent requests) |

---

### G-09 · Local Memory Strategy Benchmark

| Field | Detail |
|---|---|
| **Backlog #** | 69 |
| **Repo path** | `GenAI/067_Local_Memory_Strategy_Benchmark/` |
| **One-line idea** | Compare short-term (buffer), retrieval-based, and persistent memory strategies |
| **Learning outcome** | Memory architecture selection for conversational agents |
| **Stack** | LangChain/LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Advanced |
| **Why useful** | Memory is the least-understood component of agents; systematic comparison teaches when to use which |
| **Notebook path** | 1) Create a 20-turn conversation scenario → 2) Implement buffer memory (last N turns) → 3) Implement retrieval memory (embed + search past turns) → 4) Implement persistent summary memory → 5) Run same conversation with each strategy → 6) Evaluate recall of earlier context → 7) Key Concepts Recap |
| **Stretch goal** | Implement hybrid memory (buffer + retrieval) and compare |

---

### G-10 · Local Agent Trace Analyzer

| Field | Detail |
|---|---|
| **Backlog #** | 70 |
| **Repo path** | `GenAI/068_Local_Agent_Trace_Analyzer/` |
| **One-line idea** | Inspect and debug multi-step agent execution traces |
| **Learning outcome** | Agent observability: logging, trace visualization, and failure diagnosis |
| **Stack** | LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Advanced |
| **Why useful** | Debugging agents is the #1 pain point; teaches systematic trace analysis |
| **Notebook path** | 1) Build a multi-tool agent → 2) Add comprehensive logging to each step → 3) Run agent on success and failure scenarios → 4) Parse traces into structured format → 5) Visualize execution as a timeline → 6) Identify failure patterns → 7) Key Concepts Recap |
| **Stretch goal** | Add automated root-cause classification for common failure modes |

---

## Group H — Fine-Tuning-Adjacent Learning (Projects 71–80)

Data preparation, quality auditing, synthetic data generation, and evaluation — everything needed *before* and *after* fine-tuning.

---

### H-01 · Fine-Tuning Dataset Builder

| Field | Detail |
|---|---|
| **Backlog #** | 71 |
| **Repo path** | `GenAI/069_Fine_Tuning_Dataset_Builder/` |
| **One-line idea** | Generate and clean training pairs (instruction → response) for future fine-tuning |
| **Learning outcome** | Dataset creation for supervised fine-tuning: formats, quality filtering, deduplication |
| **Stack** | Ollama, pandas, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Data quality is the bottleneck for fine-tuning; teaches the data engineering side |
| **Notebook path** | 1) Define target task and style → 2) Generate seed examples with LLM → 3) Self-critique and filter low-quality pairs → 4) Deduplicate with embedding similarity → 5) Format as JSONL (Alpaca/ShareGPT format) → 6) Quality statistics → 7) Key Concepts Recap |
| **Stretch goal** | Add diversity scoring to ensure topic coverage |

---

### H-02 · Synthetic Data Generator for Classification

| Field | Detail |
|---|---|
| **Backlog #** | 72 |
| **Repo path** | `GenAI/070_Synthetic_Classification_Data_Generator/` |
| **One-line idea** | Generate labeled examples for text classification using a local LLM |
| **Learning outcome** | LLM-powered data augmentation with label balancing and quality checks |
| **Stack** | Ollama, pandas, scikit-learn, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Synthetic data generation is a practical alternative to expensive annotation |
| **Notebook path** | 1) Define classification categories → 2) Generate examples per category → 3) Validate label consistency → 4) Balance class distribution → 5) Train a simple classifier on synthetic data → 6) Evaluate on held-out real examples → 7) Key Concepts Recap |
| **Stretch goal** | Compare synthetic-only vs synthetic+real training performance |

---

### H-03 · Prompt vs Fine-Tune Comparison Lab

| Field | Detail |
|---|---|
| **Backlog #** | 73 |
| **Repo path** | `GenAI/071_Prompt_vs_FineTune_Comparison_Lab/` |
| **One-line idea** | Simulate the decision boundary: when is prompting enough vs when is fine-tuning needed? |
| **Learning outcome** | Decision framework for prompting vs fine-tuning based on task complexity |
| **Stack** | Ollama, eval notebook, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Prevents unnecessary fine-tuning by teaching systematic evaluation of prompt-based alternatives |
| **Notebook path** | 1) Define 3 tasks (easy, medium, hard) → 2) Test with zero-shot prompting → 3) Test with few-shot prompting → 4) Test with chain-of-thought prompting → 5) Score each approach → 6) Identify the "fine-tune boundary" → 7) Key Concepts Recap |
| **Stretch goal** | Add cost/time estimation for fine-tuning to the decision matrix |

---

### H-04 · Style Dataset Creator

| Field | Detail |
|---|---|
| **Backlog #** | 74 |
| **Repo path** | `GenAI/072_Style_Dataset_Creator/` |
| **One-line idea** | Build a tone/style dataset from example texts for future style-tuning |
| **Learning outcome** | Style analysis, feature extraction, and dataset curation |
| **Stack** | Ollama, pandas, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Style control is a frequent fine-tuning goal; teaches feature-based dataset design |
| **Notebook path** | 1) Collect sample texts in a target style → 2) Analyze style features (tone, vocabulary, structure) → 3) Generate style-matched examples → 4) Validate style consistency with LLM judge → 5) Export as training dataset → 6) Key Concepts Recap |
| **Stretch goal** | Add A/B style comparison (formal vs casual) dataset generation |

---

### H-05 · Instruction Dataset Quality Checker

| Field | Detail |
|---|---|
| **Backlog #** | 75 |
| **Repo path** | `GenAI/073_Instruction_Dataset_Quality_Checker/` |
| **One-line idea** | Detect duplicates, contradictions, and weak labels in instruction datasets |
| **Learning outcome** | Automated dataset quality auditing pipeline |
| **Stack** | Ollama, pandas, sentence-transformers, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches the "garbage in, garbage out" principle with actionable quality checks |
| **Notebook path** | 1) Load a sample instruction dataset → 2) Detect near-duplicates via embedding similarity → 3) Check for contradicting instructions → 4) Score instruction clarity → 5) Flag low-quality entries → 6) Generate quality report → 7) Key Concepts Recap |
| **Stretch goal** | Add automatic rewriting of flagged entries |

---

### H-06 · Local Distillation Lab

| Field | Detail |
|---|---|
| **Backlog #** | 76 |
| **Repo path** | `GenAI/074_Local_Distillation_Lab/` |
| **One-line idea** | Generate teacher-model outputs to create a training dataset for a smaller student model |
| **Learning outcome** | Knowledge distillation data pipeline: teacher → student dataset creation |
| **Stack** | Ollama, Jupyter |
| **Local model** | Large model (teacher) + small model (student) via Ollama |
| **Difficulty** | Advanced |
| **Why useful** | Distillation is how production teams create specialized small models |
| **Notebook path** | 1) Select teacher model (larger) and student model (smaller) → 2) Create prompt set → 3) Generate teacher outputs → 4) Format as student training data → 5) Compare student baseline vs teacher outputs → 6) Measure the quality gap → 7) Key Concepts Recap |
| **Stretch goal** | Add chain-of-thought distillation (include teacher reasoning in training data) |

---

### H-07 · Preference Pair Builder

| Field | Detail |
|---|---|
| **Backlog #** | 77 |
| **Repo path** | `GenAI/075_Preference_Pair_Builder/` |
| **One-line idea** | Create chosen/rejected response pairs for RLHF-style preference datasets |
| **Learning outcome** | Preference data creation: what makes a response "better" and how to annotate it |
| **Stack** | Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Advanced |
| **Why useful** | Preference data is the foundation of RLHF/DPO; teaches the annotation methodology |
| **Notebook path** | 1) Create prompts → 2) Generate multiple responses per prompt → 3) Score responses with rubric → 4) Select best and worst as chosen/rejected → 5) Validate pair quality with LLM judge → 6) Export in DPO format → 7) Key Concepts Recap |
| **Stretch goal** | Add multi-dimension preference scoring (helpfulness, safety, style) |

---

### H-08 · Local JSON Extraction Dataset Builder

| Field | Detail |
|---|---|
| **Backlog #** | 78 |
| **Repo path** | `GenAI/076_JSON_Extraction_Dataset_Builder/` |
| **One-line idea** | Create structured extraction training examples from documents |
| **Learning outcome** | Information extraction dataset design with schema-guided generation |
| **Stack** | Ollama, OCR (optional), Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Structured extraction is a top enterprise use case; teaches schema-driven data creation |
| **Notebook path** | 1) Define target extraction schema (e.g., invoice fields) → 2) Create source documents → 3) Generate ground-truth extractions → 4) Validate against schema → 5) Add edge cases (missing fields, ambiguous values) → 6) Export dataset → 7) Key Concepts Recap |
| **Stretch goal** | Add schema-evolution testing (what happens when a field is added?) |

---

### H-09 · Local Classification Fine-Tune Readiness Audit

| Field | Detail |
|---|---|
| **Backlog #** | 79 |
| **Repo path** | `GenAI/077_Classification_FineTune_Readiness_Audit/` |
| **One-line idea** | Check whether a classification dataset is ready for fine-tuning |
| **Learning outcome** | Pre-training data audit: class balance, text quality, label noise, leakage |
| **Stack** | pandas, Ollama, scikit-learn, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Prevents wasted compute by catching data problems before expensive training |
| **Notebook path** | 1) Load classification dataset → 2) Class distribution analysis → 3) Text length / quality statistics → 4) Detect label noise (cross-check labels with LLM) → 5) Check for train/test leakage → 6) Generate readiness report → 7) Key Concepts Recap |
| **Stretch goal** | Add automatic balancing recommendations (oversample, undersample, augment) |

---

### H-10 · Local Fine-Tuning Evals Harness

| Field | Detail |
|---|---|
| **Backlog #** | 80 |
| **Repo path** | `GenAI/078_Fine_Tuning_Evals_Harness/` |
| **One-line idea** | Build evaluation notebooks to benchmark models before and after fine-tuning |
| **Learning outcome** | Evaluation harness design: metric selection, baseline comparison, regression detection |
| **Stack** | Ollama, pandas, matplotlib, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Advanced |
| **Why useful** | Without proper evals, fine-tuning is flying blind; teaches systematic evaluation |
| **Notebook path** | 1) Define evaluation tasks and metrics → 2) Create evaluation dataset → 3) Run baseline model → 4) Record baseline scores → 5) Build comparison framework (baseline vs fine-tuned, when available) → 6) Add regression detection (did it get worse on X?) → 7) Key Concepts Recap |
| **Stretch goal** | Add automated eval report generation with pass/fail criteria |

---

## Group I — Multimodal / OCR / Speech / VLM (Projects 81–90)

Local-first multimodal AI: document OCR, image understanding, audio transcription, and combined modalities.

---

### I-01 · Local OCR + RAG Assistant

| Field | Detail |
|---|---|
| **Backlog #** | 81 |
| **Repo path** | `GenAI/079_Local_OCR_RAG_Assistant/` |
| **One-line idea** | OCR scanned documents, then answer questions over the extracted text |
| **Learning outcome** | OCR-to-RAG pipeline: image preprocessing, text extraction, embedding, retrieval |
| **Stack** | PaddleOCR or Tesseract, Ollama, LangChain, ChromaDB, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Scanned docs are ubiquitous; teaches the OCR-to-RAG bridge |
| **Notebook path** | 1) Set up PaddleOCR → 2) OCR sample scanned pages → 3) Post-process OCR text → 4) Chunk and embed → 5) Build RAG chain → 6) Query over extracted content → 7) Key Concepts Recap |
| **Stretch goal** | Add confidence-based OCR filtering (skip low-confidence regions) |

---

### I-02 · Local Invoice Extraction Copilot

| Field | Detail |
|---|---|
| **Backlog #** | 82 |
| **Repo path** | `GenAI/080_Local_Invoice_Extraction_Copilot/` |
| **One-line idea** | Extract structured fields (vendor, amount, date, items) from invoice images |
| **Learning outcome** | OCR + structured extraction with Pydantic validation |
| **Stack** | PaddleOCR, Ollama, Pydantic, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Document extraction is one of the highest-value enterprise AI applications |
| **Notebook path** | 1) Create/obtain sample invoices → 2) OCR to raw text → 3) Define Pydantic schema (InvoiceData) → 4) Build extraction chain with JSON output → 5) Validate extracted data → 6) Handle edge cases (multiple items, different formats) → 7) Key Concepts Recap |
| **Stretch goal** | Add line-item extraction with a nested Pydantic model |

---

### I-03 · Local Receipt Intelligence Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 83 |
| **Repo path** | `GenAI/081_Local_Receipt_Intelligence/` |
| **One-line idea** | Summarize and categorize expenses from receipt images |
| **Learning outcome** | End-to-end OCR → classification → summarization pipeline |
| **Stack** | PaddleOCR, Ollama, pandas, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Practical personal finance tool; teaches multi-step document intelligence |
| **Notebook path** | 1) OCR sample receipts → 2) Extract amount, merchant, date → 3) Categorize expense (food, transport, office) → 4) Aggregate into expense summary → 5) Generate spending insights → 6) Key Concepts Recap |
| **Stretch goal** | Add monthly trend analysis across multiple receipt batches |

---

### I-04 · Local Slide Deck Explainer

| Field | Detail |
|---|---|
| **Backlog #** | 84 |
| **Repo path** | `GenAI/082_Local_Slide_Deck_Explainer/` |
| **One-line idea** | Summarize presentation slides and generate speaker notes |
| **Learning outcome** | Slide-to-text extraction and multi-level summarization |
| **Stack** | python-pptx or PDF extraction, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Beginner |
| **Why useful** | Teaches document format handling (PPTX/PDF) and per-section summarization |
| **Notebook path** | 1) Load sample slide deck (PPTX or PDF) → 2) Extract text per slide → 3) Generate summary per slide → 4) Generate speaker notes per slide → 5) Generate overall deck summary → 6) Key Concepts Recap |
| **Stretch goal** | Add audience-level adaptation (executive summary vs technical deep-dive) |

---

### I-05 · Local Image Captioning Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 85 |
| **Repo path** | `GenAI/083_Local_Image_Captioning/` |
| **One-line idea** | Generate descriptive captions for images using a local VLM |
| **Learning outcome** | VLM inference via Ollama: image encoding, prompt design for vision models |
| **Stack** | Ollama (VLM), Jupyter |
| **Local model** | `llava:13b` or `minicpm-v` via Ollama |
| **Difficulty** | Intermediate |
| **Why useful** | VLMs are the new frontier; teaches local multimodal inference |
| **Notebook path** | 1) Pull VLM model via Ollama → 2) Load sample images → 3) Send image + caption prompt to VLM → 4) Generate captions for 5+ images → 5) Compare caption quality across prompt variants → 6) Key Concepts Recap |
| **Stretch goal** | Add detailed vs concise caption modes |

---

### I-06 · Local Chart Understanding Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 86 |
| **Repo path** | `GenAI/084_Local_Chart_Understanding/` |
| **One-line idea** | Explain chart images — describe trends, data points, and insights |
| **Learning outcome** | VLM-based data visualization understanding |
| **Stack** | Ollama (VLM), matplotlib (to generate test charts), Jupyter |
| **Local model** | `llava:13b` or `minicpm-v` via Ollama |
| **Difficulty** | Intermediate |
| **Why useful** | Chart understanding is a high-demand capability for business intelligence agents |
| **Notebook path** | 1) Generate sample charts (bar, line, pie) with matplotlib → 2) Save as images → 3) Send to VLM with analysis prompt → 4) Compare VLM descriptions vs ground truth → 5) Test with increasing chart complexity → 6) Key Concepts Recap |
| **Stretch goal** | Add data extraction from chart images (approximate values) |

---

### I-07 · Local Screenshot Debugging Assistant

| Field | Detail |
|---|---|
| **Backlog #** | 87 |
| **Repo path** | `GenAI/085_Local_Screenshot_Debug_Assistant/` |
| **One-line idea** | Explain UI screenshots and identify likely issues |
| **Learning outcome** | VLM-based UI analysis and error diagnosis |
| **Stack** | Ollama (VLM), Jupyter |
| **Local model** | `llava:13b` or `minicpm-v` via Ollama |
| **Difficulty** | Intermediate |
| **Why useful** | Teaches VLMs for developer tooling — a growing application area |
| **Notebook path** | 1) Collect sample UI screenshots (error dialogs, broken layouts, console errors) → 2) Build analysis prompt for VLM → 3) Generate diagnosis for each screenshot → 4) Compare VLM diagnosis vs known issue → 5) Key Concepts Recap |
| **Stretch goal** | Add fix suggestion generation based on the diagnosis |

---

### I-08 · Local Audio Transcription + Summary Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 88 |
| **Repo path** | `GenAI/086_Local_Audio_Transcription_Summary/` |
| **One-line idea** | Transcribe local audio files with Whisper and summarize with Ollama |
| **Learning outcome** | Speech-to-text-to-summary pipeline with local models |
| **Stack** | OpenAI Whisper (local), Ollama, Jupyter |
| **Local model** | `whisper` (local) + `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Audio processing is a critical modality; teaches the speech → text → insight pipeline |
| **Notebook path** | 1) Install Whisper locally → 2) Transcribe sample audio file → 3) Post-process transcript (timestamps, speaker labels if possible) → 4) Summarize with Ollama → 5) Extract key topics → 6) Key Concepts Recap |
| **Stretch goal** | Add speaker diarization with pyannote |

---

### I-09 · Local Voice Notes Organizer

| Field | Detail |
|---|---|
| **Backlog #** | 89 |
| **Repo path** | `GenAI/087_Local_Voice_Notes_Organizer/` |
| **One-line idea** | Transcribe, cluster, and summarize voice notes by topic |
| **Learning outcome** | Multi-modal pipeline: speech → text → embeddings → clustering → summarization |
| **Stack** | Whisper (local), sentence-transformers, Ollama, Jupyter |
| **Local model** | `whisper` + `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Advanced |
| **Why useful** | Teaches combining multiple AI modalities into a single coherent workflow |
| **Notebook path** | 1) Transcribe multiple voice notes → 2) Embed transcripts → 3) Cluster by topic similarity → 4) Label clusters with LLM → 5) Summarize each cluster → 6) Generate organized notes document → 7) Key Concepts Recap |
| **Stretch goal** | Add daily digest generation from all notes |

---

### I-10 · Local Multimodal Research Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 90 |
| **Repo path** | `GenAI/088_Local_Multimodal_Research/` |
| **One-line idea** | Combine text and image evidence in one answer flow |
| **Learning outcome** | Multimodal RAG: text retrieval + image understanding in unified generation |
| **Stack** | Ollama (VLM + LLM), LangChain or LlamaIndex, ChromaDB, Jupyter |
| **Local model** | `qwen3.5:9b` (text) + `llava:13b` (images) · `nomic-embed-text-v2-moe` |
| **Difficulty** | Advanced |
| **Why useful** | Multimodal RAG is the next evolution of retrieval systems |
| **Notebook path** | 1) Create a mixed knowledge base (text docs + images with captions) → 2) Embed text chunks → 3) For image queries, use VLM to generate descriptions → 4) Retrieve both text and image evidence → 5) Generate combined answer → 6) Key Concepts Recap |
| **Stretch goal** | Add image-text relevance scoring to the retrieval pipeline |

---

## Group J — Coding / Developer / Advanced (Projects 91–100)

Developer-focused agents, coding assistants, and a capstone mini-platform.

---

### J-01 · Local Coding Copilot Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 91 |
| **Repo path** | `GenAI/089_Local_Coding_Copilot/` |
| **One-line idea** | Search one repo and answer code questions with context |
| **Learning outcome** | Code-aware RAG with function-level chunking and AST parsing |
| **Stack** | Ollama, local code retrieval, LangChain, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Intermediate |
| **Why useful** | Code QA is the most common developer AI use case |
| **Notebook path** | 1) Load a local Python project → 2) Parse into function/class chunks → 3) Embed code chunks → 4) Build code QA chain → 5) Ask 5+ questions about the codebase → 6) Display answers with code references → 7) Key Concepts Recap |
| **Stretch goal** | Add cross-file dependency tracing |

---

### J-02 · Local Test Case Generator

| Field | Detail |
|---|---|
| **Backlog #** | 92 |
| **Repo path** | `GenAI/090_Local_Test_Case_Generator/` |
| **One-line idea** | Generate unit tests from code snippets and requirements |
| **Learning outcome** | Code generation for testing: edge cases, assertions, and test templates |
| **Stack** | Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Test generation is a high-value coding assistant feature |
| **Notebook path** | 1) Provide Python functions to test → 2) Build test generation prompt (pytest style) → 3) Generate test cases → 4) Include edge cases and error paths → 5) Validate generated tests are syntactically correct → 6) Key Concepts Recap |
| **Stretch goal** | Run generated tests and report pass/fail |

---

### J-03 · Local PR Review Assistant

| Field | Detail |
|---|---|
| **Backlog #** | 93 |
| **Repo path** | `GenAI/091_Local_PR_Review_Assistant/` |
| **One-line idea** | Summarize code diffs and suggest potential issues |
| **Learning outcome** | Diff parsing and code review automation |
| **Stack** | Ollama, git diff parsing, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Automated code review is a top-requested developer tool |
| **Notebook path** | 1) Generate or load sample git diffs → 2) Parse diff into structured format (file, hunks, changes) → 3) Build review prompt per file → 4) Generate review comments → 5) Categorize findings (bug, style, performance, security) → 6) Key Concepts Recap |
| **Stretch goal** | Add severity ranking and auto-approve for trivial changes |

---

### J-04 · Local Notebook Refactor Assistant

| Field | Detail |
|---|---|
| **Backlog #** | 94 |
| **Repo path** | `GenAI/092_Local_Notebook_Refactor_Assistant/` |
| **One-line idea** | Analyze a messy notebook and suggest improvements for clarity and learning |
| **Learning outcome** | Notebook structure analysis and educational design |
| **Stack** | Ollama, nbformat, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Meta-skill: teaches how to make notebooks better for learning |
| **Notebook path** | 1) Load a messy notebook → 2) Analyze cell structure (code vs markdown ratio) → 3) Check for missing explanations → 4) Suggest section reorganization → 5) Generate improved markdown cells → 6) Key Concepts Recap |
| **Stretch goal** | Auto-generate a cleaner version of the notebook |

---

### J-05 · Local Debugging Workflow Agent

| Field | Detail |
|---|---|
| **Backlog #** | 95 |
| **Repo path** | `GenAI/093_Local_Debugging_Workflow_Agent/` |
| **One-line idea** | Inspect error logs and suggest likely fixes using a structured debugging graph |
| **Learning outcome** | Systematic debugging with LangGraph: error classification → root cause → fix suggestion |
| **Stack** | LangGraph, Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Advanced |
| **Why useful** | Teaches structured problem-solving patterns applicable beyond coding |
| **Notebook path** | 1) Create sample error logs and tracebacks → 2) Build error classification node → 3) Build root cause analysis node → 4) Build fix suggestion node → 5) Build verification step node → 6) Run for 5+ error types → 7) Key Concepts Recap |
| **Stretch goal** | Add a code-fix generation step that produces a diff |

---

### J-06 · Local Documentation Writer

| Field | Detail |
|---|---|
| **Backlog #** | 96 |
| **Repo path** | `GenAI/094_Local_Documentation_Writer/` |
| **One-line idea** | Generate README files and usage notes from code context |
| **Learning outcome** | Context-aware documentation generation from source code |
| **Stack** | Ollama, LangChain, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Documentation is the most neglected part of projects; automates the tedious part |
| **Notebook path** | 1) Load a Python project's source files → 2) Extract function signatures and docstrings → 3) Build README generation chain → 4) Generate installation + usage sections → 5) Generate API reference section → 6) Key Concepts Recap |
| **Stretch goal** | Add example generation for each documented function |

---

### J-07 · Local API Spec Explainer

| Field | Detail |
|---|---|
| **Backlog #** | 97 |
| **Repo path** | `GenAI/095_Local_API_Spec_Explainer/` |
| **One-line idea** | Explain API schemas (OpenAPI/JSON Schema) and generate usage examples |
| **Learning outcome** | Schema parsing and example generation from structured specifications |
| **Stack** | Ollama, structured parsing, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | API comprehension assistance is valuable for developers working with unfamiliar APIs |
| **Notebook path** | 1) Load sample OpenAPI spec → 2) Parse endpoints and schemas → 3) Generate plain-English explanations per endpoint → 4) Generate curl/Python examples → 5) Generate error handling examples → 6) Key Concepts Recap |
| **Stretch goal** | Add comparison between two API versions (breaking change detection) |

---

### J-08 · Local Data Pipeline Reviewer

| Field | Detail |
|---|---|
| **Backlog #** | 98 |
| **Repo path** | `GenAI/096_Local_Data_Pipeline_Reviewer/` |
| **One-line idea** | Review ETL code and suggest robustness improvements |
| **Learning outcome** | Code review with domain-specific checklists (data pipelines) |
| **Stack** | Ollama, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Intermediate |
| **Why useful** | Data pipeline reliability is a universal concern; teaches domain-specific review |
| **Notebook path** | 1) Create sample ETL code (load, transform, save) → 2) Build review checklist (error handling, idempotency, logging, schema validation) → 3) Run review chain against each checklist item → 4) Generate improvement suggestions → 5) Key Concepts Recap |
| **Stretch goal** | Generate the improved version of the pipeline code |

---

### J-09 · Local AI Project Critic

| Field | Detail |
|---|---|
| **Backlog #** | 99 |
| **Repo path** | `GenAI/097_Local_AI_Project_Critic/` |
| **One-line idea** | Review an ML/GenAI project notebook and suggest concrete improvements |
| **Learning outcome** | Meta-learning: systematic project review and improvement identification |
| **Stack** | Ollama, nbformat, Jupyter |
| **Local model** | `qwen3.5:9b` |
| **Difficulty** | Advanced |
| **Why useful** | Teaches critical thinking about AI project quality — a senior engineer skill |
| **Notebook path** | 1) Load a completed project notebook → 2) Analyze code quality → 3) Check for data leakage → 4) Evaluate model selection rationale → 5) Check reproducibility → 6) Generate improvement roadmap → 7) Key Concepts Recap |
| **Stretch goal** | Add a learning recommendation chain ("study X to improve this area") |

---

### J-10 · Local AI Ops Mini-Platform Notebook

| Field | Detail |
|---|---|
| **Backlog #** | 100 |
| **Repo path** | `GenAI/098_Local_AI_Ops_Mini_Platform/` |
| **One-line idea** | Combine chat, retrieval, tools, evals, routing, and multi-agent flows into one notebook-first mini platform |
| **Learning outcome** | Capstone: integrating all skills — RAG, tools, agents, evals, routing, memory — in one system |
| **Stack** | Ollama, LangGraph, LangChain, CrewAI, ChromaDB, Jupyter |
| **Local model** | `qwen3.5:9b` · `nomic-embed-text-v2-moe` |
| **Difficulty** | Advanced |
| **Why useful** | Capstone project that demonstrates mastery of all prior concepts |
| **Notebook path** | 1) Build: simple chat mode → 2) Add: RAG retrieval mode → 3) Add: tool calling (search, code, data) → 4) Add: routing (chat vs RAG vs tool agent) → 5) Add: multi-agent crew for complex tasks → 6) Add: evaluation harness for quality monitoring → 7) Add: persistent memory → 8) Run integrated demos → 9) Key Concepts Recap |
| **Stretch goal** | Add a self-improvement loop where the system logs failures and creates new eval cases |

---

## Implementation Roadmap

### Phase 1 — Foundation (Projects 1–10)
Beginner LLM apps. Establish patterns for Ollama, LangChain, output parsing, chain composition.

### Phase 2 — RAG Mastery (Projects 11–30)
Local RAG applications and advanced retrieval engineering. Deep-dive into chunking, retrieval, reranking, evaluation.

### Phase 3 — Agentic Workflows (Projects 31–50)
LangGraph stateful workflows and CrewAI multi-agent systems. Build routing, memory, human-in-the-loop.

### Phase 4 — Tool-Using Agents (Projects 51–60)
Agents that interact with real-world data: files, databases, web, spreadsheets.

### Phase 5 — Evaluation & Quality (Projects 61–70)
Systematic evaluation, benchmarking, and debugging of LLM/agent systems.

### Phase 6 — Fine-Tuning Prep (Projects 71–80)
Data engineering for LLM training: dataset creation, quality auditing, evaluation design.

### Phase 7 — Multimodal (Projects 81–90)
OCR, VLM, speech, and combined modality workflows.

### Phase 8 — Developer & Capstone (Projects 91–100)
Coding assistants, developer tools, and the capstone mini-platform.

---

## Dependencies to Install (beyond existing requirements.txt)

```bash
# Core GenAI / Agent frameworks
pip install langchain langchain-ollama langchain-community langchain-core
pip install langgraph
pip install crewai crewai-tools
pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama
pip install dspy
pip install haystack-ai
pip install pydantic-ai
pip install autogen-agentchat

# Vector stores
pip install chromadb
pip install faiss-cpu  # or faiss-gpu

# Document processing
pip install pypdf
pip install python-pptx
pip install pdfplumber
pip install trafilatura  # web scraping
pip install beautifulsoup4

# OCR
pip install paddleocr  # already installed
pip install pytesseract

# Audio
pip install openai-whisper

# Search
pip install duckduckgo-search
pip install rank_bm25

# Evaluation
pip install sentence-transformers  # already installed

# Utilities
pip install nbformat  # notebook parsing
pip install pydantic  # already installed
```

---

## Conventions for All New Notebooks

1. **Section headers**: `## Step N — Description`
2. **Emoji markers**: 📄 📦 🔧 💡 🧠 🔍 per existing pattern
3. **One task per code cell**
4. **3–5 working examples** before wrap-up
5. **End with**: `🧠 Key Concepts Recap` (table) + `🔧 Customization Ideas` (bullets)
6. **Model reference**: `ChatOllama(model="qwen3.5:9b", temperature=X)`
7. **Embeddings reference**: `OllamaEmbeddings(model="nomic-embed-text-v2-moe")`
8. **Cloud APIs**: Never default; only mention as `[Optional Extension]`
9. **No Streamlit**: Notebook-first always

---

*This backlog is implementation-ready. Each project can be picked up independently or in sequence.*
