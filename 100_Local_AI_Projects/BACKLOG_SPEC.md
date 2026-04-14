# 100 Local-First GenAI & Agentic AI Learning Projects — Backlog Spec

> Canonical backlog/spec for the `100_Local_AI_Projects/` track in this repository.
>
> This document is grounded in the current workspace layout and is designed as an implementation-ready, notebook-first roadmap for the next 100 local AI learning projects already organized under `100_Local_AI_Projects/`.

---

## Workspace Review Summary

This backlog is aligned to the existing local AI learning workspace structure:

- `100_Local_AI_Projects/Beginner_Local_LLM_Apps`
- `100_Local_AI_Projects/Local_RAG`
- `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering`
- `100_Local_AI_Projects/LangGraph_Workflows`
- `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems`
- `100_Local_AI_Projects/Local_Tool-Using_Agents`
- `100_Local_AI_Projects/Local_Eval_and_Observability_Projects`
- `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects`
- `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM`
- `100_Local_AI_Projects/Coding_and_Developer_Agents`

The spec below preserves the spirit of the user-provided 100-project list, keeps everything local-first, and maps each idea to the repo’s current category structure.

---

## Local-First Defaults

| Area | Default |
|---|---|
| Runtime | Ollama API at `http://localhost:11434` |
| Primary LLM pattern | Local chat/instruct model via Ollama |
| Embeddings | Local embeddings via Ollama or local open-source embedding model |
| Notebook preference | Jupyter notebooks first unless a notebook is clearly a poor fit |
| Hardware | Local GPU / CUDA when available, CPU fallback otherwise |
| Vector stores | Chroma or FAISS by default; SQLite / DuckDB optional where helpful |
| OCR | PaddleOCR or Tesseract locally |
| Speech | Whisper local/open-source stack |
| VLM | Local VLM through Ollama where supported |
| Cloud APIs | Never required; only mentioned as optional extensions |
| UI default | Notebook-first, not Streamlit by default |

### Recommended local model defaults

Use these as the starting point, then swap for locally available equivalents when needed:

- **General local chat/instruct**: Ollama-served instruction model suitable for summarization, QA, rewriting, and agents
- **Embeddings**: Ollama-compatible local embedding model or a local `sentence-transformers` alternative
- **VLM**: Ollama-compatible local VLM for image/screenshot/chart tasks
- **Speech**: Whisper local model sized to the machine
- **Reranking**: local cross-encoder or local reranker when a reranking project needs one

---

## Implementation Conventions for This Track

1. Prefer **one notebook per project** for learning clarity.
2. Keep projects **educational, practical, and progressively harder**.
3. Start with **small local datasets or sample files** that can be created in the notebook.
4. Prefer **mock/local tools** over real production integrations unless the point of the lesson is tool integration.
5. Use **structured outputs** where it helps teach reliability.
6. Include **limitations, tradeoffs, and failure cases** in every project notebook.
7. When a project needs retrieval, use **local embeddings + local vector store** first.
8. When a project needs an agent, prefer **safe, inspectable tools and approval gates**.

---

## Learning Progression Map

| Group | Folder | Focus |
|---|---|---|
| 1 | `Beginner_Local_LLM_Apps` | Basic local LLM use, prompting, summarization, rewriting, structured outputs |
| 2 | `Local_RAG` | Introductory retrieval-augmented generation over local files and documents |
| 3 | `Advanced_RAG_and_Retrieval_Engineering` | Retrieval optimization, reranking, multi-hop, compression, multilingual and evaluation |
| 4 | `LangGraph_Workflows` | Stateful agent workflows, routing, approvals, memory, checkpoints |
| 5 | `CrewAI_Multi-Agent_Systems` | Multi-agent collaboration patterns and role-based workflows |
| 6 | `Local_Tool-Using_Agents` | Agents using files, SQL, spreadsheets, browser tools, CLI plans, OCR workflows |
| 7 | `Local_Eval_and_Observability_Projects` | Prompt evals, groundedness, traces, hallucination audits, benchmark notebooks |
| 8 | `Fine-Tuning-Adjacent_Learning_Projects` | Dataset building, synthetic data, readiness audits, distillation, preference data |
| 9 | `Multimodal_-_OCR_-_Speech_-_VLM` | OCR, speech, images, screenshots, charts, multimodal research |
| 10 | `Coding_and_Developer_Agents` | Repo QA, test generation, PR review, debugging, docs, API and project critique |

---

# Group 1 — Beginner Local LLM Apps

## 1. Local PDF Q&A Tutor
- **Project path:** `100_Local_AI_Projects/Beginner_Local_LLM_Apps/01_Local_PDF_QA_Tutor/`
- **One-line idea / goal:** Chat with PDFs using local embeddings and Ollama.
- **Main learning outcome:** Understand the basic local RAG pipeline from document loading through grounded answer generation.
- **Recommended tech stack:** Ollama, LangChain, Chroma or FAISS, Jupyter.
- **Suggested local model/runtime choice:** Ollama chat model + local embedding model via Ollama at `localhost:11434`.
- **Difficulty:** Beginner.
- **Why it is useful:** This is the foundation for most practical local knowledge assistants.
- **Suggested notebook-first implementation path:** Load one small PDF, split into chunks, create embeddings, store locally, retrieve top chunks, answer with citations, then inspect failure cases.
- **Optional stretch goal:** Add page-level source highlighting and a quiz mode.

## 2. Local Markdown Knowledge Bot
- **Project path:** `100_Local_AI_Projects/Beginner_Local_LLM_Apps/02_Local_Markdown_Knowledge_Bot/`
- **One-line idea / goal:** Query markdown notes and docs locally.
- **Main learning outcome:** Learn how local document collections become searchable knowledge bases.
- **Recommended tech stack:** Ollama, LlamaIndex, local vector store, Jupyter.
- **Suggested local model/runtime choice:** Ollama-served chat model plus local embeddings.
- **Difficulty:** Beginner.
- **Why it is useful:** Markdown knowledge search is a practical starting point for personal and developer AI workflows.
- **Suggested notebook-first implementation path:** Create sample markdown notes, index them, query across files, compare retrieval quality, then add metadata filtering by folder or tag.
- **Optional stretch goal:** Add wiki-link or heading-aware retrieval.

## 3. Local Meeting Notes Summarizer
- **Project path:** `100_Local_AI_Projects/Beginner_Local_LLM_Apps/03_Local_Meeting_Notes_Summarizer/`
- **One-line idea / goal:** Summarize transcripts into actions, blockers, and decisions.
- **Main learning outcome:** Practice structured summarization and extraction from long text.
- **Recommended tech stack:** Ollama, LangChain, notebooks.
- **Suggested local model/runtime choice:** Ollama local instruct model.
- **Difficulty:** Beginner.
- **Why it is useful:** It turns a common workplace task into a teachable pattern for local LLM automation.
- **Suggested notebook-first implementation path:** Start with a transcript sample, generate plain summary, then structured outputs for actions and decisions, and compare prompt variants.
- **Optional stretch goal:** Produce a manager update and follow-up email draft.

## 4. Local Resume Rewriter
- **Project path:** `100_Local_AI_Projects/Beginner_Local_LLM_Apps/04_Local_Resume_Rewriter/`
- **One-line idea / goal:** Improve resume bullets and tailor wording.
- **Main learning outcome:** Learn iterative rewriting and prompt-controlled transformation.
- **Recommended tech stack:** Ollama, LangChain, notebooks.
- **Suggested local model/runtime choice:** Ollama local instruct model.
- **Difficulty:** Beginner.
- **Why it is useful:** It is personally useful and teaches practical prompt design with measurable before/after quality.
- **Suggested notebook-first implementation path:** Use sample bullets, ask the model to critique, rewrite, tailor for a target role, and compare different prompting styles.
- **Optional stretch goal:** Add a scoring rubric for impact, clarity, and keyword alignment.

## 5. Local Cover Letter Generator
- **Project path:** `100_Local_AI_Projects/Beginner_Local_LLM_Apps/05_Local_Cover_Letter_Generator/`
- **One-line idea / goal:** Generate tailored cover letters from a job description and resume.
- **Main learning outcome:** Learn multi-input prompting and context synthesis.
- **Recommended tech stack:** Ollama, LangChain, notebooks.
- **Suggested local model/runtime choice:** Ollama local instruct model.
- **Difficulty:** Beginner.
- **Why it is useful:** It teaches how to blend multiple text sources into one polished output.
- **Suggested notebook-first implementation path:** Parse JD requirements, summarize candidate strengths from the resume, generate variants, then critique relevance and tone.
- **Optional stretch goal:** Add company-style customization and output comparison.

## 6. Local Email Reply Assistant
- **Project path:** `100_Local_AI_Projects/Beginner_Local_LLM_Apps/06_Local_Email_Reply_Assistant/`
- **One-line idea / goal:** Classify intent and draft replies.
- **Main learning outcome:** Learn intent classification plus structured reply generation.
- **Recommended tech stack:** Ollama, LangChain, Pydantic output, notebooks.
- **Suggested local model/runtime choice:** Ollama model with structured-output prompting.
- **Difficulty:** Beginner.
- **Why it is useful:** It introduces production-relevant structure and routing without needing cloud APIs.
- **Suggested notebook-first implementation path:** Create sample emails, classify urgency and intent, route to response templates, then generate tailored drafts with schema validation.
- **Optional stretch goal:** Add tone controls and escalation suggestions.

## 7. Local Research Paper Explainer
- **Project path:** `100_Local_AI_Projects/Beginner_Local_LLM_Apps/07_Local_Research_Paper_Explainer/`
- **One-line idea / goal:** Explain papers in plain English.
- **Main learning outcome:** Learn long-document summarization and section-aware explanation.
- **Recommended tech stack:** Ollama, LlamaIndex, PDF parsing, notebooks.
- **Suggested local model/runtime choice:** Ollama chat model plus local PDF parsing.
- **Difficulty:** Beginner.
- **Why it is useful:** It teaches how to turn dense technical material into beginner-friendly explanations.
- **Suggested notebook-first implementation path:** Extract sections, summarize each section, build plain-English explanations, produce glossary and key takeaway cards.
- **Optional stretch goal:** Add explanation levels such as beginner, practitioner, and expert.

## 8. Local Blog-to-Thread Converter
- **Project path:** `100_Local_AI_Projects/Beginner_Local_LLM_Apps/08_Local_Blog_to_Thread_Converter/`
- **One-line idea / goal:** Repurpose an article into thread, post, and email formats.
- **Main learning outcome:** Learn controlled content transformation across output formats.
- **Recommended tech stack:** Ollama, LangChain, notebooks.
- **Suggested local model/runtime choice:** Ollama instruct model.
- **Difficulty:** Beginner.
- **Why it is useful:** It teaches style transfer and format constraints in a simple workflow.
- **Suggested notebook-first implementation path:** Input one article, generate multiple target formats, compare tone and length constraints, and assess which prompts keep the core message best.
- **Optional stretch goal:** Add platform-aware hooks and CTA generation.

## 9. Local Study Notes Generator
- **Project path:** `100_Local_AI_Projects/Beginner_Local_LLM_Apps/09_Local_Study_Notes_Generator/`
- **One-line idea / goal:** Turn raw text into notes, flashcards, and quizzes.
- **Main learning outcome:** Learn multi-output educational content generation.
- **Recommended tech stack:** Ollama, LangChain, notebooks.
- **Suggested local model/runtime choice:** Ollama instruct model.
- **Difficulty:** Beginner.
- **Why it is useful:** It helps learners see how one source can become multiple study artifacts.
- **Suggested notebook-first implementation path:** Load lesson text, create outline notes, flashcards, self-test questions, and compare output usefulness by difficulty level.
- **Optional stretch goal:** Add spaced-repetition metadata.

## 10. Local Code Explainer
- **Project path:** `100_Local_AI_Projects/Beginner_Local_LLM_Apps/10_Local_Code_Explainer/`
- **One-line idea / goal:** Explain code snippets and detect simple issues.
- **Main learning outcome:** Learn code-focused prompting and basic code critique.
- **Recommended tech stack:** Ollama, LangChain, notebook tools.
- **Suggested local model/runtime choice:** Ollama local coding-capable model.
- **Difficulty:** Beginner.
- **Why it is useful:** It introduces developer-assistant patterns without needing a full coding agent.
- **Suggested notebook-first implementation path:** Feed short snippets, generate explanation, identify risks or bugs, suggest improvements, and compare structured vs freeform analysis.
- **Optional stretch goal:** Add language translation across Python, SQL, and JavaScript.

---

# Group 2 — Local RAG

## 11. Local Website FAQ Bot
- **Project path:** `100_Local_AI_Projects/Local_RAG/11_Local_Website_FAQ_Bot/`
- **One-line idea / goal:** Ingest one website and answer questions over it.
- **Main learning outcome:** Learn local web ingestion and grounded web-page QA.
- **Recommended tech stack:** Ollama, LangChain, crawler, local vector DB, Jupyter.
- **Suggested local model/runtime choice:** Ollama chat model plus local embeddings.
- **Difficulty:** Beginner.
- **Why it is useful:** Website Q&A is a practical RAG pattern for docs, internal wikis, and product help.
- **Suggested notebook-first implementation path:** Crawl a small docs site or local HTML files, clean text, chunk and embed, build retriever, then answer with source URLs.
- **Optional stretch goal:** Add multi-page crawl controls and sitemap support.

## 12. Local Policy Assistant
- **Project path:** `100_Local_AI_Projects/Local_RAG/12_Local_Policy_Assistant/`
- **One-line idea / goal:** Search HR, IT, or company policies with citations.
- **Main learning outcome:** Learn citation-first retrieval and policy-grounded answering.
- **Recommended tech stack:** Ollama, LangChain, Chroma, notebooks.
- **Suggested local model/runtime choice:** Ollama local chat model plus local embeddings.
- **Difficulty:** Beginner.
- **Why it is useful:** Policy assistants are clear, high-value business use cases for trustworthy local RAG.
- **Suggested notebook-first implementation path:** Create sample policy files, attach section metadata, retrieve relevant sections, then answer with exact supporting citations.
- **Optional stretch goal:** Add conflict detection across policies.

## 13. Local Multi-PDF Research Librarian
- **Project path:** `100_Local_AI_Projects/Local_RAG/13_Local_Multi_PDF_Research_Librarian/`
- **One-line idea / goal:** Answer across multiple papers with evidence.
- **Main learning outcome:** Learn multi-document retrieval and evidence aggregation.
- **Recommended tech stack:** Ollama, LlamaIndex, notebooks.
- **Suggested local model/runtime choice:** Ollama model plus local embedding pipeline.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches how to synthesize findings from several sources instead of just one file.
- **Suggested notebook-first implementation path:** Load multiple PDFs, index with metadata, retrieve across papers, summarize cross-paper evidence, and compare agreements versus disagreements.
- **Optional stretch goal:** Add paper comparison tables.

## 14. Local Financial Report Analyst
- **Project path:** `100_Local_AI_Projects/Local_RAG/14_Local_Financial_Report_Analyst/`
- **One-line idea / goal:** QA over annual reports and filings.
- **Main learning outcome:** Learn combined text-and-table retrieval for business documents.
- **Recommended tech stack:** Ollama, LangChain, tabular/text parsing, notebooks.
- **Suggested local model/runtime choice:** Ollama chat model plus local embeddings and table parsing tools.
- **Difficulty:** Intermediate.
- **Why it is useful:** Financial reports are realistic mixed-format documents and force careful grounding.
- **Suggested notebook-first implementation path:** Extract narrative and tables separately, normalize both into retrievable chunks, answer factual questions, and validate against source tables.
- **Optional stretch goal:** Add ratio calculation verification.

## 15. Local Contract Clause Finder
- **Project path:** `100_Local_AI_Projects/Local_RAG/15_Local_Contract_Clause_Finder/`
- **One-line idea / goal:** Retrieve risky or important clauses from contracts.
- **Main learning outcome:** Learn domain-focused retrieval and clause spotting.
- **Recommended tech stack:** Ollama, Haystack, local index, notebooks.
- **Suggested local model/runtime choice:** Ollama model for explanation plus local retrieval stack.
- **Difficulty:** Intermediate.
- **Why it is useful:** It demonstrates focused legal search patterns without needing a full legal platform.
- **Suggested notebook-first implementation path:** Create sample contracts, label clause types, index them, retrieve by legal concern, and summarize why each clause matters.
- **Optional stretch goal:** Add clause risk scoring.

## 16. Local Course Tutor
- **Project path:** `100_Local_AI_Projects/Local_RAG/16_Local_Course_Tutor/`
- **One-line idea / goal:** QA over lecture notes and slides.
- **Main learning outcome:** Learn topic-aware retrieval over mixed educational content.
- **Recommended tech stack:** Ollama, LangChain, notebooks.
- **Suggested local model/runtime choice:** Ollama model plus local vector store.
- **Difficulty:** Beginner.
- **Why it is useful:** It turns local course content into a reusable tutoring assistant and teaches educational RAG design.
- **Suggested notebook-first implementation path:** Load notes, slides, and readings, add week/topic metadata, retrieve by topic, and answer with study-focused explanations.
- **Optional stretch goal:** Add exam-question generation with cited evidence.

## 17. Local Personal Wiki Copilot
- **Project path:** `100_Local_AI_Projects/Local_RAG/17_Local_Personal_Wiki_Copilot/`
- **One-line idea / goal:** Query exported notes and wiki files.
- **Main learning outcome:** Learn personal-knowledge retrieval over unstructured local notes.
- **Recommended tech stack:** Ollama, LlamaIndex, notebooks.
- **Suggested local model/runtime choice:** Ollama chat model plus local embeddings.
- **Difficulty:** Beginner.
- **Why it is useful:** It is a direct bridge from personal note-taking systems to local AI knowledge assistance.
- **Suggested notebook-first implementation path:** Ingest markdown files, preserve folder/tag metadata, query across the note corpus, and test note-link-aware answers.
- **Optional stretch goal:** Add lightweight graph navigation between related notes.

## 18. Local Customer Support Memory Bot
- **Project path:** `100_Local_AI_Projects/Local_RAG/18_Local_Customer_Support_Memory_Bot/`
- **One-line idea / goal:** Retrieve similar tickets and fixes.
- **Main learning outcome:** Learn case-based retrieval from prior issue history.
- **Recommended tech stack:** Ollama, LangChain, embeddings, notebooks.
- **Suggested local model/runtime choice:** Ollama chat model plus local embeddings.
- **Difficulty:** Beginner.
- **Why it is useful:** Support memory bots show how retrieval can reduce repetitive work using past examples.
- **Suggested notebook-first implementation path:** Create a local ticket dataset, embed cases, retrieve similar historical tickets, and draft suggested resolutions from retrieved examples.
- **Optional stretch goal:** Add resolution confidence scoring.

## 19. Local Product Docs Copilot
- **Project path:** `100_Local_AI_Projects/Local_RAG/19_Local_Product_Docs_Copilot/`
- **One-line idea / goal:** Chat over internal or API docs.
- **Main learning outcome:** Learn metadata-aware retrieval across doc types.
- **Recommended tech stack:** Ollama, LangChain, notebooks.
- **Suggested local model/runtime choice:** Ollama model plus local embeddings.
- **Difficulty:** Beginner.
- **Why it is useful:** Product and API docs are one of the most common business knowledge bases.
- **Suggested notebook-first implementation path:** Index guides, API references, FAQs, and changelogs, attach doc-type metadata, and route queries to the best source type.
- **Optional stretch goal:** Add version-aware retrieval.

## 20. Local Medical Literature Finder
- **Project path:** `100_Local_AI_Projects/Local_RAG/20_Local_Medical_Literature_Finder/`
- **One-line idea / goal:** Search papers by topic and evidence level.
- **Main learning outcome:** Learn metadata filtering and evidence-aware search.
- **Recommended tech stack:** Ollama, LlamaIndex, metadata filters, notebooks.
- **Suggested local model/runtime choice:** Ollama chat model plus local embeddings.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches filtered retrieval where not all sources should be treated equally.
- **Suggested notebook-first implementation path:** Build a local abstract collection with metadata like year, study type, and evidence level, then filter and summarize results responsibly.
- **Optional stretch goal:** Add contradictory-evidence surfacing.

---

# Group 3 — Advanced RAG and Retrieval Engineering

## 21. Hybrid Retrieval Lab
- **Project path:** `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering/21_Hybrid_Retrieval_Lab/`
- **One-line idea / goal:** Compare BM25, dense, and hybrid retrieval locally.
- **Main learning outcome:** Understand sparse versus dense versus hybrid search tradeoffs.
- **Recommended tech stack:** Haystack, Ollama embeddings, notebooks.
- **Suggested local model/runtime choice:** Local embeddings via Ollama plus BM25 and hybrid fusion.
- **Difficulty:** Intermediate.
- **Why it is useful:** Retrieval choice often matters more than prompt choice in RAG quality.
- **Suggested notebook-first implementation path:** Build a labeled mini corpus, benchmark BM25, dense, and hybrid retrieval, compute retrieval metrics, and visualize which query types each method handles best.
- **Optional stretch goal:** Add reciprocal rank fusion tuning.

## 22. Query Rewriting RAG Lab
- **Project path:** `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering/22_Query_Rewriting_RAG_Lab/`
- **One-line idea / goal:** Rewrite vague questions before retrieval.
- **Main learning outcome:** Learn how query rewriting improves downstream retrieval.
- **Recommended tech stack:** LangChain, DSPy, Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama instruct model plus local embedding retriever.
- **Difficulty:** Intermediate.
- **Why it is useful:** Many RAG failures start with poor user queries rather than poor documents.
- **Suggested notebook-first implementation path:** Compare baseline retrieval versus rewritten queries, measure retrieval gains, then inspect cases where rewriting hurts.
- **Optional stretch goal:** Add HyDE or multi-query expansion.

## 23. Retrieval Reranking Lab
- **Project path:** `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering/23_Retrieval_Reranking_Lab/`
- **One-line idea / goal:** Compare no-rerank versus rerank.
- **Main learning outcome:** Learn the value of second-stage ranking in local pipelines.
- **Recommended tech stack:** Local retriever, reranker, Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama for final generation plus a local reranker/cross-encoder.
- **Difficulty:** Intermediate.
- **Why it is useful:** Reranking is often the highest-value incremental upgrade to a RAG stack.
- **Suggested notebook-first implementation path:** Retrieve top-k chunks, rerank them, compare top results and answer quality, then analyze latency versus quality.
- **Optional stretch goal:** Benchmark multiple rerankers.

## 24. Context Compression RAG
- **Project path:** `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering/24_Context_Compression_RAG/`
- **One-line idea / goal:** Compress large retrieved context before generation.
- **Main learning outcome:** Learn context-window management and relevance compression.
- **Recommended tech stack:** LangChain, Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama model for compression and answer generation.
- **Difficulty:** Intermediate.
- **Why it is useful:** Large local document sets can overwhelm context limits unless compressed well.
- **Suggested notebook-first implementation path:** Retrieve many chunks, test extractive and abstractive compression, then compare answer faithfulness and latency.
- **Optional stretch goal:** Add section-aware compression policies.

## 25. Multi-Hop RAG Research Agent
- **Project path:** `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering/25_Multi_Hop_RAG_Research_Agent/`
- **One-line idea / goal:** Use multiple retrieval hops before answering.
- **Main learning outcome:** Learn iterative retrieval and reasoning loops.
- **Recommended tech stack:** LangGraph, Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama model plus local retriever.
- **Difficulty:** Advanced.
- **Why it is useful:** Some questions require chained evidence rather than one retrieval step.
- **Suggested notebook-first implementation path:** Create multi-hop questions, run single-hop baseline, build a graph that retrieves, plans the next hop, and synthesizes a final answer, then inspect the hop trace.
- **Optional stretch goal:** Add early-stop confidence heuristics.

## 26. Table + Text Local RAG
- **Project path:** `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering/26_Table_Text_Local_RAG/`
- **One-line idea / goal:** Combine CSVs and docs in one local RAG flow.
- **Main learning outcome:** Learn unified retrieval across structured and unstructured data.
- **Recommended tech stack:** LlamaIndex, Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama local model plus local table/text ingestion.
- **Difficulty:** Intermediate.
- **Why it is useful:** Many real projects depend on both tables and prose, not one or the other.
- **Suggested notebook-first implementation path:** Load documents and CSVs, represent tables in a retrievable way, route numerical questions carefully, and compare blended versus separate retrieval.
- **Optional stretch goal:** Add SQL fallback for precise table answers.

## 27. Freshness-Aware News RAG
- **Project path:** `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering/27_Freshness_Aware_News_RAG/`
- **One-line idea / goal:** Prioritize recent documents in retrieval.
- **Main learning outcome:** Learn recency-aware ranking and metadata weighting.
- **Recommended tech stack:** LangChain, metadata filters, Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama plus time-aware local retrieval.
- **Difficulty:** Intermediate.
- **Why it is useful:** Freshness matters whenever knowledge changes over time.
- **Suggested notebook-first implementation path:** Create timestamped news items, compare freshness-aware and freshness-blind retrieval, then analyze when recency should or should not dominate.
- **Optional stretch goal:** Detect time-sensitive queries automatically.

## 28. Multilingual Local RAG
- **Project path:** `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering/28_Multilingual_Local_RAG/`
- **One-line idea / goal:** Retrieve in one language and answer in another.
- **Main learning outcome:** Learn cross-lingual retrieval and multilingual generation.
- **Recommended tech stack:** Ollama, multilingual embeddings, notebooks.
- **Suggested local model/runtime choice:** Multilingual-capable local model plus multilingual embeddings.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches language-bridging retrieval without depending on cloud APIs.
- **Suggested notebook-first implementation path:** Build a multilingual corpus, ask questions in a different language, compare retrieval quality, and inspect translation or answer consistency.
- **Optional stretch goal:** Add automatic language detection and answer-language control.

## 29. Citation Verifier for RAG
- **Project path:** `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering/29_Citation_Verifier_for_RAG/`
- **One-line idea / goal:** Check whether an answer is supported by retrieved chunks.
- **Main learning outcome:** Learn groundedness and support verification.
- **Recommended tech stack:** LangChain, local eval notebook, Ollama.
- **Suggested local model/runtime choice:** Ollama local judge/generator pattern.
- **Difficulty:** Intermediate.
- **Why it is useful:** Trustworthy RAG requires support checking, not just fluent answers.
- **Suggested notebook-first implementation path:** Generate answers, break them into claims, compare claims against retrieved chunks, and score support coverage.
- **Optional stretch goal:** Auto-revise unsupported claims.

## 30. RAG Evaluation Dashboard Notebook
- **Project path:** `100_Local_AI_Projects/Advanced_RAG_and_Retrieval_Engineering/30_RAG_Evaluation_Dashboard/`
- **One-line idea / goal:** Compare chunking, retrieval, and groundedness in one notebook.
- **Main learning outcome:** Learn systematic local RAG evaluation.
- **Recommended tech stack:** LangChain, notebooks, local evals.
- **Suggested local model/runtime choice:** Ollama local model plus local embeddings and plotting.
- **Difficulty:** Advanced.
- **Why it is useful:** It teaches measurement-driven RAG improvement rather than trial-and-error changes.
- **Suggested notebook-first implementation path:** Build an evaluation set, vary chunk size and retrieval settings, score retrieval and answer quality, then summarize the best configuration.
- **Optional stretch goal:** Add prompt-ablation and reranker comparisons.

---

# Group 4 — LangGraph Workflows

## 31. LangGraph Human Approval Workflow
- **Project path:** `100_Local_AI_Projects/LangGraph_Workflows/31_LangGraph_Human_Approval_Workflow/`
- **One-line idea / goal:** Agent pauses for human approval.
- **Main learning outcome:** Learn interruption, approval, and resume flows in LangGraph.
- **Recommended tech stack:** LangGraph, Ollama.
- **Suggested local model/runtime choice:** Ollama local model with notebook-based graph visualization.
- **Difficulty:** Intermediate.
- **Why it is useful:** Human approval is a core safety pattern for local agents that may take actions.
- **Suggested notebook-first implementation path:** Build a small graph, pause before a side effect, simulate approval/rejection, and inspect state transitions.
- **Optional stretch goal:** Add multi-stage approvals.

## 32. LangGraph Multi-Step Sales Research Flow
- **Project path:** `100_Local_AI_Projects/LangGraph_Workflows/32_LangGraph_Sales_Research_Flow/`
- **One-line idea / goal:** Company lookup to outreach draft.
- **Main learning outcome:** Learn stateful multi-step graph design.
- **Recommended tech stack:** LangGraph, local tools, Ollama.
- **Suggested local model/runtime choice:** Ollama model plus local mock company-data tools.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches how agent workflows accumulate state across dependent tasks.
- **Suggested notebook-first implementation path:** Use local mock company data, build nodes for research, pain-point extraction, and outreach drafting, then trace graph state across steps.
- **Optional stretch goal:** Add critique and revise loops.

## 33. LangGraph Incident Summary Flow
- **Project path:** `100_Local_AI_Projects/LangGraph_Workflows/33_LangGraph_Incident_Summary_Flow/`
- **One-line idea / goal:** Turn raw logs into incident summaries and next steps.
- **Main learning outcome:** Learn graph-based structured incident processing.
- **Recommended tech stack:** LangGraph, Ollama.
- **Suggested local model/runtime choice:** Ollama model plus local log samples.
- **Difficulty:** Intermediate.
- **Why it is useful:** This is a realistic ops workflow that benefits from structure and traceability.
- **Suggested notebook-first implementation path:** Parse synthetic logs, classify severity, summarize incidents, branch on severity, and output recommended actions.
- **Optional stretch goal:** Add root-cause hypothesis generation.

## 34. LangGraph Data Cleaning Approval Flow
- **Project path:** `100_Local_AI_Projects/LangGraph_Workflows/34_LangGraph_Data_Cleaning_Flow/`
- **One-line idea / goal:** Suggest and approve data transforms.
- **Main learning outcome:** Learn tool-using graphs with human review for safe data edits.
- **Recommended tech stack:** LangGraph, pandas tools, notebooks.
- **Suggested local model/runtime choice:** Ollama model plus pandas-driven local tools.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches safe automation over tabular data workflows.
- **Suggested notebook-first implementation path:** Load a dirty dataset, detect issues, propose transforms, approve selected fixes, then compare before/after quality.
- **Optional stretch goal:** Add rollback if a transform worsens data quality.

## 35. LangGraph Resume Tailoring Flow
- **Project path:** `100_Local_AI_Projects/LangGraph_Workflows/35_LangGraph_Resume_Tailoring_Flow/`
- **One-line idea / goal:** Parse JD, tailor resume, and draft cover letter.
- **Main learning outcome:** Learn multi-output graph orchestration.
- **Recommended tech stack:** LangGraph, Ollama.
- **Suggested local model/runtime choice:** Ollama local instruct model.
- **Difficulty:** Intermediate.
- **Why it is useful:** It shows how one graph can create several related deliverables from shared context.
- **Suggested notebook-first implementation path:** Build nodes for JD analysis, resume tailoring, quality review, and cover letter generation, then inspect state snapshots.
- **Optional stretch goal:** Add scoring against target-role criteria.

## 36. LangGraph Procurement Review Flow
- **Project path:** `100_Local_AI_Projects/LangGraph_Workflows/36_LangGraph_Procurement_Review_Flow/`
- **One-line idea / goal:** Compare vendors and summarize recommendations.
- **Main learning outcome:** Learn evaluation workflows and weighted scoring in graphs.
- **Recommended tech stack:** LangGraph, Ollama.
- **Suggested local model/runtime choice:** Ollama model with local sample proposal files.
- **Difficulty:** Intermediate.
- **Why it is useful:** Procurement-style review teaches structured business decision support.
- **Suggested notebook-first implementation path:** Extract features from sample proposals, score them with a rubric, route to a recommendation node, and explain the tradeoffs.
- **Optional stretch goal:** Add weight sensitivity analysis.

## 37. LangGraph Travel Planner Flow
- **Project path:** `100_Local_AI_Projects/LangGraph_Workflows/37_LangGraph_Travel_Planner_Flow/`
- **One-line idea / goal:** Gather preferences, plan, revise, and checkpoint.
- **Main learning outcome:** Learn checkpointing and iterative refinement loops.
- **Recommended tech stack:** LangGraph, Ollama.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** Travel planning is simple enough for learning but rich enough to teach graph iteration.
- **Suggested notebook-first implementation path:** Collect constraints, draft itinerary, review against budget and preferences, revise via loop, and checkpoint the plan state.
- **Optional stretch goal:** Add calendar-like day planning.

## 38. LangGraph Research Workflow with Memory
- **Project path:** `100_Local_AI_Projects/LangGraph_Workflows/38_LangGraph_Research_with_Memory/`
- **One-line idea / goal:** Accumulate research findings over time.
- **Main learning outcome:** Learn persistent memory patterns in LangGraph.
- **Recommended tech stack:** LangGraph, persistence, Ollama.
- **Suggested local model/runtime choice:** Ollama model plus local SQLite/persistent checkpointing.
- **Difficulty:** Advanced.
- **Why it is useful:** Long-running local agents need memory to stay useful across sessions.
- **Suggested notebook-first implementation path:** Run the same workflow across sessions, persist findings, merge memory into new tasks, and evaluate what memory helps or hurts.
- **Optional stretch goal:** Add memory pruning or relevance decay.

## 39. LangGraph Ticket Escalation Router
- **Project path:** `100_Local_AI_Projects/LangGraph_Workflows/39_LangGraph_Ticket_Escalation_Router/`
- **One-line idea / goal:** Auto-resolve simple tickets and escalate others.
- **Main learning outcome:** Learn routing logic and safe early-exit patterns.
- **Recommended tech stack:** LangGraph, local classifier, Ollama.
- **Suggested local model/runtime choice:** Local classifier or prompt-based classifier plus Ollama for responses.
- **Difficulty:** Intermediate.
- **Why it is useful:** Routing is a central pattern in enterprise AI workflows.
- **Suggested notebook-first implementation path:** Build simple/complex ticket sets, classify them, route to auto-resolution or escalation, then measure routing quality.
- **Optional stretch goal:** Add confidence thresholds and fallback handling.

## 40. LangGraph Compliance Checklist Flow
- **Project path:** `100_Local_AI_Projects/LangGraph_Workflows/40_LangGraph_Compliance_Checklist_Flow/`
- **One-line idea / goal:** Gather evidence and generate a checklist.
- **Main learning outcome:** Learn evidence collection and structured pass/fail reasoning.
- **Recommended tech stack:** LangGraph, local tools, Ollama.
- **Suggested local model/runtime choice:** Ollama model plus local document/evidence tools.
- **Difficulty:** Advanced.
- **Why it is useful:** Compliance workflows require traceability, structure, and justification.
- **Suggested notebook-first implementation path:** Define requirements, gather evidence from local files, verify each requirement, produce checklist output, and flag missing evidence.
- **Optional stretch goal:** Suggest remediation actions for failed checks.

---

# Group 5 — CrewAI Multi-Agent Systems

## 41. CrewAI Startup Validation Crew
- **Project path:** `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems/41_CrewAI_Startup_Validation_Crew/`
- **One-line idea / goal:** Market, competitor, pricing, and critic agents evaluate an idea.
- **Main learning outcome:** Learn role-based multi-agent collaboration.
- **Recommended tech stack:** CrewAI, Ollama.
- **Suggested local model/runtime choice:** Ollama-served local model with notebook orchestration.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches how different agent roles can improve breadth and critique.
- **Suggested notebook-first implementation path:** Define agents and tasks, run a sequential crew, compare agent outputs, then add a critic loop.
- **Optional stretch goal:** Add a go/no-go scorecard.

## 42. CrewAI Content Studio
- **Project path:** `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems/42_CrewAI_Content_Studio/`
- **One-line idea / goal:** Researcher, writer, editor, and repurposer agents create content.
- **Main learning outcome:** Learn chained task delegation with specialized agents.
- **Recommended tech stack:** CrewAI, Ollama.
- **Suggested local model/runtime choice:** Ollama local instruct model.
- **Difficulty:** Intermediate.
- **Why it is useful:** Content pipelines are a clean way to visualize multi-agent handoffs.
- **Suggested notebook-first implementation path:** Give a topic, gather research, draft content, edit it, repurpose it, and compare stage-by-stage improvements.
- **Optional stretch goal:** Add a local fact-check step.

## 43. CrewAI Lead Gen Crew
- **Project path:** `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems/43_CrewAI_Lead_Gen_Crew/`
- **One-line idea / goal:** Build ICP, research companies, personalize outreach, and draft emails.
- **Main learning outcome:** Learn practical business workflows with multi-agent specialization.
- **Recommended tech stack:** CrewAI, Ollama.
- **Suggested local model/runtime choice:** Ollama model with local mock company data.
- **Difficulty:** Intermediate.
- **Why it is useful:** It turns multi-agent workflows into a business-style sales research assistant.
- **Suggested notebook-first implementation path:** Use local company profiles, pass outputs from one agent to the next, and review personalization quality.
- **Optional stretch goal:** Add outreach variant testing.

## 44. CrewAI Job Hunt Crew
- **Project path:** `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems/44_CrewAI_Job_Hunt_Crew/`
- **One-line idea / goal:** JD analyzer, resume tailor, and interview coach work together.
- **Main learning outcome:** Learn multi-agent orchestration across related personal productivity tasks.
- **Recommended tech stack:** CrewAI, Ollama.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** It bundles multiple practical job-search tasks into a teachable crew pattern.
- **Suggested notebook-first implementation path:** Give the crew a JD and sample resume, generate tailored outputs, produce interview prep, and compare quality across agents.
- **Optional stretch goal:** Add salary negotiation strategy.

## 45. CrewAI Academic Research Crew
- **Project path:** `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems/45_CrewAI_Academic_Research_Crew/`
- **One-line idea / goal:** Search, summarize, find gaps, and produce bibliography.
- **Main learning outcome:** Learn hierarchical research workflows using specialized agents.
- **Recommended tech stack:** CrewAI, Ollama.
- **Suggested local model/runtime choice:** Ollama model with local paper abstracts or notes.
- **Difficulty:** Intermediate.
- **Why it is useful:** Academic research is a strong fit for agent specialization and iterative synthesis.
- **Suggested notebook-first implementation path:** Provide a research question, assign search/summarize/gap/bibliography tasks, then combine the deliverables into a structured memo.
- **Optional stretch goal:** Add proposal-generation agent.

## 46. CrewAI Product Launch Crew
- **Project path:** `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems/46_CrewAI_Product_Launch_Crew/`
- **One-line idea / goal:** PM, marketer, analyst, and QA agents plan a launch.
- **Main learning outcome:** Learn cross-functional multi-agent planning.
- **Recommended tech stack:** CrewAI, Ollama.
- **Suggested local model/runtime choice:** Ollama local model with local mock launch data.
- **Difficulty:** Intermediate.
- **Why it is useful:** It shows how multi-agent systems can emulate business functions without external APIs.
- **Suggested notebook-first implementation path:** Define launch inputs, assign each role a deliverable, then compile a final launch plan with identified risks.
- **Optional stretch goal:** Add budget or KPI tracking agent.

## 47. CrewAI Competitor Intelligence Crew
- **Project path:** `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems/47_CrewAI_Competitor_Intelligence_Crew/`
- **One-line idea / goal:** Analyze features, pricing, launches, and create a memo.
- **Main learning outcome:** Learn shared-memory style research crews.
- **Recommended tech stack:** CrewAI, Ollama.
- **Suggested local model/runtime choice:** Ollama model with local competitor datasets.
- **Difficulty:** Intermediate.
- **Why it is useful:** It is a realistic competitive-analysis workflow for business users.
- **Suggested notebook-first implementation path:** Feed local competitor profiles, split feature/pricing/news work across agents, then compile a memo and identify blind spots.
- **Optional stretch goal:** Add trend detection across multiple runs.

## 48. CrewAI Customer Success Crew
- **Project path:** `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems/48_CrewAI_Customer_Success_Crew/`
- **One-line idea / goal:** Analyze complaints, estimate churn risk, and draft responses.
- **Main learning outcome:** Learn routing and risk-aware responses in crews.
- **Recommended tech stack:** CrewAI, Ollama.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** It combines classification, risk assessment, and response drafting in one business flow.
- **Suggested notebook-first implementation path:** Create complaint scenarios, classify severity and churn risk, route response style, and compare agent recommendations.
- **Optional stretch goal:** Add retention playbook suggestions.

## 49. CrewAI Recruiting Crew
- **Project path:** `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems/49_CrewAI_Recruiting_Crew/`
- **One-line idea / goal:** Screen resumes, draft interview questions, and summarize candidates.
- **Main learning outcome:** Learn rubric-based evaluation in multi-agent systems.
- **Recommended tech stack:** CrewAI, Ollama.
- **Suggested local model/runtime choice:** Ollama local model with local resume samples.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches structured evaluation and ranking with explainable outputs.
- **Suggested notebook-first implementation path:** Use sample resumes and a JD, define a screening rubric, generate questions, and compare ranked candidate summaries.
- **Optional stretch goal:** Add bias-check review prompts.

## 50. CrewAI Ops Review Crew
- **Project path:** `100_Local_AI_Projects/CrewAI_Multi-Agent_Systems/50_CrewAI_Ops_Review_Crew/`
- **One-line idea / goal:** Ops analyst, risk reviewer, and summary agent create an ops review.
- **Main learning outcome:** Learn repeatable crew workflows with business-style reporting.
- **Recommended tech stack:** CrewAI, Ollama.
- **Suggested local model/runtime choice:** Ollama model with local KPI and incident samples.
- **Difficulty:** Intermediate.
- **Why it is useful:** It is a practical reporting workflow that teaches structured synthesis from mixed inputs.
- **Suggested notebook-first implementation path:** Provide mock weekly metrics and issues, assign review tasks to agents, then build a final weekly operations memo.
- **Optional stretch goal:** Add week-over-week comparisons.

---

# Group 6 — Local Tool-Using Agents

## 51. Local Web Research Agent
- **Project path:** `100_Local_AI_Projects/Local_Tool-Using_Agents/51_Local_Web_Research_Agent/`
- **One-line idea / goal:** Search, compare sources, and write cited answers.
- **Main learning outcome:** Learn tool-calling with search and source comparison.
- **Recommended tech stack:** LangChain or PydanticAI, Ollama, local browser/search tools.
- **Suggested local model/runtime choice:** Ollama local model with local-safe search tooling.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches the difference between tool use, reasoning, and grounded final output.
- **Suggested notebook-first implementation path:** Use safe search tools, compare multiple sources, rank credibility heuristically, and produce cited final answers.
- **Optional stretch goal:** Add source trust scoring.

## 52. Local Spreadsheet Analyst Agent
- **Project path:** `100_Local_AI_Projects/Local_Tool-Using_Agents/52_Local_Spreadsheet_Analyst_Agent/`
- **One-line idea / goal:** Answer questions over CSV/XLSX and generate insights.
- **Main learning outcome:** Learn agent-driven local data analysis with pandas tools.
- **Recommended tech stack:** LangChain, pandas tool, Ollama.
- **Suggested local model/runtime choice:** Ollama local model with pandas execution tools.
- **Difficulty:** Intermediate.
- **Why it is useful:** It turns tabular analysis into a natural-language local assistant pattern.
- **Suggested notebook-first implementation path:** Load local files, answer simple aggregation and filtering questions, inspect generated code/tool calls, and validate outputs.
- **Optional stretch goal:** Add chart generation.

## 53. Local SQL Analyst Agent
- **Project path:** `100_Local_AI_Projects/Local_Tool-Using_Agents/53_Local_SQL_Analyst_Agent/`
- **One-line idea / goal:** Translate natural language to SQL and summarize results.
- **Main learning outcome:** Learn safe text-to-SQL plus result interpretation.
- **Recommended tech stack:** LangChain, SQLite or Postgres, Ollama.
- **Suggested local model/runtime choice:** Ollama local model plus local database.
- **Difficulty:** Intermediate.
- **Why it is useful:** NL-to-SQL is one of the most valuable enterprise AI use cases.
- **Suggested notebook-first implementation path:** Build a local SQLite dataset, inspect schema, generate SQL, validate it, run it, and summarize results with clear explanations.
- **Optional stretch goal:** Add SQL explanation and error recovery.

## 54. Local Filesystem Agent
- **Project path:** `100_Local_AI_Projects/Local_Tool-Using_Agents/54_Local_Filesystem_Agent/`
- **One-line idea / goal:** Search, summarize, and organize files with approval.
- **Main learning outcome:** Learn safe file-manipulation agents with guardrails.
- **Recommended tech stack:** LangGraph, filesystem tools, Ollama.
- **Suggested local model/runtime choice:** Ollama local model plus sandboxed local file tools.
- **Difficulty:** Intermediate.
- **Why it is useful:** Filesystem agents are practical but risky, making them ideal for teaching safety.
- **Suggested notebook-first implementation path:** Use a sandbox folder, search and summarize files, propose organization changes, require approval, and log each action.
- **Optional stretch goal:** Add undo support.

## 55. Local GitHub Repo Reader Agent
- **Project path:** `100_Local_AI_Projects/Local_Tool-Using_Agents/55_Local_GitHub_Repo_Reader_Agent/`
- **One-line idea / goal:** Inspect a codebase and answer repo questions.
- **Main learning outcome:** Learn local code retrieval and repo-aware QA.
- **Recommended tech stack:** LangChain, local code search, Ollama.
- **Suggested local model/runtime choice:** Ollama coding-capable model plus local embedding/indexing.
- **Difficulty:** Intermediate.
- **Why it is useful:** Code-reading assistance is a central developer-agent capability.
- **Suggested notebook-first implementation path:** Chunk code by symbol, index it, answer repo questions, and return file/function references with explanations.
- **Optional stretch goal:** Add dependency graph extraction.

## 56. Local CLI Command Planner Agent
- **Project path:** `100_Local_AI_Projects/Local_Tool-Using_Agents/56_Local_CLI_Command_Planner_Agent/`
- **One-line idea / goal:** Suggest commands with approval before execution.
- **Main learning outcome:** Learn plan-before-act command generation with safety checks.
- **Recommended tech stack:** PydanticAI or LangGraph, Ollama.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches safe local automation rather than blind command execution.
- **Suggested notebook-first implementation path:** Generate proposed commands, classify risk, request approval, and simulate or sandbox execution with explanations.
- **Optional stretch goal:** Add Windows/Linux/macOS command adaptation.

## 57. Local Expense Processing Agent
- **Project path:** `100_Local_AI_Projects/Local_Tool-Using_Agents/57_Local_Expense_Processing_Agent/`
- **One-line idea / goal:** OCR receipts, categorize spend, and summarize.
- **Main learning outcome:** Learn multi-tool orchestration across OCR, extraction, and classification.
- **Recommended tech stack:** Ollama, OCR, LangGraph.
- **Suggested local model/runtime choice:** Ollama local model plus PaddleOCR/Tesseract.
- **Difficulty:** Intermediate.
- **Why it is useful:** Expense processing is a practical example of local document automation.
- **Suggested notebook-first implementation path:** OCR local receipt images, extract fields, categorize expenses, aggregate totals, and review errors due to OCR noise.
- **Optional stretch goal:** Add duplicate-receipt detection.

## 58. Local Calendar Planner Agent
- **Project path:** `100_Local_AI_Projects/Local_Tool-Using_Agents/58_Local_Calendar_Planner_Agent/`
- **One-line idea / goal:** Propose schedules from local calendar state.
- **Main learning outcome:** Learn constraint-based planning with local tools.
- **Recommended tech stack:** LangGraph, Ollama.
- **Suggested local model/runtime choice:** Ollama local model with local mocked calendar state.
- **Difficulty:** Intermediate.
- **Why it is useful:** Calendar planning is a strong example of tool-guided reasoning without external dependencies.
- **Suggested notebook-first implementation path:** Build mock availability data, generate plan proposals, validate conflicts, revise suggestions, and log why each slot was chosen.
- **Optional stretch goal:** Add priority-based rescheduling.

## 59. Local CRM Enrichment Agent
- **Project path:** `100_Local_AI_Projects/Local_Tool-Using_Agents/59_Local_CRM_Enrichment_Agent/`
- **One-line idea / goal:** Summarize accounts and suggest next actions.
- **Main learning outcome:** Learn account summarization and action recommendation over local business data.
- **Recommended tech stack:** CrewAI or LangChain, Ollama.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches enterprise-style local copilots without requiring a real CRM integration.
- **Suggested notebook-first implementation path:** Use mock account histories, summarize account state, identify opportunities or risks, and generate next-action recommendations.
- **Optional stretch goal:** Add churn-risk analysis.

## 60. Local Browser Task Agent
- **Project path:** `100_Local_AI_Projects/Local_Tool-Using_Agents/60_Local_Browser_Task_Agent/`
- **One-line idea / goal:** Navigate simple web tasks in a controlled notebook prototype.
- **Main learning outcome:** Learn local browser-agent planning with action traces.
- **Recommended tech stack:** LangChain or AutoGen, browser tools, Ollama.
- **Suggested local model/runtime choice:** Ollama local model plus local Playwright/browser automation.
- **Difficulty:** Advanced.
- **Why it is useful:** Browser agents are an important frontier, and notebooks make their traces easy to inspect.
- **Suggested notebook-first implementation path:** Use a simple local test page or safe public page, expose click/type/read tools, let the model plan actions, then inspect failures and recovery.
- **Optional stretch goal:** Add retry and recovery policies.

---

# Group 7 — Local Eval and Observability Projects

## 61. Local Prompt Evaluation Lab
- **Project path:** `100_Local_AI_Projects/Local_Eval_and_Observability_Projects/61_Local_Prompt_Evaluation_Lab/`
- **One-line idea / goal:** Compare prompt variants systematically.
- **Main learning outcome:** Learn prompt benchmarking as an experiment, not guesswork.
- **Recommended tech stack:** LangChain, notebooks, Ollama.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** Prompt iteration becomes much more effective when it is measured.
- **Suggested notebook-first implementation path:** Build a small evaluation set, test multiple prompt variants, score outputs with human-style rubrics or judge prompts, and compare results.
- **Optional stretch goal:** Add prompt-search automation.

## 62. Local Output Judge Notebook
- **Project path:** `100_Local_AI_Projects/Local_Eval_and_Observability_Projects/62_Local_Output_Judge_Notebook/`
- **One-line idea / goal:** Use one model to critique another locally.
- **Main learning outcome:** Learn local judge patterns and rubric design.
- **Recommended tech stack:** Ollama, LangChain, notebooks.
- **Suggested local model/runtime choice:** Ollama local generator and judge model pair.
- **Difficulty:** Intermediate.
- **Why it is useful:** Judging outputs locally is often the most practical evaluation approach.
- **Suggested notebook-first implementation path:** Generate outputs for a shared prompt set, judge them against a rubric, inspect judge consistency, and compare self-judge versus cross-model judge behavior.
- **Optional stretch goal:** Add judge agreement analysis across multiple judge prompts.

## 63. Local RAG A/B Testing Notebook
- **Project path:** `100_Local_AI_Projects/Local_Eval_and_Observability_Projects/63_Local_RAG_AB_Testing/`
- **One-line idea / goal:** Compare retrieval and prompt versions side by side.
- **Main learning outcome:** Learn controlled RAG experimentation.
- **Recommended tech stack:** Ollama, LangChain, notebooks.
- **Suggested local model/runtime choice:** Ollama local model plus local embeddings.
- **Difficulty:** Intermediate.
- **Why it is useful:** It shows how to improve RAG methodically rather than by intuition alone.
- **Suggested notebook-first implementation path:** Build two different RAG pipelines, run both on the same evaluation set, judge answer quality and groundedness, and summarize winners by metric.
- **Optional stretch goal:** Add significance-style comparison.

## 64. Local Tool Selection Benchmark
- **Project path:** `100_Local_AI_Projects/Local_Eval_and_Observability_Projects/64_Local_Tool_Selection_Benchmark/`
- **One-line idea / goal:** Evaluate whether an agent picks the right tool.
- **Main learning outcome:** Learn tool-routing evaluation for agent systems.
- **Recommended tech stack:** LangGraph, Ollama.
- **Suggested local model/runtime choice:** Ollama local model with local tool registry.
- **Difficulty:** Intermediate.
- **Why it is useful:** Correct tool choice is the foundation of reliable local agents.
- **Suggested notebook-first implementation path:** Define tool descriptions, create labeled tasks, compare predicted tool choices to expected ones, and analyze confusion patterns.
- **Optional stretch goal:** Evaluate multi-tool plans.

## 65. Local Hallucination Audit Notebook
- **Project path:** `100_Local_AI_Projects/Local_Eval_and_Observability_Projects/65_Local_Hallucination_Audit/`
- **One-line idea / goal:** Inspect unsupported claims in generated text.
- **Main learning outcome:** Learn claim extraction and support auditing.
- **Recommended tech stack:** Ollama, local eval harness, notebooks.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** Hallucination auditing is essential whenever outputs might be treated as facts.
- **Suggested notebook-first implementation path:** Generate factual text, split into claims, search for support locally, label unsupported claims, and summarize hallucination patterns.
- **Optional stretch goal:** Add auto-correction suggestions.

## 66. Local Groundedness Checker
- **Project path:** `100_Local_AI_Projects/Local_Eval_and_Observability_Projects/66_Local_Groundedness_Checker/`
- **One-line idea / goal:** Score answer quality against evidence.
- **Main learning outcome:** Learn evidence-grounded scoring and support analysis.
- **Recommended tech stack:** LangChain, Ollama.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches how to distinguish fluent answers from supported answers.
- **Suggested notebook-first implementation path:** Provide evidence-answer pairs, decompose answers into claims, score support quality, and analyze failure modes.
- **Optional stretch goal:** Compare groundedness across temperature or model size.

## 67. Local Structured Output Reliability Test
- **Project path:** `100_Local_AI_Projects/Local_Eval_and_Observability_Projects/67_Local_Structured_Output_Test/`
- **One-line idea / goal:** Compare JSON adherence across prompts and models.
- **Main learning outcome:** Learn structured-output robustness and schema reliability.
- **Recommended tech stack:** Ollama, PydanticAI, notebooks.
- **Suggested local model/runtime choice:** Ollama local model with schema validation.
- **Difficulty:** Intermediate.
- **Why it is useful:** Tool-using and data-extraction workflows depend on reliable structured outputs.
- **Suggested notebook-first implementation path:** Define schemas of varying complexity, test prompt styles and parsing approaches, measure parse rates, and inspect failure cases.
- **Optional stretch goal:** Add retry-with-error-feedback loops.

## 68. Local Cost/Latency Notebook
- **Project path:** `100_Local_AI_Projects/Local_Eval_and_Observability_Projects/68_Local_Cost_Latency_Benchmark/`
- **One-line idea / goal:** Compare speed and quality across local models.
- **Main learning outcome:** Learn practical model selection under local constraints.
- **Recommended tech stack:** Ollama, notebooks.
- **Suggested local model/runtime choice:** Several local Ollama models of different sizes.
- **Difficulty:** Beginner.
- **Why it is useful:** Latency and throughput often determine whether a local project feels usable.
- **Suggested notebook-first implementation path:** Run the same tasks across several local models, record latency and quality, then recommend which model fits which task.
- **Optional stretch goal:** Add concurrency and throughput measurements.

## 69. Local Memory Strategy Benchmark
- **Project path:** `100_Local_AI_Projects/Local_Eval_and_Observability_Projects/69_Local_Memory_Strategy_Benchmark/`
- **One-line idea / goal:** Compare short-term, retrieval, and persistent memory.
- **Main learning outcome:** Learn memory architecture tradeoffs for agents.
- **Recommended tech stack:** LangChain or LangGraph, Ollama.
- **Suggested local model/runtime choice:** Ollama local model plus local persistence.
- **Difficulty:** Advanced.
- **Why it is useful:** Memory design strongly affects agent quality and failure modes.
- **Suggested notebook-first implementation path:** Simulate long conversations, plug in different memory strategies, then measure factual recall, stability, and confusion over time.
- **Optional stretch goal:** Add hybrid-memory comparison.

## 70. Local Agent Trace Analyzer
- **Project path:** `100_Local_AI_Projects/Local_Eval_and_Observability_Projects/70_Local_Agent_Trace_Analyzer/`
- **One-line idea / goal:** Inspect failures in multi-step agent traces.
- **Main learning outcome:** Learn observability and trace debugging for agents.
- **Recommended tech stack:** LangGraph, notebooks, Ollama.
- **Suggested local model/runtime choice:** Ollama local model plus structured trace logging.
- **Difficulty:** Advanced.
- **Why it is useful:** Debugging agent traces is one of the biggest practical challenges in agent development.
- **Suggested notebook-first implementation path:** Capture step-by-step traces, visualize them, classify common failures, and suggest interventions such as prompt changes or better tool descriptions.
- **Optional stretch goal:** Add automated root-cause tagging.

---

# Group 8 — Fine-Tuning-Adjacent Learning Projects

## 71. Fine-Tuning Dataset Builder
- **Project path:** `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects/71_Fine_Tuning_Dataset_Builder/`
- **One-line idea / goal:** Generate and clean training pairs for future tuning.
- **Main learning outcome:** Learn instruction-dataset construction and cleaning.
- **Recommended tech stack:** Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** Good fine-tuning starts with good data, not just more compute.
- **Suggested notebook-first implementation path:** Define a target task, generate seed examples, deduplicate, quality-check, and export clean training pairs.
- **Optional stretch goal:** Add diversity and coverage scoring.

## 72. Synthetic Data Generator for Classification
- **Project path:** `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects/72_Synthetic_Data_Generator/`
- **One-line idea / goal:** Generate labeled examples with a local LLM.
- **Main learning outcome:** Learn local synthetic data generation for supervised tasks.
- **Recommended tech stack:** Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** It helps learners see when local LLMs can reduce manual labeling effort.
- **Suggested notebook-first implementation path:** Define label set, generate examples per class, audit label quality, and train a small baseline classifier on the result.
- **Optional stretch goal:** Compare synthetic-only versus mixed-data training.

## 73. Prompt vs Fine-Tune Comparison Lab
- **Project path:** `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects/73_Prompt_vs_FineTune_Comparison/`
- **One-line idea / goal:** Simulate when prompting is enough versus when tuning is worth it.
- **Main learning outcome:** Learn decision-making before committing to fine-tuning.
- **Recommended tech stack:** Ollama, eval notebook.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches judgment about whether fine-tuning is necessary at all.
- **Suggested notebook-first implementation path:** Compare zero-shot, few-shot, and richer prompt patterns on a target task, measure performance, and define a tuning-readiness threshold.
- **Optional stretch goal:** Add an estimated cost-benefit matrix.

## 74. Style Dataset Creator
- **Project path:** `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects/74_Style_Dataset_Creator/`
- **One-line idea / goal:** Build a tone/style dataset from examples.
- **Main learning outcome:** Learn style-focused dataset design.
- **Recommended tech stack:** Ollama, pandas, notebooks.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** Style control is a common fine-tuning goal and a concrete way to learn data curation.
- **Suggested notebook-first implementation path:** Collect style examples, extract style cues, generate more examples, validate style consistency, and package a usable dataset.
- **Optional stretch goal:** Add side-by-side style-transfer evaluation.

## 75. Instruction Dataset Quality Checker
- **Project path:** `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects/75_Instruction_Quality_Checker/`
- **One-line idea / goal:** Detect duplicates, contradictions, and weak labels.
- **Main learning outcome:** Learn local data-quality auditing for instruction datasets.
- **Recommended tech stack:** notebooks, Ollama, pandas.
- **Suggested local model/runtime choice:** Ollama local model plus local embedding similarity checks.
- **Difficulty:** Intermediate.
- **Why it is useful:** Data quality problems often matter more than model choice.
- **Suggested notebook-first implementation path:** Load or create a small instruction dataset, detect near-duplicates and contradictions, flag weak examples, and summarize quality findings.
- **Optional stretch goal:** Add auto-rewrite suggestions for flagged items.

## 76. Local Distillation Lab
- **Project path:** `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects/76_Local_Distillation_Lab/`
- **One-line idea / goal:** Prototype teacher-student data generation locally.
- **Main learning outcome:** Learn distillation as a data workflow rather than a training recipe.
- **Recommended tech stack:** Ollama, notebooks.
- **Suggested local model/runtime choice:** Larger local Ollama model as teacher, smaller one as student baseline.
- **Difficulty:** Advanced.
- **Why it is useful:** Distillation is a practical path to cheaper or faster local models.
- **Suggested notebook-first implementation path:** Generate teacher outputs on a prompt set, package them as training-style data, compare teacher and student behaviors, and identify compression tradeoffs.
- **Optional stretch goal:** Add reasoning-trace distillation experiments.

## 77. Preference Pair Builder
- **Project path:** `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects/77_Preference_Pair_Builder/`
- **One-line idea / goal:** Create chosen/rejected response pairs.
- **Main learning outcome:** Learn preference-data creation for DPO/RLHF-style workflows.
- **Recommended tech stack:** Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama local model or model-pair setup.
- **Difficulty:** Advanced.
- **Why it is useful:** Preference datasets teach what “better” means in practice.
- **Suggested notebook-first implementation path:** Generate multiple responses per prompt, define judging rules, select preferred and rejected pairs, and inspect ambiguous examples.
- **Optional stretch goal:** Add multi-criteria preference labels.

## 78. Local JSON Extraction Dataset Builder
- **Project path:** `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects/78_JSON_Extraction_Dataset_Builder/`
- **One-line idea / goal:** Create extraction examples from documents.
- **Main learning outcome:** Learn schema-first extraction dataset design.
- **Recommended tech stack:** Ollama, OCR, notebooks.
- **Suggested local model/runtime choice:** Ollama local model plus local OCR for document sources.
- **Difficulty:** Intermediate.
- **Why it is useful:** Structured extraction is one of the clearest enterprise fine-tuning targets.
- **Suggested notebook-first implementation path:** Define schema, collect source docs, generate JSON outputs, validate them, and curate hard edge cases for training.
- **Optional stretch goal:** Add schema evolution tests.

## 79. Local Classification Fine-Tune Readiness Audit
- **Project path:** `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects/79_Classification_FineTune_Readiness/`
- **One-line idea / goal:** Audit whether a dataset is tuning-ready.
- **Main learning outcome:** Learn readiness checks for classification tuning.
- **Recommended tech stack:** notebooks, pandas, Ollama.
- **Suggested local model/runtime choice:** Ollama local model with local data-quality analysis tools.
- **Difficulty:** Intermediate.
- **Why it is useful:** It prevents wasteful tuning runs on poor or leaky datasets.
- **Suggested notebook-first implementation path:** Analyze label balance, text quality, duplication, leakage, and noise, then generate a tuning-readiness report.
- **Optional stretch goal:** Add remediation suggestions per issue type.

## 80. Local Fine-Tuning Evals Harness
- **Project path:** `100_Local_AI_Projects/Fine-Tuning-Adjacent_Learning_Projects/80_Local_FineTuning_Evals_Harness/`
- **One-line idea / goal:** Build evaluation notebooks for future tuning runs.
- **Main learning outcome:** Learn baseline-versus-tuned evaluation design.
- **Recommended tech stack:** notebooks, Ollama.
- **Suggested local model/runtime choice:** Ollama baseline model with placeholder comparison hooks for later tuned models.
- **Difficulty:** Advanced.
- **Why it is useful:** Fine-tuning without evaluation is guesswork; this project teaches what to measure.
- **Suggested notebook-first implementation path:** Build an evaluation set, score the baseline, define pass/fail thresholds, and structure the notebook so later tuned checkpoints can be compared fairly.
- **Optional stretch goal:** Add regression alerts by category.

---

# Group 9 — Multimodal / OCR / Speech / VLM

## 81. Local OCR + RAG Assistant
- **Project path:** `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM/81_Local_OCR_RAG_Assistant/`
- **One-line idea / goal:** OCR scanned docs and then answer questions.
- **Main learning outcome:** Learn OCR-to-RAG pipeline design.
- **Recommended tech stack:** PaddleOCR or Tesseract local, Ollama, LangChain.
- **Suggested local model/runtime choice:** Ollama local chat model plus local OCR and embeddings.
- **Difficulty:** Intermediate.
- **Why it is useful:** It bridges image-heavy documents with local text-based QA.
- **Suggested notebook-first implementation path:** OCR scanned pages, clean extracted text, build a retriever, answer with citations, and inspect OCR error propagation.
- **Optional stretch goal:** Add OCR confidence filtering.

## 82. Local Invoice Extraction Copilot
- **Project path:** `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM/82_Local_Invoice_Extraction_Copilot/`
- **One-line idea / goal:** Extract structured invoice fields locally.
- **Main learning outcome:** Learn OCR plus structured extraction with schema validation.
- **Recommended tech stack:** OCR, Ollama, Pydantic outputs.
- **Suggested local model/runtime choice:** Ollama local model plus local OCR.
- **Difficulty:** Intermediate.
- **Why it is useful:** Invoice extraction is a classic practical document-AI problem.
- **Suggested notebook-first implementation path:** OCR invoice images, define extraction schema, validate outputs, compare formats, and review edge cases like missing fields.
- **Optional stretch goal:** Add line-item extraction.

## 83. Local Receipt Intelligence Notebook
- **Project path:** `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM/83_Local_Receipt_Intelligence/`
- **One-line idea / goal:** Summarize and categorize expenses from receipts.
- **Main learning outcome:** Learn document extraction plus local business categorization.
- **Recommended tech stack:** OCR, Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama local model plus local OCR.
- **Difficulty:** Intermediate.
- **Why it is useful:** It turns OCR output into simple financial intelligence.
- **Suggested notebook-first implementation path:** Extract merchant/date/amount, categorize spend, aggregate totals, and create a mini monthly report from local sample receipts.
- **Optional stretch goal:** Add anomaly spotting for unusual expenses.

## 84. Local Slide Deck Explainer
- **Project path:** `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM/84_Local_Slide_Deck_Explainer/`
- **One-line idea / goal:** Summarize slides and create speaking notes.
- **Main learning outcome:** Learn presentation parsing and slide-level summarization.
- **Recommended tech stack:** OCR/text extraction, Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama local model with local PPT/PDF extraction.
- **Difficulty:** Beginner.
- **Why it is useful:** It teaches document understanding in a format many learners actually use.
- **Suggested notebook-first implementation path:** Extract slide text, summarize slide-by-slide, create speaker notes, and generate an overall executive summary.
- **Optional stretch goal:** Add audience-specific versions.

## 85. Local Image Captioning Notebook
- **Project path:** `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM/85_Local_Image_Captioning/`
- **One-line idea / goal:** Generate captions for images locally.
- **Main learning outcome:** Learn local VLM inference through Ollama.
- **Recommended tech stack:** Local VLM via Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama-served local VLM.
- **Difficulty:** Intermediate.
- **Why it is useful:** It is a clean introduction to multimodal local inference.
- **Suggested notebook-first implementation path:** Run a local VLM on several images, compare concise versus detailed caption prompts, and inspect hallucinated visual details.
- **Optional stretch goal:** Add domain-specific caption styles.

## 86. Local Chart Understanding Notebook
- **Project path:** `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM/86_Local_Chart_Understanding/`
- **One-line idea / goal:** Explain chart images and trends.
- **Main learning outcome:** Learn visual reasoning over charts with local VLMs.
- **Recommended tech stack:** Local VLM via Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama local VLM.
- **Difficulty:** Intermediate.
- **Why it is useful:** Chart understanding is useful for business and analytics copilots.
- **Suggested notebook-first implementation path:** Create charts locally, ask the VLM to summarize trends and anomalies, then compare output against known chart data.
- **Optional stretch goal:** Add approximate data-point extraction.

## 87. Local Screenshot Debugging Assistant
- **Project path:** `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM/87_Local_Screenshot_Debugging/`
- **One-line idea / goal:** Explain UI screenshots and likely issues.
- **Main learning outcome:** Learn VLM-assisted debugging from images.
- **Recommended tech stack:** Local VLM, Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama local VLM.
- **Difficulty:** Intermediate.
- **Why it is useful:** It teaches a practical developer-support use case for local multimodal models.
- **Suggested notebook-first implementation path:** Use screenshots with known issues, prompt the VLM to identify problems, compare against ground truth, and discuss limitations.
- **Optional stretch goal:** Add likely-fix suggestions.

## 88. Local Audio Transcription + Summary Notebook
- **Project path:** `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM/88_Local_Audio_Transcription_Summary/`
- **One-line idea / goal:** Transcribe local audio and summarize it.
- **Main learning outcome:** Learn speech-to-text plus local text summarization.
- **Recommended tech stack:** Whisper local, Ollama, notebooks.
- **Suggested local model/runtime choice:** Whisper local model plus Ollama summarizer.
- **Difficulty:** Intermediate.
- **Why it is useful:** Speech pipelines are a core building block for local meeting and notes tools.
- **Suggested notebook-first implementation path:** Transcribe a local audio clip, clean the transcript, summarize it, extract tasks or key ideas, and compare model sizes or prompts.
- **Optional stretch goal:** Add speaker segmentation.

## 89. Local Voice Notes Organizer
- **Project path:** `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM/89_Local_Voice_Notes_Organizer/`
- **One-line idea / goal:** Cluster and summarize voice notes.
- **Main learning outcome:** Learn speech-to-text plus embedding-based grouping.
- **Recommended tech stack:** Whisper, embeddings, Ollama.
- **Suggested local model/runtime choice:** Whisper local model plus local embeddings and Ollama summarizer.
- **Difficulty:** Advanced.
- **Why it is useful:** It combines several local AI stages into a coherent productivity workflow.
- **Suggested notebook-first implementation path:** Transcribe multiple notes, embed transcripts, cluster them, label cluster themes, and generate an organized digest.
- **Optional stretch goal:** Add a daily or weekly summary report.

## 90. Local Multimodal Research Notebook
- **Project path:** `100_Local_AI_Projects/Multimodal_-_OCR_-_Speech_-_VLM/90_Local_Multimodal_Research/`
- **One-line idea / goal:** Combine text and image evidence in one answer flow.
- **Main learning outcome:** Learn local multimodal retrieval and synthesis.
- **Recommended tech stack:** Local VLM plus LangChain or LlamaIndex, Ollama.
- **Suggested local model/runtime choice:** Ollama local VLM plus local text model and embeddings.
- **Difficulty:** Advanced.
- **Why it is useful:** It teaches how local AI systems can combine more than one modality without cloud services.
- **Suggested notebook-first implementation path:** Build a small mixed text/image corpus, retrieve relevant text and image evidence, summarize each modality, and synthesize a final answer.
- **Optional stretch goal:** Add cross-modal relevance scoring.

---

# Group 10 — Coding and Developer Agents

## 91. Local Coding Copilot Notebook
- **Project path:** `100_Local_AI_Projects/Coding_and_Developer_Agents/91_Local_Coding_Copilot/`
- **One-line idea / goal:** Search one repo and answer code questions.
- **Main learning outcome:** Learn local code retrieval and codebase QA.
- **Recommended tech stack:** Ollama, local code retrieval, LangChain.
- **Suggested local model/runtime choice:** Ollama local coding-capable model plus local embeddings.
- **Difficulty:** Intermediate.
- **Why it is useful:** Repo QA is one of the most important practical developer-assistant tasks.
- **Suggested notebook-first implementation path:** Parse a small local repo, chunk by file and symbol, retrieve relevant code, and answer architecture or usage questions with references.
- **Optional stretch goal:** Add call-graph or dependency hints.

## 92. Local Test Case Generator
- **Project path:** `100_Local_AI_Projects/Coding_and_Developer_Agents/92_Local_Test_Case_Generator/`
- **One-line idea / goal:** Generate tests from code snippets and requirements.
- **Main learning outcome:** Learn local code generation for testing and edge-case reasoning.
- **Recommended tech stack:** Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama coding-capable model.
- **Difficulty:** Intermediate.
- **Why it is useful:** Test generation is high-value and very teachable in a notebook setting.
- **Suggested notebook-first implementation path:** Provide functions and expected behaviors, generate tests, inspect missing edge cases, and revise prompts for better coverage.
- **Optional stretch goal:** Execute generated tests in a sandbox.

## 93. Local PR Review Assistant
- **Project path:** `100_Local_AI_Projects/Coding_and_Developer_Agents/93_Local_PR_Review_Assistant/`
- **One-line idea / goal:** Summarize code diffs and suggest issues.
- **Main learning outcome:** Learn diff parsing and review-style critique.
- **Recommended tech stack:** Ollama, git diff parsing.
- **Suggested local model/runtime choice:** Ollama local coding-capable model.
- **Difficulty:** Intermediate.
- **Why it is useful:** It turns code review into a structured local assistant workflow.
- **Suggested notebook-first implementation path:** Load or create sample diffs, summarize changes, identify bug or style concerns, and classify feedback severity.
- **Optional stretch goal:** Add auto-approval for trivial safe changes.

## 94. Local Notebook Refactor Assistant
- **Project path:** `100_Local_AI_Projects/Coding_and_Developer_Agents/94_Local_Notebook_Refactor_Assistant/`
- **One-line idea / goal:** Improve a messy notebook into a better learning notebook.
- **Main learning outcome:** Learn notebook quality review and instructional refactoring.
- **Recommended tech stack:** Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** It reinforces notebook craftsmanship, which matters across this whole track.
- **Suggested notebook-first implementation path:** Load a notebook, analyze code/markdown balance, identify missing explanations, propose a cleaner structure, and draft improved markdown.
- **Optional stretch goal:** Generate a revised notebook draft.

## 95. Local Debugging Workflow Agent
- **Project path:** `100_Local_AI_Projects/Coding_and_Developer_Agents/95_Local_Debugging_Workflow_Agent/`
- **One-line idea / goal:** Inspect logs and suggest likely fixes.
- **Main learning outcome:** Learn graph-based debugging with classification and fix planning.
- **Recommended tech stack:** LangGraph, Ollama.
- **Suggested local model/runtime choice:** Ollama local coding-capable model.
- **Difficulty:** Advanced.
- **Why it is useful:** It teaches systematic debugging rather than one-shot guesswork.
- **Suggested notebook-first implementation path:** Build nodes for error parsing, likely root cause, suggested fix, and verification checklist, then test on varied error examples.
- **Optional stretch goal:** Generate candidate patch diffs.

## 96. Local Documentation Writer
- **Project path:** `100_Local_AI_Projects/Coding_and_Developer_Agents/96_Local_Documentation_Writer/`
- **One-line idea / goal:** Generate README and usage notes from code and context.
- **Main learning outcome:** Learn documentation synthesis from local project artifacts.
- **Recommended tech stack:** Ollama, LangChain.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** Documentation generation is a practical developer assistant that stays local.
- **Suggested notebook-first implementation path:** Ingest code, comments, and config files, summarize project purpose, generate usage instructions, and critique clarity.
- **Optional stretch goal:** Add example generation.

## 97. Local API Spec Explainer
- **Project path:** `100_Local_AI_Projects/Coding_and_Developer_Agents/97_Local_API_Spec_Explainer/`
- **One-line idea / goal:** Explain API schema and generate examples.
- **Main learning outcome:** Learn schema parsing and example generation.
- **Recommended tech stack:** Ollama, structured parsing.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Intermediate.
- **Why it is useful:** API understanding support is useful for both learners and developers.
- **Suggested notebook-first implementation path:** Load OpenAPI or JSON Schema, extract endpoints and models, explain them, and generate example requests and responses.
- **Optional stretch goal:** Compare API versions for breaking changes.

## 98. Local Data Pipeline Reviewer
- **Project path:** `100_Local_AI_Projects/Coding_and_Developer_Agents/98_Local_Data_Pipeline_Reviewer/`
- **One-line idea / goal:** Review ETL code and suggest robustness fixes.
- **Main learning outcome:** Learn checklist-based code review for data workflows.
- **Recommended tech stack:** Ollama.
- **Suggested local model/runtime choice:** Ollama local coding-capable model.
- **Difficulty:** Intermediate.
- **Why it is useful:** Data pipeline reliability is broadly relevant and easy to turn into review heuristics.
- **Suggested notebook-first implementation path:** Provide ETL snippets, evaluate them against reliability criteria, and generate prioritized improvement suggestions.
- **Optional stretch goal:** Produce a revised version of the ETL code.

## 99. Local AI Project Critic
- **Project path:** `100_Local_AI_Projects/Coding_and_Developer_Agents/99_Local_AI_Project_Critic/`
- **One-line idea / goal:** Review an ML or GenAI project and suggest improvements.
- **Main learning outcome:** Learn structured project critique and roadmap generation.
- **Recommended tech stack:** Ollama, notebooks.
- **Suggested local model/runtime choice:** Ollama local model.
- **Difficulty:** Advanced.
- **Why it is useful:** It trains higher-level review judgment across project quality, rigor, and communication.
- **Suggested notebook-first implementation path:** Inspect a local notebook or project folder, score it on clarity, reproducibility, grounding, and usefulness, then generate an improvement roadmap.
- **Optional stretch goal:** Add learning-resource recommendations per weakness.

## 100. Local AI Ops Mini-Platform Notebook
- **Project path:** `100_Local_AI_Projects/Coding_and_Developer_Agents/100_Local_AI_Ops_Mini_Platform/`
- **One-line idea / goal:** Combine chat, retrieval, tools, evals, routing, and multi-agent flows into one notebook-first mini platform.
- **Main learning outcome:** Integrate the major ideas of the full local AI track into one capstone.
- **Recommended tech stack:** Ollama, LangGraph, LangChain, CrewAI, local vector DB, notebooks.
- **Suggested local model/runtime choice:** Ollama local chat model plus local embeddings and optional local tool stack.
- **Difficulty:** Advanced.
- **Why it is useful:** It serves as the capstone that ties together local inference, tools, RAG, routing, memory, and evaluation.
- **Suggested notebook-first implementation path:** Start with local chat, add retrieval mode, add tools, add routing, add agent workflows, then add evaluation and observability so the notebook becomes a small local AI lab.
- **Optional stretch goal:** Add self-improvement loops that log failures into new eval cases.

---

## Delivery Priorities

### Phase 1 — Foundational local LLM fluency
Projects `01–10`
- Focus on local prompting, structured outputs, summarization, and transformation.

### Phase 2 — Practical local RAG
Projects `11–20`
- Focus on ingestion, chunking, retrieval, metadata, and grounded answering.

### Phase 3 — Retrieval engineering depth
Projects `21–30`
- Focus on benchmarking, reranking, compression, multilingual retrieval, and evaluation.

### Phase 4 — Agent workflow orchestration
Projects `31–50`
- Focus on LangGraph workflows and CrewAI multi-agent collaboration.

### Phase 5 — Real tool use
Projects `51–60`
- Focus on safe tools, local files, SQL, spreadsheets, browser tasks, and action planning.

### Phase 6 — Evals and observability
Projects `61–70`
- Focus on prompt testing, groundedness, hallucination audits, traces, and memory comparisons.

### Phase 7 — Fine-tuning-adjacent readiness
Projects `71–80`
- Focus on dataset design, quality, distillation thinking, and readiness checks.

### Phase 8 — Multimodal local AI
Projects `81–90`
- Focus on OCR, speech, screenshots, charts, VLMs, and multimodal evidence.

### Phase 9 — Developer agents and capstone integration
Projects `91–100`
- Focus on coding copilots, review assistants, debugging workflows, and the integrated capstone.

---

## Suggested Build Rules for All 100 Projects

- Default to **Jupyter notebooks** for the learning artifact.
- Keep each project **local-first** and **Ollama-first**.
- Prefer **small, reproducible local sample data** over remote dependencies.
- If external data is helpful, treat it as **optional setup**, not a hard dependency.
- Prefer **local/open-source OCR, speech, retrieval, and database components**.
- Add **clear limitations and failure cases** instead of overstating model capability.
- Avoid Streamlit unless a notebook is clearly the wrong fit.
- Keep every project beginner-friendly in explanation, even when technically advanced.

---

## Validation Checklist for This Backlog

- [x] Anchored to the real `100_Local_AI_Projects/` folder structure.
- [x] Covers all 100 requested projects.
- [x] Preserves the spirit of the exact user-provided project list.
- [x] Uses local-first, Ollama-first assumptions.
- [x] Prefers notebook-first implementations.
- [x] Avoids cloud-first assumptions.
- [x] Keeps projects progressive, educational, and practical.
- [x] Avoids adding or changing runtime code in existing projects.

---

This document is intended to be the implementation-ready planning spec for the local AI learning track under `100_Local_AI_Projects/`.

