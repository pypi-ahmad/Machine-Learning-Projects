import os
import json

projects = [
    # Group 1: Beginner Local LLM Apps
    (1, "Beginner Local LLM Apps", "1. Local PDF Q&A Tutor", "chat with PDFs using local embeddings and Ollama", "Ollama, LangChain, Chroma/FAISS, Jupyter"),
    (2, "Beginner Local LLM Apps", "2. Local Markdown Knowledge Bot", "query markdown notes and docs", "Ollama, LlamaIndex, local vector store, Jupyter"),
    (3, "Beginner Local LLM Apps", "3. Local Meeting Notes Summarizer", "summarize transcripts into actions and blockers", "Ollama, LangChain, notebooks"),
    (4, "Beginner Local LLM Apps", "4. Local Resume Rewriter", "improve bullets and tailor resume text", "Ollama, LangChain, notebooks"),
    (5, "Beginner Local LLM Apps", "5. Local Cover Letter Generator", "generate tailored cover letters from JD + resume", "Ollama, LangChain"),
    (6, "Beginner Local LLM Apps", "6. Local Email Reply Assistant", "classify intent and draft replies", "Ollama, LangChain, Pydantic output"),
    (7, "Beginner Local LLM Apps", "7. Local Research Paper Explainer", "explain papers in plain English", "Ollama, LlamaIndex, PDF parsing"),
    (8, "Beginner Local LLM Apps", "8. Local Blog-to-Thread Converter", "repurpose article to thread/post/email", "Ollama, LangChain"),
    (9, "Beginner Local LLM Apps", "9. Local Study Notes Generator", "turn raw text into notes and quizzes", "Ollama, LangChain"),
    (10, "Beginner Local LLM Apps", "10. Local Code Explainer", "explain code snippets and detect simple issues", "Ollama, LangChain, notebook tools"),

    # Group 2: Local RAG
    (11, "Local RAG", "11. Local Website FAQ Bot", "ingest one website and answer questions", "Ollama, LangChain, crawler + vector DB"),
    (12, "Local RAG", "12. Local Policy Assistant", "search HR/IT/company policies with citations", "Ollama, LangChain, Chroma"),
    (13, "Local RAG", "13. Local Multi-PDF Research Librarian", "answer across multiple papers with evidence", "Ollama, LlamaIndex"),
    (14, "Local RAG", "14. Local Financial Report Analyst", "QA over annual reports and filings", "Ollama, LangChain, tabular/text parsing"),
    (15, "Local RAG", "15. Local Contract Clause Finder", "retrieve risky or important clauses", "Ollama, Haystack, local index"),
    (16, "Local RAG", "16. Local Course Tutor", "QA over lecture notes/slides", "Ollama, LangChain"),
    (17, "Local RAG", "17. Local Personal Wiki Copilot", "query exported notes/wiki files", "Ollama, LlamaIndex"),
    (18, "Local RAG", "18. Local Customer Support Memory Bot", "retrieve similar tickets and fixes", "Ollama, LangChain, embeddings"),
    (19, "Local RAG", "19. Local Product Docs Copilot", "chat over internal or API docs", "Ollama, LangChain"),
    (20, "Local RAG", "20. Local Medical Literature Finder", "search papers by topic and evidence", "Ollama, LlamaIndex, metadata filters"),

    # Group 3: Advanced RAG and retrieval engineering
    (21, "Advanced RAG and Retrieval Engineering", "21. Hybrid Retrieval Lab", "compare BM25, dense, and hybrid locally", "Haystack, Ollama embeddings, notebooks"),
    (22, "Advanced RAG and Retrieval Engineering", "22. Query Rewriting RAG Lab", "rewrite vague questions before retrieval", "LangChain, DSPy, Ollama"),
    (23, "Advanced RAG and Retrieval Engineering", "23. Retrieval Reranking Lab", "compare no-rerank vs rerank", "local retriever + reranker + Ollama"),
    (24, "Advanced RAG and Retrieval Engineering", "24. Context Compression RAG", "compress large retrieved context before answer generation", "LangChain, Ollama"),
    (25, "Advanced RAG and Retrieval Engineering", "25. Multi-Hop RAG Research Agent", "use multiple retrieval hops before answering", "LangGraph, Ollama"),
    (26, "Advanced RAG and Retrieval Engineering", "26. Table + Text Local RAG", "combine CSVs and docs", "LlamaIndex, Ollama"),
    (27, "Advanced RAG and Retrieval Engineering", "27. Freshness-Aware News RAG", "prioritize recent documents", "LangChain, metadata filters, Ollama"),
    (28, "Advanced RAG and Retrieval Engineering", "28. Multilingual Local RAG", "retrieve in one language, answer in another", "Ollama, multilingual embeddings"),
    (29, "Advanced RAG and Retrieval Engineering", "29. Citation Verifier for RAG", "check if answer is supported by chunks", "LangChain, local eval notebook"),
    (30, "Advanced RAG and Retrieval Engineering", "30. RAG Evaluation Dashboard Notebook", "compare chunking/retrieval/groundedness", "LangChain, notebooks, local evals"),

    # Group 4: LangGraph Workflows
    (31, "LangGraph Workflows", "31. LangGraph Human Approval Workflow", "agent pauses for approval", "LangGraph, Ollama"),
    (32, "LangGraph Workflows", "32. LangGraph Multi-Step Sales Research Flow", "company lookup to outreach draft", "LangGraph, local tools"),
    (33, "LangGraph Workflows", "33. LangGraph Incident Summary Flow", "logs to incident summary to next steps", "LangGraph, Ollama"),
    (34, "LangGraph Workflows", "34. LangGraph Data Cleaning Approval Flow", "suggest and approve transforms", "LangGraph, pandas tools"),
    (35, "LangGraph Workflows", "35. LangGraph Resume Tailoring Flow", "parse JD, tailor resume, draft cover letter", "LangGraph, Ollama"),
    (36, "LangGraph Workflows", "36. LangGraph Procurement Review Flow", "compare vendor options and summarize", "LangGraph, Ollama"),
    (37, "LangGraph Workflows", "37. LangGraph Travel Planner Flow", "gather preferences, plan, revise with checkpoints", "LangGraph, Ollama"),
    (38, "LangGraph Workflows", "38. LangGraph Research Workflow with Memory", "accumulate findings over time", "LangGraph, persistence, Ollama"),
    (39, "LangGraph Workflows", "39. LangGraph Ticket Escalation Router", "auto-resolve vs escalate", "LangGraph, local classifier + Ollama"),
    (40, "LangGraph Workflows", "40. LangGraph Compliance Checklist Flow", "gather evidence and generate checklist", "LangGraph, local tools"),

    # Group 5: CrewAI Multi-Agent Systems
    (41, "CrewAI Multi-Agent Systems", "41. CrewAI Startup Validation Crew", "market, competitor, pricing, critic agents", "CrewAI, Ollama"),
    (42, "CrewAI Multi-Agent Systems", "42. CrewAI Content Studio", "researcher, writer, editor, repurposer", "CrewAI, Ollama"),
    (43, "CrewAI Multi-Agent Systems", "43. CrewAI Lead Gen Crew", "ICP, company research, personalization, email drafter", "CrewAI, Ollama"),
    (44, "CrewAI Multi-Agent Systems", "44. CrewAI Job Hunt Crew", "JD analyzer, resume tailor, interview coach", "CrewAI, Ollama"),
    (45, "CrewAI Multi-Agent Systems", "45. CrewAI Academic Research Crew", "search, summarize, gap finder, bibliography", "CrewAI, Ollama"),
    (46, "CrewAI Multi-Agent Systems", "46. CrewAI Product Launch Crew", "PM, marketer, analyst, QA agents", "CrewAI, Ollama"),
    (47, "CrewAI Multi-Agent Systems", "47. CrewAI Competitor Intelligence Crew", "features, pricing, launches, memo", "CrewAI, Ollama"),
    (48, "CrewAI Multi-Agent Systems", "48. CrewAI Customer Success Crew", "complaint analyst, churn risk, response writer", "CrewAI, Ollama"),
    (49, "CrewAI Multi-Agent Systems", "49. CrewAI Recruiting Crew", "resume screener, interviewer, summarizer", "CrewAI, Ollama"),
    (50, "CrewAI Multi-Agent Systems", "50. CrewAI Ops Review Crew", "ops analyst, risk reviewer, summary agent", "CrewAI, Ollama"),

    # Group 6: Local Tool-Using Agents
    (51, "Local Tool-Using Agents", "51. Local Web Research Agent", "search, compare sources, write cited answer", "LangChain or PydanticAI, Ollama, local browser/search tools"),
    (52, "Local Tool-Using Agents", "52. Local Spreadsheet Analyst Agent", "answer questions over CSV/XLSX and generate insights", "LangChain, pandas tool, Ollama"),
    (53, "Local Tool-Using Agents", "53. Local SQL Analyst Agent", "NL to SQL to summary", "LangChain, sqlite/postgres, Ollama"),
    (54, "Local Tool-Using Agents", "54. Local Filesystem Agent", "search, summarize, and organize files with approval", "LangGraph, filesystem tools, Ollama"),
    (55, "Local Tool-Using Agents", "55. Local GitHub Repo Reader Agent", "inspect codebase and answer repo questions", "LangChain, local code search, Ollama"),
    (56, "Local Tool-Using Agents", "56. Local CLI Command Planner Agent", "suggest commands with approval", "PydanticAI or LangGraph, Ollama"),
    (57, "Local Tool-Using Agents", "57. Local Expense Processing Agent", "OCR receipt, categorize, summarize", "Ollama, OCR, LangGraph"),
    (58, "Local Tool-Using Agents", "58. Local Calendar Planner Agent", "propose schedules with mocked/local calendar state", "LangGraph, Ollama"),
    (59, "Local Tool-Using Agents", "59. Local CRM Enrichment Agent", "summarize account info and next actions", "CrewAI or LangChain, Ollama"),
    (60, "Local Tool-Using Agents", "60. Local Browser Task Agent", "navigate simple web tasks in a controlled notebook prototype", "LangChain or AutoGen, browser tools, Ollama"),

    # Group 7: Local Eval and Observability Projects
    (61, "Local Eval and Observability Projects", "61. Local Prompt Evaluation Lab", "compare prompt variants systematically", "LangChain, notebooks, Ollama"),
    (62, "Local Eval and Observability Projects", "62. Local Output Judge Notebook", "use one model to critique another locally", "Ollama, LangChain"),
    (63, "Local Eval and Observability Projects", "63. Local RAG A/B Testing Notebook", "compare retrieval/prompt versions", "Ollama, LangChain"),
    (64, "Local Eval and Observability Projects", "64. Local Tool Selection Benchmark", "evaluate whether an agent picks the right tool", "LangGraph, Ollama"),
    (65, "Local Eval and Observability Projects", "65. Local Hallucination Audit Notebook", "inspect unsupported claims in generated text", "Ollama, local eval harness"),
    (66, "Local Eval and Observability Projects", "66. Local Groundedness Checker", "score answer vs evidence quality", "LangChain, Ollama"),
    (67, "Local Eval and Observability Projects", "67. Local Structured Output Reliability Test", "compare JSON adherence across prompts/models", "Ollama, PydanticAI"),
    (68, "Local Eval and Observability Projects", "68. Local Cost/Latency Notebook", "compare speed and output quality across local models", "Ollama, notebooks"),
    (69, "Local Eval and Observability Projects", "69. Local Memory Strategy Benchmark", "compare short-term vs retrieval vs persistent memory", "LangChain/LangGraph, Ollama"),
    (70, "Local Eval and Observability Projects", "70. Local Agent Trace Analyzer", "inspect failures in multi-step agent traces", "LangGraph, notebooks, Ollama"),

    # Group 8: Fine-Tuning-Adjacent Learning Projects
    (71, "Fine-Tuning-Adjacent Learning Projects", "71. Fine-Tuning Dataset Builder", "generate and clean training pairs for future tuning", "Ollama, notebooks"),
    (72, "Fine-Tuning-Adjacent Learning Projects", "72. Synthetic Data Generator for Classification", "generate labeled examples with local LLM", "Ollama, notebooks"),
    (73, "Fine-Tuning-Adjacent Learning Projects", "73. Prompt vs Fine-Tune Comparison Lab", "simulate the decision boundary before actual tuning", "Ollama, eval notebook"),
    (74, "Fine-Tuning-Adjacent Learning Projects", "74. Style Dataset Creator", "build a tone/style dataset from examples", "Ollama, pandas, notebooks"),
    (75, "Fine-Tuning-Adjacent Learning Projects", "75. Instruction Dataset Quality Checker", "detect duplicates, contradictions, weak labels", "notebooks, Ollama, pandas"),
    (76, "Fine-Tuning-Adjacent Learning Projects", "76. Local Distillation Lab", "teacher-student data generation prototype", "Ollama, notebooks"),
    (77, "Fine-Tuning-Adjacent Learning Projects", "77. Preference Pair Builder", "create chosen/rejected response datasets", "Ollama, notebooks"),
    (78, "Fine-Tuning-Adjacent Learning Projects", "78. Local JSON Extraction Dataset Builder", "create extraction examples from documents", "Ollama, OCR, notebooks"),
    (79, "Fine-Tuning-Adjacent Learning Projects", "79. Local Classification Fine-Tune Readiness Audit", "check whether a dataset is tuning-ready", "notebooks, pandas, Ollama"),
    (80, "Fine-Tuning-Adjacent Learning Projects", "80. Local Fine-Tuning Evals Harness", "build evaluation notebooks for future tuning runs", "notebooks, Ollama"),

    # Group 9: Multimodal / OCR / Speech / VLM
    (81, "Multimodal - OCR - Speech - VLM", "81. Local OCR + RAG Assistant", "OCR scanned docs then answer questions", "PaddleOCR/Tesseract local, Ollama, LangChain"),
    (82, "Multimodal - OCR - Speech - VLM", "82. Local Invoice Extraction Copilot", "extract structured fields from invoices", "OCR, Ollama, Pydantic outputs"),
    (83, "Multimodal - OCR - Speech - VLM", "83. Local Receipt Intelligence Notebook", "summarize and categorize expenses from receipts", "OCR, Ollama, notebooks"),
    (84, "Multimodal - OCR - Speech - VLM", "84. Local Slide Deck Explainer", "summarize PPT/PDF slides and generate speaking notes", "OCR/text extraction, Ollama"),
    (85, "Multimodal - OCR - Speech - VLM", "85. Local Image Captioning Notebook", "generate captions for images locally", "local VLM via Ollama, notebooks"),
    (86, "Multimodal - OCR - Speech - VLM", "86. Local Chart Understanding Notebook", "explain chart images and trends", "local VLM via Ollama"),
    (87, "Multimodal - OCR - Speech - VLM", "87. Local Screenshot Debugging Assistant", "explain UI screenshot and likely issue", "local VLM, Ollama, notebooks"),
    (88, "Multimodal - OCR - Speech - VLM", "88. Local Audio Transcription + Summary Notebook", "transcribe local audio and summarize", "Whisper local + Ollama"),
    (89, "Multimodal - OCR - Speech - VLM", "89. Local Voice Notes Organizer", "cluster and summarize voice notes", "Whisper, embeddings, Ollama"),
    (90, "Multimodal - OCR - Speech - VLM", "90. Local Multimodal Research Notebook", "combine text + image evidence in one answer flow", "local VLM + LangChain/LlamaIndex + Ollama"),

    # Group 10: Coding / Developer Agents
    (91, "Coding and Developer Agents", "91. Local Coding Copilot Notebook", "search one repo and answer code questions", "Ollama, local code retrieval, LangChain"),
    (92, "Coding and Developer Agents", "92. Local Test Case Generator", "generate tests from code snippets and requirements", "Ollama, notebooks"),
    (93, "Coding and Developer Agents", "93. Local PR Review Assistant", "summarize code diff and suggest issues", "Ollama, git diff parsing"),
    (94, "Coding and Developer Agents", "94. Local Notebook Refactor Assistant", "improve a messy notebook into a cleaner learning notebook", "Ollama, notebooks"),
    (95, "Coding and Developer Agents", "95. Local Debugging Workflow Agent", "inspect error logs and suggest likely fixes", "LangGraph, Ollama"),
    (96, "Coding and Developer Agents", "96. Local Documentation Writer", "generate README and usage notes from code/context", "Ollama, LangChain"),
    (97, "Coding and Developer Agents", "97. Local API Spec Explainer", "explain API schema and generate examples", "Ollama, structured parsing"),
    (98, "Coding and Developer Agents", "98. Local Data Pipeline Reviewer", "review ETL notebook/code and suggest robustness fixes", "Ollama"),
    (99, "Coding and Developer Agents", "99. Local AI Project Critic", "review one ML/GenAI project and suggest next improvements", "Ollama, notebooks"),
    (100, "Coding and Developer Agents", "100. Local AI Ops Mini-Platform Notebook", "combine chat, retrieval, tools, evals", "Ollama, LangGraph, LangChain, CrewAI, local vector DB, notebooks"),
]

def sanitize(name):
    import re
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name).strip('_')

base_dir = "100_Local_AI_Projects"
os.makedirs(base_dir, exist_ok=True)

for num, group, title, desc, stack in projects:
    group_sanitized = sanitize(group)
    title_sanitized = sanitize(title)
    project_dir = os.path.join(base_dir, group_sanitized, title_sanitized)
    os.makedirs(project_dir, exist_ok=True)
    
    nb_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    f"\n",
                    f"**Goal**: {desc}\n",
                    f"**Stack**: {stack}\n",
                    f"\n",
                    f"This is a local-first learning project defaulting to Ollama."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install prerequisites\n",
                    "# !pip install -q ollama langchain langchain-ollama ..."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from langchain_ollama import ChatOllama\n",
                    "\n",
                    "llm = ChatOllama(model=\"qwen3.5:9b\")\n",
                    "print(\"LLM initialized successfully locally.\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
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
    
    nb_path = os.path.join(project_dir, "notebook.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_content, f, indent=2)

print("Generated 100 projects successfully!")