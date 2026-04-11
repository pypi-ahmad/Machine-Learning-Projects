"""Group 1 — Projects 1-10: Beginner Local LLM Apps."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── Project 1: Local PDF Q&A Tutor ──────────────────────────────────
    paths.append(write_nb(1, "01_Local_PDF_QA_Tutor", [
        md("""
        # Project 1 — Local PDF Q&A Tutor
        ## Chat with PDFs Using Local Embeddings and Ollama

        **What you'll learn:**
        - Load and parse PDF documents with PyPDF
        - Split text into overlapping chunks for better retrieval
        - Create embeddings with a local Ollama model
        - Store vectors in ChromaDB
        - Build a RetrievalQA chain that answers questions with citations

        **Stack:** Ollama · LangChain · ChromaDB · PyPDF · Jupyter

        **Prerequisites:** `ollama pull nomic-embed-text` and `ollama pull qwen3:8b`

        ```
        PDF → Split into chunks → Embed each chunk → Store in Chroma
                                                               ↓
        User question → Embed question → Find similar chunks → Feed to LLM → Answer
        ```
        """),
        code("""
        # Cell 1 — Install dependencies
        # !pip install -q langchain langchain-ollama langchain-community chromadb pypdf
        """),
        md("## Step 1 — Configure LLM and Embeddings"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        llm = ChatOllama(model="qwen3:8b", temperature=0.3)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        vec = embeddings.embed_query("test")
        print(f"Embedding dim: {len(vec)} — model is working!")
        """),
        md("## Step 2 — Create a Sample PDF (self-contained demo)"),
        code("""
        from pathlib import Path
        import textwrap

        SAMPLE_DIR = Path("sample_data"); SAMPLE_DIR.mkdir(exist_ok=True)

        sample_text = textwrap.dedent(\"\"\"
            Machine Learning Fundamentals

            Supervised learning uses labeled data to train models that predict outcomes.
            Common algorithms include linear regression for continuous targets and
            logistic regression for classification. Decision trees split data based on
            feature thresholds, while random forests combine many trees to reduce overfitting.

            Deep Learning Overview

            Neural networks consist of layers of interconnected nodes. Convolutional
            Neural Networks (CNNs) excel at image recognition. Recurrent Neural Networks
            (RNNs) and LSTMs handle sequential data such as text and time series.

            Transformers and Attention

            The transformer architecture replaced recurrence with self-attention mechanisms.
            This enables parallel processing, leading to BERT for understanding and GPT for
            generation. Large Language Models are scaled-up transformers.

            Retrieval-Augmented Generation (RAG)

            RAG combines a retriever with a generator. This reduces hallucination by grounding
            the LLM in actual source material. Key components: document chunking, embedding
            models, vector databases, and prompt engineering.
        \"\"\")
        (SAMPLE_DIR / "ml_fundamentals.txt").write_text(sample_text, encoding="utf-8")
        print("Sample document saved.")
        """),
        md("""
        ## Step 3 — Load and Chunk the Document
        `RecursiveCharacterTextSplitter` splits on paragraph → sentence → word boundaries.
        """),
        code("""
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        loader = TextLoader("sample_data/ml_fundamentals.txt", encoding="utf-8")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,
            separators=["\\n\\n", "\\n", ". ", " "])
        chunks = splitter.split_documents(docs)

        print(f"Original: {len(docs[0].page_content)} chars → {len(chunks)} chunks")
        for i, c in enumerate(chunks):
            print(f"  Chunk {i}: {len(c.page_content)} chars — {c.page_content[:60]}...")
        """),
        md("## Step 4 — Create the Vector Store with ChromaDB"),
        code("""
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings,
            persist_directory="sample_data/chroma_db",
            collection_name="pdf_qa_tutor",
        )
        results = vectorstore.similarity_search("What is RAG?", k=2)
        print("Top 2 results for 'What is RAG?':")
        for i, r in enumerate(results):
            print(f"  [{i+1}] {r.page_content[:100]}...")
        """),
        md("## Step 5 — Build the QA Chain with Citations"),
        code("""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=\"\"\"Use the following context to answer the question. If you cannot
        find the answer, say "I don't have enough information."

        Context:
        {context}

        Question: {question}

        Answer (cite relevant details from the context):\"\"\"
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )
        print("QA chain ready!")
        """),
        md("## Step 6 — Ask Questions"),
        code("""
        questions = [
            "What is the difference between CNNs and RNNs?",
            "How does RAG reduce hallucination?",
            "What did the transformer architecture replace?",
        ]
        for q in questions:
            print(f"\\n{'='*60}\\nQ: {q}")
            result = qa_chain.invoke({"query": q})
            print(f"A: {result['result']}")
            print(f"Sources: {len(result['source_documents'])} chunks used")
        """),
        md("## Step 7 — Interactive Q&A Helper"),
        code("""
        def ask(question: str) -> str:
            result = qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]
            output = f"Answer: {answer}\\n\\nSources ({len(sources)}):\\n"
            for i, src in enumerate(sources):
                output += f"  [{i+1}] {src.page_content[:80]}...\\n"
            return output

        print(ask("What are the key components of RAG?"))
        """),
        md("""
        ## What You Learned
        - **PDF loading** → text extraction
        - **Chunking** → overlapping splits for context retention
        - **Local embeddings** → Ollama nomic-embed-text
        - **Vector store** → ChromaDB for similarity search
        - **RetrievalQA** → end-to-end Q&A with citations
        """),
    ]))

    # ── Project 2: Local Markdown Knowledge Bot ─────────────────────────
    paths.append(write_nb(1, "02_Local_Markdown_Knowledge_Bot", [
        md("""
        # Project 2 — Local Markdown Knowledge Bot
        ## Query Your Markdown Notes with LlamaIndex + Ollama

        **What you'll learn:**
        - Parse markdown files into structured documents
        - Build a LlamaIndex VectorStoreIndex over your notes
        - Query your knowledge base conversationally
        - Handle metadata (titles, tags, dates) for better retrieval

        **Stack:** Ollama · LlamaIndex · Jupyter
        """),
        code("# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama"),
        md("## Step 1 — Generate Sample Markdown Notes"),
        code("""
        from pathlib import Path

        notes_dir = Path("sample_notes"); notes_dir.mkdir(exist_ok=True)

        notes = {
            "python_tips.md": \"\"\"# Python Tips
        ## List Comprehensions
        List comprehensions are a concise way to create lists:
        `squares = [x**2 for x in range(10)]`

        ## Context Managers
        Use `with` statements for resource management. They ensure cleanup happens
        even if exceptions occur. Common for file handling, database connections.

        ## Type Hints
        Type hints improve code readability: `def greet(name: str) -> str:`
        \"\"\",
            "git_workflow.md": \"\"\"# Git Workflow Guide
        ## Branching Strategy
        - `main` — production-ready code
        - `develop` — integration branch
        - `feature/*` — new features
        - `hotfix/*` — urgent production fixes

        ## Common Commands
        - `git rebase -i HEAD~3` — interactive rebase last 3 commits
        - `git stash` — temporarily save uncommitted changes
        - `git cherry-pick <hash>` — apply specific commit to current branch
        \"\"\",
            "docker_basics.md": \"\"\"# Docker Basics
        ## Key Concepts
        - **Image**: Read-only template with instructions for creating a container
        - **Container**: Runnable instance of an image
        - **Dockerfile**: Script that defines how to build an image
        - **Volume**: Persistent data storage that survives container restarts

        ## Essential Commands
        ```bash
        docker build -t myapp .
        docker run -p 8080:80 myapp
        docker-compose up -d
        ```
        \"\"\",
        }
        for fname, content in notes.items():
            (notes_dir / fname).write_text(content, encoding="utf-8")
        print(f"Created {len(notes)} sample markdown notes in {notes_dir}/")
        """),
        md("## Step 2 — Configure LlamaIndex with Local Ollama"),
        code("""
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding

        Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        print("LlamaIndex configured with local Ollama models.")
        """),
        md("## Step 3 — Load Documents and Build Index"),
        code("""
        documents = SimpleDirectoryReader("sample_notes").load_data()
        print(f"Loaded {len(documents)} documents")
        for doc in documents:
            print(f"  - {doc.metadata.get('file_name', 'unknown')}: {len(doc.text)} chars")

        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        print("\\nVector index built successfully!")
        """),
        md("## Step 4 — Query Your Notes"),
        code("""
        query_engine = index.as_query_engine(similarity_top_k=3)

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
            print(f"Sources: {[n.metadata.get('file_name','?') for n in response.source_nodes]}")
        """),
        md("## Step 5 — Chat Mode with Memory"),
        code("""
        from llama_index.core.chat_engine import CondenseQuestionChatEngine

        chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine)

        conversation = [
            "Tell me about Docker volumes",
            "How do I persist data with them?",
            "What command starts containers in the background?",
        ]
        for msg in conversation:
            print(f"\\nUser: {msg}")
            response = chat_engine.chat(msg)
            print(f"Bot: {response}")
        """),
        md("## Step 6 — Add Metadata Filtering"),
        code("""
        from llama_index.core import Document

        # Rebuild with explicit metadata
        enriched_docs = []
        for doc in documents:
            fname = doc.metadata.get('file_name', '')
            topic = fname.replace('.md', '').replace('_', ' ').title()
            enriched_docs.append(Document(
                text=doc.text,
                metadata={"topic": topic, "file": fname, "type": "notes"}
            ))

        enriched_index = VectorStoreIndex.from_documents(enriched_docs)
        enriched_qe = enriched_index.as_query_engine(similarity_top_k=2)

        resp = enriched_qe.query("What are essential Docker commands?")
        print(f"Answer: {resp}")
        print(f"Metadata: {[n.metadata for n in resp.source_nodes]}")
        """),
        md("""
        ## What You Learned
        - **Markdown loading** with SimpleDirectoryReader
        - **LlamaIndex VectorStoreIndex** for semantic search
        - **Chat engine** with conversation memory
        - **Metadata enrichment** for better filtering
        """),
    ]))

    # ── Project 3: Local Meeting Notes Summarizer ───────────────────────
    paths.append(write_nb(1, "03_Local_Meeting_Notes_Summarizer", [
        md("""
        # Project 3 — Local Meeting Notes Summarizer
        ## Summarize Transcripts into Actions, Decisions, and Blockers

        **What you'll learn:**
        - Structured summarization with LangChain
        - Output parsing into defined sections
        - Chain-of-thought extraction of action items
        - Iterative refinement with map-reduce

        **Stack:** Ollama · LangChain · Pydantic · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Set Up Local LLM"),
        code("""
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)
        print("LLM ready for summarization!")
        """),
        md("## Step 2 — Create Sample Meeting Transcripts"),
        code("""
        transcripts = {
            "sprint_planning": \"\"\"
        Sprint Planning Meeting — 2025-01-15
        Attendees: Alice (PM), Bob (Backend), Carol (Frontend), Dave (QA)

        Alice: Let's review the backlog. We have 15 tickets for this sprint.
        Bob: The API refactoring is almost done. I need 2 more days for the auth module.
        Carol: The dashboard redesign is blocked by the new API endpoints. Bob, can you
        prioritize the /users endpoint?
        Bob: Sure, I'll have that ready by Wednesday.
        Dave: I found 3 critical bugs in the payment flow. They need to be fixed before release.
        Alice: Let's mark those as P0. Bob, can you look at the payment bugs first?
        Bob: I can start on that tomorrow.
        Carol: I also need the design tokens from the design team. I'll follow up today.
        Alice: Good. Let's aim to have all P0 items done by Thursday. Sprint demo is Friday.
        Dave: I'll prepare the test cases for the new endpoints by Wednesday.
        \"\"\",
            "stakeholder_review": \"\"\"
        Stakeholder Review — 2025-01-20
        Attendees: VP Product, Engineering Lead, Marketing, Support Lead

        VP Product: Q4 revenue grew 18%. The new pricing tier drove most of the growth.
        Engineering: We shipped the real-time analytics feature. Latency is under 200ms.
        Marketing: The launch campaign reached 50K users. Conversion rate was 3.2%.
        Support Lead: Ticket volume increased 25% after the launch. Most issues are about
        the new billing system. We need better documentation.
        VP Product: Let's prioritize help docs this sprint. Can engineering add tooltips?
        Engineering: Yes, we can add contextual help. Need 1 week.
        Marketing: We should also update the knowledge base articles.
        VP Product: Decision: Prioritize documentation and tooltips over new features next sprint.
        \"\"\"
        }
        print(f"Created {len(transcripts)} sample transcripts")
        """),
        md("## Step 3 — Structured Summarization Chain"),
        code("""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"You are a meeting summarizer. Extract a structured summary with:

        ## Summary
        (2-3 sentence overview)

        ## Key Decisions
        - (each decision made)

        ## Action Items
        - [ ] (task) — Owner: (person) — Due: (date if mentioned)

        ## Blockers
        - (any blockers or dependencies identified)

        ## Follow-ups Needed
        - (items requiring follow-up)

        Be concise and factual. Only include information explicitly stated.\"\"\"),
            ("human", "Summarize this meeting transcript:\\n\\n{transcript}")
        ])

        summary_chain = summary_prompt | llm | StrOutputParser()

        for name, transcript in transcripts.items():
            print(f"\\n{'='*70}")
            print(f"Meeting: {name}")
            print('='*70)
            result = summary_chain.invoke({"transcript": transcript})
            print(result)
        """),
        md("## Step 4 — Pydantic-Structured Output"),
        code("""
        from pydantic import BaseModel, Field
        from typing import Optional

        class ActionItem(BaseModel):
            task: str = Field(description="The specific task to be done")
            owner: str = Field(description="Person responsible")
            due_date: Optional[str] = Field(None, description="Due date if mentioned")
            priority: str = Field(default="normal", description="P0/P1/normal")

        class MeetingSummary(BaseModel):
            title: str = Field(description="Meeting title")
            summary: str = Field(description="2-3 sentence overview")
            decisions: list[str] = Field(description="Key decisions made")
            action_items: list[ActionItem] = Field(description="Action items with owners")
            blockers: list[str] = Field(description="Blockers identified")

        structured_llm = llm.with_structured_output(MeetingSummary)

        result = structured_llm.invoke(
            f"Extract a structured summary from this meeting:\\n\\n{transcripts['sprint_planning']}"
        )
        print(f"Title: {result.title}")
        print(f"Summary: {result.summary}")
        print(f"\\nDecisions:")
        for d in result.decisions:
            print(f"  • {d}")
        print(f"\\nAction Items:")
        for ai in result.action_items:
            print(f"  [{ai.priority}] {ai.task} — Owner: {ai.owner} — Due: {ai.due_date}")
        print(f"\\nBlockers:")
        for b in result.blockers:
            print(f"  ⚠ {b}")
        """),
        md("## Step 5 — Batch Processing with Map-Reduce"),
        code("""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # Map: summarize each transcript individually
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this meeting into 3-5 bullet points of key outcomes."),
            ("human", "{transcript}")
        ])
        map_chain = map_prompt | llm | StrOutputParser()

        # Reduce: combine individual summaries
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"You are given summaries from multiple meetings. Produce a
        consolidated weekly report with: Overall Progress, Combined Action Items,
        Cross-cutting Blockers, and Priorities for Next Week.\"\"\"),
            ("human", "Individual meeting summaries:\\n\\n{summaries}")
        ])
        reduce_chain = reduce_prompt | llm | StrOutputParser()

        # Execute map-reduce
        individual_summaries = []
        for name, transcript in transcripts.items():
            summary = map_chain.invoke({"transcript": transcript})
            individual_summaries.append(f"### {name}\\n{summary}")
            print(f"Summarized: {name}")

        combined = "\\n\\n".join(individual_summaries)
        weekly_report = reduce_chain.invoke({"summaries": combined})
        print(f"\\n{'='*70}")
        print("WEEKLY CONSOLIDATED REPORT")
        print('='*70)
        print(weekly_report)
        """),
        md("""
        ## What You Learned
        - **Structured prompts** for meeting summarization
        - **Pydantic output parsing** for machine-readable summaries
        - **Map-reduce pattern** for multi-document summarization
        - **Action item extraction** with metadata (owner, priority, due date)
        """),
    ]))

    # ── Project 4: Local Resume Rewriter ────────────────────────────────
    paths.append(write_nb(1, "04_Local_Resume_Rewriter", [
        md("""
        # Project 4 — Local Resume Rewriter
        ## Improve Resume Bullets Using STAR Format + Local LLM

        **What you'll learn:**
        - Parse resume sections (experience, skills, education)
        - Rewrite bullets using STAR methodology
        - Tailor content to a specific job description
        - Score resume-JD fit quantitatively

        **Stack:** Ollama · LangChain · Pydantic · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.3)
        print("LLM ready!")
        """),
        md("## Step 2 — Sample Resume and Job Description"),
        code("""
        sample_resume = \"\"\"
        EXPERIENCE
        Software Engineer, Acme Corp (2022-2024)
        - Worked on backend services
        - Helped improve system performance
        - Did code reviews and mentored junior developers
        - Built APIs for the mobile app

        Data Analyst Intern, BigData Inc (2021-2022)
        - Analyzed data and made reports
        - Used Python and SQL
        - Helped with dashboard creation

        SKILLS
        Python, SQL, Docker, AWS, React, Git

        EDUCATION
        B.S. Computer Science, State University, 2021
        \"\"\"

        target_jd = \"\"\"
        Senior Backend Engineer — TechCo
        Requirements:
        - 3+ years building scalable backend services
        - Experience with Python, FastAPI, PostgreSQL
        - Proven track record of improving system performance (latency, throughput)
        - Experience mentoring engineers and leading code reviews
        - Familiarity with CI/CD pipelines and containerization (Docker, K8s)
        - Strong communication and documentation skills
        \"\"\"
        print("Resume and JD loaded.")
        """),
        md("## Step 3 — Rewrite Bullets with STAR Method"),
        code("""
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"You are an expert resume coach. Rewrite each resume bullet
        using the STAR method (Situation, Task, Action, Result). Make bullets:
        - Start with a strong action verb
        - Include quantifiable metrics where possible
        - Be specific about technologies and impact
        - Keep each bullet to 1-2 lines

        If a bullet is vague, infer reasonable details to make it concrete.\"\"\"),
            ("human", \"\"\"Rewrite these resume bullets in STAR format:

        {bullets}

        Return each original bullet followed by its improved version.\"\"\")
        ])

        rewrite_chain = rewrite_prompt | llm | StrOutputParser()

        result = rewrite_chain.invoke({"bullets": sample_resume})
        print(result)
        """),
        md("## Step 4 — Tailor Resume to Job Description"),
        code("""
        tailor_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"You are a resume tailoring expert. Given a resume and target JD:
        1. Identify which resume experiences match JD requirements
        2. Rewrite bullets to emphasize relevant skills and outcomes
        3. Suggest which skills to highlight/add
        4. Recommend section ordering for maximum impact\"\"\"),
            ("human", \"\"\"Resume:
        {resume}

        Target Job Description:
        {jd}

        Produce a tailored resume with rewritten bullets optimized for this JD.\"\"\")
        ])

        tailor_chain = tailor_prompt | llm | StrOutputParser()
        tailored = tailor_chain.invoke({"resume": sample_resume, "jd": target_jd})
        print(tailored)
        """),
        md("## Step 5 — Score Resume-JD Fit"),
        code("""
        from pydantic import BaseModel, Field

        class FitScore(BaseModel):
            overall_score: int = Field(description="1-100 fit score")
            matched_requirements: list[str] = Field(description="JD requirements matched")
            gaps: list[str] = Field(description="JD requirements not addressed")
            recommendations: list[str] = Field(description="Specific improvements to make")

        scoring_llm = llm.with_structured_output(FitScore)
        fit = scoring_llm.invoke(
            f\"\"\"Score this resume against the job description.

        Resume: {sample_resume}

        Job Description: {target_jd}

        Provide a detailed fit analysis.\"\"\"
        )

        print(f"Overall Fit Score: {fit.overall_score}/100")
        print(f"\\nMatched Requirements ({len(fit.matched_requirements)}):")
        for m in fit.matched_requirements:
            print(f"  ✓ {m}")
        print(f"\\nGaps ({len(fit.gaps)}):")
        for g in fit.gaps:
            print(f"  ✗ {g}")
        print(f"\\nRecommendations:")
        for r in fit.recommendations:
            print(f"  → {r}")
        """),
        md("""
        ## What You Learned
        - **STAR method** for resume bullet rewriting
        - **JD-resume tailoring** with targeted prompts
        - **Structured scoring** with Pydantic outputs
        - **Quantitative fit analysis** to prioritize improvements
        """),
    ]))

    # ── Project 5: Local Cover Letter Generator ─────────────────────────
    paths.append(write_nb(1, "05_Local_Cover_Letter_Generator", [
        md("""
        # Project 5 — Local Cover Letter Generator
        ## Generate Tailored Cover Letters from JD + Resume

        **Stack:** Ollama · LangChain · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.4)
        """),
        md("## Step 2 — Input Data"),
        code("""
        resume_highlights = \"\"\"
        - 3 years Python backend (FastAPI, Django)
        - Built microservices handling 10K RPM
        - Led migration from monolith to microservices, reducing deploy time 60%
        - Mentored 4 junior engineers through code reviews
        - Experience with PostgreSQL, Redis, Docker, Kubernetes
        \"\"\"

        job_description = \"\"\"
        Senior Backend Engineer at CloudScale Inc.
        - Build and maintain Python microservices (FastAPI)
        - Design scalable APIs for 100K+ concurrent users
        - Lead technical design reviews
        - Mentor team of 5-8 engineers
        - Experience with cloud deployments (AWS/GCP)
        \"\"\"

        company_info = \"\"\"
        CloudScale Inc. is a B2B SaaS company in the observability space.
        They value engineering craftsmanship and open-source contributions.
        Their tech blog emphasizes clean code and testing.
        \"\"\"
        print("Input data ready.")
        """),
        md("## Step 3 — Multi-Paragraph Cover Letter Generation"),
        code("""
        cover_letter_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"You write professional cover letters. Structure:
        1. Opening — show enthusiasm + mention the role by name
        2. Why you — 2 paragraphs connecting YOUR experience to THEIR needs
        3. Why them — show you researched the company
        4. Closing — call to action, thank them

        Rules:
        - Be specific, not generic. Reference actual projects and metrics.
        - Match the company's tone (formal/startup-casual)
        - Keep under 400 words
        - Never fabricate experience not in the resume\"\"\"),
            ("human", \"\"\"Write a cover letter for this role.

        Resume highlights:
        {resume}

        Job Description:
        {jd}

        Company Info:
        {company}

        Generate the complete cover letter.\"\"\")
        ])

        chain = cover_letter_prompt | llm | StrOutputParser()

        letter = chain.invoke({
            "resume": resume_highlights,
            "jd": job_description,
            "company": company_info,
        })
        print(letter)
        """),
        md("## Step 4 — Tone Variation Generator"),
        code("""
        tone_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the following cover letter in a {tone} tone. Keep the core "
             "content but adjust the language, formality level, and energy."),
            ("human", "{letter}")
        ])
        tone_chain = tone_prompt | llm | StrOutputParser()

        for tone in ["confident and assertive", "humble and collaborative", "concise and technical"]:
            print(f"\\n{'='*60}")
            print(f"TONE: {tone}")
            print('='*60)
            variant = tone_chain.invoke({"letter": letter, "tone": tone})
            print(variant[:500] + "...")
        """),
        md("## Step 5 — Quality Check"),
        code("""
        from pydantic import BaseModel, Field

        class LetterReview(BaseModel):
            score: int = Field(description="Quality score 1-10")
            strengths: list[str]
            weaknesses: list[str]
            fabrication_risk: bool = Field(description="Does it claim experience not in resume?")

        reviewer = llm.with_structured_output(LetterReview)
        review = reviewer.invoke(
            f\"\"\"Review this cover letter against the resume.

        Cover Letter: {letter}

        Resume: {resume_highlights}

        Score the letter and identify any fabricated claims.\"\"\"
        )
        print(f"Score: {review.score}/10")
        print(f"Fabrication Risk: {review.fabrication_risk}")
        print(f"Strengths: {review.strengths}")
        print(f"Weaknesses: {review.weaknesses}")
        """),
        md("""
        ## What You Learned
        - **Multi-section cover letter** generation from structured inputs
        - **Tone variation** for different company cultures
        - **Self-review pipeline** to catch fabrication and quality issues
        """),
    ]))

    # ── Project 6: Local Email Reply Assistant ──────────────────────────
    paths.append(write_nb(1, "06_Local_Email_Reply_Assistant", [
        md("""
        # Project 6 — Local Email Reply Assistant
        ## Classify Email Intent and Draft Context-Aware Replies

        **Stack:** Ollama · LangChain · Pydantic · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pydantic import BaseModel, Field
        from enum import Enum

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)
        """),
        md("## Step 2 — Sample Emails"),
        code("""
        emails = [
            {
                "from": "client@example.com",
                "subject": "Urgent: Production API returning 500 errors",
                "body": \"\"\"Hi Team,
        Since 2pm today our dashboard is showing 500 errors on the /analytics endpoint.
        This is blocking our quarterly reporting. Can someone look into this ASAP?
        - Sarah, VP Engineering\"\"\"
            },
            {
                "from": "hr@company.com",
                "subject": "Team offsite planning",
                "body": \"\"\"Hi everyone,
        We're planning a team offsite for next month. Please fill out the survey by Friday
        with your date preferences and activity suggestions.
        Thanks, HR Team\"\"\"
            },
            {
                "from": "vendor@saas.io",
                "subject": "Your subscription renewal",
                "body": \"\"\"Dear Customer,
        Your annual subscription ($2,400/yr) renews on Feb 15. We've added new features
        including advanced analytics and API rate limit increases. If you'd like to
        discuss pricing or upgrade options, let me know.
        Best, Account Manager\"\"\"
            },
        ]
        print(f"Loaded {len(emails)} sample emails")
        """),
        md("## Step 3 — Email Classification"),
        code("""
        class EmailCategory(str, Enum):
            URGENT_ISSUE = "urgent_issue"
            ACTION_REQUIRED = "action_required"
            INFORMATION = "information"
            SALES = "sales"
            SCHEDULING = "scheduling"

        class EmailClassification(BaseModel):
            category: EmailCategory
            urgency: int = Field(description="1-5 urgency level")
            sentiment: str = Field(description="positive, neutral, negative")
            key_entities: list[str] = Field(description="People, products, dates mentioned")
            requires_response: bool
            suggested_response_time: str = Field(description="e.g., 'within 1 hour', 'by Friday'")

        classifier = llm.with_structured_output(EmailClassification)

        for email in emails:
            print(f"\\nSubject: {email['subject']}")
            result = classifier.invoke(
                f"Classify this email:\\nFrom: {email['from']}\\n"
                f"Subject: {email['subject']}\\nBody: {email['body']}"
            )
            print(f"  Category: {result.category.value}")
            print(f"  Urgency: {result.urgency}/5")
            print(f"  Sentiment: {result.sentiment}")
            print(f"  Entities: {result.key_entities}")
            print(f"  Needs Response: {result.requires_response} ({result.suggested_response_time})")
        """),
        md("## Step 4 — Context-Aware Reply Drafting"),
        code("""
        reply_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"You draft professional email replies. Rules:
        - Match the formality level of the original email
        - For urgent issues: acknowledge, provide timeline, ask clarifying questions
        - For scheduling: confirm availability or suggest alternatives
        - For sales: be polite but noncommittal unless clearly interested
        - Keep replies concise (under 150 words)
        - Include a clear next step or call to action\"\"\"),
            ("human", \"\"\"Draft a reply to this email:
        From: {sender}
        Subject: {subject}
        Body: {body}

        My role: Senior Software Engineer
        My context: {context}\"\"\")
        ])

        reply_chain = reply_prompt | llm | StrOutputParser()

        contexts = [
            "I'm the on-call engineer this week. I can check the logs.",
            "I'm available most dates next month except the 15th-17th.",
            "Our contract is up for review. Budget is tight this quarter.",
        ]

        for email, ctx in zip(emails, contexts):
            print(f"\\n{'='*60}")
            print(f"RE: {email['subject']}")
            print('='*60)
            reply = reply_chain.invoke({
                "sender": email["from"],
                "subject": email["subject"],
                "body": email["body"],
                "context": ctx,
            })
            print(reply)
        """),
        md("## Step 5 — Batch Processing Pipeline"),
        code("""
        def process_email(email, context=""):
            classification = classifier.invoke(
                f"From: {email['from']}\\nSubject: {email['subject']}\\nBody: {email['body']}"
            )
            reply = reply_chain.invoke({
                "sender": email["from"], "subject": email["subject"],
                "body": email["body"], "context": context,
            })
            return {"classification": classification, "reply": reply}

        # Process all emails
        results = []
        for email, ctx in zip(emails, contexts):
            r = process_email(email, ctx)
            results.append(r)

        # Priority-sorted output
        sorted_results = sorted(results, key=lambda x: x["classification"].urgency, reverse=True)
        for r in sorted_results:
            c = r["classification"]
            print(f"[Urgency {c.urgency}] {c.category.value} — Response: {c.suggested_response_time}")
        """),
        md("""
        ## What You Learned
        - **Email classification** with structured Pydantic output
        - **Context-aware reply drafting** with persona and situation context
        - **Priority-based batch processing** pipeline
        """),
    ]))

    # ── Project 7: Local Research Paper Explainer ───────────────────────
    paths.append(write_nb(1, "07_Local_Research_Paper_Explainer", [
        md("""
        # Project 7 — Local Research Paper Explainer
        ## Explain Research Papers in Plain English

        **Stack:** Ollama · LlamaIndex · Jupyter
        """),
        code("# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama"),
        md("## Step 1 — Setup and Sample Paper"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)

        sample_paper = {
            "title": "Attention Is All You Need",
            "abstract": \"\"\"The dominant sequence transduction models are based on complex
        recurrent or convolutional neural networks that include an encoder and a decoder.
        The best performing models also connect the encoder and decoder through an
        attention mechanism. We propose a new simple network architecture, the Transformer,
        based solely on attention mechanisms, dispensing with recurrence and convolutions
        entirely. Experiments on two machine translation tasks show these models to be
        superior in quality while being more parallelizable and requiring significantly
        less time to train.\"\"\",
            "sections": {
                "Introduction": \"\"\"Recurrent neural networks, long short-term memory and gated
        recurrent neural networks have been established as state of the art approaches in
        sequence modeling and transduction problems. The Transformer is the first
        transduction model relying entirely on self-attention to compute representations.\"\"\",
                "Self-Attention": \"\"\"An attention function maps a query and a set of key-value
        pairs to an output. We call our particular attention Scaled Dot-Product Attention.
        Multi-Head Attention allows the model to jointly attend to information from
        different representation subspaces at different positions.\"\"\",
                "Architecture": \"\"\"The encoder maps an input sequence to a sequence of continuous
        representations. The decoder generates an output sequence of symbols one at a time.
        The Transformer follows this overall architecture using stacked self-attention and
        point-wise, fully connected layers for both encoder and decoder.\"\"\"
            }
        }
        print(f"Loaded paper: {sample_paper['title']}")
        """),
        md("## Step 2 — Section-by-Section Explanation"),
        code("""
        explain_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"You explain research paper sections in plain English.
        Target audience: smart undergraduate who hasn't studied this field.
        - Use analogies and everyday examples
        - Define technical terms when first used
        - Keep explanations under 200 words per section
        - Highlight the key insight of each section\"\"\"),
            ("human", \"\"\"Paper: {title}

        Section: {section_name}
        Content: {section_content}

        Explain this section in plain English.\"\"\")
        ])

        explain_chain = explain_prompt | llm | StrOutputParser()

        print(f"# {sample_paper['title']}\\n")
        print("## Abstract (Plain English)")
        abstract_explained = explain_chain.invoke({
            "title": sample_paper["title"],
            "section_name": "Abstract",
            "section_content": sample_paper["abstract"],
        })
        print(abstract_explained)

        for section_name, content in sample_paper["sections"].items():
            print(f"\\n## {section_name} (Plain English)")
            explanation = explain_chain.invoke({
                "title": sample_paper["title"],
                "section_name": section_name,
                "section_content": content,
            })
            print(explanation)
        """),
        md("## Step 3 — Key Takeaways Extractor"),
        code("""
        from pydantic import BaseModel, Field

        class PaperTakeaways(BaseModel):
            one_sentence_summary: str
            key_contribution: str
            main_technique: str
            practical_applications: list[str]
            limitations: list[str]
            prerequisites_to_understand: list[str]

        takeaway_llm = llm.with_structured_output(PaperTakeaways)

        full_text = f"Title: {sample_paper['title']}\\nAbstract: {sample_paper['abstract']}"
        for name, content in sample_paper["sections"].items():
            full_text += f"\\n\\n{name}: {content}"

        takeaways = takeaway_llm.invoke(f"Extract key takeaways from this paper:\\n\\n{full_text}")
        print(f"One-Sentence Summary: {takeaways.one_sentence_summary}")
        print(f"Key Contribution: {takeaways.key_contribution}")
        print(f"Main Technique: {takeaways.main_technique}")
        print(f"\\nApplications:")
        for a in takeaways.practical_applications:
            print(f"  • {a}")
        print(f"\\nLimitations:")
        for l in takeaways.limitations:
            print(f"  • {l}")
        """),
        md("## Step 4 — Generate Quiz Questions"),
        code("""
        quiz_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate 5 quiz questions about this research paper. Include "
             "multiple choice (4 options each) and short answer questions. Provide the "
             "correct answers."),
            ("human", "{paper_text}")
        ])

        quiz_chain = quiz_prompt | llm | StrOutputParser()
        quiz = quiz_chain.invoke({"paper_text": full_text})
        print("Quiz Questions:")
        print(quiz)
        """),
        md("""
        ## What You Learned
        - **Section-by-section explanation** with audience-aware prompts
        - **Structured takeaway extraction** with Pydantic
        - **Quiz generation** from academic content
        """),
    ]))

    # ── Project 8: Local Blog-to-Thread Converter ───────────────────────
    paths.append(write_nb(1, "08_Local_Blog_to_Thread_Converter", [
        md("""
        # Project 8 — Local Blog-to-Thread Converter
        ## Repurpose Long-Form Content into Threads, Posts, and Emails

        **Stack:** Ollama · LangChain · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.5)

        blog_post = \"\"\"
        # Why RAG Beats Fine-Tuning for Most Enterprise Use Cases

        The debate between Retrieval-Augmented Generation and fine-tuning has evolved
        significantly. While fine-tuning creates specialized models, RAG offers advantages
        that make it the better default choice for most enterprise applications.

        **Cost Efficiency**: Fine-tuning requires curating datasets, running training jobs,
        and maintaining multiple model versions. RAG needs only a vector database and an
        embedding pipeline — often 10x cheaper to set up and maintain.

        **Data Freshness**: Fine-tuned models are frozen at training time. When your company
        policies, products, or documentation change, you need to retrain. RAG systems
        update instantly — just re-embed the new documents.

        **Transparency**: RAG provides source citations. Users can verify answers against
        the original documents. Fine-tuned models are black boxes — you can't trace where
        an answer came from.

        **Composability**: A single RAG system can serve multiple domains by swapping
        document collections. Fine-tuning requires separate models for each domain.

        **When Fine-Tuning Wins**: Style/tone adaptation, consistent formatting requirements,
        and high-throughput latency-sensitive applications where retrieval adds overhead.

        The pragmatic approach: Start with RAG, measure gaps, and fine-tune only when
        you have clear evidence that RAG alone can't meet your quality bar.
        \"\"\"
        print(f"Blog post loaded: {len(blog_post)} chars")
        """),
        md("## Step 2 — Convert to Twitter/X Thread"),
        code("""
        thread_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"Convert blog posts into engaging Twitter/X threads. Rules:
        - First tweet: hook that makes people want to read more
        - 5-10 tweets total, each under 280 characters
        - Use numbered format (1/, 2/, etc.)
        - Include a mix of insights, data points, and opinions
        - End with a takeaway or call to action
        - Use line breaks for readability\"\"\"),
            ("human", "Convert this blog post to a Twitter thread:\\n\\n{blog}")
        ])

        thread_chain = thread_prompt | llm | StrOutputParser()
        thread = thread_chain.invoke({"blog": blog_post})
        print("TWITTER THREAD:")
        print(thread)
        """),
        md("## Step 3 — Convert to LinkedIn Post"),
        code("""
        linkedin_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"Convert blog posts into LinkedIn posts. Rules:
        - Opening hook (first line visible before 'See more')
        - Professional but conversational tone
        - Use bullet points or numbered lists
        - Include a personal insight or opinion
        - End with a question to drive engagement
        - 150-300 words
        - Add relevant hashtags\"\"\"),
            ("human", "Convert this to a LinkedIn post:\\n\\n{blog}")
        ])

        linkedin_chain = linkedin_prompt | llm | StrOutputParser()
        post = linkedin_chain.invoke({"blog": blog_post})
        print("LINKEDIN POST:")
        print(post)
        """),
        md("## Step 4 — Convert to Newsletter Email"),
        code("""
        email_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"Convert blog posts into newsletter emails. Rules:
        - Compelling subject line
        - Personal greeting
        - TL;DR at the top (3 bullets)
        - Expanded discussion with examples
        - Clear CTA (reply, share, or link)
        - Casual but authoritative tone
        - Under 500 words\"\"\"),
            ("human", "Convert this blog to a newsletter email:\\n\\n{blog}")
        ])

        email_chain = email_prompt | llm | StrOutputParser()
        newsletter = email_chain.invoke({"blog": blog_post})
        print("NEWSLETTER EMAIL:")
        print(newsletter)
        """),
        md("## Step 5 — All-Formats Pipeline"),
        code("""
        formats = {
            "twitter_thread": thread_chain,
            "linkedin_post": linkedin_chain,
            "newsletter_email": email_chain,
        }

        print("=== CONTENT REPURPOSING PIPELINE ===")
        for fmt_name, chain in formats.items():
            print(f"\\n{'='*60}")
            print(f"FORMAT: {fmt_name}")
            print('='*60)
            result = chain.invoke({"blog": blog_post})
            print(result[:300] + "..." if len(result) > 300 else result)
            print(f"\\n[{len(result)} chars generated]")
        """),
        md("""
        ## What You Learned
        - **Platform-specific content adaptation** (Twitter, LinkedIn, Email)
        - **Tone and format constraints** via system prompts
        - **Batch content repurposing** pipeline
        """),
    ]))

    # ── Project 9: Local Study Notes Generator ──────────────────────────
    paths.append(write_nb(1, "09_Local_Study_Notes_Generator", [
        md("""
        # Project 9 — Local Study Notes Generator
        ## Turn Raw Text into Cornell Notes, Flashcards, and Quizzes

        **Stack:** Ollama · LangChain · Pydantic · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pydantic import BaseModel, Field

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)

        study_material = \"\"\"
        Operating Systems — Process Management

        A process is a program in execution. Each process has a Process Control Block (PCB)
        containing: process state, program counter, CPU registers, memory management info,
        and I/O status.

        Process States: New → Ready → Running → Waiting → Terminated.
        The scheduler moves processes between Ready and Running queues.

        Context switching saves the state of the current process and loads the state of
        the next. It involves saving/loading PCBs and is pure overhead — no useful work
        is done during a context switch. Typical cost: 1-10 microseconds.

        Scheduling Algorithms:
        - FCFS (First Come First Served): Simple but causes convoy effect
        - SJF (Shortest Job First): Optimal average wait time but requires knowing burst time
        - Round Robin: Fair, uses time quantum. Too small = overhead, too large = FCFS
        - Priority Scheduling: Risk of starvation, solved with aging
        \"\"\"
        print(f"Study material loaded: {len(study_material)} chars")
        """),
        md("## Step 2 — Generate Cornell Notes"),
        code("""
        cornell_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"Generate Cornell Notes format from the study material:

        CUES (left column) | NOTES (right column)
        Key questions/terms | Detailed notes, examples, diagrams
                           |
        ─────────────────────────────────────────
        SUMMARY: 3-5 sentence summary of the entire topic

        Rules:
        - Cues should be questions that the notes answer
        - Notes should be concise bullet points
        - Include examples where applicable
        - Summary should capture the main concepts\"\"\"),
            ("human", "Generate Cornell Notes for:\\n\\n{material}")
        ])

        cornell_chain = cornell_prompt | llm | StrOutputParser()
        cornell_notes = cornell_chain.invoke({"material": study_material})
        print(cornell_notes)
        """),
        md("## Step 3 — Generate Flashcards"),
        code("""
        class Flashcard(BaseModel):
            front: str = Field(description="Question or term")
            back: str = Field(description="Answer or definition")
            difficulty: str = Field(description="easy, medium, hard")

        class FlashcardDeck(BaseModel):
            topic: str
            cards: list[Flashcard]

        flashcard_llm = llm.with_structured_output(FlashcardDeck)
        deck = flashcard_llm.invoke(
            f"Generate 10 flashcards from this material:\\n\\n{study_material}"
        )
        print(f"Topic: {deck.topic}")
        print(f"Generated {len(deck.cards)} flashcards\\n")
        for i, card in enumerate(deck.cards):
            print(f"Card {i+1} [{card.difficulty}]:")
            print(f"  Q: {card.front}")
            print(f"  A: {card.back}\\n")
        """),
        md("## Step 4 — Generate Practice Quiz"),
        code("""
        class QuizQuestion(BaseModel):
            question: str
            options: list[str] = Field(description="4 options (A-D)")
            correct_answer: str = Field(description="Letter of correct option")
            explanation: str

        class Quiz(BaseModel):
            questions: list[QuizQuestion]

        quiz_llm = llm.with_structured_output(Quiz)
        quiz = quiz_llm.invoke(
            f"Generate 5 multiple choice questions:\\n\\n{study_material}"
        )
        print("=== PRACTICE QUIZ ===\\n")
        for i, q in enumerate(quiz.questions):
            print(f"Q{i+1}: {q.question}")
            for opt in q.options:
                print(f"   {opt}")
            print(f"   Answer: {q.correct_answer}")
            print(f"   Explanation: {q.explanation}\\n")
        """),
        md("## Step 5 — Study Session Simulator"),
        code("""
        def study_session(material, num_rounds=2):
            \"\"\"Run an automated study → quiz → review cycle.\"\"\"
            print("=== STUDY SESSION START ===\\n")

            # Round 1: Key concepts
            concepts_prompt = ChatPromptTemplate.from_messages([
                ("system", "List the 5 most important concepts from this material, "
                 "each with a one-sentence explanation."),
                ("human", "{material}")
            ])
            concepts = (concepts_prompt | llm | StrOutputParser()).invoke({"material": material})
            print("KEY CONCEPTS TO LEARN:")
            print(concepts)

            # Round 2: Self-test
            quiz = quiz_llm.invoke(f"Generate 3 quiz questions:\\n\\n{material}")
            print("\\n=== SELF-TEST ===")
            score = 0
            for i, q in enumerate(quiz.questions):
                print(f"\\nQ{i+1}: {q.question}")
                for opt in q.options:
                    print(f"  {opt}")
                print(f"  Correct: {q.correct_answer} — {q.explanation}")

            print(f"\\n=== SESSION COMPLETE ===")

        study_session(study_material)
        """),
        md("""
        ## What You Learned
        - **Cornell Notes** generation from raw text
        - **Flashcard decks** with structured output
        - **Quiz generation** with explanations
        - **Study session automation** combining multiple techniques
        """),
    ]))

    # ── Project 10: Local Code Explainer ────────────────────────────────
    paths.append(write_nb(1, "10_Local_Code_Explainer", [
        md("""
        # Project 10 — Local Code Explainer
        ## Explain Code Snippets and Detect Simple Issues

        **Stack:** Ollama · LangChain · Pydantic · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pydantic import BaseModel, Field

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)

        code_samples = {
            "binary_search": \"\"\"
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        \"\"\",
            "buggy_function": \"\"\"
        def calculate_average(numbers):
            total = 0
            for i in range(len(numbers)):
                total += numbers[i]
            average = total / len(numbers)
            return average

        result = calculate_average([])
        \"\"\",
            "complex_decorator": \"\"\"
        import functools
        import time

        def retry(max_attempts=3, delay=1):
            def decorator(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    for attempt in range(max_attempts):
                        try:
                            return func(*args, **kwargs)
                        except Exception as e:
                            if attempt == max_attempts - 1:
                                raise
                            time.sleep(delay * (2 ** attempt))
                    return None
                return wrapper
            return decorator
        \"\"\",
        }
        print(f"Loaded {len(code_samples)} code samples")
        """),
        md("## Step 2 — Line-by-Line Explanation"),
        code("""
        explain_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"You are a patient code tutor. Explain code line by line:
        - Explain what each line does in plain English
        - Note the algorithm or pattern being used
        - Explain time/space complexity
        - Use analogies for complex concepts
        - Target audience: intermediate Python developer\"\"\"),
            ("human", "Explain this code line by line:\\n```python\\n{code}\\n```")
        ])

        explain_chain = explain_prompt | llm | StrOutputParser()

        for name, sample in code_samples.items():
            print(f"\\n{'='*60}")
            print(f"Code: {name}")
            print('='*60)
            explanation = explain_chain.invoke({"code": sample})
            print(explanation)
        """),
        md("## Step 3 — Bug Detection"),
        code("""
        class Bug(BaseModel):
            line_number: int = Field(description="Approximate line number")
            severity: str = Field(description="critical, warning, info")
            description: str
            fix_suggestion: str

        class CodeReview(BaseModel):
            bugs: list[Bug]
            overall_quality: int = Field(description="1-10 code quality score")
            improvements: list[str]

        reviewer = llm.with_structured_output(CodeReview)

        for name, sample in code_samples.items():
            print(f"\\nReviewing: {name}")
            review = reviewer.invoke(f"Review this code for bugs and issues:\\n```python\\n{sample}\\n```")
            print(f"  Quality: {review.overall_quality}/10")
            for bug in review.bugs:
                print(f"  [{bug.severity}] Line ~{bug.line_number}: {bug.description}")
                print(f"    Fix: {bug.fix_suggestion}")
            for imp in review.improvements:
                print(f"  Suggestion: {imp}")
        """),
        md("## Step 4 — Code Refactoring Suggestions"),
        code("""
        refactor_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"Suggest how to refactor this code. Provide:
        1. The refactored code
        2. What changed and why
        3. Any Pythonic improvements (list comprehensions, generators, etc.)
        4. Error handling additions needed\"\"\"),
            ("human", "Refactor this code:\\n```python\\n{code}\\n```")
        ])

        refactor_chain = refactor_prompt | llm | StrOutputParser()
        print("Refactoring suggestions for 'buggy_function':")
        result = refactor_chain.invoke({"code": code_samples["buggy_function"]})
        print(result)
        """),
        md("## Step 5 — Complexity Analysis"),
        code("""
        complexity_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"Analyze the time and space complexity of this code.
        Provide:
        - Big-O time complexity with explanation
        - Big-O space complexity
        - Best/worst/average case analysis
        - Suggest optimization if complexity can be improved\"\"\"),
            ("human", "Analyze complexity:\\n```python\\n{code}\\n```")
        ])

        complexity_chain = complexity_prompt | llm | StrOutputParser()

        for name, sample in code_samples.items():
            print(f"\\n{'='*60}")
            print(f"Complexity: {name}")
            print('='*60)
            analysis = complexity_chain.invoke({"code": sample})
            print(analysis)
        """),
        md("""
        ## What You Learned
        - **Line-by-line code explanation** for any snippet
        - **Automated bug detection** with structured output
        - **Refactoring suggestions** with Pythonic improvements
        - **Complexity analysis** with Big-O notation
        """),
    ]))

    print(f"Group 1 complete: {len(paths)} notebooks written")
    for p in paths:
        print(f"  ✓ {p}")
    return paths

if __name__ == "__main__":
    build()
