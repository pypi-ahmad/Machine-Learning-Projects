"""Group 2 — Projects 11-20: Local RAG."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── Project 11: Local Website FAQ Bot ───────────────────────────────
    paths.append(write_nb(2, "11_Local_Website_FAQ_Bot", [
        md("""
        # Project 11 — Local Website FAQ Bot
        ## Ingest One Website and Answer Questions Over It

        **What you'll learn:**
        - Crawl/scrape a website into documents
        - Chunk HTML content intelligently
        - Build a local RAG pipeline for website Q&A
        - Handle noisy web content (nav, footers, boilerplate)

        **Stack:** Ollama · LangChain · ChromaDB · BeautifulSoup · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb beautifulsoup4"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        print("Models ready!")
        """),
        md("## Step 2 — Create Sample Website Content"),
        code("""
        from pathlib import Path
        import json

        data_dir = Path("sample_data"); data_dir.mkdir(exist_ok=True)

        # Simulated website pages (in practice, use WebBaseLoader or a crawler)
        pages = [
            {"url": "/pricing", "title": "Pricing", "content": \"\"\"
        Our Pricing Plans:
        - Starter: $9/mo — 1 user, 5GB storage, email support
        - Professional: $29/mo — 5 users, 50GB storage, priority support, API access
        - Enterprise: $99/mo — Unlimited users, 500GB, dedicated support, SSO, audit logs
        All plans include a 14-day free trial. Annual billing saves 20%.
        \"\"\"},
            {"url": "/features", "title": "Features", "content": \"\"\"
        Key Features:
        - Real-time Collaboration: Edit documents simultaneously with your team
        - AI-Powered Search: Find any document in seconds with semantic search
        - Version History: Track all changes with unlimited version history
        - Integrations: Connect with Slack, Jira, GitHub, and 50+ other tools
        - Security: SOC 2 Type II certified, end-to-end encryption at rest and in transit
        - API Access: RESTful API with comprehensive documentation
        \"\"\"},
            {"url": "/faq", "title": "FAQ", "content": \"\"\"
        Frequently Asked Questions:
        Q: Can I cancel anytime?
        A: Yes, you can cancel your subscription at any time. No cancellation fees.

        Q: Do you offer refunds?
        A: We offer a full refund within the first 30 days of any paid plan.

        Q: Is my data secure?
        A: Yes. We use AES-256 encryption and are SOC 2 Type II certified.

        Q: Can I export my data?
        A: Yes, you can export all your data in standard formats (CSV, JSON, PDF).

        Q: Do you support single sign-on (SSO)?
        A: SSO is available on the Enterprise plan.
        \"\"\"},
            {"url": "/about", "title": "About Us", "content": \"\"\"
        About TechDocs Inc.
        Founded in 2020, TechDocs helps teams organize and search their documentation
        efficiently. Our platform serves over 10,000 companies worldwide, from startups
        to Fortune 500 enterprises. Headquartered in San Francisco with remote teams
        across 15 countries.
        \"\"\"},
        ]

        (data_dir / "website_pages.json").write_text(json.dumps(pages, indent=2))
        print(f"Created {len(pages)} simulated website pages")
        """),
        md("## Step 3 — Load and Chunk Website Content"),
        code("""
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        documents = []
        for page in pages:
            doc = Document(
                page_content=page["content"].strip(),
                metadata={"url": page["url"], "title": page["title"], "source": "website"}
            )
            documents.append(doc)

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        print(f"Pages: {len(documents)} → Chunks: {len(chunks)}")
        for c in chunks:
            print(f"  [{c.metadata['title']}] {len(c.page_content)} chars")
        """),
        md("## Step 4 — Build Vector Store"),
        code("""
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings,
            persist_directory="sample_data/website_chroma",
            collection_name="website_faq",
        )
        print(f"Vector store created with {len(chunks)} chunks")

        # Test retrieval
        results = vectorstore.similarity_search("How much does the enterprise plan cost?", k=2)
        for r in results:
            print(f"  [{r.metadata['title']}] {r.page_content[:80]}...")
        """),
        md("## Step 5 — Build FAQ Bot Chain"),
        code("""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        faq_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=\"\"\"You are a helpful FAQ bot for TechDocs Inc. Answer the customer's
        question using ONLY the website content provided. If the answer isn't in the
        context, say "I don't have that information on our website. Please contact
        support@techdocs.com for help."

        Website Content:
        {context}

        Customer Question: {question}

        Answer (be specific, cite page sections when possible):\"\"\"
        )

        faq_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": faq_prompt},
        )
        print("FAQ bot ready!")
        """),
        md("## Step 6 — Test the FAQ Bot"),
        code("""
        questions = [
            "How much does the Professional plan cost?",
            "Do you offer SSO?",
            "Can I get a refund if I cancel after 2 weeks?",
            "How many companies use TechDocs?",
            "What integrations do you support?",
            "Do you have a mobile app?",  # Not in the content
        ]

        for q in questions:
            print(f"\\nQ: {q}")
            result = faq_chain.invoke({"query": q})
            print(f"A: {result['result']}")
            sources = [s.metadata['title'] for s in result['source_documents']]
            print(f"Sources: {sources}")
        """),
        md("""
        ## What You Learned
        - **Website content ingestion** with metadata
        - **Domain-specific RAG** for customer-facing FAQ bots
        - **Graceful fallback** when info is not in the knowledge base
        """),
    ]))

    # ── Project 12: Local Policy Assistant ──────────────────────────────
    paths.append(write_nb(2, "12_Local_Policy_Assistant", [
        md("""
        # Project 12 — Local Policy Assistant
        ## Search HR/IT/Company Policies with Citations

        **Stack:** Ollama · LangChain · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        """),
        md("## Step 2 — Create Sample Policy Documents"),
        code("""
        from pathlib import Path

        policies_dir = Path("sample_data/policies"); policies_dir.mkdir(parents=True, exist_ok=True)

        policies = {
            "remote_work_policy.txt": \"\"\"Remote Work Policy — Effective Jan 2025
        Section 1: Eligibility
        All full-time employees who have completed 90 days are eligible for remote work.
        Contractors and part-time employees require manager approval.

        Section 2: Equipment
        The company provides a laptop, monitor, and $500 home office stipend.
        Equipment must be returned within 14 days of separation.

        Section 3: Working Hours
        Core hours are 10am-3pm in your local timezone. Employees must be
        available on Slack during core hours. Total weekly hours: 40.

        Section 4: Security
        All work must be done on company-approved devices. Use the VPN for
        accessing internal systems. Do not use public WiFi without VPN.\"\"\",

            "pto_policy.txt": \"\"\"Paid Time Off Policy — Effective Jan 2025
        Section 1: Accrual
        Full-time employees accrue 1.67 days per month (20 days/year).
        Maximum accrual cap: 30 days. Unused PTO does not roll over beyond the cap.

        Section 2: Requesting PTO
        Submit requests via HR Portal at least 5 business days in advance.
        Requests of 5+ consecutive days require VP approval.

        Section 3: Holidays
        The company observes 10 federal holidays plus 2 floating holidays.
        Floating holidays must be used within the calendar year.

        Section 4: Sick Leave
        Employees receive 10 sick days per year, separate from PTO.
        A doctor's note is required for absences exceeding 3 consecutive days.\"\"\",

            "expense_policy.txt": \"\"\"Expense Reimbursement Policy — Effective Jan 2025
        Section 1: Eligible Expenses
        Business travel, client meals, conference fees, and work-related software.

        Section 2: Limits
        Meals: $50/person for client dinners, $25 for solo meals.
        Hotels: Up to $250/night in major metros, $175 elsewhere.
        Flights: Economy class for domestic, business class for 6+ hour flights.

        Section 3: Submission
        Submit expenses within 30 days via Concur. Receipts required for all
        expenses over $25. Late submissions may be denied.

        Section 4: Approval
        Expenses under $500: Manager approval.
        Expenses $500-$5000: Director approval.
        Expenses over $5000: VP approval.\"\"\",
        }

        for fname, content in policies.items():
            (policies_dir / fname).write_text(content, encoding="utf-8")
        print(f"Created {len(policies)} policy documents")
        """),
        md("## Step 3 — Index Policies with Metadata"),
        code("""
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        loader = DirectoryLoader("sample_data/policies", glob="*.txt", loader_cls=TextLoader,
                                 loader_kwargs={"encoding": "utf-8"})
        docs = loader.load()

        # Enrich metadata
        for doc in docs:
            fname = Path(doc.metadata["source"]).name
            doc.metadata["policy_name"] = fname.replace("_policy.txt", "").replace("_", " ").title()
            doc.metadata["type"] = "company_policy"

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(chunks, embeddings,
            persist_directory="sample_data/policy_chroma", collection_name="policies")

        print(f"Indexed {len(chunks)} policy chunks from {len(docs)} documents")
        """),
        md("## Step 4 — Policy Q&A with Section Citations"),
        code("""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        policy_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=\"\"\"You are the company's HR policy assistant. Answer questions
        using ONLY the policy documents provided. Always cite the specific
        policy name and section number.

        Format: Answer the question, then cite as "Source: [Policy Name], Section X"

        If the answer is not in the policies, say "This is not covered in our
        current policies. Please contact HR at hr@company.com."

        Policy excerpts:
        {context}

        Employee Question: {question}

        Answer (with citations):\"\"\"
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": policy_prompt},
        )

        questions = [
            "How many PTO days do I get per year?",
            "What's the hotel reimbursement limit in New York?",
            "Am I eligible for remote work as a contractor?",
            "Do I need approval for a $600 expense?",
            "What happens to unused PTO?",
            "Can I use public WiFi when working remotely?",
        ]

        for q in questions:
            print(f"\\nQ: {q}")
            result = qa.invoke({"query": q})
            print(f"A: {result['result']}")
            policies_cited = set(s.metadata.get('policy_name', '?') for s in result['source_documents'])
            print(f"Policies referenced: {policies_cited}")
        """),
        md("""
        ## What You Learned
        - **Multi-document policy indexing** with metadata enrichment
        - **Citation-aware retrieval** with specific section references
        - **Graceful fallback** for uncovered topics
        """),
    ]))

    # ── Project 13: Local Multi-PDF Research Librarian ──────────────────
    paths.append(write_nb(2, "13_Local_Multi_PDF_Research_Librarian", [
        md("""
        # Project 13 — Local Multi-PDF Research Librarian
        ## Answer Questions Across Multiple Papers with Evidence

        **Stack:** Ollama · LlamaIndex · Jupyter
        """),
        code("# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama"),
        md("## Step 1 — Setup"),
        code("""
        from llama_index.core import Settings, VectorStoreIndex, Document
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding

        Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        """),
        md("## Step 2 — Create Sample Research Papers"),
        code("""
        papers = [
            Document(text=\"\"\"Title: Attention Mechanisms in Neural Networks
        Author: Smith et al., 2023

        Abstract: This paper surveys attention mechanisms from basic additive attention
        to multi-head self-attention used in transformers. We find that attention improves
        model performance by 15-30% on sequence tasks while adding minimal parameters.

        Key Findings:
        1. Self-attention scales quadratically with sequence length O(n²)
        2. Multi-head attention captures different types of relationships
        3. Relative position encodings outperform absolute ones
        4. Flash attention reduces memory from O(n²) to O(n)\"\"\",
            metadata={"paper_id": "P001", "year": 2023, "topic": "attention"}),

            Document(text=\"\"\"Title: Efficient Retrieval-Augmented Generation
        Author: Johnson et al., 2024

        Abstract: We propose improvements to RAG pipelines that reduce latency by 40%
        while maintaining answer quality. Key innovations include hierarchical retrieval,
        context compression, and adaptive chunk sizing.

        Key Findings:
        1. Hierarchical retrieval (summary→detail) reduces retrievals by 60%
        2. Context compression removes 40% of tokens with <2% quality loss
        3. Adaptive chunks (200-800 tokens) outperform fixed-size chunks
        4. Hybrid BM25+dense retrieval improves recall by 25%\"\"\",
            metadata={"paper_id": "P002", "year": 2024, "topic": "RAG"}),

            Document(text=\"\"\"Title: Local Language Models for Enterprise
        Author: Williams et al., 2024

        Abstract: We evaluate running 7B-13B parameter models locally for enterprise
        use cases. Results show local models achieve 85-92% of GPT-4 quality on
        domain-specific tasks when combined with RAG and fine-tuning.

        Key Findings:
        1. 7B models with RAG match GPT-4 on closed-domain tasks
        2. Fine-tuning on 1000 examples improves accuracy by 20%
        3. Quantized models (4-bit) run at 30+ tokens/sec on consumer GPUs
        4. Privacy-sensitive industries prefer local deployment 3:1\"\"\",
            metadata={"paper_id": "P003", "year": 2024, "topic": "local_llm"}),
        ]

        print(f"Created {len(papers)} sample research papers")
        """),
        md("## Step 3 — Build Cross-Paper Index"),
        code("""
        index = VectorStoreIndex.from_documents(papers, show_progress=True)
        query_engine = index.as_query_engine(similarity_top_k=3)
        print("Cross-paper index built!")
        """),
        md("## Step 4 — Cross-Reference Research Questions"),
        code("""
        research_questions = [
            "What are the key findings about attention mechanism efficiency?",
            "How do local models compare to GPT-4 for enterprise use?",
            "What techniques reduce RAG latency?",
            "Compare the findings about retrieval improvements across papers.",
        ]

        for q in research_questions:
            print(f"\\n{'='*60}")
            print(f"Research Q: {q}")
            response = query_engine.query(q)
            print(f"\\nAnswer: {response}")
            print(f"\\nEvidence from:")
            for node in response.source_nodes:
                pid = node.metadata.get('paper_id', '?')
                score = node.score if hasattr(node, 'score') else 'N/A'
                print(f"  [{pid}] relevance={score}: {node.text[:80]}...")
        """),
        md("## Step 5 — Research Synthesis"),
        code("""
        from llama_index.core.query_engine import CitationQueryEngine

        # Build a citation-aware query engine
        citation_engine = CitationQueryEngine.from_args(index, similarity_top_k=3, citation_chunk_size=256)

        synthesis_q = "Synthesize the key findings across all papers about improving AI system efficiency."
        response = citation_engine.query(synthesis_q)
        print("RESEARCH SYNTHESIS:")
        print(response)
        print("\\nCitations:")
        for i, node in enumerate(response.source_nodes):
            print(f"  [{i+1}] {node.text[:100]}...")
        """),
        md("""
        ## What You Learned
        - **Multi-document indexing** with paper metadata
        - **Cross-reference queries** across papers
        - **Citation-aware synthesis** combining evidence
        """),
    ]))

    # ── Project 14: Local Financial Report Analyst ──────────────────────
    paths.append(write_nb(2, "14_Local_Financial_Report_Analyst", [
        md("""
        # Project 14 — Local Financial Report Analyst
        ## QA Over Annual Reports with Table + Text Parsing

        **Stack:** Ollama · LangChain · pandas · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb pandas"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import pandas as pd

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        """),
        md("## Step 2 — Create Sample Financial Data"),
        code("""
        from pathlib import Path
        Path("sample_data").mkdir(exist_ok=True)

        # Simulated financial statements
        income_data = {
            "Year": [2022, 2023, 2024],
            "Revenue_M": [450, 580, 720],
            "COGS_M": [180, 220, 270],
            "Gross_Profit_M": [270, 360, 450],
            "Operating_Expenses_M": [150, 180, 210],
            "Net_Income_M": [85, 128, 175],
        }
        df = pd.DataFrame(income_data)
        df.to_csv("sample_data/income_statement.csv", index=False)
        print(df.to_string(index=False))

        narrative = \"\"\"
        Annual Report 2024 — TechCorp Inc.

        Financial Highlights:
        Revenue grew 24% year-over-year to $720M, driven by enterprise SaaS expansion.
        Gross margins improved to 62.5% from 62.1% as we optimized cloud infrastructure.
        Net income reached $175M, a 37% increase, reflecting operating leverage.

        Strategic Initiatives:
        - Launched AI-powered analytics platform (Q2 2024), contributing $45M in new ARR
        - Expanded to 3 new international markets (Germany, Japan, Brazil)
        - Acquired DataViz Corp for $50M to enhance visualization capabilities

        Risk Factors:
        - Enterprise sales cycles lengthening from 45 to 60 days
        - Currency headwinds expected to impact 2025 revenue by 2-3%
        - Key competitor launched similar AI features at lower price point
        \"\"\"
        Path("sample_data/annual_report.txt").write_text(narrative, encoding="utf-8")
        print("\\nFinancial data and narrative report created.")
        """),
        md("## Step 3 — Build Combined Text + Table Index"),
        code("""
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        # Load narrative
        narrative_doc = Document(
            page_content=Path("sample_data/annual_report.txt").read_text(encoding="utf-8"),
            metadata={"type": "narrative", "source": "annual_report"}
        )

        # Convert table to natural language for indexing
        table_text = "Income Statement Summary:\\n"
        for _, row in df.iterrows():
            table_text += (f"Year {int(row['Year'])}: Revenue ${row['Revenue_M']}M, "
                          f"Gross Profit ${row['Gross_Profit_M']}M "
                          f"({row['Gross_Profit_M']/row['Revenue_M']*100:.1f}% margin), "
                          f"Net Income ${row['Net_Income_M']}M\\n")

        table_doc = Document(
            page_content=table_text,
            metadata={"type": "financial_table", "source": "income_statement"}
        )

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        all_chunks = splitter.split_documents([narrative_doc, table_doc])

        vectorstore = Chroma.from_documents(all_chunks, embeddings,
            persist_directory="sample_data/financial_chroma", collection_name="financial")
        print(f"Indexed {len(all_chunks)} chunks (narrative + tables)")
        """),
        md("## Step 4 — Financial Q&A"),
        code("""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        fin_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=\"\"\"You are a financial analyst assistant. Answer questions about the
        company using the provided financial data and reports.

        When citing numbers, be precise and show calculations where applicable.

        Data:
        {context}

        Question: {question}

        Analysis:\"\"\"
        )

        fin_qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": fin_prompt},
        )

        questions = [
            "What was the revenue growth rate from 2023 to 2024?",
            "What are the main risk factors for 2025?",
            "How much did the AI analytics platform contribute in ARR?",
            "Calculate the net income margin trend over 3 years.",
        ]

        for q in questions:
            print(f"\\nQ: {q}")
            result = fin_qa.invoke({"query": q})
            print(f"A: {result['result']}")
        """),
        md("## Step 5 — Trend Analysis with pandas"),
        code("""
        # Compute key metrics
        df["Gross_Margin_Pct"] = (df["Gross_Profit_M"] / df["Revenue_M"] * 100).round(1)
        df["Net_Margin_Pct"] = (df["Net_Income_M"] / df["Revenue_M"] * 100).round(1)
        df["Revenue_Growth_Pct"] = df["Revenue_M"].pct_change().mul(100).round(1)

        print("Key Financial Metrics:")
        print(df[["Year", "Revenue_M", "Gross_Margin_Pct", "Net_Margin_Pct", "Revenue_Growth_Pct"]].to_string(index=False))

        # Generate natural language summary of trends
        trend_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a financial analyst. Summarize key trends from the data."),
            ("human", "Analyze these trends:\\n{data}")
        ])
        trend_chain = trend_prompt | llm | StrOutputParser()
        analysis = trend_chain.invoke({"data": df.to_string()})
        print(f"\\nTrend Analysis:\\n{analysis}")
        """),
        md("""
        ## What You Learned
        - **Combined table + text RAG** for financial analysis
        - **Table-to-text conversion** for indexing structured data
        - **Financial metrics computation** with pandas
        - **Trend analysis** combining quantitative and qualitative data
        """),
    ]))

    # ── Project 15: Local Contract Clause Finder ────────────────────────
    paths.append(write_nb(2, "15_Local_Contract_Clause_Finder", [
        md("""
        # Project 15 — Local Contract Clause Finder
        ## Retrieve Risky or Important Clauses from Contracts

        **Stack:** Ollama · LangChain · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        llm = ChatOllama(model="qwen3:8b", temperature=0.1)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        """),
        md("## Step 2 — Sample Contract Content"),
        code("""
        from langchain.schema import Document

        contract_clauses = [
            Document(page_content=\"\"\"LIMITATION OF LIABILITY: In no event shall either party
        be liable for any indirect, incidental, special, consequential, or punitive
        damages, regardless of the cause of action. The total cumulative liability
        shall not exceed the fees paid in the 12 months preceding the claim.\"\"\",
                metadata={"clause_type": "liability", "section": "8.1", "risk": "high"}),

            Document(page_content=\"\"\"TERMINATION FOR CONVENIENCE: Either party may terminate
        this agreement for any reason with 30 days written notice. Upon termination,
        Customer shall pay all fees for services rendered through the termination date.\"\"\",
                metadata={"clause_type": "termination", "section": "9.2", "risk": "medium"}),

            Document(page_content=\"\"\"NON-COMPETE: During the term and for 24 months after
        termination, neither party shall directly solicit employees of the other party.
        This restriction applies globally with no geographic limitation.\"\"\",
                metadata={"clause_type": "non-compete", "section": "11.1", "risk": "high"}),

            Document(page_content=\"\"\"DATA PROCESSING: Provider shall process personal data
        only as instructed by Customer. Provider implements AES-256 encryption at rest
        and TLS 1.3 in transit. Data breach notification within 72 hours.\"\"\",
                metadata={"clause_type": "data_protection", "section": "7.3", "risk": "medium"}),

            Document(page_content=\"\"\"INTELLECTUAL PROPERTY: All IP created during the
        engagement belongs to the Customer. Provider retains rights to pre-existing IP
        and general knowledge. Provider grants Customer a perpetual license to
        pre-existing IP incorporated into deliverables.\"\"\",
                metadata={"clause_type": "ip", "section": "6.1", "risk": "high"}),

            Document(page_content=\"\"\"PAYMENT TERMS: Invoices are due within 30 days of receipt.
        Late payments accrue interest at 1.5% per month or the maximum legal rate.
        Customer may dispute invoices within 15 days of receipt.\"\"\",
                metadata={"clause_type": "payment", "section": "4.2", "risk": "low"}),
        ]
        print(f"Loaded {len(contract_clauses)} contract clauses")
        """),
        md("## Step 3 — Build Contract Index"),
        code("""
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma.from_documents(
            contract_clauses, embeddings,
            persist_directory="sample_data/contract_chroma",
            collection_name="contracts",
        )
        print("Contract index ready!")
        """),
        md("## Step 4 — Risk-Aware Clause Search"),
        code("""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        clause_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=\"\"\"You are a contract review assistant. Analyze the contract clauses
        and answer the question. For each relevant clause:
        1. Quote the key language
        2. Assess risk level (low/medium/high)
        3. Flag any concerning terms
        4. Suggest modifications to better protect the client

        Contract clauses:
        {context}

        Question: {question}

        Analysis:\"\"\"
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": clause_prompt},
        )

        queries = [
            "What are the liability limitations in this contract?",
            "Is there a non-compete clause and what are its terms?",
            "How is intellectual property handled?",
            "What are the data protection obligations?",
        ]

        for q in queries:
            print(f"\\n{'='*60}\\nQ: {q}")
            result = qa.invoke({"query": q})
            print(f"A: {result['result']}")
        """),
        md("## Step 5 — Risk Summary Report"),
        code("""
        from pydantic import BaseModel, Field

        class ClauseRisk(BaseModel):
            clause_type: str
            section: str
            risk_level: str
            concern: str
            recommendation: str

        class ContractRiskReport(BaseModel):
            overall_risk: str = Field(description="low/medium/high/critical")
            high_risk_clauses: list[ClauseRisk]
            recommendations: list[str]

        risk_llm = llm.with_structured_output(ContractRiskReport)

        all_text = "\\n\\n".join([c.page_content for c in contract_clauses])
        report = risk_llm.invoke(f"Generate a risk report for these contract clauses:\\n\\n{all_text}")

        print(f"Overall Risk: {report.overall_risk}")
        print(f"\\nHigh-Risk Clauses:")
        for c in report.high_risk_clauses:
            print(f"  Section {c.section} ({c.clause_type}): {c.concern}")
            print(f"    → {c.recommendation}")
        print(f"\\nTop Recommendations:")
        for r in report.recommendations:
            print(f"  • {r}")
        """),
        md("""
        ## What You Learned
        - **Legal clause indexing** with risk metadata
        - **Risk-aware retrieval** and clause analysis
        - **Structured risk reports** with actionable recommendations
        """),
    ]))

    # ── Project 16: Local Course Tutor ──────────────────────────────────
    paths.append(write_nb(2, "16_Local_Course_Tutor", [
        md("""
        # Project 16 — Local Course Tutor
        ## QA Over Lecture Notes and Slides

        **Stack:** Ollama · LangChain · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup and Sample Lecture Material"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain.schema import Document
        from pathlib import Path

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        lectures = [
            Document(page_content=\"\"\"Lecture 1: Introduction to Databases
        A database is an organized collection of structured data. DBMS (Database
        Management System) provides tools to create, read, update, delete data.

        Types: Relational (SQL), Document (MongoDB), Key-Value (Redis), Graph (Neo4j).

        ACID Properties:
        - Atomicity: Transactions are all-or-nothing
        - Consistency: Data remains valid after transactions
        - Isolation: Concurrent transactions don't interfere
        - Durability: Committed data survives failures\"\"\",
                metadata={"lecture": 1, "topic": "intro", "week": 1}),

            Document(page_content=\"\"\"Lecture 2: SQL Fundamentals
        SQL operations: SELECT, INSERT, UPDATE, DELETE.

        Joins: INNER JOIN returns matching rows. LEFT JOIN includes all left rows.
        FULL OUTER JOIN includes all rows from both tables.

        Aggregation: GROUP BY with COUNT, SUM, AVG, MAX, MIN.
        HAVING filters groups (vs WHERE which filters rows).

        Subqueries: Nested SELECT statements. Correlated subqueries reference outer query.\"\"\",
                metadata={"lecture": 2, "topic": "sql", "week": 2}),

            Document(page_content=\"\"\"Lecture 3: Normalization
        Normalization reduces data redundancy.
        1NF: Eliminate repeating groups, atomic values only.
        2NF: 1NF + no partial dependencies on composite keys.
        3NF: 2NF + no transitive dependencies.
        BCNF: Every determinant is a candidate key.

        Denormalization: Intentionally adding redundancy for read performance.
        Common in data warehouses and analytics workloads.\"\"\",
                metadata={"lecture": 3, "topic": "normalization", "week": 3}),
        ]
        print(f"Loaded {len(lectures)} lectures")
        """),
        md("## Step 2 — Build Course Index"),
        code("""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_documents(lectures)

        vectorstore = Chroma.from_documents(chunks, embeddings,
            persist_directory="sample_data/course_chroma", collection_name="db_course")
        print(f"Course index: {len(chunks)} chunks from {len(lectures)} lectures")
        """),
        md("## Step 3 — Tutor Q&A"),
        code("""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        tutor_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=\"\"\"You are a patient CS tutor helping a student study for exams.
        Use the lecture material to answer. If the concept is complex:
        1. Start with a simple analogy
        2. Give the technical explanation
        3. Provide an example
        Reference which lecture the information comes from.

        Lecture Material:
        {context}

        Student Question: {question}

        Tutor:\"\"\"
        )

        tutor = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": tutor_prompt},
        )

        student_questions = [
            "What's the difference between INNER JOIN and LEFT JOIN?",
            "Explain ACID properties with real-world examples",
            "What is 3NF and why does it matter?",
            "When would you choose MongoDB over PostgreSQL?",
        ]

        for q in student_questions:
            print(f"\\nStudent: {q}")
            result = tutor.invoke({"query": q})
            print(f"Tutor: {result['result']}")
        """),
        md("## Step 4 — Generate Study Guide"),
        code("""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        guide_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"Generate an exam study guide from the lecture material.
        Include: key terms with definitions, important formulas/rules,
        common exam questions, and tips for remembering concepts.\"\"\"),
            ("human", "Lectures:\\n{lectures}")
        ])

        guide_chain = guide_prompt | llm | StrOutputParser()
        all_lectures = "\\n\\n---\\n\\n".join([l.page_content for l in lectures])
        study_guide = guide_chain.invoke({"lectures": all_lectures})
        print("EXAM STUDY GUIDE:")
        print(study_guide)
        """),
        md("""
        ## What You Learned
        - **Lecture content indexing** with week/topic metadata
        - **Pedagogical prompting** (analogy → technical → example)
        - **Study guide generation** from course material
        """),
    ]))

    # ── Project 17: Local Personal Wiki Copilot ────────────────────────
    paths.append(write_nb(2, "17_Local_Personal_Wiki_Copilot", [
        md("""
        # Project 17 — Local Personal Wiki Copilot
        ## Query Exported Notes/Wiki Files Locally

        **Stack:** Ollama · LlamaIndex · Jupyter
        """),
        code("# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama"),
        md("## Step 1 — Setup"),
        code("""
        from llama_index.core import Settings, VectorStoreIndex, Document
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding

        Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        """),
        md("## Step 2 — Create Sample Wiki"),
        code("""
        wiki_pages = [
            Document(text=\"\"\"# Project Phoenix
        Status: Active | Lead: Alice | Started: 2024-Q3

        ## Goals
        - Migrate legacy monolith to microservices
        - Reduce deployment time from 4 hours to 15 minutes
        - Achieve 99.95% uptime SLA

        ## Architecture Decisions
        - ADR-001: Use Kubernetes for orchestration (approved)
        - ADR-002: Use PostgreSQL over MySQL (approved)
        - ADR-003: Event-driven architecture with Kafka (in review)

        ## Current Status
        Phase 1 (Auth service) complete. Phase 2 (Payment service) in progress.
        Blocked by: Legacy database schema migration.\"\"\",
            metadata={"page": "Project Phoenix", "type": "project", "tags": "architecture,migration"}),

            Document(text=\"\"\"# Onboarding Checklist
        ## Week 1
        - [ ] Set up dev environment (see Dev Setup page)
        - [ ] Complete security training
        - [ ] Get access to GitHub, Jira, Slack
        - [ ] Meet with team lead and buddy

        ## Week 2
        - [ ] Complete first PR (good-first-issue label)
        - [ ] Shadow a production deployment
        - [ ] Read Architecture Decision Records

        ## Tools
        - IDE: VS Code with recommended extensions
        - Terminal: Use Oh My Zsh with our custom theme
        - VPN: Required for staging/production access\"\"\",
            metadata={"page": "Onboarding", "type": "process", "tags": "hr,new-hire"}),

            Document(text=\"\"\"# Incident Response Playbook
        ## Severity Levels
        - SEV1: Customer-facing outage, all hands on deck
        - SEV2: Degraded service, team lead + on-call
        - SEV3: Internal tooling issue, on-call engineer

        ## Response Steps
        1. Acknowledge the alert within 5 minutes
        2. Create incident channel: #inc-YYYY-MM-DD-brief-description
        3. Assign Incident Commander and Communications Lead
        4. Begin investigation, post updates every 15 min for SEV1
        5. After resolution: blameless postmortem within 48 hours

        ## Escalation
        On-call → Team Lead → Engineering Director → CTO\"\"\",
            metadata={"page": "Incident Response", "type": "playbook", "tags": "ops,incidents"}),
        ]
        print(f"Created {len(wiki_pages)} wiki pages")
        """),
        md("## Step 3 — Build Wiki Index"),
        code("""
        index = VectorStoreIndex.from_documents(wiki_pages, show_progress=True)
        query_engine = index.as_query_engine(similarity_top_k=3)
        print("Wiki search index ready!")
        """),
        md("## Step 4 — Query the Wiki"),
        code("""
        queries = [
            "What's the current status of Project Phoenix?",
            "What do I need to do in my first week as a new hire?",
            "What's the process for a SEV1 incident?",
            "What database did we choose and why?",
        ]

        for q in queries:
            print(f"\\nQ: {q}")
            response = query_engine.query(q)
            print(f"A: {response}")
            pages = [n.metadata.get('page', '?') for n in response.source_nodes]
            print(f"Found in: {pages}")
        """),
        md("## Step 5 — Wiki Chat with Context"),
        code("""
        from llama_index.core.chat_engine import CondenseQuestionChatEngine

        chat = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine)

        conversation = [
            "Tell me about our architecture decisions",
            "Which one is still in review?",
            "What event system are we considering?",
        ]
        for msg in conversation:
            print(f"\\nUser: {msg}")
            resp = chat.chat(msg)
            print(f"Wiki: {resp}")
        """),
        md("""
        ## What You Learned
        - **Wiki page indexing** with metadata tags
        - **Cross-page search** for internal knowledge
        - **Conversational wiki** with context memory
        """),
    ]))

    # ── Project 18: Local Customer Support Memory Bot ───────────────────
    paths.append(write_nb(2, "18_Local_Customer_Support_Memory_Bot", [
        md("""
        # Project 18 — Local Customer Support Memory Bot
        ## Retrieve Similar Tickets and Suggested Fixes

        **Stack:** Ollama · LangChain · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain.schema import Document

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        """),
        md("## Step 2 — Historical Ticket Database"),
        code("""
        tickets = [
            Document(page_content=\"\"\"Ticket #1042: User cannot login after password reset
        Customer: john@example.com | Priority: High | Status: Resolved
        Issue: After resetting password, user gets "Invalid credentials" error.
        Root Cause: Password reset token expired (15-min window).
        Resolution: Generated new reset link, extended token expiry to 30 min.
        Time to resolve: 2 hours\"\"\",
                metadata={"ticket_id": 1042, "category": "auth", "resolved": True}),

            Document(page_content=\"\"\"Ticket #1055: Dashboard loading slowly
        Customer: corp@bigco.com | Priority: Medium | Status: Resolved
        Issue: Dashboard takes 30+ seconds to load for accounts with 10K+ records.
        Root Cause: Missing database index on the analytics table.
        Resolution: Added composite index on (account_id, created_at). Load time: <2s.
        Time to resolve: 4 hours\"\"\",
                metadata={"ticket_id": 1055, "category": "performance", "resolved": True}),

            Document(page_content=\"\"\"Ticket #1063: API rate limit exceeded
        Customer: dev@startup.io | Priority: Medium | Status: Resolved
        Issue: Getting 429 errors during automated data imports.
        Root Cause: Customer hitting 100 req/min limit with bulk import script.
        Resolution: Provided batch import endpoint (/api/v2/bulk-import).
        Time to resolve: 1 hour\"\"\",
                metadata={"ticket_id": 1063, "category": "api", "resolved": True}),

            Document(page_content=\"\"\"Ticket #1078: Email notifications not sending
        Customer: admin@school.edu | Priority: High | Status: Resolved
        Issue: No email notifications for new comments in the last 24 hours.
        Root Cause: SMTP relay service (SendGrid) had regional outage.
        Resolution: Switched to backup SMTP provider. Queued emails sent.
        Time to resolve: 6 hours\"\"\",
                metadata={"ticket_id": 1078, "category": "notifications", "resolved": True}),

            Document(page_content=\"\"\"Ticket #1091: Data export missing columns
        Customer: analyst@finance.com | Priority: Low | Status: Resolved
        Issue: CSV export missing 'created_at' and 'updated_at' columns.
        Root Cause: Export template not updated after schema migration.
        Resolution: Updated export template, added regression test.
        Time to resolve: 3 hours\"\"\",
                metadata={"ticket_id": 1091, "category": "data", "resolved": True}),
        ]
        print(f"Historical ticket database: {len(tickets)} resolved tickets")
        """),
        md("## Step 3 — Build Ticket Memory"),
        code("""
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma.from_documents(tickets, embeddings,
            persist_directory="sample_data/support_chroma", collection_name="tickets")
        print("Support ticket memory indexed!")
        """),
        md("## Step 4 — Similar Ticket Finder + Resolution Suggester"),
        code("""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        support_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=\"\"\"You are a support agent assistant. A new ticket has come in.
        Use the similar historical tickets to:
        1. Identify the most likely root cause
        2. Suggest a resolution based on past fixes
        3. Estimate time to resolve
        4. Recommend which team should handle it

        Similar past tickets:
        {context}

        New ticket description: {question}

        Analysis and recommended resolution:\"\"\"
        )

        support_qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": support_prompt},
        )

        new_tickets = [
            "User reports they can't login. They say the password reset email link is expired.",
            "Customer's API calls are getting rejected with HTTP 429 errors.",
            "The weekly report email didn't arrive for any users this morning.",
        ]

        for ticket in new_tickets:
            print(f"\\n{'='*60}")
            print(f"NEW TICKET: {ticket}")
            result = support_qa.invoke({"query": ticket})
            print(f"\\nSUGGESTED RESPONSE:\\n{result['result']}")
            similar = [s.metadata.get('ticket_id') for s in result['source_documents']]
            print(f"\\nSimilar tickets: {similar}")
        """),
        md("""
        ## What You Learned
        - **Ticket knowledge base** with resolution history
        - **Similar ticket retrieval** for faster resolution
        - **Resolution suggestion** from historical patterns
        """),
    ]))

    # ── Project 19: Local Product Docs Copilot ──────────────────────────
    paths.append(write_nb(2, "19_Local_Product_Docs_Copilot", [
        md("""
        # Project 19 — Local Product Docs Copilot
        ## Chat Over Internal or API Documentation

        **Stack:** Ollama · LangChain · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        llm = ChatOllama(model="qwen3:8b", temperature=0.2)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        """),
        md("## Step 2 — Sample API Documentation"),
        code("""
        from langchain.schema import Document

        api_docs = [
            Document(page_content=\"\"\"## Authentication
        All API requests require a Bearer token in the Authorization header.

        POST /api/v1/auth/token
        Body: {"email": "user@example.com", "password": "..."}
        Response: {"token": "eyJ...", "expires_in": 3600}

        The token expires in 1 hour. Use the refresh endpoint to get a new token.
        POST /api/v1/auth/refresh
        Header: Authorization: Bearer <expired_token>
        Response: {"token": "new_eyJ...", "expires_in": 3600}\"\"\",
                metadata={"section": "authentication", "version": "v1"}),

            Document(page_content=\"\"\"## Users API
        GET /api/v1/users — List all users (paginated, 50 per page)
        GET /api/v1/users/:id — Get user by ID
        POST /api/v1/users — Create user (requires admin role)
        PATCH /api/v1/users/:id — Update user
        DELETE /api/v1/users/:id — Soft delete user

        Query Parameters:
        - page (int): Page number, default 1
        - per_page (int): Items per page, max 100
        - search (string): Filter by name or email
        - role (string): Filter by role (admin, user, viewer)\"\"\",
                metadata={"section": "users", "version": "v1"}),

            Document(page_content=\"\"\"## Webhooks
        POST /api/v1/webhooks — Register a webhook
        Body: {"url": "https://...", "events": ["user.created", "user.updated"]}

        Available events: user.created, user.updated, user.deleted,
        document.created, document.shared.

        Webhook payload format:
        {"event": "user.created", "data": {...}, "timestamp": "2025-01-15T10:30:00Z"}

        Retry policy: 3 attempts with exponential backoff (1s, 5s, 25s).
        Webhooks are disabled after 10 consecutive failures.\"\"\",
                metadata={"section": "webhooks", "version": "v1"}),

            Document(page_content=\"\"\"## Rate Limits
        Default limits: 100 requests/minute per API key.
        Bulk endpoints: 10 requests/minute.
        Auth endpoints: 20 requests/minute.

        Rate limit headers included in every response:
        X-RateLimit-Limit: 100
        X-RateLimit-Remaining: 95
        X-RateLimit-Reset: 1705312800

        When rate limited, respond with HTTP 429 and Retry-After header.\"\"\",
                metadata={"section": "rate_limits", "version": "v1"}),
        ]
        print(f"Loaded {len(api_docs)} API documentation sections")
        """),
        md("## Step 3 — Build Docs Index"),
        code("""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(api_docs)

        vectorstore = Chroma.from_documents(chunks, embeddings,
            persist_directory="sample_data/docs_chroma", collection_name="api_docs")
        print(f"Docs index: {len(chunks)} chunks")
        """),
        md("## Step 4 — Developer Q&A"),
        code("""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        docs_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=\"\"\"You are a developer support assistant for our API.
        Answer technical questions using the documentation. Include:
        - Exact endpoint URLs and HTTP methods
        - Example request/response bodies
        - Important headers or parameters
        - Common gotchas or tips

        Documentation:
        {context}

        Developer Question: {question}

        Technical Answer:\"\"\"
        )

        docs_qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": docs_prompt},
        )

        dev_questions = [
            "How do I authenticate my API requests?",
            "What's the rate limit and what happens when I exceed it?",
            "How do I set up a webhook for new user events?",
            "How do I search for users by email?",
        ]

        for q in dev_questions:
            print(f"\\nQ: {q}")
            result = docs_qa.invoke({"query": q})
            print(f"A: {result['result']}")
        """),
        md("## Step 5 — Code Example Generator"),
        code("""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        codegen_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a working Python code example using the requests library. "
             "Include error handling and comments."),
            ("human", "Generate Python code for: {task}\\n\\nAPI docs: {docs}")
        ])
        codegen = codegen_prompt | llm | StrOutputParser()

        # Get relevant docs and generate code
        task = "Authenticate and list all admin users"
        relevant = vectorstore.similarity_search(task, k=2)
        docs_text = "\\n".join([d.page_content for d in relevant])
        code_example = codegen.invoke({"task": task, "docs": docs_text})
        print("Generated Code Example:")
        print(code_example)
        """),
        md("""
        ## What You Learned
        - **API documentation indexing** with section metadata
        - **Developer Q&A** with endpoint-specific answers
        - **Code example generation** from API docs
        """),
    ]))

    # ── Project 20: Local Medical Literature Finder ─────────────────────
    paths.append(write_nb(2, "20_Local_Medical_Literature_Finder", [
        md("""
        # Project 20 — Local Medical Literature Finder
        ## Search Papers by Topic with Evidence Grading

        **Stack:** Ollama · LlamaIndex · Metadata Filters · Jupyter
        """),
        code("# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama"),
        md("## Step 1 — Setup"),
        code("""
        from llama_index.core import Settings, VectorStoreIndex, Document
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding

        Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        """),
        md("## Step 2 — Sample Medical Literature"),
        code("""
        papers = [
            Document(text=\"\"\"Title: Efficacy of Metformin in Type 2 Diabetes Management
        Authors: Chen et al., 2024 | Journal: Lancet Diabetes
        Study Type: Randomized Controlled Trial | Evidence Level: I
        Sample Size: 2,400 patients | Duration: 24 months

        Findings: Metformin reduced HbA1c by 1.2% compared to placebo (p<0.001).
        Secondary outcomes showed 15% reduction in cardiovascular events.
        Adverse effects: GI symptoms in 20% of patients (vs 8% placebo).
        Conclusion: Metformin remains first-line therapy for T2DM.\"\"\",
                metadata={"topic": "diabetes", "year": 2024, "evidence_level": "I",
                         "study_type": "RCT", "journal": "Lancet"}),

            Document(text=\"\"\"Title: Machine Learning for Early Cancer Detection
        Authors: Park et al., 2024 | Journal: Nature Medicine
        Study Type: Retrospective Cohort | Evidence Level: III
        Sample Size: 50,000 imaging studies | Duration: 5 years

        Findings: Deep learning model detected early-stage lung cancer with
        94% sensitivity and 88% specificity, outperforming radiologists (87%/85%).
        False positive rate was 12%, requiring further investigation.
        Conclusion: AI-assisted screening shows promise but needs prospective validation.\"\"\",
                metadata={"topic": "oncology", "year": 2024, "evidence_level": "III",
                         "study_type": "retrospective", "journal": "Nature Medicine"}),

            Document(text=\"\"\"Title: Cognitive Behavioral Therapy vs SSRIs for Anxiety
        Authors: Williams et al., 2023 | Journal: JAMA Psychiatry
        Study Type: Meta-Analysis | Evidence Level: I
        Studies Included: 45 RCTs | Total Patients: 12,000

        Findings: CBT and SSRIs showed equivalent efficacy at 12 weeks.
        CBT showed lower relapse rate at 12 months (15% vs 30%, p<0.01).
        Combined therapy showed 25% better outcomes than either alone.
        Conclusion: CBT should be offered as first-line, alone or combined.\"\"\",
                metadata={"topic": "psychiatry", "year": 2023, "evidence_level": "I",
                         "study_type": "meta-analysis", "journal": "JAMA"}),
        ]
        print(f"Loaded {len(papers)} medical papers")
        """),
        md("## Step 3 — Build Literature Index"),
        code("""
        index = VectorStoreIndex.from_documents(papers, show_progress=True)
        query_engine = index.as_query_engine(similarity_top_k=3)
        print("Literature index ready!")
        """),
        md("## Step 4 — Evidence-Based Search"),
        code("""
        queries = [
            "What is the first-line treatment for type 2 diabetes?",
            "How effective is AI for cancer screening?",
            "Compare CBT and medication for anxiety disorders",
        ]

        for q in queries:
            print(f"\\n{'='*60}")
            print(f"Clinical Q: {q}")
            response = query_engine.query(q)
            print(f"\\nEvidence Summary: {response}")
            print("\\nSources:")
            for node in response.source_nodes:
                print(f"  [{node.metadata.get('evidence_level', '?')}] "
                      f"{node.metadata.get('journal', '?')} "
                      f"({node.metadata.get('study_type', '?')}, "
                      f"{node.metadata.get('year', '?')})")
        """),
        md("## Step 5 — Evidence Quality Grading"),
        code("""
        from pydantic import BaseModel, Field

        class EvidenceGrade(BaseModel):
            grade: str = Field(description="A (strong), B (moderate), C (weak), D (very weak)")
            reasoning: str
            level_of_evidence: str
            recommendation_strength: str
            caveats: list[str]

        # Use ChatOllama directly for structured grading
        from langchain_ollama import ChatOllama
        grading_llm = ChatOllama(model="qwen3:8b", temperature=0.1)
        grader = grading_llm.with_structured_output(EvidenceGrade)

        for paper in papers:
            grade = grader.invoke(
                f"Grade the evidence quality of this study:\\n\\n{paper.text}"
            )
            title = paper.text.split("\\n")[0].replace("Title: ", "")
            print(f"\\n{title}")
            print(f"  Grade: {grade.grade}")
            print(f"  Evidence Level: {grade.level_of_evidence}")
            print(f"  Recommendation: {grade.recommendation_strength}")
            print(f"  Caveats: {grade.caveats}")
        """),
        md("""
        ## What You Learned
        - **Medical literature indexing** with evidence metadata
        - **Evidence-based retrieval** with quality indicators
        - **Structured evidence grading** using medical standards
        """),
    ]))

    print(f"Group 2 complete: {len(paths)} notebooks written")
    for p in paths:
        print(f"  ✓ {p}")
    return paths

if __name__ == "__main__":
    build()
