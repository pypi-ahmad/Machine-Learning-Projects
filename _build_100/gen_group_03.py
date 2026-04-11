"""Group 3 — Projects 21-30: Advanced RAG and Retrieval Engineering."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── Project 21: Hybrid Retrieval Lab ────────────────────────────────
    paths.append(write_nb(3, "21_Hybrid_Retrieval_Lab", [
        md("""
        # Project 21 — Hybrid Retrieval Lab
        ## Compare BM25, Dense, and Hybrid Retrieval Locally

        **What you'll learn:**
        - Implement BM25 (sparse) retrieval
        - Implement dense vector retrieval
        - Combine both into a hybrid retriever with Reciprocal Rank Fusion
        - Measure and compare retrieval quality

        **Stack:** Ollama · LangChain · ChromaDB · rank-bm25 · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb rank-bm25"),
        md("## Step 1 — Setup and Sample Corpus"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain.schema import Document

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        corpus = [
            Document(page_content="Python is a high-level programming language known for readability. "
                "It supports multiple paradigms including OOP and functional programming.",
                metadata={"id": 1, "topic": "python"}),
            Document(page_content="Rust provides memory safety without garbage collection through "
                "its ownership system. It competes with C++ for systems programming.",
                metadata={"id": 2, "topic": "rust"}),
            Document(page_content="Machine learning models learn patterns from data. Supervised "
                "learning uses labeled examples while unsupervised finds hidden structures.",
                metadata={"id": 3, "topic": "ml"}),
            Document(page_content="Vector databases store embeddings for similarity search. "
                "Popular options include ChromaDB, FAISS, Pinecone, and Weaviate.",
                metadata={"id": 4, "topic": "vectordb"}),
            Document(page_content="RAG combines retrieval with generation. The retriever finds "
                "relevant documents and the generator produces answers grounded in evidence.",
                metadata={"id": 5, "topic": "rag"}),
            Document(page_content="BM25 is a bag-of-words retrieval function that ranks documents "
                "based on term frequency and inverse document frequency with saturation.",
                metadata={"id": 6, "topic": "retrieval"}),
            Document(page_content="Transformers use self-attention to process sequences in parallel. "
                "The architecture consists of encoder and decoder stacks with multi-head attention.",
                metadata={"id": 7, "topic": "transformers"}),
            Document(page_content="Fine-tuning adapts a pre-trained model to specific tasks using "
                "smaller domain-specific datasets. LoRA reduces the parameters needed.",
                metadata={"id": 8, "topic": "finetuning"}),
        ]
        print(f"Corpus: {len(corpus)} documents")
        """),
        md("## Step 2 — BM25 Sparse Retriever"),
        code("""
        from rank_bm25 import BM25Okapi
        import numpy as np

        # Tokenize for BM25
        tokenized_corpus = [doc.page_content.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        def bm25_search(query, k=3):
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            top_k = np.argsort(scores)[::-1][:k]
            results = []
            for idx in top_k:
                results.append({"doc": corpus[idx], "score": float(scores[idx]), "rank": len(results)+1})
            return results

        print("BM25 results for 'vector database similarity search':")
        for r in bm25_search("vector database similarity search"):
            print(f"  [{r['rank']}] score={r['score']:.3f} — {r['doc'].page_content[:60]}...")
        """),
        md("## Step 3 — Dense Vector Retriever"),
        code("""
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma.from_documents(corpus, embeddings, collection_name="hybrid_lab")
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        def dense_search(query, k=3):
            results = vectorstore.similarity_search_with_score(query, k=k)
            return [{"doc": doc, "score": float(score), "rank": i+1}
                    for i, (doc, score) in enumerate(results)]

        print("Dense results for 'vector database similarity search':")
        for r in dense_search("vector database similarity search"):
            print(f"  [{r['rank']}] score={r['score']:.3f} — {r['doc'].page_content[:60]}...")
        """),
        md("## Step 4 — Hybrid Retrieval with Reciprocal Rank Fusion"),
        code("""
        def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
            \"\"\"Combine BM25 and dense results using RRF.\"\"\"
            fused_scores = {}

            for result_list in [bm25_results, dense_results]:
                for r in result_list:
                    doc_id = r["doc"].metadata["id"]
                    rank = r["rank"]
                    if doc_id not in fused_scores:
                        fused_scores[doc_id] = {"doc": r["doc"], "score": 0, "sources": []}
                    fused_scores[doc_id]["score"] += 1.0 / (k + rank)
                    fused_scores[doc_id]["sources"].append(
                        f"{'BM25' if result_list == bm25_results else 'Dense'} rank={rank}"
                    )

            sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
            return sorted_results

        def hybrid_search(query, k=3):
            bm25_r = bm25_search(query, k=5)
            dense_r = dense_search(query, k=5)
            fused = reciprocal_rank_fusion(bm25_r, dense_r)
            return fused[:k]

        print("Hybrid (RRF) results for 'vector database similarity search':")
        for i, r in enumerate(hybrid_search("vector database similarity search")):
            print(f"  [{i+1}] rrf_score={r['score']:.4f} — {r['doc'].page_content[:60]}...")
            print(f"       Sources: {r['sources']}")
        """),
        md("## Step 5 — Comparative Evaluation"),
        code("""
        test_queries = [
            ("What is BM25?", [6]),  # Expected: doc about BM25
            ("How does RAG work?", [5]),  # Expected: RAG doc
            ("memory safe systems language", [2]),  # Expected: Rust doc
            ("neural network architecture", [7]),  # Expected: Transformers doc
        ]

        print(f"{'Query':<35} {'BM25':>6} {'Dense':>6} {'Hybrid':>6}")
        print("-" * 60)

        for query, expected_ids in test_queries:
            bm25_top = bm25_search(query, k=1)[0]["doc"].metadata["id"]
            dense_top = dense_search(query, k=1)[0]["doc"].metadata["id"]
            hybrid_top = hybrid_search(query, k=1)[0]["doc"].metadata["id"]

            def check(result_id):
                return "✓" if result_id in expected_ids else "✗"

            print(f"{query:<35} {check(bm25_top):>6} {check(dense_top):>6} {check(hybrid_top):>6}")
        """),
        md("""
        ## What You Learned
        - **BM25 sparse retrieval** — keyword-based, good for exact matches
        - **Dense vector retrieval** — semantic, good for paraphrases
        - **Reciprocal Rank Fusion** — combines both for robust results
        - **Comparative evaluation** with ground truth
        """),
    ]))

    # ── Project 22: Query Rewriting RAG Lab ─────────────────────────────
    paths.append(write_nb(3, "22_Query_Rewriting_RAG_Lab", [
        md("""
        # Project 22 — Query Rewriting RAG Lab
        ## Rewrite Vague Questions Before Retrieval for Better Results

        **Stack:** Ollama · LangChain · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain.schema import Document
        from langchain_community.vectorstores import Chroma

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Knowledge base
        docs = [
            Document(page_content="LangChain is a framework for building LLM applications. "
                "It provides chains, agents, and retrieval components.", metadata={"topic": "langchain"}),
            Document(page_content="ChromaDB is an open-source vector database for AI applications. "
                "It supports in-memory and persistent storage modes.", metadata={"topic": "chroma"}),
            Document(page_content="Ollama runs large language models locally. It supports Llama, "
                "Mistral, Qwen, and many other model families.", metadata={"topic": "ollama"}),
            Document(page_content="RAG retrieval quality depends on chunk size, overlap, embedding "
                "model choice, and the retrieval algorithm used.", metadata={"topic": "rag_quality"}),
            Document(page_content="Query rewriting transforms user queries into forms better suited "
                "for retrieval. Techniques include HyDE, step-back, and multi-query.", metadata={"topic": "rewriting"}),
        ]

        vectorstore = Chroma.from_documents(docs, embeddings, collection_name="query_rewrite_lab")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        print("Knowledge base indexed!")
        """),
        md("## Step 2 — Baseline (No Rewriting)"),
        code("""
        vague_queries = [
            "how do I make it work better?",       # Vague
            "that vector thing",                     # Ambiguous
            "what runs models on my laptop?",        # Informal
            "the python library for LLM apps",       # Imprecise
        ]

        print("=== BASELINE (No Rewriting) ===")
        for q in vague_queries:
            results = retriever.invoke(q)
            top_topic = results[0].metadata["topic"] if results else "none"
            print(f"  Q: {q!r:<45} → Top result: {top_topic}")
        """),
        md("## Step 3 — Query Rewriting Strategies"),
        code("""
        # Strategy 1: Simple clarification rewrite
        clarify_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite this vague query into a clear, specific search query. "
             "Add context that makes the query unambiguous. Return ONLY the rewritten query."),
            ("human", "{query}")
        ])
        clarify_chain = clarify_prompt | llm | StrOutputParser()

        # Strategy 2: HyDE (Hypothetical Document Embeddings)
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a short paragraph that would be the ideal answer to this "
             "question. This will be used to find similar documents. Write as if you are "
             "a technical document, not a conversational answer."),
            ("human", "{query}")
        ])
        hyde_chain = hyde_prompt | llm | StrOutputParser()

        # Strategy 3: Multi-query expansion
        multi_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate 3 different versions of this search query that might "
             "retrieve relevant documents. Return one query per line, no numbering."),
            ("human", "{query}")
        ])
        multi_chain = multi_prompt | llm | StrOutputParser()

        # Strategy 4: Step-back prompting
        stepback_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a more general, higher-level question that would help "
             "answer the specific question. Return ONLY the step-back question."),
            ("human", "{query}")
        ])
        stepback_chain = stepback_prompt | llm | StrOutputParser()

        print("Query rewriting strategies defined!")
        """),
        md("## Step 4 — Compare All Strategies"),
        code("""
        print("=== STRATEGY COMPARISON ===\\n")

        for q in vague_queries:
            print(f"Original: {q!r}")

            # Clarification
            clarified = clarify_chain.invoke({"query": q})
            results_c = retriever.invoke(clarified)
            print(f"  Clarified: {clarified!r}")
            print(f"    → {results_c[0].metadata['topic'] if results_c else 'none'}")

            # HyDE
            hypothetical = hyde_chain.invoke({"query": q})
            results_h = vectorstore.similarity_search(hypothetical, k=2)
            print(f"  HyDE doc: {hypothetical[:60]}...")
            print(f"    → {results_h[0].metadata['topic'] if results_h else 'none'}")

            # Multi-query
            multi = multi_chain.invoke({"query": q})
            sub_queries = [sq.strip() for sq in multi.strip().split("\\n") if sq.strip()]
            all_results = set()
            for sq in sub_queries[:3]:
                for r in retriever.invoke(sq):
                    all_results.add(r.metadata["topic"])
            print(f"  Multi-query ({len(sub_queries)} variants) → topics: {all_results}")

            # Step-back
            stepped = stepback_chain.invoke({"query": q})
            results_s = retriever.invoke(stepped)
            print(f"  Step-back: {stepped!r}")
            print(f"    → {results_s[0].metadata['topic'] if results_s else 'none'}")
            print()
        """),
        md("## Step 5 — End-to-End Rewriting RAG Pipeline"),
        code("""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        def rewriting_rag(query, strategy="clarify"):
            \"\"\"Full pipeline: rewrite → retrieve → answer.\"\"\"
            if strategy == "clarify":
                rewritten = clarify_chain.invoke({"query": query})
            elif strategy == "hyde":
                rewritten = hyde_chain.invoke({"query": query})
            elif strategy == "stepback":
                rewritten = stepback_chain.invoke({"query": query})
            else:
                rewritten = query

            retrieved = retriever.invoke(rewritten)
            context = "\\n".join([d.page_content for d in retrieved])

            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the question using the provided context."),
                ("human", "Context: {context}\\n\\nQuestion: {question}")
            ])
            answer_chain = answer_prompt | llm | StrOutputParser()
            answer = answer_chain.invoke({"context": context, "question": query})

            return {"rewritten": rewritten, "answer": answer, "sources": [d.metadata["topic"] for d in retrieved]}

        # Test the full pipeline
        result = rewriting_rag("how do I make retrieval better?", strategy="clarify")
        print(f"Rewritten: {result['rewritten']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        """),
        md("""
        ## What You Learned
        - **Query clarification** — rewrite vague queries
        - **HyDE** — use hypothetical documents for better embedding match
        - **Multi-query expansion** — cast a wider retrieval net
        - **Step-back prompting** — generalize for broader context
        """),
    ]))

    # ── Project 23: Retrieval Reranking Lab ─────────────────────────────
    paths.append(write_nb(3, "23_Retrieval_Reranking_Lab", [
        md("""
        # Project 23 — Retrieval Reranking Lab
        ## Compare No-Rerank vs LLM-Based Reranking

        **Stack:** Ollama · LangChain · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain.schema import Document
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.0)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Build a larger corpus where initial retrieval might bring back less relevant results
        corpus = [
            Document(page_content="Python list comprehensions provide a concise way to create lists. "
                "The syntax is [expr for item in iterable if condition].", metadata={"id": 1}),
            Document(page_content="Python dictionaries store key-value pairs. Common methods: "
                "get(), keys(), values(), items(), update().", metadata={"id": 2}),
            Document(page_content="Python generators use yield to produce values lazily, saving "
                "memory for large datasets. Generator expressions use parentheses.", metadata={"id": 3}),
            Document(page_content="Python decorators are functions that modify other functions. "
                "Common use cases: logging, caching, auth, timing.", metadata={"id": 4}),
            Document(page_content="Python context managers handle setup and cleanup. Use the with "
                "statement. Implement via __enter__/__exit__ or @contextmanager.", metadata={"id": 5}),
            Document(page_content="Python async/await enables concurrent I/O operations. asyncio "
                "provides event loop, tasks, and coroutine management.", metadata={"id": 6}),
            Document(page_content="Python type hints improve code documentation: def greet(name: str) "
                "-> str. Use mypy for static type checking.", metadata={"id": 7}),
            Document(page_content="Python dataclasses reduce boilerplate for data containers. "
                "Use @dataclass decorator. Supports defaults, ordering, freezing.", metadata={"id": 8}),
        ]

        vectorstore = Chroma.from_documents(corpus, embeddings, collection_name="rerank_lab")
        print(f"Corpus indexed: {len(corpus)} documents")
        """),
        md("## Step 2 — Initial Retrieval (No Reranking)"),
        code("""
        def retrieve_no_rerank(query, k=5):
            results = vectorstore.similarity_search_with_score(query, k=k)
            return [(doc, score) for doc, score in results]

        query = "How do I efficiently process large amounts of data in Python?"

        print(f"Query: {query}\\n")
        print("=== NO RERANKING (raw vector similarity) ===")
        no_rerank_results = retrieve_no_rerank(query)
        for i, (doc, score) in enumerate(no_rerank_results):
            print(f"  [{i+1}] score={score:.4f} id={doc.metadata['id']} — {doc.page_content[:60]}...")
        """),
        md("## Step 3 — LLM-Based Reranker"),
        code("""
        rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"You are a relevance judge. Given a query and a document,
        rate the document's relevance on a scale of 0-10.
        Return ONLY the number, nothing else.

        Scoring guide:
        0-2: Not relevant
        3-5: Somewhat relevant
        6-8: Relevant
        9-10: Highly relevant\"\"\"),
            ("human", "Query: {query}\\n\\nDocument: {document}\\n\\nRelevance score (0-10):")
        ])
        rerank_chain = rerank_prompt | llm | StrOutputParser()

        def rerank_with_llm(query, initial_results, top_k=3):
            scored = []
            for doc, vector_score in initial_results:
                try:
                    relevance = rerank_chain.invoke({"query": query, "document": doc.page_content})
                    score = float(relevance.strip().split()[0])
                except (ValueError, IndexError):
                    score = 0.0
                scored.append((doc, score, vector_score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        print("=== WITH LLM RERANKING ===")
        reranked = rerank_with_llm(query, no_rerank_results)
        for i, (doc, llm_score, vec_score) in enumerate(reranked):
            print(f"  [{i+1}] llm_score={llm_score:.0f} vec_score={vec_score:.4f} "
                  f"id={doc.metadata['id']} — {doc.page_content[:60]}...")
        """),
        md("## Step 4 — Side-by-Side Comparison"),
        code("""
        test_queries = [
            "How do I efficiently process large amounts of data in Python?",
            "What's the best way to add caching to my functions?",
            "How do I handle concurrent network requests?",
            "What's a clean way to define data structures?",
        ]

        for q in test_queries:
            print(f"\\n{'='*60}")
            print(f"Q: {q}")

            raw = retrieve_no_rerank(q, k=5)
            reranked = rerank_with_llm(q, raw, top_k=3)

            raw_order = [doc.metadata['id'] for doc, _ in raw[:3]]
            reranked_order = [doc.metadata['id'] for doc, _, _ in reranked]

            changed = raw_order != reranked_order
            print(f"  Raw top-3:     {raw_order}")
            print(f"  Reranked top-3: {reranked_order} {'← CHANGED' if changed else '(same)'}")
        """),
        md("## Step 5 — End-to-End QA: No-Rerank vs Rerank"),
        code("""
        from langchain.prompts import PromptTemplate

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Answer using the context.\\nContext: {context}\\nQuestion: {question}\\nAnswer:"
        )

        def qa_no_rerank(query):
            docs = vectorstore.similarity_search(query, k=3)
            context = "\\n".join([d.page_content for d in docs])
            chain = qa_prompt | llm | StrOutputParser()
            return chain.invoke({"context": context, "question": query})

        def qa_with_rerank(query):
            raw = retrieve_no_rerank(query, k=5)
            reranked = rerank_with_llm(query, raw, top_k=3)
            context = "\\n".join([d.page_content for d, _, _ in reranked])
            chain = qa_prompt | llm | StrOutputParser()
            return chain.invoke({"context": context, "question": query})

        q = "How do I process large datasets efficiently in Python?"
        print(f"Q: {q}\\n")
        print("Without reranking:")
        print(qa_no_rerank(q))
        print("\\nWith LLM reranking:")
        print(qa_with_rerank(q))
        """),
        md("""
        ## What You Learned
        - **Vector similarity** retrieval as a fast first pass
        - **LLM reranking** for higher precision at the cost of latency
        - **Relevance scoring** with structured LLM prompts
        - **Side-by-side comparison** methodology
        """),
    ]))

    # ── Project 24: Context Compression RAG ─────────────────────────────
    paths.append(write_nb(3, "24_Context_Compression_RAG", [
        md("""
        # Project 24 — Context Compression RAG
        ## Compress Retrieved Context Before Answer Generation

        **Stack:** Ollama · LangChain · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup with Large Documents"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain.schema import Document
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Long documents that retrieve chunks with noise
        docs = [Document(page_content=c, metadata={"id": i}) for i, c in enumerate([
            \"\"\"Introduction to Kubernetes. Kubernetes (K8s) is an open-source container
        orchestration platform. It automates deployment, scaling, and management of
        containerized applications. Originally designed by Google, it's now maintained
        by CNCF. The name Kubernetes comes from Greek meaning 'helmsman'. It was
        released in 2014 and has since become the de facto standard for container
        orchestration. Many cloud providers offer managed Kubernetes services.\"\"\",

            \"\"\"Kubernetes Architecture. The control plane consists of the API server,
        etcd (key-value store), scheduler, and controller manager. Worker nodes run
        kubelet, kube-proxy, and container runtime. Pods are the smallest deployable
        units. A pod can contain one or more containers that share networking and storage.\"\"\",

            \"\"\"Kubernetes Deployments. A Deployment manages ReplicaSets and provides
        declarative updates. Rolling updates allow zero-downtime updates by gradually
        replacing pods. Strategy types: RollingUpdate and Recreate. You can set
        maxSurge and maxUnavailable to control the update pace.\"\"\",

            \"\"\"Kubernetes Services. Services provide stable networking for pods.
        Types: ClusterIP (internal), NodePort (external on node ports),
        LoadBalancer (cloud LB), ExternalName (DNS alias). Services use label
        selectors to route traffic to matching pods.\"\"\",
        ])]

        vectorstore = Chroma.from_documents(docs, embeddings, collection_name="compress_lab")
        print(f"Indexed {len(docs)} documents")
        """),
        md("## Step 2 — Without Compression (Full Context)"),
        code("""
        query = "How do I do zero-downtime updates in Kubernetes?"
        retrieved = vectorstore.similarity_search(query, k=3)
        full_context = "\\n\\n".join([d.page_content for d in retrieved])

        print(f"Full context length: {len(full_context)} chars")
        print(f"Retrieved {len(retrieved)} chunks\\n")

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using ONLY the provided context. Be specific."),
            ("human", "Context: {context}\\n\\nQuestion: {question}")
        ])
        qa_chain = qa_prompt | llm | StrOutputParser()

        answer_full = qa_chain.invoke({"context": full_context, "question": query})
        print(f"Answer (full context): {answer_full}")
        """),
        md("## Step 3 — With Context Compression"),
        code("""
        # Compression step: extract only relevant sentences
        compress_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"Extract ONLY the sentences from the context that are directly
        relevant to answering the question. Remove all irrelevant information.
        Return the extracted sentences as-is, do not paraphrase.\"\"\"),
            ("human", "Question: {question}\\n\\nContext: {context}\\n\\nRelevant sentences only:")
        ])
        compress_chain = compress_prompt | llm | StrOutputParser()

        compressed = compress_chain.invoke({"context": full_context, "question": query})
        print(f"Compressed context: {len(compressed)} chars ({len(compressed)/len(full_context)*100:.0f}% of original)")
        print(f"\\nCompressed content:\\n{compressed}")

        answer_compressed = qa_chain.invoke({"context": compressed, "question": query})
        print(f"\\nAnswer (compressed): {answer_compressed}")
        """),
        md("## Step 4 — Comparison: Full vs Compressed"),
        code("""
        queries = [
            "What is the control plane in Kubernetes?",
            "What types of Kubernetes services exist?",
            "How do rolling updates work?",
        ]

        for q in queries:
            retrieved = vectorstore.similarity_search(q, k=3)
            full = "\\n\\n".join([d.page_content for d in retrieved])
            compressed = compress_chain.invoke({"context": full, "question": q})

            print(f"\\nQ: {q}")
            print(f"  Full: {len(full)} chars → Compressed: {len(compressed)} chars "
                  f"({len(compressed)/max(len(full),1)*100:.0f}%)")

            ans_full = qa_chain.invoke({"context": full, "question": q})
            ans_comp = qa_chain.invoke({"context": compressed, "question": q})
            print(f"  Full answer: {ans_full[:100]}...")
            print(f"  Compressed:  {ans_comp[:100]}...")
        """),
        md("""
        ## What You Learned
        - **Context compression** removes noise before generation
        - **Trade-offs:** compression adds latency but improves focus
        - **Measurement:** track context reduction ratio vs answer quality
        """),
    ]))

    # ── Project 25: Multi-Hop RAG Research Agent ────────────────────────
    paths.append(write_nb(3, "25_Multi_Hop_RAG_Research_Agent", [
        md("""
        # Project 25 — Multi-Hop RAG Research Agent
        ## Use Multiple Retrieval Hops Before Answering Complex Questions

        **Stack:** LangGraph · Ollama · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community langgraph chromadb"),
        md("## Step 1 — Setup Knowledge Base"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain.schema import Document
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        knowledge = [
            Document(page_content="The Transformer architecture was introduced by Vaswani et al. "
                "in 2017 in the paper 'Attention Is All You Need'. It uses self-attention "
                "instead of recurrence.", metadata={"topic": "transformer", "id": 1}),
            Document(page_content="BERT (Bidirectional Encoder Representations from Transformers) "
                "was created by Google in 2018. It uses masked language modeling and was "
                "pre-trained on BookCorpus and Wikipedia.", metadata={"topic": "bert", "id": 2}),
            Document(page_content="GPT-3 has 175 billion parameters and was trained by OpenAI. "
                "It uses autoregressive language modeling. Few-shot learning emerged as a "
                "key capability.", metadata={"topic": "gpt3", "id": 3}),
            Document(page_content="RAG was introduced by Lewis et al. in 2020. It combines "
                "a retriever (DPR) with a generator (BART). This reduces hallucination by "
                "grounding generation in retrieved documents.", metadata={"topic": "rag", "id": 4}),
            Document(page_content="DPR (Dense Passage Retrieval) uses dual BERT encoders — one "
                "for questions and one for passages. It outperforms BM25 on open-domain QA "
                "benchmarks.", metadata={"topic": "dpr", "id": 5}),
            Document(page_content="Self-attention computes query, key, value matrices from input. "
                "Attention scores = softmax(QK^T / sqrt(d)). Multi-head attention uses "
                "multiple parallel attention heads.", metadata={"topic": "attention", "id": 6}),
        ]

        vectorstore = Chroma.from_documents(knowledge, embeddings, collection_name="multihop")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        print(f"Knowledge base: {len(knowledge)} documents")
        """),
        md("## Step 2 — Define Multi-Hop Graph with LangGraph"),
        code("""
        from langgraph.graph import StateGraph, END
        from typing import TypedDict, Annotated
        import operator

        class ResearchState(TypedDict):
            question: str
            sub_questions: list[str]
            retrieved_context: Annotated[list[str], operator.add]
            hop_count: int
            final_answer: str

        def decompose_question(state: ResearchState) -> ResearchState:
            \"\"\"Break complex question into sub-questions.\"\"\"
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Break this complex question into 2-3 simpler sub-questions "
                 "that need to be answered first. Return one question per line."),
                ("human", "{question}")
            ])
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"question": state["question"]})
            subs = [q.strip() for q in result.strip().split("\\n") if q.strip()][:3]
            print(f"  Decomposed into {len(subs)} sub-questions")
            return {"sub_questions": subs, "hop_count": 0}

        def retrieve_hop(state: ResearchState) -> ResearchState:
            \"\"\"Retrieve documents for current sub-question.\"\"\"
            hop = state["hop_count"]
            if hop < len(state["sub_questions"]):
                sub_q = state["sub_questions"][hop]
                docs = retriever.invoke(sub_q)
                context = [f"[Hop {hop+1}: {sub_q}]\\n" + d.page_content for d in docs]
                print(f"  Hop {hop+1}: Retrieved {len(docs)} docs for: {sub_q[:50]}...")
                return {"retrieved_context": context, "hop_count": hop + 1}
            return {"hop_count": hop}

        def should_continue(state: ResearchState) -> str:
            if state["hop_count"] < len(state["sub_questions"]):
                return "retrieve_more"
            return "synthesize"

        def synthesize(state: ResearchState) -> ResearchState:
            \"\"\"Combine all retrieved evidence into a final answer.\"\"\"
            all_context = "\\n\\n".join(state["retrieved_context"])
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Synthesize a comprehensive answer using ALL the retrieved evidence. "
                 "Cite which hop/source each piece of information comes from."),
                ("human", "Question: {question}\\n\\nEvidence:\\n{context}\\n\\nSynthesized answer:")
            ])
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"question": state["question"], "context": all_context})
            print(f"  Synthesized from {len(state['retrieved_context'])} evidence pieces")
            return {"final_answer": answer}

        # Build the graph
        graph = StateGraph(ResearchState)
        graph.add_node("decompose", decompose_question)
        graph.add_node("retrieve", retrieve_hop)
        graph.add_node("synthesize", synthesize)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "retrieve")
        graph.add_conditional_edges("retrieve", should_continue, {
            "retrieve_more": "retrieve",
            "synthesize": "synthesize",
        })
        graph.add_edge("synthesize", END)

        app = graph.compile()
        print("Multi-hop research graph compiled!")
        """),
        md("## Step 3 — Run Multi-Hop Queries"),
        code("""
        complex_questions = [
            "How does the attention mechanism used in BERT relate to the original Transformer?",
            "What retrieval method does RAG use, and how does it compare to keyword search?",
            "Trace the evolution from Transformers to GPT-3 — what are the key architectural changes?",
        ]

        for q in complex_questions:
            print(f"\\n{'='*60}")
            print(f"Q: {q}")
            print("-"*60)
            result = app.invoke({"question": q, "sub_questions": [], "retrieved_context": [], "hop_count": 0, "final_answer": ""})
            print(f"\\nFinal Answer:")
            print(result["final_answer"])
            print(f"\\nTotal evidence pieces: {len(result['retrieved_context'])}")
        """),
        md("""
        ## What You Learned
        - **Question decomposition** for complex multi-part queries
        - **Iterative retrieval** (multi-hop) using LangGraph
        - **Evidence synthesis** across multiple retrieval rounds
        - **Graph-based workflow** with conditional branching
        """),
    ]))

    # ── Project 26: Table + Text Local RAG ──────────────────────────────
    paths.append(write_nb(3, "26_Table_Text_Local_RAG", [
        md("""
        # Project 26 — Table + Text Local RAG
        ## Combine CSVs and Text Documents in One RAG Pipeline

        **Stack:** Ollama · LlamaIndex · pandas · Jupyter
        """),
        code("# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama pandas"),
        md("## Step 1 — Setup"),
        code("""
        from llama_index.core import Settings, VectorStoreIndex, Document
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding
        import pandas as pd
        from pathlib import Path

        Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        """),
        md("## Step 2 — Create Mixed Data Sources"),
        code("""
        Path("sample_data").mkdir(exist_ok=True)

        # CSV: Sales data
        sales_df = pd.DataFrame({
            "Quarter": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
            "Revenue_M": [45.2, 52.1, 58.7, 67.3],
            "Units_Sold": [12000, 14500, 16200, 19800],
            "Top_Product": ["Widget Pro", "Widget Pro", "Widget Max", "Widget Max"],
            "Region": ["North America", "Europe", "North America", "Asia Pacific"],
        })
        sales_df.to_csv("sample_data/quarterly_sales.csv", index=False)

        # Text: Narrative report
        narrative = \"\"\"
        Annual Business Review 2024

        Our company achieved record revenue of $223.3M in 2024, representing 35% YoY growth.
        The Widget Max product line, launched in Q3, exceeded expectations with rapid adoption
        in the Asia Pacific market.

        Key wins included the Fortune 500 deal in Q2 ($8M ARR) and the government contract
        in Q4 ($12M over 3 years). The sales team expanded from 45 to 72 people.

        Challenges: Supply chain disruptions affected Q1 delivery times. Customer churn
        increased to 8% from 5% due to a competitor's aggressive pricing.
        \"\"\"
        Path("sample_data/business_review.txt").write_text(narrative, encoding="utf-8")
        print("Created CSV and text data sources")
        print(sales_df.to_string(index=False))
        """),
        md("## Step 3 — Convert Table to Text and Build Combined Index"),
        code("""
        # Convert CSV rows to natural language
        table_docs = []
        for _, row in sales_df.iterrows():
            text = (f"In {row['Quarter']}, revenue was ${row['Revenue_M']}M "
                    f"with {row['Units_Sold']:,} units sold. "
                    f"Top product: {row['Top_Product']}. "
                    f"Strongest region: {row['Region']}.")
            table_docs.append(Document(text=text, metadata={"source": "sales_table", "type": "data"}))

        # Add summary statistics
        summary_text = (f"Full year 2024: Total revenue ${sales_df['Revenue_M'].sum():.1f}M, "
                       f"Total units {sales_df['Units_Sold'].sum():,}, "
                       f"avg quarterly revenue ${sales_df['Revenue_M'].mean():.1f}M")
        table_docs.append(Document(text=summary_text, metadata={"source": "sales_summary", "type": "data"}))

        # Add narrative document
        text_doc = Document(text=narrative, metadata={"source": "business_review", "type": "narrative"})

        all_docs = table_docs + [text_doc]
        index = VectorStoreIndex.from_documents(all_docs, show_progress=True)
        query_engine = index.as_query_engine(similarity_top_k=4)
        print(f"Combined index: {len(all_docs)} documents (table + text)")
        """),
        md("## Step 4 — Query Across Tables and Text"),
        code("""
        queries = [
            "What was the total revenue in 2024?",
            "Which product performed best in Q4?",
            "What were the main challenges this year?",
            "How did Asia Pacific perform?",
            "Tell me about the Fortune 500 deal.",
        ]

        for q in queries:
            print(f"\\nQ: {q}")
            response = query_engine.query(q)
            print(f"A: {response}")
            sources = [n.metadata.get('source', '?') for n in response.source_nodes]
            print(f"Sources: {sources}")
        """),
        md("""
        ## What You Learned
        - **Table-to-text conversion** for indexing structured data
        - **Combined retrieval** across heterogeneous sources
        - **Source attribution** showing data vs narrative evidence
        """),
    ]))

    # ── Project 27: Freshness-Aware News RAG ────────────────────────────
    paths.append(write_nb(3, "27_Freshness_Aware_News_RAG", [
        md("""
        # Project 27 — Freshness-Aware News RAG
        ## Prioritize Recent Documents in Retrieval

        **Stack:** Ollama · LangChain · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup with Timestamped Articles"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain.schema import Document
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from datetime import datetime, timedelta

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Simulated news articles at different dates
        base_date = datetime(2025, 1, 15)
        articles = [
            Document(page_content="AI startup raises $50M Series B to build enterprise RAG solutions. "
                "The company plans to hire 100 engineers.", metadata={
                "date": (base_date - timedelta(days=1)).isoformat(), "days_ago": 1, "source": "techcrunch"}),
            Document(page_content="New open-source embedding model achieves state-of-the-art on MTEB "
                "benchmark. Available on HuggingFace.", metadata={
                "date": (base_date - timedelta(days=7)).isoformat(), "days_ago": 7, "source": "arxiv"}),
            Document(page_content="AI startup raises $20M Series A for enterprise RAG platform. "
                "Focuses on healthcare and legal verticals.", metadata={
                "date": (base_date - timedelta(days=90)).isoformat(), "days_ago": 90, "source": "techcrunch"}),
            Document(page_content="Ollama releases version 0.5 with multi-model support and improved "
                "performance. Memory usage reduced by 30%.", metadata={
                "date": (base_date - timedelta(days=3)).isoformat(), "days_ago": 3, "source": "github"}),
            Document(page_content="Study shows RAG systems reduce hallucination by 70% compared to "
                "base LLMs in enterprise settings.", metadata={
                "date": (base_date - timedelta(days=30)).isoformat(), "days_ago": 30, "source": "research"}),
        ]

        vectorstore = Chroma.from_documents(articles, embeddings, collection_name="news_rag")
        print(f"Indexed {len(articles)} articles with timestamps")
        """),
        md("## Step 2 — Standard Retrieval (Ignores Freshness)"),
        code("""
        query = "What's the latest news about AI startups and RAG?"
        results = vectorstore.similarity_search_with_score(query, k=3)

        print("Standard retrieval (no freshness weighting):")
        for doc, score in results:
            print(f"  [{doc.metadata['days_ago']}d ago] score={score:.4f} — {doc.page_content[:60]}...")
        """),
        md("## Step 3 — Freshness-Weighted Retrieval"),
        code("""
        import math

        def freshness_search(query, k=3, freshness_weight=0.3, max_days=180):
            \"\"\"Combine semantic similarity with time recency.\"\"\"
            results = vectorstore.similarity_search_with_score(query, k=len(articles))
            scored = []
            for doc, sim_score in results:
                days_ago = doc.metadata.get("days_ago", max_days)
                # Freshness: exponential decay
                freshness = math.exp(-days_ago / 30.0)
                # Combined score (lower sim_score = better in Chroma)
                combined = (1 - freshness_weight) * (1 / (1 + sim_score)) + freshness_weight * freshness
                scored.append((doc, combined, sim_score, freshness))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k]

        print("Freshness-weighted retrieval:")
        for doc, combined, sim, fresh in freshness_search(query):
            print(f"  [{doc.metadata['days_ago']}d ago] combined={combined:.4f} "
                  f"sim={sim:.4f} fresh={fresh:.4f} — {doc.page_content[:60]}...")
        """),
        md("## Step 4 — Compare Results"),
        code("""
        queries = [
            "Latest AI startup funding news",
            "What's new with Ollama?",
            "RAG hallucination research",
        ]

        for q in queries:
            print(f"\\nQ: {q}")
            standard = vectorstore.similarity_search(q, k=2)
            fresh = freshness_search(q, k=2)

            std_ids = [(d.metadata['days_ago'], d.metadata['source']) for d in standard]
            fresh_ids = [(d.metadata['days_ago'], d.metadata['source']) for d, _, _, _ in fresh]

            print(f"  Standard: {std_ids}")
            print(f"  Fresh:    {fresh_ids}")
        """),
        md("""
        ## What You Learned
        - **Time-aware retrieval** using exponential freshness decay
        - **Score combination** blending semantic similarity with recency
        - **Configurable freshness weight** for different use cases
        """),
    ]))

    # ── Project 28: Multilingual Local RAG ──────────────────────────────
    paths.append(write_nb(3, "28_Multilingual_Local_RAG", [
        md("""
        # Project 28 — Multilingual Local RAG
        ## Retrieve in One Language, Answer in Another

        **Stack:** Ollama · LangChain · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup Multilingual Corpus"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain.schema import Document
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Documents in multiple languages
        docs = [
            Document(page_content="Python is a versatile programming language used for web development, "
                "data science, and artificial intelligence.", metadata={"lang": "en", "id": 1}),
            Document(page_content="Python est un langage de programmation polyvalent utilisé pour le "
                "développement web, la science des données et l'intelligence artificielle.",
                metadata={"lang": "fr", "id": 2}),
            Document(page_content="Python ist eine vielseitige Programmiersprache für Webentwicklung, "
                "Datenwissenschaft und künstliche Intelligenz.",
                metadata={"lang": "de", "id": 3}),
            Document(page_content="Machine learning algorithms learn from data to make predictions. "
                "Common types include supervised, unsupervised, and reinforcement learning.",
                metadata={"lang": "en", "id": 4}),
            Document(page_content="Les algorithmes d'apprentissage automatique apprennent à partir "
                "des données. Types courants: supervisé, non supervisé, et par renforcement.",
                metadata={"lang": "fr", "id": 5}),
        ]

        vectorstore = Chroma.from_documents(docs, embeddings, collection_name="multilingual_rag")
        print(f"Indexed {len(docs)} documents in {len(set(d.metadata['lang'] for d in docs))} languages")
        """),
        md("## Step 2 — Cross-Lingual Retrieval Test"),
        code("""
        # Query in English, retrieve from all languages
        query_en = "What is Python used for?"
        results = vectorstore.similarity_search_with_score(query_en, k=5)

        print(f"Query (EN): {query_en}\\n")
        for doc, score in results:
            lang = doc.metadata['lang']
            print(f"  [{lang}] score={score:.4f} — {doc.page_content[:60]}...")

        # Query in French
        query_fr = "Qu'est-ce que l'apprentissage automatique?"
        results_fr = vectorstore.similarity_search_with_score(query_fr, k=5)

        print(f"\\nQuery (FR): {query_fr}\\n")
        for doc, score in results_fr:
            lang = doc.metadata['lang']
            print(f"  [{lang}] score={score:.4f} — {doc.page_content[:60]}...")
        """),
        md("## Step 3 — Answer in Target Language"),
        code("""
        translate_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using the provided context. "
             "IMPORTANT: Always respond in {target_lang}, regardless of the context language."),
            ("human", "Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer in {target_lang}:")
        ])
        translate_chain = translate_answer_prompt | llm | StrOutputParser()

        # English query → French answer
        retrieved = vectorstore.similarity_search(query_en, k=3)
        context = "\\n".join([d.page_content for d in retrieved])

        print("EN question → FR answer:")
        answer_fr = translate_chain.invoke({
            "context": context, "question": query_en, "target_lang": "French"
        })
        print(f"  Q: {query_en}")
        print(f"  A: {answer_fr}")

        print("\\nEN question → DE answer:")
        answer_de = translate_chain.invoke({
            "context": context, "question": query_en, "target_lang": "German"
        })
        print(f"  Q: {query_en}")
        print(f"  A: {answer_de}")
        """),
        md("""
        ## What You Learned
        - **Cross-lingual retrieval** with multilingual embeddings
        - **Language-controlled generation** — answer in any target language
        - **Embedding model behavior** across languages
        """),
    ]))

    # ── Project 29: Citation Verifier for RAG ───────────────────────────
    paths.append(write_nb(3, "29_Citation_Verifier_for_RAG", [
        md("""
        # Project 29 — Citation Verifier for RAG
        ## Check if LLM Answers Are Supported by Retrieved Chunks

        **Stack:** Ollama · LangChain · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama"),
        md("## Step 1 — Setup"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pydantic import BaseModel, Field

        llm = ChatOllama(model="qwen3:8b", temperature=0.0)
        """),
        md("## Step 2 — Test Cases: Answers + Source Evidence"),
        code("""
        test_cases = [
            {
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris. It has a population of about 2.1 million.",
                "sources": ["Paris is the capital and most populous city of France, with a population of 2,102,650."],
                "expected": "supported",
            },
            {
                "question": "When was Python created?",
                "answer": "Python was created in 1991 by Guido van Rossum. It was designed at Microsoft.",
                "sources": ["Python was conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands."],
                "expected": "partially_supported",  # year close but 'Microsoft' is wrong
            },
            {
                "question": "How fast is GPT-4?",
                "answer": "GPT-4 can process 1 million tokens per second and costs $0.001 per query.",
                "sources": ["GPT-4 is a large language model created by OpenAI."],
                "expected": "not_supported",
            },
        ]
        print(f"Prepared {len(test_cases)} test cases for verification")
        """),
        md("## Step 3 — Claim Extraction"),
        code("""
        class ExtractedClaims(BaseModel):
            claims: list[str] = Field(description="Individual factual claims from the answer")

        claim_extractor = llm.with_structured_output(ExtractedClaims)

        for tc in test_cases:
            claims = claim_extractor.invoke(
                f"Extract individual factual claims from this answer:\\n\\n{tc['answer']}"
            )
            tc["claims"] = claims.claims
            print(f"Q: {tc['question']}")
            print(f"  Claims extracted: {len(claims.claims)}")
            for c in claims.claims:
                print(f"    • {c}")
            print()
        """),
        md("## Step 4 — Verify Each Claim Against Sources"),
        code("""
        class ClaimVerification(BaseModel):
            claim: str
            verdict: str = Field(description="supported, contradicted, or not_mentioned")
            evidence: str = Field(description="Quote from source supporting/contradicting, or 'none'")
            confidence: float = Field(description="0.0 to 1.0")

        class VerificationReport(BaseModel):
            claim_results: list[ClaimVerification]
            overall_verdict: str = Field(description="supported, partially_supported, not_supported")
            groundedness_score: float = Field(description="0.0 to 1.0 — fraction of supported claims")

        verifier = llm.with_structured_output(VerificationReport)

        for tc in test_cases:
            sources_text = "\\n".join(tc["sources"])
            claims_text = "\\n".join([f"- {c}" for c in tc["claims"]])

            report = verifier.invoke(
                f\"\"\"Verify each claim against the source evidence.

        Claims:
        {claims_text}

        Source Evidence:
        {sources_text}

        For each claim, determine if it is supported, contradicted, or not mentioned
        in the sources.\"\"\"
            )

            print(f"\\nQ: {tc['question']}")
            print(f"Answer: {tc['answer']}")
            print(f"Verdict: {report.overall_verdict} (expected: {tc['expected']})")
            print(f"Groundedness: {report.groundedness_score:.0%}")
            for cr in report.claim_results:
                icon = {"supported": "✓", "contradicted": "✗", "not_mentioned": "?"}
                print(f"  {icon.get(cr.verdict, '?')} [{cr.verdict}] {cr.claim}")
                if cr.evidence != "none":
                    print(f"    Evidence: {cr.evidence[:80]}...")
        """),
        md("""
        ## What You Learned
        - **Claim extraction** from generated answers
        - **Evidence-based verification** of individual claims
        - **Groundedness scoring** for RAG quality assessment
        - **Distinguishing** supported, contradicted, and unsupported claims
        """),
    ]))

    # ── Project 30: RAG Evaluation Dashboard Notebook ───────────────────
    paths.append(write_nb(3, "30_RAG_Evaluation_Dashboard", [
        md("""
        # Project 30 — RAG Evaluation Dashboard Notebook
        ## Compare Chunking, Retrieval, and Groundedness Across Strategies

        **Stack:** Ollama · LangChain · ChromaDB · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Setup Evaluation Framework"),
        code("""
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import json

        llm = ChatOllama(model="qwen3:8b", temperature=0.0)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Source document
        source_text = \"\"\"
        Kubernetes is a container orchestration platform. It manages containerized
        applications across multiple hosts. Key components include pods (smallest unit),
        services (stable networking), deployments (declarative updates), and ingress
        (external access). The control plane runs the API server, scheduler, and etcd.
        Worker nodes run kubelet and container runtime. Horizontal Pod Autoscaler (HPA)
        scales pods based on CPU/memory metrics. Kubernetes supports rolling updates
        for zero-downtime deployments. Namespaces provide logical isolation. ConfigMaps
        and Secrets manage configuration. Persistent Volumes handle storage needs.
        \"\"\"

        # Ground truth QA pairs
        eval_set = [
            {"q": "What is the smallest unit in Kubernetes?", "a": "Pods"},
            {"q": "What provides stable networking?", "a": "Services"},
            {"q": "What does HPA scale based on?", "a": "CPU/memory metrics"},
            {"q": "How does Kubernetes handle zero-downtime updates?", "a": "Rolling updates"},
            {"q": "What manages configuration?", "a": "ConfigMaps and Secrets"},
        ]
        print(f"Source: {len(source_text)} chars, Eval set: {len(eval_set)} QA pairs")
        """),
        md("## Step 2 — Compare Chunking Strategies"),
        code("""
        chunk_configs = [
            {"name": "small", "chunk_size": 100, "overlap": 10},
            {"name": "medium", "chunk_size": 250, "overlap": 30},
            {"name": "large", "chunk_size": 500, "overlap": 50},
        ]

        doc = Document(page_content=source_text, metadata={"source": "k8s_doc"})
        results = {}

        for config in chunk_configs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["chunk_size"], chunk_overlap=config["overlap"])
            chunks = splitter.split_documents([doc])

            vs = Chroma.from_documents(chunks, embeddings,
                collection_name=f"eval_{config['name']}")

            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="Answer using the context only.\\nContext: {context}\\nQ: {question}\\nA:")

            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff",
                retriever=vs.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_prompt},
            )

            config_results = []
            for item in eval_set:
                result = qa.invoke({"query": item["q"]})
                answer = result["result"]
                gt = item["a"]
                # Simple contains check for evaluation
                correct = gt.lower() in answer.lower()
                config_results.append({
                    "question": item["q"],
                    "expected": gt,
                    "got": answer[:100],
                    "correct": correct,
                    "n_sources": len(result["source_documents"]),
                })

            accuracy = sum(r["correct"] for r in config_results) / len(config_results)
            results[config["name"]] = {"accuracy": accuracy, "n_chunks": len(chunks), "details": config_results}
            print(f"  {config['name']}: {len(chunks)} chunks, accuracy={accuracy:.0%}")
        """),
        md("## Step 3 — Dashboard Summary"),
        code("""
        print("\\n" + "="*60)
        print("RAG EVALUATION DASHBOARD")
        print("="*60)

        print(f"\\n{'Strategy':<12} {'Chunks':>8} {'Accuracy':>10} {'Status':>10}")
        print("-"*45)
        for name, data in results.items():
            status = "✓ GOOD" if data["accuracy"] >= 0.8 else "⚠ CHECK" if data["accuracy"] >= 0.6 else "✗ POOR"
            print(f"{name:<12} {data['n_chunks']:>8} {data['accuracy']:>9.0%} {status:>10}")

        best = max(results.items(), key=lambda x: x[1]["accuracy"])
        print(f"\\nBest strategy: {best[0]} ({best[1]['accuracy']:.0%} accuracy)")

        # Per-question breakdown for best strategy
        print(f"\\nDetailed results for '{best[0]}':")
        for r in best[1]["details"]:
            icon = "✓" if r["correct"] else "✗"
            print(f"  {icon} Q: {r['question'][:40]}...")
            print(f"    Expected: {r['expected']} | Got: {r['got'][:60]}")
        """),
        md("## Step 4 — Groundedness Evaluation"),
        code("""
        judge_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rate if the answer is grounded in the context. "
             "Return a score from 0-10 and one word: 'grounded' or 'hallucinated'."),
            ("human", "Context: {context}\\nAnswer: {answer}\\nScore and verdict:")
        ])
        judge_chain = judge_prompt | llm | StrOutputParser()

        best_strategy = best[0]
        best_results = best[1]["details"]

        print("Groundedness Check:")
        for r in best_results:
            verdict = judge_chain.invoke({"context": source_text, "answer": r["got"]})
            print(f"  Q: {r['question'][:40]}... → {verdict.strip()[:30]}")
        """),
        md("""
        ## What You Learned
        - **Systematic evaluation** of RAG configurations
        - **Chunking strategy comparison** with ground truth
        - **Groundedness scoring** of generated answers
        - **Dashboard-style reporting** for RAG quality metrics
        """),
    ]))

    print(f"Group 3 complete: {len(paths)} notebooks written")
    for p in paths:
        print(f"  ✓ {p}")
    return paths

if __name__ == "__main__":
    build()
