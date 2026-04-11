"""Group 6 — Projects 51-60: Local Tool-Using Agents."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── Project 51: Local Web Research Agent ────────────────────────────
    paths.append(write_nb(5, "51_Local_Web_Research_Agent", [
        md("""
        # Project 51 — Local Web Research Agent
        ## Search, Compare Sources, Write Cited Answers

        **Stack:** LangChain · Ollama · DuckDuckGo · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community duckduckgo-search"),
        md("## Step 1 — Setup with Tool Definitions"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.tools import tool
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)

        @tool
        def web_search(query: str) -> str:
            \"\"\"Search the web for information. Returns top results.\"\"\"
            # Simulated search results for offline use
            results = {
                "RAG vs fine-tuning": [
                    "RAG retrieves documents at inference; fine-tuning modifies model weights. RAG is cheaper to update.",
                    "Fine-tuning excels at style/tone adaptation; RAG excels at factual accuracy with citations.",
                ],
                "Ollama models 2025": [
                    "Ollama supports Llama 3, Mistral, Qwen, Phi-3, Gemma and more for local execution.",
                    "Ollama 0.5 introduced multi-model serving and 30% memory reduction.",
                ],
            }
            for key, vals in results.items():
                if any(w in query.lower() for w in key.lower().split()):
                    return "\\n".join(vals)
            return "No specific results found. Try a more specific query."

        @tool
        def summarize_text(text: str) -> str:
            \"\"\"Summarize a long text into key points.\"\"\"
            chain = ChatPromptTemplate.from_messages([
                ("system", "Summarize in 3 bullet points."),
                ("human", "{text}")
            ]) | llm | StrOutputParser()
            return chain.invoke({"text": text})

        tools = [web_search, summarize_text]
        print(f"Defined {len(tools)} tools: {[t.name for t in tools]}")
        """),
        md("## Step 2 — Build Research Agent"),
        code("""
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import MessagesPlaceholder

        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", \"\"\"You are a research agent with access to web search and summarization tools.
        For each question:
        1. Search for relevant information
        2. Summarize findings
        3. Write a cited answer

        Always cite your sources.\"\"\"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # For models that don't support tool calling natively, use ReAct
        from langchain.agents import create_react_agent

        react_prompt = ChatPromptTemplate.from_template(
            \"\"\"Answer the question using the available tools.

        Tools: {tools}
        Tool Names: {tool_names}

        Question: {input}

        Think step by step.
        {agent_scratchpad}\"\"\"
        )

        # Direct approach without full agent (works reliably with local models)
        def research(question):
            search_result = web_search.invoke(question)
            summary = summarize_text.invoke(search_result)
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "Write a comprehensive answer using the research. Cite sources."),
                ("human", "Question: {question}\\nResearch: {research}\\nSummary: {summary}")
            ])
            chain = answer_prompt | llm | StrOutputParser()
            return chain.invoke({"question": question, "research": search_result, "summary": summary})

        print("Research agent ready!")
        """),
        md("## Step 3 — Run Research Queries"),
        code("""
        queries = [
            "What's the difference between RAG and fine-tuning?",
            "What models does Ollama support in 2025?",
        ]

        for q in queries:
            print(f"\\n{'='*60}")
            print(f"Research Q: {q}")
            print("="*60)
            answer = research(q)
            print(answer)
        """),
        md("""
        ## What You Learned
        - **Tool-using agents** with search and summarization
        - **Research pipeline:** search → summarize → cite → answer
        - **Offline-friendly** design with simulated search results
        """),
    ]))

    # ── Project 52: Local Spreadsheet Analyst Agent ─────────────────────
    paths.append(write_nb(5, "52_Local_Spreadsheet_Analyst_Agent", [
        md("""
        # Project 52 — Local Spreadsheet Analyst Agent
        ## Answer Questions Over CSV/XLSX and Generate Insights

        **Stack:** LangChain · Ollama · pandas · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-experimental pandas"),
        md("## Step 1 — Create Sample Dataset"),
        code("""
        import pandas as pd
        from pathlib import Path

        Path("sample_data").mkdir(exist_ok=True)

        sales = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "region": ["North","South","East","West"]*3,
            "product": ["Widget A","Widget B","Widget C"]*4,
            "revenue": [12000,15000,9000,11000,13500,16000,10500,12500,14000,17000,11500,13000],
            "units": [240,300,180,220,270,320,210,250,280,340,230,260],
            "returns": [12,5,8,15,10,3,14,9,7,2,11,6],
        })
        sales.to_csv("sample_data/sales_data.csv", index=False)
        print(sales.to_string(index=False))
        print(f"\\nShape: {sales.shape}")
        """),
        md("## Step 2 — Setup Pandas Agent"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)

        def analyze_data(question, df):
            \"\"\"Use LLM to generate and execute pandas code for analysis.\"\"\"
            # Generate code
            code_prompt = ChatPromptTemplate.from_messages([
                ("system", \"\"\"You are a data analyst. Given a pandas DataFrame `df` with columns:
        {columns}

        Sample data:
        {sample}

        Write Python code using pandas to answer the question.
        Return ONLY the Python code, no explanation. Use print() for output.\"\"\"),
                ("human", "{question}")
            ])
            code_chain = code_prompt | llm | StrOutputParser()
            generated_code = code_chain.invoke({
                "columns": str(list(df.columns)),
                "sample": df.head(3).to_string(),
                "question": question,
            })

            # Clean up code
            code_lines = []
            for line in generated_code.split("\\n"):
                line = line.strip()
                if line and not line.startswith("```"):
                    code_lines.append(line)
            clean_code = "\\n".join(code_lines)

            # Execute safely
            import io, contextlib
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    exec(clean_code, {"df": df, "pd": pd, "print": print})
                result = output.getvalue()
            except Exception as e:
                result = f"Error: {e}\\nGenerated code:\\n{clean_code}"

            return {"code": clean_code, "result": result}

        print("Spreadsheet analyst ready!")
        """),
        md("## Step 3 — Ask Questions About the Data"),
        code("""
        questions = [
            "What is the total revenue by region?",
            "Which product has the highest return rate?",
            "What's the average revenue per unit sold?",
            "Show the month with the highest revenue",
        ]

        for q in questions:
            print(f"\\nQ: {q}")
            analysis = analyze_data(q, sales)
            print(f"Code: {analysis['code'][:100]}...")
            print(f"Result: {analysis['result']}")
        """),
        md("## Step 4 — Generate Insights Report"),
        code("""
        insight_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a business analyst. Given this sales data, provide "
             "5 key insights and 3 actionable recommendations."),
            ("human", "Data summary:\\n{data}")
        ])
        insight_chain = insight_prompt | llm | StrOutputParser()

        data_summary = f\"\"\"
        Total Revenue: ${sales['revenue'].sum():,}
        Avg Monthly Revenue: ${sales['revenue'].mean():,.0f}
        Revenue by Region: {sales.groupby('region')['revenue'].sum().to_dict()}
        Revenue by Product: {sales.groupby('product')['revenue'].sum().to_dict()}
        Total Returns: {sales['returns'].sum()} ({sales['returns'].sum()/sales['units'].sum()*100:.1f}%)
        Best Month: {sales.loc[sales['revenue'].idxmax(), 'date'].strftime('%B %Y')}
        \"\"\"

        insights = insight_chain.invoke({"data": data_summary})
        print("BUSINESS INSIGHTS REPORT")
        print("="*40)
        print(insights)
        """),
        md("""
        ## What You Learned
        - **NL-to-pandas** code generation for data analysis
        - **Safe code execution** with output capture
        - **Automated insight generation** from data summaries
        """),
    ]))

    # ── Project 53: Local SQL Analyst Agent ─────────────────────────────
    paths.append(write_nb(5, "53_Local_SQL_Analyst_Agent", [
        md("""
        # Project 53 — Local SQL Analyst Agent
        ## Natural Language → SQL → Results → Summary

        **Stack:** LangChain · Ollama · SQLite · Jupyter
        """),
        code("# !pip install -q langchain langchain-ollama langchain-community"),
        md("## Step 1 — Create Sample SQLite Database"),
        code("""
        import sqlite3
        from pathlib import Path

        Path("sample_data").mkdir(exist_ok=True)
        conn = sqlite3.connect("sample_data/company.db")
        cur = conn.cursor()

        cur.executescript(\"\"\"
        DROP TABLE IF EXISTS employees;
        DROP TABLE IF EXISTS departments;
        DROP TABLE IF EXISTS projects;

        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget REAL
        );

        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dept_id INTEGER REFERENCES departments(id),
            salary REAL,
            hire_date TEXT,
            role TEXT
        );

        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dept_id INTEGER REFERENCES departments(id),
            status TEXT,
            budget_used REAL
        );

        INSERT INTO departments VALUES (1,'Engineering',500000),(2,'Marketing',200000),(3,'Sales',300000);

        INSERT INTO employees VALUES
        (1,'Alice',1,120000,'2022-01-15','Senior Engineer'),
        (2,'Bob',1,95000,'2023-03-01','Engineer'),
        (3,'Carol',2,85000,'2022-06-15','Marketing Manager'),
        (4,'Dave',3,90000,'2021-11-01','Sales Lead'),
        (5,'Eve',1,110000,'2022-09-01','Staff Engineer'),
        (6,'Frank',2,75000,'2024-01-10','Marketing Associate'),
        (7,'Grace',3,82000,'2023-07-01','Sales Rep'),
        (8,'Hank',1,130000,'2021-03-15','Engineering Manager');

        INSERT INTO projects VALUES
        (1,'API Redesign',1,'active',150000),
        (2,'Brand Refresh',2,'completed',80000),
        (3,'Q1 Campaign',3,'active',120000),
        (4,'ML Pipeline',1,'active',200000),
        (5,'Website Relaunch',2,'planning',50000);
        \"\"\")
        conn.commit()

        # Show schema
        for table in ['departments', 'employees', 'projects']:
            cur.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
            print(f"\\n{table}: {len(rows)} rows")
            for row in rows[:3]:
                print(f"  {row}")
        conn.close()
        print("\\nDatabase created!")
        """),
        md("## Step 2 — Build NL-to-SQL Agent"),
        code("""
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.0)

        schema = \"\"\"
        departments(id, name, budget)
        employees(id, name, dept_id, salary, hire_date, role)
        projects(id, name, dept_id, status, budget_used)
        \"\"\"

        def nl_to_sql(question):
            # Step 1: Generate SQL
            sql_prompt = ChatPromptTemplate.from_messages([
                ("system", f\"\"\"You are a SQL expert. Given this schema:
        {schema}
        Generate a SQLite query to answer the question. Return ONLY the SQL, no explanation.\"\"\"),
                ("human", "{question}")
            ])
            sql_chain = sql_prompt | llm | StrOutputParser()
            sql = sql_chain.invoke({"question": question}).strip()
            sql = sql.replace("```sql", "").replace("```", "").strip()

            # Step 2: Execute
            conn = sqlite3.connect("sample_data/company.db")
            try:
                results = conn.execute(sql).fetchall()
                columns = [desc[0] for desc in conn.execute(sql).description] if results else []
            except Exception as e:
                conn.close()
                return {"sql": sql, "error": str(e)}
            conn.close()

            # Step 3: Summarize
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize the SQL results in natural language."),
                ("human", "Question: {question}\\nSQL: {sql}\\nResults: {results}")
            ])
            summary_chain = summary_prompt | llm | StrOutputParser()
            summary = summary_chain.invoke({
                "question": question, "sql": sql,
                "results": str(list(zip(columns, *zip(*results))) if results else "No results"),
            })

            return {"sql": sql, "results": results, "columns": columns, "summary": summary}

        print("SQL analyst ready!")
        """),
        md("## Step 3 — Run Queries"),
        code("""
        questions = [
            "What is the average salary by department?",
            "List all active projects and their budgets",
            "Who is the highest paid employee?",
            "How many employees were hired in 2023?",
            "What percentage of the engineering budget has been used by projects?",
        ]

        for q in questions:
            print(f"\\nQ: {q}")
            result = nl_to_sql(q)
            if "error" in result:
                print(f"  SQL: {result['sql']}")
                print(f"  Error: {result['error']}")
            else:
                print(f"  SQL: {result['sql']}")
                print(f"  Results: {result['results'][:5]}")
                print(f"  Summary: {result['summary']}")
        """),
        md("""
        ## What You Learned
        - **NL-to-SQL generation** with schema context
        - **Safe SQL execution** with error handling
        - **Result summarization** in natural language
        """),
    ]))

    # ── Projects 54-60: Tool-Using Agent patterns ──────────────────────
    tool_projects = [
        (54, "54_Local_Filesystem_Agent", "Local Filesystem Agent",
         "Search, summarize, and organize files with approval",
         "filesystem tools, pathlib, approval flow",
         """
        from langchain_ollama import ChatOllama
        from langchain_core.tools import tool
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pathlib import Path
        import os

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)

        @tool
        def list_files(directory: str) -> str:
            \"\"\"List files in a directory.\"\"\"
            p = Path(directory)
            if not p.exists():
                return f"Directory not found: {directory}"
            files = list(p.iterdir())
            return "\\n".join([f"{'[DIR]' if f.is_dir() else '[FILE]'} {f.name}" for f in files[:20]])

        @tool
        def read_file(filepath: str) -> str:
            \"\"\"Read a text file and return its contents (first 500 chars).\"\"\"
            try:
                return Path(filepath).read_text(encoding="utf-8")[:500]
            except Exception as e:
                return f"Error reading file: {e}"

        @tool
        def file_info(filepath: str) -> str:
            \"\"\"Get file metadata: size, modified date, type.\"\"\"
            p = Path(filepath)
            if not p.exists():
                return "File not found"
            stat = p.stat()
            return f"Name: {p.name}, Size: {stat.st_size} bytes, Extension: {p.suffix}"

        # Test the tools
        files = list_files.invoke(".")
        print("Files in current directory:")
        print(files[:500])
        """,
         """
        def analyze_directory(directory="."):
            files = list_files.invoke(directory)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Analyze this file listing. Categorize files by type and suggest organization."),
                ("human", "Directory: {dir}\\nFiles:\\n{files}")
            ])
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"dir": directory, "files": files})

        analysis = analyze_directory(".")
        print("Directory Analysis:")
        print(analysis)
        """),

        (55, "55_Local_GitHub_Repo_Reader_Agent", "Local GitHub Repo Reader Agent",
         "Inspect codebase and answer questions about code",
         "code search, AST parsing, embeddings",
         """
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pathlib import Path
        import ast

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)

        def extract_python_info(filepath):
            \"\"\"Extract functions, classes, and docstrings from a Python file.\"\"\"
            try:
                source = Path(filepath).read_text(encoding="utf-8")
                tree = ast.parse(source)
                info = {"functions": [], "classes": [], "imports": []}
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        doc = ast.get_docstring(node) or ""
                        info["functions"].append({"name": node.name, "doc": doc[:100], "line": node.lineno})
                    elif isinstance(node, ast.ClassDef):
                        doc = ast.get_docstring(node) or ""
                        info["classes"].append({"name": node.name, "doc": doc[:100], "line": node.lineno})
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        names = [a.name for a in node.names]
                        info["imports"].extend(names)
                return info
            except Exception as e:
                return {"error": str(e)}

        # Demo: analyze this very script
        demo_code = '''
        def calculate_roi(investment, returns):
            \"\"\"Calculate return on investment.\"\"\"
            return (returns - investment) / investment * 100

        class DataProcessor:
            \"\"\"Processes raw data into clean formats.\"\"\"
            def __init__(self, data):
                self.data = data
            def clean(self):
                return [x.strip() for x in self.data if x]
        '''
        Path("sample_data").mkdir(exist_ok=True)
        Path("sample_data/demo_module.py").write_text(demo_code)
        info = extract_python_info("sample_data/demo_module.py")
        print(f"Functions: {[f['name'] for f in info['functions']]}")
        print(f"Classes: {[c['name'] for c in info['classes']]}")
        """,
         """
        def explain_code(filepath):
            source = Path(filepath).read_text(encoding="utf-8")[:2000]
            info = extract_python_info(filepath)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Explain this code. Cover: purpose, key functions/classes, and potential improvements."),
                ("human", "Code:\\n{code}\\n\\nStructure: {info}")
            ])
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"code": source, "info": str(info)})

        explanation = explain_code("sample_data/demo_module.py")
        print("Code Explanation:")
        print(explanation)
        """),

        (56, "56_Local_CLI_Command_Planner_Agent", "Local CLI Command Planner Agent",
         "Suggest CLI commands with explanation and approval",
         "command generation, safety checks",
         """
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pydantic import BaseModel, Field

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)

        class CommandSuggestion(BaseModel):
            command: str = Field(description="The CLI command")
            explanation: str = Field(description="What the command does")
            risk_level: str = Field(description="safe, moderate, dangerous")
            alternatives: list[str] = Field(description="Alternative approaches")

        command_llm = llm.with_structured_output(CommandSuggestion)
        print("CLI planner ready!")
        """,
         """
        tasks = [
            "Find all Python files larger than 1MB in the current directory",
            "Show disk usage for each subdirectory",
            "Find and delete all __pycache__ directories",
            "Check which process is using port 8080",
        ]

        for task in tasks:
            print(f"\\nTask: {task}")
            suggestion = command_llm.invoke(f"Suggest a CLI command for: {task}")
            print(f"  Command: {suggestion.command}")
            print(f"  Risk: {suggestion.risk_level}")
            print(f"  Explanation: {suggestion.explanation}")
            if suggestion.alternatives:
                print(f"  Alternatives: {suggestion.alternatives}")
        """),

        (57, "57_Local_Expense_Processing_Agent", "Local Expense Processing Agent",
         "Parse receipt text, categorize expenses, generate summary",
         "text extraction, categorization, reporting",
         """
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pydantic import BaseModel, Field
        import json

        llm = ChatOllama(model="qwen3:8b", temperature=0.0)

        class ExpenseItem(BaseModel):
            vendor: str
            amount: float
            category: str = Field(description="meals, travel, supplies, software, other")
            date: str
            description: str
            reimbursable: bool

        class ExpenseReport(BaseModel):
            items: list[ExpenseItem]
            total: float
            by_category: dict[str, float]

        expense_llm = llm.with_structured_output(ExpenseReport)

        receipts = [
            "Uber ride Jan 15 2025 - $32.50 from office to airport for client meeting",
            "Starbucks Jan 15 2025 - $12.80 coffee with client Jane Smith",
            "AWS Jan 2025 - $450 monthly cloud hosting for dev environment",
            "Office Depot Jan 16 - $85.20 printer paper and notebooks for team",
            "Delta Airlines Jan 15 - $380 roundtrip SFO to NYC client visit",
        ]
        print(f"Processing {len(receipts)} receipts...")
        """,
         """
        receipt_text = "\\n".join(receipts)
        report = expense_llm.invoke(
            f"Parse these receipts into an expense report:\\n\\n{receipt_text}"
        )

        print("EXPENSE REPORT")
        print("="*50)
        for item in report.items:
            status = "✓" if item.reimbursable else "✗"
            print(f"  {status} ${item.amount:>8.2f} | {item.category:<10} | {item.vendor} — {item.description}")

        print(f"\\nTotal: ${report.total:,.2f}")
        print(f"By Category: {json.dumps(report.by_category, indent=2)}")
        """),

        (58, "58_Local_Calendar_Planner_Agent", "Local Calendar Planner Agent",
         "Analyze schedule, detect conflicts, propose optimal meeting times",
         "time management, conflict detection",
         """
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pydantic import BaseModel, Field

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)

        # Simulated calendar
        calendar = [
            {"title": "Team Standup", "start": "09:00", "end": "09:30", "day": "Mon-Fri"},
            {"title": "Product Review", "start": "10:00", "end": "11:00", "day": "Tuesday"},
            {"title": "1:1 with Manager", "start": "14:00", "end": "14:30", "day": "Wednesday"},
            {"title": "Sprint Planning", "start": "10:00", "end": "12:00", "day": "Monday"},
            {"title": "Focus Time", "start": "13:00", "end": "16:00", "day": "Thursday"},
            {"title": "Team Lunch", "start": "12:00", "end": "13:00", "day": "Friday"},
        ]
        print(f"Calendar: {len(calendar)} recurring events")
        for e in calendar:
            print(f"  {e['day']}: {e['start']}-{e['end']} — {e['title']}")
        """,
         """
        import json

        class MeetingSuggestion(BaseModel):
            proposed_times: list[str] = Field(description="List of available time slots")
            conflicts: list[str] = Field(description="Any conflicts detected")
            recommendation: str

        planner = llm.with_structured_output(MeetingSuggestion)

        requests = [
            "Schedule a 60-minute design review on Tuesday",
            "Find a 2-hour block for deep work on Wednesday",
            "Schedule a team-wide 30-min sync any day this week",
        ]

        for req in requests:
            print(f"\\nRequest: {req}")
            suggestion = planner.invoke(
                f"Calendar: {json.dumps(calendar)}\\n\\nRequest: {req}\\n\\n"
                f"Available hours: 9am-5pm. Find the best time slot."
            )
            print(f"  Proposed: {suggestion.proposed_times}")
            if suggestion.conflicts:
                print(f"  Conflicts: {suggestion.conflicts}")
            print(f"  Recommendation: {suggestion.recommendation}")
        """),

        (59, "59_Local_CRM_Enrichment_Agent", "Local CRM Enrichment Agent",
         "Summarize account info, generate next actions, enrich CRM data",
         "account analysis, action planning",
         """
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pydantic import BaseModel, Field

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)

        accounts = [
            {"name": "Acme Corp", "industry": "Manufacturing", "size": "500 employees",
             "contract": "$50K/yr", "renewal": "2025-06-01", "health": "at-risk",
             "last_contact": "45 days ago", "open_tickets": 3,
             "notes": "Unhappy with recent downtime. Considering competitor."},
            {"name": "TechStart Inc", "industry": "SaaS", "size": "50 employees",
             "contract": "$12K/yr", "renewal": "2025-09-01", "health": "healthy",
             "last_contact": "7 days ago", "open_tickets": 0,
             "notes": "Interested in upgrading to enterprise tier. Champion: CTO Sarah."},
        ]
        print(f"CRM accounts: {len(accounts)}")
        """,
         """
        class AccountAction(BaseModel):
            account: str
            risk_level: str
            priority_actions: list[str]
            talking_points: list[str]
            upsell_opportunity: str
            recommended_contact_date: str

        enricher = llm.with_structured_output(AccountAction)

        for acct in accounts:
            print(f"\\n{'='*50}")
            print(f"Account: {acct['name']}")
            result = enricher.invoke(f"Analyze this CRM account and suggest actions:\\n{acct}")
            print(f"  Risk: {result.risk_level}")
            print(f"  Priority Actions:")
            for a in result.priority_actions:
                print(f"    → {a}")
            print(f"  Talking Points:")
            for t in result.talking_points:
                print(f"    • {t}")
            print(f"  Upsell: {result.upsell_opportunity}")
            print(f"  Next Contact: {result.recommended_contact_date}")
        """),

        (60, "60_Local_Browser_Task_Agent", "Local Browser Task Agent",
         "Plan and prototype web task automation",
         "task planning, DOM simulation",
         """
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pydantic import BaseModel, Field

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)

        class BrowserStep(BaseModel):
            step_number: int
            action: str = Field(description="click, type, navigate, scroll, wait, extract")
            target: str = Field(description="CSS selector or URL")
            value: str = Field(default="", description="Text to type or data to extract")
            explanation: str

        class BrowserPlan(BaseModel):
            task: str
            steps: list[BrowserStep]
            estimated_time: str

        planner = llm.with_structured_output(BrowserPlan)
        print("Browser task planner ready!")
        """,
         """
        tasks = [
            "Log into a web application and download the latest report",
            "Search for a product on an e-commerce site and compare prices",
            "Fill out a multi-step registration form with provided data",
        ]

        for task in tasks:
            print(f"\\n{'='*50}")
            print(f"Task: {task}")
            plan = planner.invoke(f"Create a step-by-step browser automation plan for: {task}")
            print(f"Estimated time: {plan.estimated_time}")
            for step in plan.steps:
                print(f"  Step {step.step_number}: [{step.action}] {step.target}")
                print(f"    {step.explanation}")
                if step.value:
                    print(f"    Value: {step.value}")
        """),
    ]

    for proj_num, folder, title, desc, tools_used, setup_code, main_code in tool_projects:
        paths.append(write_nb(5, folder, [
            md(f"""
        # Project {proj_num} — {title}
        ## {desc}

        **Tools:** {tools_used}
        **Stack:** LangChain · Ollama · Jupyter
        """),
            code("# !pip install -q langchain langchain-ollama langchain-community pydantic"),
            md("## Step 1 — Setup and Tool Definitions"),
            code(setup_code),
            md("## Step 2 — Run the Agent"),
            code(main_code),
            md(f"""
        ## What You Learned
        - **Tool-augmented LLM** for {desc.lower()}
        - **Structured output** with Pydantic models
        - **Local execution** — all processing on your machine
        """),
        ]))

    print(f"Group 6 complete: {len(paths)} notebooks written")
    for p in paths:
        print(f"  ✓ {p}")
    return paths

if __name__ == "__main__":
    build()
