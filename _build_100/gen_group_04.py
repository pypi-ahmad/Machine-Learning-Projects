"""Group 4 — Projects 31-40: LangGraph Workflows."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── Project 31: LangGraph Human Approval Workflow ───────────────────
    paths.append(write_nb(4, "31_LangGraph_Human_Approval_Workflow", [
        md("""
        # Project 31 — LangGraph Human Approval Workflow
        ## Agent Pauses for Human Approval Before Executing Actions

        **Stack:** LangGraph · Ollama · Jupyter
        """),
        code("# !pip install -q langgraph langchain langchain-ollama"),
        md("## Step 1 — Setup"),
        code("""
        from langgraph.graph import StateGraph, END
        from typing import TypedDict, Literal
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)

        class WorkflowState(TypedDict):
            request: str
            analysis: str
            proposed_action: str
            approval_status: str  # pending, approved, rejected
            result: str
        """),
        md("## Step 2 — Define Workflow Nodes"),
        code("""
        def analyze_request(state: WorkflowState) -> WorkflowState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Analyze this request and determine what action is needed. "
                 "Classify risk as LOW, MEDIUM, or HIGH."),
                ("human", "{request}")
            ])
            chain = prompt | llm | StrOutputParser()
            analysis = chain.invoke({"request": state["request"]})
            print(f"  📋 Analysis complete")
            return {"analysis": analysis}

        def propose_action(state: WorkflowState) -> WorkflowState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Based on the analysis, propose a specific action to take. "
                 "Be concrete about what will happen."),
                ("human", "Request: {request}\\nAnalysis: {analysis}")
            ])
            chain = prompt | llm | StrOutputParser()
            action = chain.invoke({"request": state["request"], "analysis": state["analysis"]})
            print(f"  🎯 Proposed action ready")
            return {"proposed_action": action, "approval_status": "pending"}

        def simulate_human_approval(state: WorkflowState) -> WorkflowState:
            \"\"\"In a real app, this would pause and wait for human input.
            Here we simulate approval based on risk level.\"\"\"
            analysis = state.get("analysis", "")
            # Auto-approve LOW risk, prompt for MEDIUM/HIGH
            if "LOW" in analysis.upper():
                print(f"  ✅ Auto-approved (low risk)")
                return {"approval_status": "approved"}
            elif "HIGH" in analysis.upper():
                print(f"  ❌ Rejected (high risk — requires manual review)")
                return {"approval_status": "rejected"}
            else:
                print(f"  ✅ Approved (medium risk)")
                return {"approval_status": "approved"}

        def execute_action(state: WorkflowState) -> WorkflowState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Execute the approved action and report the result."),
                ("human", "Action: {action}")
            ])
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"action": state["proposed_action"]})
            print(f"  🚀 Action executed")
            return {"result": result}

        def reject_action(state: WorkflowState) -> WorkflowState:
            return {"result": f"Action REJECTED. Proposed action was: {state['proposed_action'][:100]}..."}

        def route_approval(state: WorkflowState) -> Literal["execute", "reject"]:
            return "execute" if state["approval_status"] == "approved" else "reject"
        """),
        md("## Step 3 — Build the Graph"),
        code("""
        graph = StateGraph(WorkflowState)

        graph.add_node("analyze", analyze_request)
        graph.add_node("propose", propose_action)
        graph.add_node("approve", simulate_human_approval)
        graph.add_node("execute", execute_action)
        graph.add_node("reject", reject_action)

        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "propose")
        graph.add_edge("propose", "approve")
        graph.add_conditional_edges("approve", route_approval, {
            "execute": "execute",
            "reject": "reject",
        })
        graph.add_edge("execute", END)
        graph.add_edge("reject", END)

        app = graph.compile()
        print("Human approval workflow compiled!")
        """),
        md("## Step 4 — Test With Different Risk Levels"),
        code("""
        requests = [
            "Update the user's email address in the system",          # Low risk
            "Deploy the new feature to production servers",           # Medium risk
            "Delete all user data for GDPR compliance request",       # High risk
        ]

        for req in requests:
            print(f"\\n{'='*60}")
            print(f"Request: {req}")
            print("-"*60)
            result = app.invoke({
                "request": req, "analysis": "", "proposed_action": "",
                "approval_status": "pending", "result": ""
            })
            print(f"\\nFinal Result: {result['result'][:200]}")
        """),
        md("""
        ## What You Learned
        - **Human-in-the-loop** approval gates in LangGraph
        - **Risk-based routing** with conditional edges
        - **Workflow state management** across nodes
        """),
    ]))

    # ── Project 32: LangGraph Multi-Step Sales Research Flow ────────────
    paths.append(write_nb(4, "32_LangGraph_Sales_Research_Flow", [
        md("""
        # Project 32 — LangGraph Multi-Step Sales Research Flow
        ## Company Lookup → Analysis → Outreach Draft

        **Stack:** LangGraph · Ollama · Jupyter
        """),
        code("# !pip install -q langgraph langchain langchain-ollama"),
        md("## Step 1 — Setup"),
        code("""
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.3)

        class SalesState(TypedDict):
            company_name: str
            company_profile: str
            pain_points: str
            value_proposition: str
            outreach_email: str
        """),
        md("## Step 2 — Define Sales Research Nodes"),
        code("""
        def research_company(state: SalesState) -> SalesState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a sales researcher. Create a company profile including: "
                 "industry, estimated size, likely tech stack, and recent news/trends. "
                 "Use your knowledge to make reasonable inferences."),
                ("human", "Research this company: {company}")
            ])
            chain = prompt | llm | StrOutputParser()
            profile = chain.invoke({"company": state["company_name"]})
            print(f"  📊 Company profile researched")
            return {"company_profile": profile}

        def identify_pain_points(state: SalesState) -> SalesState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Based on the company profile, identify 3-5 likely pain points "
                 "that our AI/ML platform could solve. Be specific to their industry."),
                ("human", "Company: {company}\\nProfile: {profile}")
            ])
            chain = prompt | llm | StrOutputParser()
            pains = chain.invoke({"company": state["company_name"], "profile": state["company_profile"]})
            print(f"  🎯 Pain points identified")
            return {"pain_points": pains}

        def craft_value_prop(state: SalesState) -> SalesState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Create a tailored value proposition that addresses the identified "
                 "pain points. Focus on measurable outcomes (ROI, time saved, etc.)"),
                ("human", "Company: {company}\\nPain Points: {pains}")
            ])
            chain = prompt | llm | StrOutputParser()
            vp = chain.invoke({"company": state["company_name"], "pains": state["pain_points"]})
            print(f"  💎 Value proposition crafted")
            return {"value_proposition": vp}

        def draft_outreach(state: SalesState) -> SalesState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", \"\"\"Write a personalized cold outreach email. Rules:
        - Subject line that creates curiosity
        - Opening that references something specific about their company
        - 2-3 sentences connecting their pain to our solution
        - Clear CTA (15-min call)
        - Under 150 words
        - Professional but not corporate\"\"\"),
                ("human", \"\"\"Company: {company}
        Profile: {profile}
        Pain Points: {pains}
        Value Prop: {vp}

        Write the outreach email.\"\"\")
            ])
            chain = prompt | llm | StrOutputParser()
            email = chain.invoke({
                "company": state["company_name"], "profile": state["company_profile"],
                "pains": state["pain_points"], "vp": state["value_proposition"],
            })
            print(f"  ✉️ Outreach email drafted")
            return {"outreach_email": email}
        """),
        md("## Step 3 — Build Sales Pipeline Graph"),
        code("""
        graph = StateGraph(SalesState)
        graph.add_node("research", research_company)
        graph.add_node("pain_points", identify_pain_points)
        graph.add_node("value_prop", craft_value_prop)
        graph.add_node("outreach", draft_outreach)

        graph.set_entry_point("research")
        graph.add_edge("research", "pain_points")
        graph.add_edge("pain_points", "value_prop")
        graph.add_edge("value_prop", "outreach")
        graph.add_edge("outreach", END)

        app = graph.compile()
        print("Sales research pipeline compiled!")
        """),
        md("## Step 4 — Run for Target Companies"),
        code("""
        companies = ["Stripe", "Shopify", "Datadog"]

        for company in companies:
            print(f"\\n{'='*60}")
            print(f"RESEARCHING: {company}")
            print("="*60)
            result = app.invoke({
                "company_name": company, "company_profile": "", "pain_points": "",
                "value_proposition": "", "outreach_email": "",
            })
            print(f"\\n--- OUTREACH EMAIL ---")
            print(result["outreach_email"])
        """),
        md("""
        ## What You Learned
        - **Multi-step sales research** pipeline with LangGraph
        - **Progressive context building** across nodes
        - **Personalized outreach** from automated research
        """),
    ]))

    # ── Project 33: LangGraph Incident Summary Flow ─────────────────────
    paths.append(write_nb(4, "33_LangGraph_Incident_Summary_Flow", [
        md("""
        # Project 33 — LangGraph Incident Summary Flow
        ## Logs → Severity Classification → Summary → Next Steps

        **Stack:** LangGraph · Ollama · Jupyter
        """),
        code("# !pip install -q langgraph langchain langchain-ollama"),
        md("## Step 1 — Setup"),
        code("""
        from langgraph.graph import StateGraph, END
        from typing import TypedDict, Literal
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from pydantic import BaseModel, Field

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)

        class IncidentState(TypedDict):
            raw_logs: str
            severity: str
            affected_services: str
            root_cause_hypothesis: str
            summary: str
            next_steps: str
        """),
        md("## Step 2 — Define Incident Nodes"),
        code("""
        def classify_severity(state: IncidentState) -> IncidentState:
            class SeverityClassification(BaseModel):
                severity: str = Field(description="SEV1, SEV2, or SEV3")
                affected_services: str
                user_impact: str

            classifier = llm.with_structured_output(SeverityClassification)
            result = classifier.invoke(
                f"Classify this incident from logs:\\n\\n{state['raw_logs']}"
            )
            print(f"  🚨 Severity: {result.severity}")
            return {"severity": result.severity, "affected_services": result.affected_services}

        def analyze_root_cause(state: IncidentState) -> IncidentState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an SRE expert. Analyze these logs and propose the "
                 "most likely root cause. Consider: infrastructure, code, configuration, "
                 "and external dependencies."),
                ("human", "Logs: {logs}\\nAffected: {services}")
            ])
            chain = prompt | llm | StrOutputParser()
            hypothesis = chain.invoke({"logs": state["raw_logs"], "services": state["affected_services"]})
            print(f"  🔍 Root cause analyzed")
            return {"root_cause_hypothesis": hypothesis}

        def generate_summary(state: IncidentState) -> IncidentState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", \"\"\"Generate a concise incident summary with:
        - What happened (1-2 sentences)
        - Impact (users, services, duration)
        - Root cause hypothesis
        - Current status\"\"\"),
                ("human", \"\"\"Severity: {severity}
        Affected: {affected}
        Root Cause: {root_cause}
        Logs: {logs}\"\"\")
            ])
            chain = prompt | llm | StrOutputParser()
            summary = chain.invoke({
                "severity": state["severity"], "affected": state["affected_services"],
                "root_cause": state["root_cause_hypothesis"], "logs": state["raw_logs"],
            })
            print(f"  📝 Summary generated")
            return {"summary": summary}

        def recommend_next_steps(state: IncidentState) -> IncidentState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Based on the incident, recommend immediate next steps and "
                 "longer-term prevention measures. Be specific and actionable."),
                ("human", "Summary: {summary}\\nRoot Cause: {root_cause}")
            ])
            chain = prompt | llm | StrOutputParser()
            steps = chain.invoke({"summary": state["summary"], "root_cause": state["root_cause_hypothesis"]})
            print(f"  📋 Next steps recommended")
            return {"next_steps": steps}
        """),
        md("## Step 3 — Build Incident Graph"),
        code("""
        graph = StateGraph(IncidentState)
        graph.add_node("classify", classify_severity)
        graph.add_node("root_cause", analyze_root_cause)
        graph.add_node("summarize", generate_summary)
        graph.add_node("next_steps", recommend_next_steps)

        graph.set_entry_point("classify")
        graph.add_edge("classify", "root_cause")
        graph.add_edge("root_cause", "summarize")
        graph.add_edge("summarize", "next_steps")
        graph.add_edge("next_steps", END)

        app = graph.compile()
        print("Incident response workflow compiled!")
        """),
        md("## Step 4 — Process Sample Incidents"),
        code("""
        incidents = [
            \"\"\"[2025-01-15 14:23:01] ERROR api-gateway: Connection refused to payment-service:8080
        [2025-01-15 14:23:02] ERROR payment-service: OutOfMemoryError: Java heap space
        [2025-01-15 14:23:03] ERROR api-gateway: 503 Service Unavailable - /api/checkout
        [2025-01-15 14:23:05] WARN load-balancer: payment-service health check failed (3/3)
        [2025-01-15 14:23:10] ERROR api-gateway: 1,247 requests failed in last 60 seconds\"\"\",

            \"\"\"[2025-01-15 09:00:01] WARN cron-service: Daily report job started
        [2025-01-15 09:15:22] ERROR cron-service: Report generation timeout after 900s
        [2025-01-15 09:15:23] WARN cron-service: Retrying report generation (attempt 2/3)
        [2025-01-15 09:30:45] ERROR cron-service: Report generation failed — database query timeout
        [2025-01-15 09:30:46] INFO notification: Admin notified of report failure\"\"\",
        ]

        for i, logs in enumerate(incidents):
            print(f"\\n{'='*60}")
            print(f"INCIDENT {i+1}")
            print("="*60)
            result = app.invoke({
                "raw_logs": logs, "severity": "", "affected_services": "",
                "root_cause_hypothesis": "", "summary": "", "next_steps": "",
            })
            print(f"\\n--- INCIDENT REPORT ---")
            print(f"Severity: {result['severity']}")
            print(f"\\nSummary:\\n{result['summary']}")
            print(f"\\nNext Steps:\\n{result['next_steps']}")
        """),
        md("""
        ## What You Learned
        - **Log-based incident classification** with structured output
        - **Root cause analysis** workflow
        - **Automated incident reports** with actionable next steps
        """),
    ]))

    # ── Project 34: LangGraph Data Cleaning Approval Flow ──────────────
    paths.append(write_nb(4, "34_LangGraph_Data_Cleaning_Flow", [
        md("""
        # Project 34 — LangGraph Data Cleaning Approval Flow
        ## Suggest Transforms → Review → Apply with Approval

        **Stack:** LangGraph · Ollama · pandas · Jupyter
        """),
        code("# !pip install -q langgraph langchain langchain-ollama pandas"),
        md("## Step 1 — Setup with Dirty Dataset"),
        code("""
        from langgraph.graph import StateGraph, END
        from typing import TypedDict, Literal
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import pandas as pd
        import json

        llm = ChatOllama(model="qwen3:8b", temperature=0.1)

        # Create a messy dataset
        dirty_data = pd.DataFrame({
            "name": ["John Smith", "jane doe", "  Bob Johnson ", "ALICE WONG", "Charlie", None],
            "email": ["john@email.com", "not-an-email", "bob@email.com", "alice@email.com", "", "charlie@email.com"],
            "age": [25, 30, -5, 150, 28, 22],
            "salary": [50000, 60000, 55000, None, 45000, 52000],
            "dept": ["Engineering", "engineering", "Eng", "ENGINEERING", "Sales", "Sales"],
        })
        print("Dirty Dataset:")
        print(dirty_data.to_string())
        print(f"\\nIssues: {dirty_data.isnull().sum().sum()} nulls, inconsistent casing, invalid values")

        class CleaningState(TypedDict):
            data_summary: str
            suggested_transforms: str
            approved_transforms: list[str]
            cleaning_log: str
            clean_data_summary: str
        """),
        md("## Step 2 — Profile Data and Suggest Transforms"),
        code("""
        def profile_data(state: CleaningState) -> CleaningState:
            summary = f\"\"\"Dataset Shape: {dirty_data.shape}
        Columns: {list(dirty_data.columns)}
        Nulls: {dirty_data.isnull().sum().to_dict()}
        Sample:\\n{dirty_data.head().to_string()}\"\"\"
            print(f"  📊 Data profiled: {dirty_data.shape}")
            return {"data_summary": summary}

        def suggest_transforms(state: CleaningState) -> CleaningState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", \"\"\"You are a data quality expert. Analyze the dataset and suggest
        specific cleaning transforms. For each issue, suggest a numbered transform like:
        1. [column] — issue — transform
        Be specific about what values to change.\"\"\"),
                ("human", "Dataset summary:\\n{summary}")
            ])
            chain = prompt | llm | StrOutputParser()
            suggestions = chain.invoke({"summary": state["data_summary"]})
            print(f"  🔧 Transforms suggested")
            return {"suggested_transforms": suggestions}

        def auto_approve(state: CleaningState) -> CleaningState:
            \"\"\"Simulate approval — in production, this would be a human review step.\"\"\"
            transforms = [
                "Trim whitespace from name column",
                "Title-case all names",
                "Standardize dept to 'Engineering' or 'Sales'",
                "Replace invalid ages (<0 or >120) with median",
                "Fill null salary with column median",
                "Validate email format",
            ]
            print(f"  ✅ Approved {len(transforms)} transforms")
            return {"approved_transforms": transforms}

        def apply_transforms(state: CleaningState) -> CleaningState:
            global dirty_data
            clean = dirty_data.copy()
            log = []

            # Apply each transform
            clean["name"] = clean["name"].fillna("Unknown").str.strip().str.title()
            log.append(f"Names: trimmed, title-cased, filled {dirty_data['name'].isnull().sum()} nulls")

            dept_map = {"eng": "Engineering", "engineering": "Engineering"}
            clean["dept"] = clean["dept"].str.lower().map(lambda x: dept_map.get(x, x.title() if isinstance(x,str) else x))
            log.append("Dept: standardized to consistent casing")

            median_age = clean["age"][clean["age"].between(0, 120)].median()
            invalid_ages = ~clean["age"].between(0, 120)
            clean.loc[invalid_ages, "age"] = int(median_age)
            log.append(f"Age: replaced {invalid_ages.sum()} invalid values with median ({median_age:.0f})")

            median_salary = clean["salary"].median()
            null_salaries = clean["salary"].isnull().sum()
            clean["salary"] = clean["salary"].fillna(median_salary)
            log.append(f"Salary: filled {null_salaries} nulls with median ({median_salary:.0f})")

            import re
            email_pattern = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$')
            invalid_emails = clean["email"].apply(lambda x: not bool(email_pattern.match(str(x))) if x else True)
            clean.loc[invalid_emails, "email"] = "invalid@placeholder.com"
            log.append(f"Email: flagged {invalid_emails.sum()} invalid entries")

            cleaning_log = "\\n".join([f"  • {l}" for l in log])
            print(f"  🧹 Applied {len(log)} transforms")
            return {
                "cleaning_log": cleaning_log,
                "clean_data_summary": f"Clean dataset:\\n{clean.to_string()}"
            }
        """),
        md("## Step 3 — Build Cleaning Graph"),
        code("""
        graph = StateGraph(CleaningState)
        graph.add_node("profile", profile_data)
        graph.add_node("suggest", suggest_transforms)
        graph.add_node("approve", auto_approve)
        graph.add_node("apply", apply_transforms)

        graph.set_entry_point("profile")
        graph.add_edge("profile", "suggest")
        graph.add_edge("suggest", "approve")
        graph.add_edge("approve", "apply")
        graph.add_edge("apply", END)

        app = graph.compile()
        print("Data cleaning workflow compiled!")
        """),
        md("## Step 4 — Run the Pipeline"),
        code("""
        result = app.invoke({
            "data_summary": "", "suggested_transforms": "",
            "approved_transforms": [], "cleaning_log": "", "clean_data_summary": "",
        })

        print("\\nSuggested Transforms:")
        print(result["suggested_transforms"][:500])
        print("\\nCleaning Log:")
        print(result["cleaning_log"])
        print("\\nResult:")
        print(result["clean_data_summary"])
        """),
        md("""
        ## What You Learned
        - **Automated data profiling** to detect issues
        - **LLM-suggested transforms** for data quality
        - **Approval workflow** before applying changes
        - **Structured cleaning log** for auditability
        """),
    ]))

    # ── Project 35: LangGraph Resume Tailoring Flow ─────────────────────
    paths.append(write_nb(4, "35_LangGraph_Resume_Tailoring_Flow", [
        md("""
        # Project 35 — LangGraph Resume Tailoring Flow
        ## Parse JD → Match Skills → Tailor Resume → Draft Cover Letter

        **Stack:** LangGraph · Ollama · Jupyter
        """),
        code("# !pip install -q langgraph langchain langchain-ollama"),
        md("## Step 1 — Setup"),
        code("""
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.3)

        class ResumeState(TypedDict):
            resume: str
            job_description: str
            jd_requirements: str
            skill_matches: str
            tailored_resume: str
            cover_letter: str

        resume = \"\"\"Software Engineer with 3 years experience. Built REST APIs with Python/FastAPI.
        Led migration from monolith to microservices reducing deploy time by 60%.
        Mentored 3 junior developers. Experience with PostgreSQL, Redis, Docker, AWS.\"\"\"

        jd = \"\"\"Senior Backend Engineer — DataFlow Inc.
        Requirements: 3+ years Python backend, distributed systems experience,
        Kubernetes expertise, strong SQL skills, experience with data pipelines,
        leadership and mentoring ability, excellent communication.\"\"\"
        """),
        md("## Step 2 — Define Pipeline Nodes"),
        code("""
        def parse_jd(state: ResumeState) -> ResumeState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract a structured list of requirements from this JD. "
                 "Categorize as: must-have, nice-to-have."),
                ("human", "{jd}")
            ])
            chain = prompt | llm | StrOutputParser()
            reqs = chain.invoke({"jd": state["job_description"]})
            return {"jd_requirements": reqs}

        def match_skills(state: ResumeState) -> ResumeState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Compare the resume against JD requirements. List: "
                 "MATCHED skills, PARTIALLY MATCHED, and GAPS."),
                ("human", "Resume: {resume}\\n\\nRequirements: {reqs}")
            ])
            chain = prompt | llm | StrOutputParser()
            matches = chain.invoke({"resume": state["resume"], "reqs": state["jd_requirements"]})
            return {"skill_matches": matches}

        def tailor_resume(state: ResumeState) -> ResumeState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Rewrite the resume to better match the JD. "
                 "Emphasize matching skills, use JD keywords, add metrics. "
                 "Do NOT fabricate experience."),
                ("human", "Resume: {resume}\\nMatches: {matches}\\nJD: {jd}")
            ])
            chain = prompt | llm | StrOutputParser()
            tailored = chain.invoke({
                "resume": state["resume"], "matches": state["skill_matches"],
                "jd": state["job_description"],
            })
            return {"tailored_resume": tailored}

        def write_cover_letter(state: ResumeState) -> ResumeState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Write a cover letter connecting resume strengths to JD needs. "
                 "Under 200 words."),
                ("human", "Tailored Resume: {resume}\\nJD: {jd}\\nGaps: {matches}")
            ])
            chain = prompt | llm | StrOutputParser()
            letter = chain.invoke({
                "resume": state["tailored_resume"], "jd": state["job_description"],
                "matches": state["skill_matches"],
            })
            return {"cover_letter": letter}
        """),
        md("## Step 3 — Build and Run"),
        code("""
        graph = StateGraph(ResumeState)
        graph.add_node("parse_jd", parse_jd)
        graph.add_node("match_skills", match_skills)
        graph.add_node("tailor_resume", tailor_resume)
        graph.add_node("write_cover_letter", write_cover_letter)

        graph.set_entry_point("parse_jd")
        graph.add_edge("parse_jd", "match_skills")
        graph.add_edge("match_skills", "tailor_resume")
        graph.add_edge("tailor_resume", "write_cover_letter")
        graph.add_edge("write_cover_letter", END)

        app = graph.compile()

        result = app.invoke({
            "resume": resume, "job_description": jd,
            "jd_requirements": "", "skill_matches": "",
            "tailored_resume": "", "cover_letter": "",
        })

        print("=== SKILL ANALYSIS ===")
        print(result["skill_matches"])
        print("\\n=== TAILORED RESUME ===")
        print(result["tailored_resume"])
        print("\\n=== COVER LETTER ===")
        print(result["cover_letter"])
        """),
        md("""
        ## What You Learned
        - **JD parsing** into structured requirements
        - **Skill gap analysis** comparing resume vs JD
        - **Automated resume tailoring** preserving authenticity
        - **Multi-step document generation** pipeline
        """),
    ]))

    # ── Project 36-40: Remaining LangGraph projects ─────────────────────
    for proj_num, folder, title, desc, nodes in [
        (36, "36_LangGraph_Procurement_Review_Flow",
         "LangGraph Procurement Review Flow",
         "Compare vendor options and generate procurement recommendation",
         ["gather_requirements", "compare_vendors", "score_vendors", "generate_recommendation"]),
        (37, "37_LangGraph_Travel_Planner_Flow",
         "LangGraph Travel Planner Flow",
         "Gather preferences, plan itinerary, revise with checkpoints",
         ["gather_prefs", "plan_itinerary", "review_budget", "finalize_plan"]),
        (38, "38_LangGraph_Research_with_Memory",
         "LangGraph Research Workflow with Memory",
         "Accumulate research findings over multiple iterations",
         ["search", "evaluate", "accumulate", "synthesize"]),
        (39, "39_LangGraph_Ticket_Escalation_Router",
         "LangGraph Ticket Escalation Router",
         "Auto-resolve simple tickets or escalate complex ones",
         ["classify_ticket", "attempt_auto_resolve", "escalate", "send_response"]),
        (40, "40_LangGraph_Compliance_Checklist_Flow",
         "LangGraph Compliance Checklist Flow",
         "Gather evidence and generate compliance checklist with gap analysis",
         ["define_requirements", "gather_evidence", "assess_gaps", "generate_report"]),
    ]:
        node_defs = ""
        for i, node_name in enumerate(nodes):
            node_defs += f"""
        def {node_name}(state: WorkflowState) -> WorkflowState:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are step {i+1} of a {title} pipeline. "
                 "Your role: {node_name.replace('_', ' ')}. "
                 "Process the input and produce structured output for the next step."),
                ("human", "Input: {{input}}\\nPrevious steps: {{context}}")
            ])
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({{"input": state["input_data"], "context": state.get("accumulated", "")}})
            accumulated = state.get("accumulated", "") + f"\\n\\n[{node_name}]: " + result
            print(f"  Step {i+1}/{len(nodes)}: {node_name}")
            return {{"accumulated": accumulated, "final_output": result}}
"""

        edge_code = ""
        for i in range(len(nodes) - 1):
            edge_code += f'        graph.add_edge("{nodes[i]}", "{nodes[i+1]}")\n'

        node_adds = "\n".join([f'        graph.add_node("{n}", {n})' for n in nodes])

        paths.append(write_nb(4, folder, [
            md(f"""
        # Project {proj_num} — {title}
        ## {desc}

        **Stack:** LangGraph · Ollama · Jupyter

        **Workflow:** {' → '.join([n.replace('_', ' ').title() for n in nodes])}
        """),
            code("# !pip install -q langgraph langchain langchain-ollama"),
            md("## Step 1 — Setup"),
            code(f"""
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatOllama(model="qwen3:8b", temperature=0.2)

        class WorkflowState(TypedDict):
            input_data: str
            accumulated: str
            final_output: str
        """),
            md("## Step 2 — Define Workflow Nodes"),
            code(node_defs),
            md("## Step 3 — Build and Compile Graph"),
            code(f"""
        graph = StateGraph(WorkflowState)
{node_adds}

        graph.set_entry_point("{nodes[0]}")
{edge_code}        graph.add_edge("{nodes[-1]}", END)

        app = graph.compile()
        print("{title} — workflow compiled!")
        """),
            md("## Step 4 — Run the Workflow"),
            code(f"""
        sample_input = "Analyze and process this request through the {title} pipeline."
        result = app.invoke({{
            "input_data": sample_input, "accumulated": "", "final_output": "",
        }})

        print("=== WORKFLOW RESULT ===")
        print(result["final_output"])
        print("\\n=== FULL TRACE ===")
        print(result["accumulated"][:1000])
        """),
            md(f"""
        ## What You Learned
        - **Multi-node LangGraph workflow** for {desc.lower()}
        - **Progressive context accumulation** across steps
        - **Structured pipeline execution** with trace logging
        """),
        ]))

    print(f"Group 4 complete: {len(paths)} notebooks written")
    for p in paths:
        print(f"  ✓ {p}")
    return paths

if __name__ == "__main__":
    build()
