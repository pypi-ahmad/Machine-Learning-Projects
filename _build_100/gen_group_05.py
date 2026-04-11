"""Group 5 — Projects 41-50: CrewAI Multi-Agent Systems."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    crews = [
        (41, "41_CrewAI_Startup_Validation_Crew", "CrewAI Startup Validation Crew",
         "Validate a startup idea with multi-agent analysis",
         [("Market Researcher", "Research the market size, trends, and target audience for the startup idea. Provide data-driven insights.", "market_analysis"),
          ("Competitor Analyst", "Identify top 5 competitors, their strengths/weaknesses, pricing, and market positioning.", "competitor_report"),
          ("Financial Analyst", "Estimate startup costs, revenue projections, unit economics, and break-even timeline.", "financial_model"),
          ("Devil's Advocate", "Challenge the startup idea with tough questions, risks, and potential failure modes.", "risk_assessment")],
         "Validate this startup idea: An AI-powered local tutoring platform that matches students with tutors based on learning style."),

        (42, "42_CrewAI_Content_Studio", "CrewAI Content Studio",
         "Multi-agent content creation pipeline: research → write → edit → repurpose",
         [("Content Researcher", "Research the topic thoroughly. Find key facts, statistics, expert opinions, and trending angles.", "research_brief"),
          ("Content Writer", "Write a compelling 800-word blog post based on the research brief. Use clear structure with headers.", "blog_draft"),
          ("Editor", "Edit the draft for clarity, grammar, tone consistency, and engagement. Improve the hook and CTA.", "edited_article"),
          ("Content Repurposer", "Create 3 derivative pieces: a Twitter thread (5 tweets), LinkedIn post, and email newsletter summary.", "content_variants")],
         "Topic: How small businesses can use AI to compete with enterprises in 2025"),

        (43, "43_CrewAI_Lead_Gen_Crew", "CrewAI Lead Gen Crew",
         "ICP definition → company research → personalization → email drafting",
         [("ICP Analyst", "Define the Ideal Customer Profile for an AI analytics SaaS product. Include industry, company size, tech stack, pain points.", "icp_definition"),
          ("Company Researcher", "Research 3 target companies matching the ICP. Find key decision makers, recent initiatives, and potential hooks.", "company_profiles"),
          ("Personalization Specialist", "Create personalized talking points for each company, connecting our product to their specific needs.", "personalization_notes"),
          ("Email Copywriter", "Draft a personalized cold outreach email for each company. Under 150 words. Clear CTA for a 15-min demo.", "outreach_emails")],
         "Generate outreach for an AI-powered analytics platform targeting mid-market SaaS companies."),

        (44, "44_CrewAI_Job_Hunt_Crew", "CrewAI Job Hunt Crew",
         "JD analysis → resume tailoring → interview prep → coaching",
         [("JD Analyzer", "Parse the job description. Extract must-have skills, nice-to-haves, company values, and culture signals.", "jd_analysis"),
          ("Resume Tailor", "Rewrite the resume to emphasize relevant experience matching the JD requirements. Use action verbs and metrics.", "tailored_resume"),
          ("Interview Coach", "Generate 10 likely interview questions based on the JD and company. Include behavioral, technical, and situational questions.", "interview_prep"),
          ("Strategy Advisor", "Provide a comprehensive job application strategy: networking tips, portfolio suggestions, and follow-up plan.", "application_strategy")],
         "Prepare for a Senior ML Engineer role at a fintech startup. Resume: 4 years Python, ML models, deployed production systems."),

        (45, "45_CrewAI_Academic_Research_Crew", "CrewAI Academic Research Crew",
         "Literature search → summarization → gap analysis → bibliography",
         [("Literature Scout", "Find and summarize key research papers on the topic. Include authors, year, methodology, and findings.", "literature_review"),
          ("Methodology Analyst", "Compare research methodologies across papers. Identify common approaches, datasets, and evaluation metrics.", "methodology_analysis"),
          ("Gap Finder", "Identify gaps in the existing research. What questions remain unanswered? Where are the opportunities?", "research_gaps"),
          ("Bibliography Curator", "Create an organized bibliography with APA citations and annotated summaries for each source.", "annotated_bibliography")],
         "Research topic: Effectiveness of RAG systems versus fine-tuning for domain-specific NLP tasks"),

        (46, "46_CrewAI_Product_Launch_Crew", "CrewAI Product Launch Crew",
         "PM → Marketing → Analyst → QA multi-agent product launch",
         [("Product Manager", "Create a product requirements document (PRD) for the launch including target users, key features, success metrics, and timeline.", "prd"),
          ("Marketing Strategist", "Develop a go-to-market strategy with messaging, channels, content calendar, and launch day plan.", "gtm_strategy"),
          ("Data Analyst", "Define the analytics plan: key metrics, dashboards needed, A/B tests to run, and tracking requirements.", "analytics_plan"),
          ("QA Lead", "Create a launch readiness checklist: testing scenarios, rollback plan, monitoring alerts, and support escalation process.", "launch_checklist")],
         "Launch a new AI writing assistant feature for an existing productivity SaaS platform."),

        (47, "47_CrewAI_Competitor_Intelligence_Crew", "CrewAI Competitor Intelligence Crew",
         "Feature comparison → pricing analysis → SWOT → strategic memo",
         [("Feature Analyst", "Compare features across the top 3 competitors. Create a feature matrix showing strengths and gaps.", "feature_matrix"),
          ("Pricing Analyst", "Analyze competitor pricing models. Compare tiers, value per dollar, and positioning strategy.", "pricing_analysis"),
          ("SWOT Analyst", "Conduct SWOT analysis for each competitor. Identify our relative advantages and vulnerabilities.", "swot_analysis"),
          ("Strategy Summarizer", "Write an executive memo summarizing competitive landscape and recommending strategic actions.", "strategic_memo")],
         "Analyze competitors for a local-first AI code assistant tool competing with GitHub Copilot, Cursor, and Codeium."),

        (48, "48_CrewAI_Customer_Success_Crew", "CrewAI Customer Success Crew",
         "Complaint analysis → churn risk → response drafting → action plan",
         [("Complaint Analyst", "Analyze the customer complaint. Classify severity, identify root cause, and determine if it's a systemic issue.", "complaint_analysis"),
          ("Churn Risk Assessor", "Evaluate churn risk based on the complaint, account history, and revenue importance. Score 1-10.", "churn_assessment"),
          ("Response Writer", "Draft a personalized response to the customer. Acknowledge the issue, provide timeline, offer compensation if appropriate.", "customer_response"),
          ("Action Planner", "Create an internal action plan to resolve the issue and prevent recurrence. Include owners and deadlines.", "action_plan")],
         "Customer complaint: 'Your API has been down 3 times this month. We're paying $5000/month and considering switching to a competitor.'"),

        (49, "49_CrewAI_Recruiting_Crew", "CrewAI Recruiting Crew",
         "Resume screening → interview design → evaluation → recommendation",
         [("Resume Screener", "Screen the resume against job requirements. Score on technical skills, experience level, and cultural indicators.", "resume_screening"),
          ("Interview Designer", "Design a structured interview with specific questions targeting identified skill areas and gaps.", "interview_plan"),
          ("Technical Evaluator", "Create a take-home technical assessment aligned with the role. Include rubric and evaluation criteria.", "technical_assessment"),
          ("Hiring Advisor", "Provide a hiring recommendation based on all inputs. Include risk factors and onboarding suggestions.", "hiring_recommendation")],
         "Evaluate a candidate for Senior Backend Engineer: 5 years Python, Django, PostgreSQL, some Kubernetes, previously at a startup."),

        (50, "50_CrewAI_Ops_Review_Crew", "CrewAI Ops Review Crew",
         "Operations analysis → risk review → optimization → summary report",
         [("Ops Analyst", "Analyze the operational data and identify key trends: uptime, latency, error rates, resource utilization.", "ops_analysis"),
          ("Risk Reviewer", "Identify operational risks from the analysis. Classify by probability and impact. Recommend mitigations.", "risk_review"),
          ("Optimization Advisor", "Suggest specific optimizations for infrastructure, processes, and tooling based on the analysis.", "optimization_plan"),
          ("Report Writer", "Write an executive operations review report summarizing findings, risks, and recommended actions.", "ops_report")],
         "Review Q4 operations: 99.8% uptime, p95 latency 450ms (target 200ms), 3 incidents, infrastructure costs up 25%."),
    ]

    for proj_num, folder, title, desc, agents_config, sample_input in crews:
        agent_defs = ""
        task_defs = ""
        agent_names = []

        for role, backstory, output_name in agents_config:
            var_name = role.lower().replace(" ", "_").replace("'", "")
            agent_names.append(var_name)
            agent_defs += f"""
        {var_name} = Agent(
            role='{role}',
            goal='{backstory.split(".")[0]}',
            backstory='{backstory}',
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
"""
            task_defs += f"""
        {output_name}_task = Task(
            description='{backstory}',
            expected_output='Detailed {output_name.replace("_", " ")}',
            agent={var_name},
        )
"""

        task_names = [f"{cfg[2]}_task" for cfg in agents_config]

        paths.append(write_nb(6, folder, [
            md(f"""
        # Project {proj_num} — {title}
        ## {desc}

        **Agents:** {' → '.join([a[0] for a in agents_config])}

        **Stack:** CrewAI · Ollama · Jupyter
        """),
            code("# !pip install -q crewai langchain-community"),
            md("## Step 1 — Setup Local LLM"),
            code("""
        from langchain_community.llms import Ollama

        llm = Ollama(model="qwen3:8b", temperature=0.3)
        print("Local Ollama LLM ready for CrewAI!")
        """),
            md("## Step 2 — Define Agents"),
            code(f"""
        from crewai import Agent
{agent_defs}
        print("Agents defined: {', '.join([a[0] for a in agents_config])}")
        """),
            md("## Step 3 — Define Tasks"),
            code(f"""
        from crewai import Task
{task_defs}
        tasks = [{', '.join(task_names)}]
        print(f"{{len(tasks)}} tasks defined")
        """),
            md("## Step 4 — Assemble and Run Crew"),
            code(f"""
        from crewai import Crew, Process

        crew = Crew(
            agents=[{', '.join(agent_names)}],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        print("Crew assembled! Running {title}...")
        result = crew.kickoff(inputs={{"input": "{sample_input}"}})
        """),
            md("## Step 5 — Review Results"),
            code("""
        print("="*60)
        print("CREW OUTPUT")
        print("="*60)
        print(result)
        """),
            md(f"""
        ## What You Learned
        - **Multi-agent orchestration** with CrewAI
        - **Sequential task delegation** with specialized agents
        - **Local LLM execution** — no API keys needed
        - **Agent roles:** {', '.join([a[0] for a in agents_config])}
        """),
        ]))

    print(f"Group 5 complete: {len(paths)} notebooks written")
    for p in paths:
        print(f"  ✓ {p}")
    return paths

if __name__ == "__main__":
    build()
