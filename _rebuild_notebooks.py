"""Rebuild notebooks 48 and 100 with proper JSON format."""
import json

# ── Notebook 48 ──
nb48 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Project 48 \u2014 CrewAI Customer Success Crew\n",
                "## Complaint analysis \u2192 churn risk \u2192 response drafting \u2192 action plan\n",
                "\n",
                "**Agents:** Complaint Analyst \u2192 Churn Risk Assessor \u2192 Response Writer \u2192 Action Planner\n",
                "\n",
                "**Stack:** CrewAI \u00b7 Ollama \u00b7 Jupyter"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# !pip install -q crewai langchain-community"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 1 \u2014 Setup Local LLM"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from langchain_community.llms import Ollama\n",
                "\n",
                "llm = Ollama(model=\"qwen3:8b\", temperature=0.3)\n",
                "print(\"Local Ollama LLM ready for CrewAI!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 2 \u2014 Define Agents"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from crewai import Agent\n",
                "\n",
                "complaint_analyst = Agent(\n",
                "    role='Complaint Analyst',\n",
                "    goal='Analyze the customer complaint',\n",
                "    backstory=\"Analyze the customer complaint. Classify severity, identify root cause, and determine if it is a systemic issue.\",\n",
                "    verbose=True,\n",
                "    allow_delegation=False,\n",
                "    llm=llm,\n",
                ")\n",
                "\n",
                "churn_risk_assessor = Agent(\n",
                "    role='Churn Risk Assessor',\n",
                "    goal='Evaluate churn risk based on the complaint, account history, and revenue importance',\n",
                "    backstory='Evaluate churn risk based on the complaint, account history, and revenue importance. Score 1-10.',\n",
                "    verbose=True,\n",
                "    allow_delegation=False,\n",
                "    llm=llm,\n",
                ")\n",
                "\n",
                "response_writer = Agent(\n",
                "    role='Response Writer',\n",
                "    goal='Draft a personalized response to the customer',\n",
                "    backstory='Draft a personalized response to the customer. Acknowledge the issue, provide timeline, offer compensation if appropriate.',\n",
                "    verbose=True,\n",
                "    allow_delegation=False,\n",
                "    llm=llm,\n",
                ")\n",
                "\n",
                "action_planner = Agent(\n",
                "    role='Action Planner',\n",
                "    goal='Create an internal action plan to resolve the issue and prevent recurrence',\n",
                "    backstory='Create an internal action plan to resolve the issue and prevent recurrence. Include owners and deadlines.',\n",
                "    verbose=True,\n",
                "    allow_delegation=False,\n",
                "    llm=llm,\n",
                ")\n",
                "\n",
                "print(\"Agents defined: Complaint Analyst, Churn Risk Assessor, Response Writer, Action Planner\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 3 \u2014 Define Tasks"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from crewai import Task\n",
                "\n",
                "complaint_analysis_task = Task(\n",
                "    description=\"Analyze the customer complaint. Classify severity, identify root cause, and determine if it is a systemic issue.\",\n",
                "    expected_output='Detailed complaint analysis',\n",
                "    agent=complaint_analyst,\n",
                ")\n",
                "\n",
                "churn_assessment_task = Task(\n",
                "    description='Evaluate churn risk based on the complaint, account history, and revenue importance. Score 1-10.',\n",
                "    expected_output='Detailed churn assessment',\n",
                "    agent=churn_risk_assessor,\n",
                ")\n",
                "\n",
                "customer_response_task = Task(\n",
                "    description='Draft a personalized response to the customer. Acknowledge the issue, provide timeline, offer compensation if appropriate.',\n",
                "    expected_output='Detailed customer response',\n",
                "    agent=response_writer,\n",
                ")\n",
                "\n",
                "action_plan_task = Task(\n",
                "    description='Create an internal action plan to resolve the issue and prevent recurrence. Include owners and deadlines.',\n",
                "    expected_output='Detailed action plan',\n",
                "    agent=action_planner,\n",
                ")\n",
                "\n",
                "tasks = [complaint_analysis_task, churn_assessment_task, customer_response_task, action_plan_task]\n",
                "print(f\"{len(tasks)} tasks defined\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 4 \u2014 Assemble and Run Crew"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from crewai import Crew, Process\n",
                "\n",
                "crew = Crew(\n",
                "    agents=[complaint_analyst, churn_risk_assessor, response_writer, action_planner],\n",
                "    tasks=tasks,\n",
                "    process=Process.sequential,\n",
                "    verbose=True,\n",
                ")\n",
                "\n",
                "print(\"Crew assembled! Running CrewAI Customer Success Crew...\")\n",
                "result = crew.kickoff(inputs={\"input\": \"Customer complaint: Your API has been down 3 times this month. We are paying $5000/month and considering switching to a competitor.\"})"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 5 \u2014 Review Results"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=\"*60)\n",
                "print(\"CREW OUTPUT\")\n",
                "print(\"=\"*60)\n",
                "print(result)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## What You Learned\n",
                "- **Multi-agent orchestration** with CrewAI\n",
                "- **Sequential task delegation** with specialized agents\n",
                "- **Local LLM execution** \u2014 no API keys needed\n",
                "- **Agent roles:** Complaint Analyst, Churn Risk Assessor, Response Writer, Action Planner"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

path48 = r'E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\CrewAI_Multi-Agent_Systems\48_CrewAI_Customer_Success_Crew\notebook.ipynb'
with open(path48, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb48, f, indent=1, ensure_ascii=False)
    f.write('\n')
print(f'Rebuilt notebook 48: {len(nb48["cells"])} cells')

# Verify
with open(path48, 'r', encoding='utf-8') as f:
    check = json.load(f)
print(f'Verified: {len(check["cells"])} cells')

# ── Notebook 100 ──
path100 = r'E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\Coding_and_Developer_Agents\100_Local_AI_Ops_Mini_Platform\notebook.ipynb'

# Read the current content (may be in wrong encoding)
try:
    with open(path100, 'r', encoding='utf-8') as f:
        content = f.read()
except:
    with open(path100, 'r', encoding='utf-8-sig') as f:
        content = f.read()

# Fix: ensure \r\n -> \n in the raw content before parsing
content = content.replace('\r\n', '\n').replace('\r', '\n')
nb100 = json.loads(content)
print(f'Read notebook 100: {len(nb100.get("cells", []))} cells')

# Re-save with proper encoding
with open(path100, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb100, f, indent=1, ensure_ascii=False)
    f.write('\n')
print(f'Re-saved notebook 100')

# Verify
with open(path100, 'r', encoding='utf-8') as f:
    check100 = json.load(f)
print(f'Verified notebook 100: {len(check100["cells"])} cells')

