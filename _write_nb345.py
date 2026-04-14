"""Temporary script to write Projects 03, 04, 05 notebooks. Delete after use."""
import json

def write_notebook(nb_path, cells_data):
    cells = []
    for cell_type, src in cells_data:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": cell_type, "metadata": {}, "source": source}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py",
                              "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python",
                              "pygments_lexer": "ipython3", "version": "3.10.0"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Written {len(cells)} cells to {nb_path}")


# ============================================================================
# PROJECT 3 — Local Meeting Notes Summarizer
# ============================================================================
base = r"E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\Beginner_Local_LLM_Apps"

p3 = [
("markdown", """# Project 3 — Local Meeting Notes Summarizer

## Summarize Transcripts into Actions, Decisions, and Blockers

**Goal:** Take a raw meeting transcript and produce a structured summary with
action items, decisions made, blockers identified, and key discussion points —
all using a local Ollama model.

**Stack:** Ollama · LangChain · Jupyter

### What You'll Learn

1. Design **structured summarization prompts** that extract specific sections
2. Compare **plain summary** vs **structured extraction**
3. Use **Pydantic models** for validated structured output
4. Handle **long transcripts** with chunked summarization
5. Analyze **prompt variant quality**

### Prerequisites

- Ollama running with `qwen3:8b` pulled
- Python 3.9+"""),

("code", """# Install dependencies (uncomment and run once)
# !pip install -q langchain langchain-ollama langchain-core"""),

("markdown", """## Step 1 — Verify Ollama"""),

("code", """import requests
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    r.raise_for_status()
    print(f"Ollama is running — {len(r.json().get('models', []))} model(s)")
except Exception as e:
    print(f"Cannot reach Ollama: {e}\\n  Run: ollama pull qwen3:8b")"""),

("markdown", """## Step 2 — Configure LLM"""),

("code", """from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b", temperature=0)
resp = llm.invoke("Say 'ready' in one word.")
print(f"LLM ready: {resp.content[:80]}")"""),

("markdown", """## Step 3 — Create a Sample Meeting Transcript

We create a realistic (but synthetic) meeting transcript with multiple speakers,
decisions, action items, and blockers embedded naturally in the conversation."""),

("code", """transcript = \"\"\"
Meeting: Q2 Platform Planning
Date: 2025-03-15
Attendees: Sarah (PM), Mike (Engineering Lead), Lisa (Design), Tom (QA)

Sarah: Let's start with the API redesign status. Mike, where are we?

Mike: We've completed the schema migration for the user service. The auth
endpoints are 90% done. We're blocked on the rate limiting implementation
because we need the infrastructure team to provision the Redis cluster.

Sarah: That's a blocker. Tom, can you escalate that to infra?

Tom: I'll file a priority ticket today and follow up by Wednesday.

Lisa: On the design side, the new dashboard mockups are ready for review.
I've shared them in Figma. The mobile responsive version needs another
iteration based on the feedback from the usability study.

Sarah: Great. Let's schedule a design review for Thursday. Mike, can your
team start the frontend implementation next week?

Mike: Yes, but we need the final API contracts first. I propose we freeze
the API spec by Friday so frontend can start Monday.

Sarah: Agreed. Decision: API spec freeze by Friday March 21st.

Tom: For QA, I've set up the automated regression suite for the new endpoints.
We found 3 critical bugs in the batch processing module. Two are fixed, one
needs Mike's team to review.

Mike: I'll assign someone to look at that critical bug tomorrow.

Sarah: Any other blockers?

Lisa: We need the brand team to approve the new color palette. I've been
waiting two weeks. This blocks the final design handoff.

Sarah: I'll escalate that today. Let's wrap up. Next meeting same time next week.
\"\"\"

print(f"Transcript: {len(transcript)} characters, ~{len(transcript.split())} words")"""),

("markdown", """## Step 4 — Plain Summary

First, let's see what a simple "summarize this" prompt produces.
This establishes our baseline before trying structured extraction."""),

("code", """from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

plain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a meeting assistant. Summarize the following meeting transcript concisely."),
    ("human", "{transcript}"),
])

plain_chain = plain_prompt | llm | StrOutputParser()
plain_summary = plain_chain.invoke({"transcript": transcript})

print("=== Plain Summary ===")
print(plain_summary)"""),

("markdown", """## Step 5 — Structured Summary Extraction

Now let's use a more specific prompt that extracts **distinct sections**:
actions, decisions, blockers, and key points. This is much more useful
for follow-up than a plain paragraph summary."""),

("code", """structured_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"You are a meeting assistant. Analyze the transcript and extract:

1. **DECISIONS** — What was agreed upon
2. **ACTION ITEMS** — Who will do what, by when
3. **BLOCKERS** — What is preventing progress
4. **KEY DISCUSSION POINTS** — Main topics discussed

Format each section clearly with bullet points. Be specific about owners and deadlines.\"\"\"),
    ("human", "{transcript}"),
])

structured_chain = structured_prompt | llm | StrOutputParser()
structured_summary = structured_chain.invoke({"transcript": transcript})

print("=== Structured Summary ===")
print(structured_summary)"""),

("markdown", """## Step 6 — JSON Extraction with Validation

For downstream automation (ticketing, email drafts), we want **machine-readable output**.
Let's ask the model to produce JSON and validate the structure."""),

("code", """import json

json_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Extract meeting information as valid JSON with this exact structure:
{{
  "meeting_title": "...",
  "date": "...",
  "decisions": ["decision 1", "decision 2"],
  "action_items": [
    {{"owner": "Name", "task": "description", "deadline": "date or null"}}
  ],
  "blockers": [
    {{"description": "...", "owner": "who is affected", "escalation": "who will escalate"}}
  ],
  "key_topics": ["topic 1", "topic 2"]
}}

Return ONLY valid JSON, no markdown formatting.\"\"\"),
    ("human", "{transcript}"),
])

json_chain = json_prompt | llm | StrOutputParser()
raw_output = json_chain.invoke({"transcript": transcript})

# Try to parse the JSON
print("=== Raw LLM Output ===")
print(raw_output[:500])

# Clean and parse
try:
    # Strip potential markdown code fences
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("\\n", 1)[0]
    # Try to find JSON in output (handle thinking tags)
    if "{" in cleaned:
        json_start = cleaned.index("{")
        json_end = cleaned.rindex("}") + 1
        cleaned = cleaned[json_start:json_end]
    parsed = json.loads(cleaned)
    print("\\n=== Parsed JSON ===")
    print(json.dumps(parsed, indent=2))
    print(f"\\nExtracted: {len(parsed.get('action_items', []))} action items, "
          f"{len(parsed.get('blockers', []))} blockers, "
          f"{len(parsed.get('decisions', []))} decisions")
except json.JSONDecodeError as e:
    print(f"\\nJSON parsing failed: {e}")
    print("This is a common challenge with local models — see Limitations below.")"""),

("markdown", """## Step 7 — Compare Prompt Variants

Different prompt styles produce different quality summaries.
Let's compare a few approaches to see which works best for our model."""),

("code", """prompt_variants = {
    "bullet_style": "Summarize this meeting as bullet points grouped by topic.",
    "executive": "Write a 3-sentence executive summary of this meeting for a VP.",
    "email_followup": "Draft a follow-up email summarizing this meeting for all attendees. Include action items with owners.",
}

for style, instruction in prompt_variants.items():
    variant_prompt = ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "{transcript}"),
    ])
    result = (variant_prompt | llm | StrOutputParser()).invoke({"transcript": transcript})
    print(f"\\n{'='*60}")
    print(f"Style: {style}")
    print(f"{'='*60}")
    print(result[:400])
    print("..." if len(result) > 400 else "")"""),

("markdown", """## Step 8 — Failure Cases

Let's test edge cases: very short input, gibberish, and non-meeting content."""),

("code", """edge_cases = {
    "very_short": "Bob: Let's meet tomorrow. Alice: OK.",
    "no_actions": "Team discussed the weather and had coffee. No decisions were made.",
    "non_meeting": "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
}

for case_name, text in edge_cases.items():
    result = (structured_prompt | llm | StrOutputParser()).invoke({"transcript": text})
    print(f"\\n--- {case_name} ---")
    print(f"Input: {text[:60]}...")
    print(f"Output: {result[:200]}...")"""),

("markdown", """## Limitations & Tradeoffs

| Aspect | What happens | How to improve |
|--------|-------------|----------------|
| **JSON reliability** | Local models may produce malformed JSON | Add retry with error feedback |
| **Long transcripts** | May exceed context window | Use map-reduce or chunking |
| **Speaker diarization** | Assumes speakers are labeled in text | Add pre-processing for unlabeled audio |
| **Hallucinated actions** | Model may invent action items | Cross-check against transcript |
| **Thinking tags** | qwen3 may include reasoning | Post-process to strip tags |

### What this project does NOT cover
- Audio transcription (see Project 88)
- Speaker identification from audio
- Integration with task management tools"""),

("markdown", """## What You Learned

1. **Plain vs structured summarization** — structured prompts extract more useful information
2. **JSON extraction** — asking for structured output enables downstream automation
3. **Prompt variants** — different styles serve different audiences (executive, team, email)
4. **Edge case handling** — testing unusual inputs reveals model limitations
5. **Output validation** — parsing and validating LLM output is essential for reliability

## Exercises

1. **Add your own transcript** — paste a real meeting transcript and compare outputs
2. **Add a priority field** — extend the JSON schema to include priority for each action item
3. **Handle long meetings** — split a transcript into 5-minute chunks and merge summaries
4. **Build a manager update** — create a prompt that generates a weekly status email from multiple meeting summaries

---

*Next project: **04 — Local Resume Rewriter** (iterative rewriting with Ollama)*"""),
]

print("Writing Project 3...")
write_notebook(f"{base}\\03_Local_Meeting_Notes_Summarizer\\notebook.ipynb", p3)


# ============================================================================
# PROJECT 4 — Local Resume Rewriter
# ============================================================================
p4 = [
("markdown", """# Project 4 — Local Resume Rewriter

## Improve Resume Bullets and Tailor Wording with a Local LLM

**Goal:** Take rough resume bullet points, critique them, rewrite them using
the STAR method, and tailor them to a target job description — all locally.

**Stack:** Ollama · LangChain · Jupyter

### What You'll Learn

1. **Critique** resume bullets for impact, clarity, and specificity
2. **Rewrite** bullets using the STAR (Situation, Task, Action, Result) method
3. **Tailor** bullets to match a target job description's keywords
4. **Score** resume-JD alignment quantitatively
5. Compare **different prompting strategies** for rewriting

### Prerequisites

- Ollama running with `qwen3:8b` pulled"""),

("code", """# Install dependencies (uncomment and run once)
# !pip install -q langchain langchain-ollama langchain-core"""),

("markdown", """## Step 1 — Verify Ollama and Configure LLM"""),

("code", """import requests
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    print(f"Ollama running — {len(r.json().get('models', []))} models")
except: print("Start Ollama first!")

llm = ChatOllama(model="qwen3:8b", temperature=0.3)
print("LLM configured")"""),

("markdown", """## Step 2 — Sample Resume Bullets

These are typical **weak** resume bullets that many people write.
They lack specificity, measurable impact, and action verbs."""),

("code", """weak_bullets = [
    "Responsible for managing a team and delivering projects on time.",
    "Worked on data analysis and made reports for management.",
    "Helped improve the company website and increased traffic.",
    "Used Python and SQL for various data tasks.",
    "Participated in code reviews and helped maintain code quality.",
]

print("=== Original Resume Bullets ===")
for i, b in enumerate(weak_bullets, 1):
    print(f"  {i}. {b}")"""),

("markdown", """## Step 3 — Critique the Bullets

Before rewriting, let's have the LLM identify what's weak about
each bullet. This teaches the model (and us) what to fix."""),

("code", """critique_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"You are a resume coach. For each bullet point, provide a brief critique:
- Is it specific or vague?
- Does it show measurable impact?
- Does it use strong action verbs?
- What is missing?

Be concise — 2-3 sentences per bullet.\"\"\"),
    ("human", "Critique these resume bullets:\\n\\n{bullets}"),
])

bullets_text = "\\n".join(f"{i}. {b}" for i, b in enumerate(weak_bullets, 1))
critique = (critique_prompt | llm | StrOutputParser()).invoke({"bullets": bullets_text})

print("=== Critique ===")
print(critique)"""),

("markdown", """## Step 4 — Rewrite Using STAR Method

The **STAR method** (Situation, Task, Action, Result) creates impactful
bullet points. We ask the LLM to transform each weak bullet into a
strong STAR-formatted version."""),

("code", """star_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"You are a resume writing expert. Rewrite each bullet point using the STAR method:
- **Situation**: Brief context
- **Action**: What you specifically did (strong verb)
- **Result**: Measurable outcome with numbers when possible

Make reasonable assumptions about metrics. Keep each bullet to 1-2 lines.
Return ONLY the rewritten bullets, numbered.\"\"\"),
    ("human", "Rewrite these resume bullets:\\n\\n{bullets}"),
])

rewritten = (star_prompt | llm | StrOutputParser()).invoke({"bullets": bullets_text})

print("=== STAR-Rewritten Bullets ===")
print(rewritten)"""),

("markdown", """## Step 5 — Tailor to a Job Description

Now let's align the rewritten bullets with a specific job posting.
The model should emphasize **keywords and skills** from the JD."""),

("code", """sample_jd = \"\"\"
Senior Data Engineer — TechCorp

Requirements:
- 5+ years experience with Python, SQL, and cloud data platforms
- Experience building and maintaining ETL/ELT pipelines
- Strong understanding of data modeling and warehousing
- Experience with Apache Spark, Airflow, or similar tools
- Track record of improving data quality and pipeline reliability
- Leadership experience mentoring junior engineers
- Excellent communication skills for cross-functional collaboration
\"\"\"

tailor_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"You are a resume tailoring expert. Given a job description and resume bullets,
rewrite the bullets to better match the job requirements. Emphasize relevant
keywords, skills, and experiences from the JD. Keep the content truthful
but strategically aligned.\"\"\"),
    ("human", "Job Description:\\n{jd}\\n\\nResume Bullets to tailor:\\n{bullets}"),
])

tailored = (tailor_prompt | llm | StrOutputParser()).invoke({
    "jd": sample_jd,
    "bullets": rewritten,
})

print("=== Tailored for Data Engineer Role ===")
print(tailored)"""),

("markdown", """## Step 6 — Score Resume-JD Fit

Let's ask the model to evaluate how well the tailored resume
matches the job description, with a numerical score."""),

("code", """score_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Rate how well these resume bullets match the job description.

Score each criterion from 1-10:
1. Keyword alignment (do the bullets mention JD skills?)
2. Experience level match
3. Impact/metrics quality
4. Overall fit

Provide scores and a brief explanation for each.\"\"\"),
    ("human", "Job Description:\\n{jd}\\n\\nResume Bullets:\\n{bullets}"),
])

# Score original vs tailored
print("=== Scoring Original Bullets ===")
orig_score = (score_prompt | llm | StrOutputParser()).invoke({"jd": sample_jd, "bullets": bullets_text})
print(orig_score)

print("\\n=== Scoring Tailored Bullets ===")
tailored_score = (score_prompt | llm | StrOutputParser()).invoke({"jd": sample_jd, "bullets": tailored})
print(tailored_score)"""),

("markdown", """## Step 7 — Side-by-Side Comparison"""),

("code", """print("=== Before vs After ===\\n")
print("ORIGINAL:")
for i, b in enumerate(weak_bullets, 1):
    print(f"  {i}. {b}")
print("\\nTAILORED:")
print(tailored)"""),

("markdown", """## Limitations & Tradeoffs

| Aspect | What happens | How to improve |
|--------|-------------|----------------|
| **Fabricated metrics** | Model may invent numbers | Ask user to provide real metrics |
| **Keyword stuffing** | Over-optimizing for JD keywords | Balance relevance with authenticity |
| **Generic rewrites** | Without context, rewrites may be vague | Provide role-specific context |
| **Thinking tags** | qwen3 may include reasoning in output | Post-process or use non-thinking model |

### What this project does NOT cover
- ATS (Applicant Tracking System) parsing simulation
- Multi-page resume formatting
- Portfolio/project section optimization"""),

("markdown", """## What You Learned

1. **Resume critique** — identifying weaknesses before rewriting
2. **STAR method** — structured approach to impactful bullet points
3. **JD tailoring** — aligning resume language with job requirements
4. **Fit scoring** — quantitative evaluation of resume-JD alignment
5. **Before/after comparison** — measuring improvement visually

## Exercises

1. **Use your own resume** — paste your actual bullets and run the pipeline
2. **Try multiple JDs** — tailor the same resume for 3 different roles
3. **Add a skills section** — extend the pipeline to also rewrite a skills summary
4. **Iterative refinement** — feed the critique back and rewrite again

---

*Next project: **05 — Local Cover Letter Generator** (multi-input context synthesis)*"""),
]

print("Writing Project 4...")
write_notebook(f"{base}\\04_Local_Resume_Rewriter\\notebook.ipynb", p4)


# ============================================================================
# PROJECT 5 — Local Cover Letter Generator
# ============================================================================
p5 = [
("markdown", """# Project 5 — Local Cover Letter Generator

## Generate Tailored Cover Letters from JD + Resume

**Goal:** Combine a job description and resume into a polished, tailored
cover letter — learning multi-input prompting and context synthesis.

**Stack:** Ollama · LangChain · Jupyter

### What You'll Learn

1. **Multi-input prompting** — blending JD and resume into one prompt
2. **Context extraction** — pulling key requirements and candidate strengths
3. **Tone control** — adjusting formality, enthusiasm, and length
4. **Variant generation** — creating multiple letter styles for comparison
5. **Quality critique** — self-evaluation of generated letters

### Prerequisites

- Ollama running with `qwen3:8b` pulled"""),

("code", """# Install dependencies (uncomment and run once)
# !pip install -q langchain langchain-ollama langchain-core"""),

("markdown", """## Step 1 — Verify Ollama and Configure LLM"""),

("code", """import requests
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    print(f"Ollama running — {len(r.json().get('models', []))} models")
except: print("Start Ollama first!")

llm = ChatOllama(model="qwen3:8b", temperature=0.4)
print("LLM configured")"""),

("markdown", """## Step 2 — Sample Job Description and Resume

We provide both inputs that will be synthesized into a cover letter."""),

("code", """job_description = \"\"\"
Senior Machine Learning Engineer — DataFlow Inc.

About the Role:
We're looking for an experienced ML engineer to lead our recommendation
system team. You'll design and deploy models that serve millions of users.

Requirements:
- 4+ years in ML engineering or applied data science
- Strong Python skills and experience with PyTorch or TensorFlow
- Experience deploying ML models to production at scale
- Knowledge of recommendation systems, NLP, or computer vision
- Experience with cloud platforms (AWS, GCP, or Azure)
- Excellent communication skills for cross-team collaboration
- Experience mentoring junior engineers is a plus

What We Offer:
- Competitive salary and equity
- Flexible remote work policy
- Learning and conference budget
\"\"\"

resume_text = \"\"\"
Jane Chen — ML Engineer
Email: jane.chen@email.com | GitHub: github.com/janechen

EXPERIENCE:
Machine Learning Engineer, TechStartup Inc. (2021-Present)
- Built and deployed a product recommendation engine serving 2M+ daily users
- Reduced model inference latency by 40% through model optimization and caching
- Led migration from TensorFlow to PyTorch, improving training speed by 2x
- Mentored 2 junior engineers on ML best practices and code review

Data Scientist, Analytics Corp (2019-2021)
- Developed NLP pipeline for customer feedback classification (92% accuracy)
- Built A/B testing framework for model comparison, adopted by 3 teams
- Deployed models on AWS SageMaker with CI/CD pipelines

SKILLS: Python, PyTorch, TensorFlow, scikit-learn, SQL, AWS, Docker, Git
EDUCATION: M.S. Computer Science, State University (2019)
\"\"\"

print(f"JD: {len(job_description.split())} words")
print(f"Resume: {len(resume_text.split())} words")"""),

("markdown", """## Step 3 — Extract Key Requirements and Strengths

Before generating the letter, let's explicitly extract what the JD needs
and what the candidate offers. This intermediate step improves letter quality."""),

("code", """extract_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Analyze the job description and resume. Extract:

1. **Top 5 JD Requirements** — the most important skills/experiences needed
2. **Candidate's Matching Strengths** — where the resume directly matches
3. **Gaps** — JD requirements not clearly addressed in the resume
4. **Unique Value** — candidate strengths beyond what's required

Be specific and concise.\"\"\"),
    ("human", "Job Description:\\n{jd}\\n\\nResume:\\n{resume}"),
])

analysis = (extract_prompt | llm | StrOutputParser()).invoke({
    "jd": job_description,
    "resume": resume_text,
})

print("=== JD-Resume Analysis ===")
print(analysis)"""),

("markdown", """## Step 4 — Generate the Cover Letter

Now we generate a complete cover letter that synthesizes both inputs,
highlighting the candidate's relevant experience and enthusiasm."""),

("code", """cover_letter_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Write a professional cover letter for this job application.

Guidelines:
- Open with enthusiasm for the specific role and company
- Highlight 2-3 most relevant experiences that match JD requirements
- Use specific metrics and achievements from the resume
- Show knowledge of what the company does
- Close with a clear call to action
- Keep it to 3-4 paragraphs, under 400 words
- Tone: confident but not arrogant, professional but personable\"\"\"),
    ("human", "Job Description:\\n{jd}\\n\\nResume:\\n{resume}"),
])

cover_letter = (cover_letter_prompt | llm | StrOutputParser()).invoke({
    "jd": job_description,
    "resume": resume_text,
})

print("=== Generated Cover Letter ===")
print(cover_letter)"""),

("markdown", """## Step 5 — Generate Tone Variants

Different companies prefer different communication styles.
Let's generate variants to see how tone affects the letter."""),

("code", """tone_variants = {
    "formal": "Write in a highly formal, traditional business letter style.",
    "conversational": "Write in a warm, conversational but professional tone.",
    "technical": "Write in a technical style, emphasizing engineering depth and specific technologies.",
}

for tone_name, instruction in tone_variants.items():
    variant_prompt = ChatPromptTemplate.from_messages([
        ("system", f"Write a cover letter for this job application. {instruction} Keep it under 300 words."),
        ("human", "Job Description:\\n{jd}\\n\\nResume:\\n{resume}"),
    ])
    result = (variant_prompt | llm | StrOutputParser()).invoke({
        "jd": job_description,
        "resume": resume_text,
    })
    print(f"\\n{'='*60}")
    print(f"Tone: {tone_name}")
    print(f"{'='*60}")
    print(result[:500])
    print("..." if len(result) > 500 else "")"""),

("markdown", """## Step 6 — Self-Critique the Letter

Let's use the LLM as a **reviewer** to evaluate its own output.
This teaches the pattern of generation → evaluation → revision."""),

("code", """critique_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"You are a hiring manager reviewing this cover letter for the given job.

Rate each aspect from 1-10 and explain briefly:
1. **Relevance** — Does it address the job requirements?
2. **Specificity** — Does it use concrete examples and metrics?
3. **Tone** — Is it appropriate for the role and company?
4. **Persuasiveness** — Would you want to interview this candidate?
5. **Length** — Is it the right length?

Then give one specific suggestion for improvement.\"\"\"),
    ("human", "Job Description:\\n{jd}\\n\\nCover Letter:\\n{letter}"),
])

critique = (critique_prompt | llm | StrOutputParser()).invoke({
    "jd": job_description,
    "letter": cover_letter,
})

print("=== Cover Letter Critique ===")
print(critique)"""),

("markdown", """## Limitations & Tradeoffs

| Aspect | What happens | How to improve |
|--------|-------------|----------------|
| **Generic phrasing** | Model may produce cliché language | Provide more specific company context |
| **Company knowledge** | Model doesn't know real company details | Add company research as extra input |
| **Fabrication** | May embellish beyond what's in the resume | Cross-check against actual resume |
| **Format variation** | Different runs produce different structures | Use a template with fill-in-the-blank |
| **Length control** | May be too long or too short | Specify word count in prompt |

### What this project does NOT cover
- PDF formatting and export
- ATS keyword optimization
- Multiple application tracking
- Company-specific research integration"""),

("markdown", """## What You Learned

1. **Multi-input prompting** — blending two text sources into one coherent output
2. **Context extraction** — analyzing JD and resume before generation
3. **Tone variants** — controlling style through prompt instructions
4. **Self-critique** — using the model to evaluate its own output
5. **Quality assessment** — structured evaluation with scoring rubrics

## Exercises

1. **Use your own resume + a real JD** — generate a letter for a role you're interested in
2. **Add company research** — include a paragraph about the company and see if the letter improves
3. **Iterative refinement** — take the critique feedback and regenerate an improved version
4. **A/B test** — have a friend rate two letter variants without knowing which is which

---

*Next project: **06 — Local Email Reply Assistant** (intent classification + reply drafting)*"""),
]

print("Writing Project 5...")
write_notebook(f"{base}\\05_Local_Cover_Letter_Generator\\notebook.ipynb", p5)

print("\nAll 3 notebooks written successfully!")

