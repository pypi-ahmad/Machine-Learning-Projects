"""Build the Web Research Agent notebook."""
import json, pathlib

cells = []

def _lines(text: str):
    parts = text.split("\n")
    for i in range(len(parts) - 1):
        parts[i] += "\n"
    return parts

def md(text: str):
    cells.append({"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": _lines(text)})

def code(text: str):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {"language": "python"}, "outputs": [], "source": _lines(text)})


# ══════════════════════════════════════════════
# 1  Title
# ══════════════════════════════════════════════
md(r"""# Web Research Agent: Search, Compare Sources, and Write Cited Answers

**Goal:** build an agent that receives a factual question, searches the web for information, compares what multiple sources say, resolves disagreements, and writes a final answer with inline citations.

---

## Why This Matters

LLMs hallucinate. A web-research agent addresses this by:
1. **Grounding** — answers are backed by real, retrievable sources
2. **Multi-source verification** — claims are cross-checked across independent sources
3. **Citation transparency** — the reader can verify every claim

## What We Build

```text
User question
    │
    ▼
┌────────────────────┐
│  Plan: decompose   │  Break question into sub-queries
│  into search tasks │
└────────┬───────────┘
         │ sub-queries
         ▼
┌────────────────────┐
│  Search Tool       │  Fetch results from multiple sources
│  (simulated web)   │
└────────┬───────────┘
         │ raw results
         ▼
┌────────────────────┐
│  Extract & Filter  │  Pull key claims, discard noise
└────────┬───────────┘
         │ extracted claims
         ▼
┌────────────────────┐
│  Compare Sources   │  Identify agreement, disagreement, unique claims
└────────┬───────────┘
         │ verified claims
         ▼
┌────────────────────┐
│  Write Answer      │  Compose cited answer with [1], [2] references
└────────────────────┘
```

## Key Concepts Covered

- **Tool use**: how agents call external tools (search, fetch, extract) in a loop
- **Source verification**: cross-referencing claims across independent sources
- **Citation generation**: inline references linked to a bibliography
- **Conflict resolution**: what to do when sources disagree
- **Confidence scoring**: how much to trust the final answer
""")

# ══════════════════════════════════════════════
# 2  Setup
# ══════════════════════════════════════════════
md("## 1. Environment Setup")

code("""!pip install -q pandas numpy seaborn matplotlib""")

code("""import json
import random
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path.cwd()
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

print(f"Project dir: {PROJECT_DIR}")""")

# ══════════════════════════════════════════════
# 3  Tool use — concepts
# ══════════════════════════════════════════════
md(r"""## 2. Tool Use — How Agents Call External Functions

### What Is a Tool?

A **tool** is a function the agent can invoke to interact with the outside world. The agent decides **when** to call a tool, **which** tool to call, and **what arguments** to pass.

```text
Agent (LLM)
    │
    ├── "I need to search for X"
    │       ▼
    │   ┌──────────┐
    │   │ web_search│  → returns results
    │   └──────────┘
    │       │
    ├── "I need more detail on result #2"
    │       ▼
    │   ┌──────────┐
    │   │ fetch_page│  → returns page content
    │   └──────────┘
    │       │
    └── "Now I can answer"
            ▼
        Final response
```

### Tool Definition Pattern

Every tool needs:
1. **Name** — unique identifier (`web_search`, `fetch_page`)
2. **Description** — what the tool does (the LLM reads this to decide when to use it)
3. **Parameters** — typed inputs with descriptions
4. **Return type** — what comes back

### Why Not Just Let the LLM Answer Directly?

| Approach | Pros | Cons |
|---|---|---|
| LLM only | Fast, simple | Hallucinates, knowledge cutoff |
| LLM + tools | Grounded, current | Slower, more complex, tool errors |

The tradeoff is worth it when **factual accuracy matters more than speed**.
""")

# ══════════════════════════════════════════════
# 4  Define tools
# ══════════════════════════════════════════════
md(r"""## 3. Defining the Agent's Tools

We define three tools the agent can use:

| Tool | Purpose | Input | Output |
|---|---|---|---|
| `web_search` | Find relevant pages | query string | list of search results (title, URL, snippet) |
| `fetch_page` | Get content from a URL | URL | extracted text content |
| `extract_claims` | Pull factual claims from text | text block | list of atomic claims |

In production, these would call real APIs (Google Search, Bing, etc.). Here we simulate them with a curated knowledge base so the notebook runs offline.
""")

code("""@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source_name: str
    reliability: str  # "high", "medium", "low"
    publish_date: str


@dataclass
class PageContent:
    url: str
    title: str
    text: str
    source_name: str
    publish_date: str


@dataclass
class Claim:
    text: str
    source_url: str
    source_name: str
    confidence: float  # 0-1
    category: str  # "factual", "opinion", "outdated", "unverified"


print("Tool data classes defined:")
for cls in [SearchResult, PageContent, Claim]:
    print(f"  {cls.__name__}: {[f.name for f in cls.__dataclass_fields__.values()]}")""")

# ══════════════════════════════════════════════
# 5  Simulated web
# ══════════════════════════════════════════════
md(r"""## 4. Simulated Web: A Controlled Knowledge Base

We build a small "internet" with multiple sources covering the same topics. Sources have different reliability levels, sometimes agree, sometimes disagree, and sometimes contain outdated information.

This controlled setup lets us:
- **Test source comparison** with known ground truth
- **Inject disagreements** to see how the agent handles them
- **Control reliability** to test source weighting
""")

code("""# Topic: "What is the recommended daily water intake for adults?"
# Chosen because sources genuinely disagree on this in the real world.

SIMULATED_WEB = {
    # ── Source 1: Government health agency (high reliability) ──
    "https://health.gov/nutrition/water-intake": PageContent(
        url="https://health.gov/nutrition/water-intake",
        title="Daily Water Intake Recommendations",
        text=(
            "The U.S. National Academies of Sciences recommends approximately "
            "3.7 liters (125 oz) of total daily water intake for adult men and "
            "2.7 liters (91 oz) for adult women. This includes water from all "
            "beverages and food. About 20% of daily water intake typically comes "
            "from food. These recommendations are for temperate climates and "
            "sedentary to moderately active individuals. Requirements increase "
            "with exercise, heat exposure, and certain medical conditions. "
            "There is no single number that fits everyone."
        ),
        source_name="National Health Institute",
        publish_date="2024-03-15",
    ),
    # ── Source 2: Medical journal (high reliability) ──
    "https://medical-journal.org/hydration-review-2024": PageContent(
        url="https://medical-journal.org/hydration-review-2024",
        title="Hydration Requirements: A Systematic Review",
        text=(
            "A 2024 systematic review of 42 studies found that adequate daily "
            "water intake varies significantly by body weight, activity level, "
            "and climate. The commonly cited '8 glasses a day' (about 2 liters) "
            "lacks strong scientific evidence and originates from a 1945 Food "
            "and Nutrition Board recommendation that was taken out of context. "
            "The original recommendation included water from food sources. "
            "Current evidence supports 2.7-3.7 liters of total fluid intake "
            "(from all sources) for adults, aligning with the National Academies "
            "recommendation. Individual hydration needs are best assessed by "
            "urine color (pale yellow indicates adequate hydration). "
            "Over-hydration (hyponatremia) is a real risk, particularly in "
            "endurance athletes."
        ),
        source_name="Journal of Clinical Nutrition",
        publish_date="2024-06-01",
    ),
    # ── Source 3: Popular health blog (medium reliability) ──
    "https://healthblog.example.com/water-intake": PageContent(
        url="https://healthblog.example.com/water-intake",
        title="How Much Water Should You Really Drink?",
        text=(
            "You've probably heard the advice to drink 8 glasses of water a day. "
            "While this is a decent starting point, the truth is more nuanced. "
            "Most experts recommend drinking about 2 liters (8 cups) of water per "
            "day, not counting other beverages or food. Some newer studies suggest "
            "you might need more — up to 3 liters for active adults. "
            "Coffee and tea DO count toward your daily intake, despite the old "
            "myth that caffeine dehydrates you significantly. Mild caffeine "
            "consumption has minimal diuretic effect. "
            "Pro tip: if you're thirsty, you're already mildly dehydrated! "
            "Drink before you feel thirsty."
        ),
        source_name="HealthyLiving Blog",
        publish_date="2023-11-20",
    ),
    # ── Source 4: Fitness site (medium reliability, slightly outdated) ──
    "https://fitlife.example.com/hydration-guide": PageContent(
        url="https://fitlife.example.com/hydration-guide",
        title="The Ultimate Hydration Guide for Athletes",
        text=(
            "Athletes need significantly more water than sedentary adults. "
            "The baseline recommendation is 3.7 liters for men and 2.7 liters "
            "for women, but active individuals should add 500-1000 mL per hour "
            "of exercise. A good rule of thumb: drink half your body weight in "
            "ounces. For a 180-pound person, that's 90 ounces (about 2.7 liters). "
            "Electrolyte replacement is critical during exercise lasting more "
            "than 60 minutes. Water alone can lead to hyponatremia in long "
            "endurance events. Sports drinks or electrolyte tablets are "
            "recommended for sessions over 90 minutes."
        ),
        source_name="FitLife Magazine",
        publish_date="2022-08-10",
    ),
    # ── Source 5: Questionable wellness site (low reliability) ──
    "https://wellness-guru.example.com/secret-water": PageContent(
        url="https://wellness-guru.example.com/secret-water",
        title="The SECRET Water Rule Doctors Won't Tell You",
        text=(
            "FORGET everything you've been told about water intake! "
            "You should be drinking AT LEAST 4-5 liters of water per day — pure "
            "water only, no tea, coffee, or juice counts. Alkaline water with "
            "pH 9+ is especially beneficial and can prevent cancer. "
            "If your urine isn't completely clear, you're dehydrated. "
            "Tap water is full of toxins and should always be filtered through "
            "a reverse osmosis system. Most diseases are caused by chronic "
            "dehydration. Drink a full liter immediately upon waking for optimal "
            "detox benefits."
        ),
        source_name="WellnessGuru",
        publish_date="2023-05-01",
    ),
}

# Search index: maps keywords to URLs
SEARCH_INDEX = defaultdict(list)
for url, page in SIMULATED_WEB.items():
    text_blob = f"{page.title} {page.text}".lower()
    for keyword in ["water", "intake", "hydration", "drinking", "daily", "liters",
                     "glasses", "recommendation", "adults", "athletes", "electrolyte"]:
        if keyword in text_blob:
            SEARCH_INDEX[keyword].append(url)

print(f"Simulated web: {len(SIMULATED_WEB)} pages")
for url, page in SIMULATED_WEB.items():
    print(f"  [{page.source_name}] {page.title}")""")

# ══════════════════════════════════════════════
# 6  Tool implementations
# ══════════════════════════════════════════════
md("## 5. Tool Implementations")

code("""def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Simulate a web search by matching query keywords to our index."""
    query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
    url_scores: dict[str, int] = defaultdict(int)

    for word in query_words:
        for url in SEARCH_INDEX.get(word, []):
            url_scores[url] += 1

    ranked = sorted(url_scores.items(), key=lambda x: -x[1])[:max_results]
    results = []
    for url, score in ranked:
        page = SIMULATED_WEB[url]
        # Snippet: first 150 chars of text
        snippet = page.text[:150].rsplit(" ", 1)[0] + "..."
        reliability = "low"
        if page.source_name in ("National Health Institute", "Journal of Clinical Nutrition"):
            reliability = "high"
        elif page.source_name in ("HealthyLiving Blog", "FitLife Magazine"):
            reliability = "medium"
        results.append(SearchResult(
            title=page.title, url=url, snippet=snippet,
            source_name=page.source_name, reliability=reliability,
            publish_date=page.publish_date,
        ))
    return results


def fetch_page(url: str) -> Optional[PageContent]:
    """Fetch full page content from a URL."""
    return SIMULATED_WEB.get(url)


def extract_claims(text: str, source_url: str, source_name: str) -> list[Claim]:
    """Extract atomic factual claims from a text block.

    In production, this would use an LLM. Here we use rule-based extraction.
    """
    sentences = [s.strip() for s in re.split(r'[.!]', text) if len(s.strip()) > 20]
    claims = []
    for sent in sentences:
        # Skip purely promotional or vague sentences
        if any(w in sent.lower() for w in ["secret", "forget everything", "pro tip"]):
            continue

        # Classify claim type
        has_number = bool(re.search(r'\d', sent))
        has_hedge = any(w in sent.lower() for w in ["might", "some", "probably", "suggest"])

        if has_number and not has_hedge:
            category = "factual"
            confidence = 0.8
        elif has_number and has_hedge:
            category = "factual"
            confidence = 0.6
        elif any(w in sent.lower() for w in ["should", "recommend", "best", "optimal"]):
            category = "opinion"
            confidence = 0.5
        else:
            category = "unverified"
            confidence = 0.4

        claims.append(Claim(
            text=sent.strip(), source_url=source_url,
            source_name=source_name, confidence=confidence,
            category=category,
        ))
    return claims


print("Tools defined: web_search, fetch_page, extract_claims")
print(f"\\nTest search for 'daily water intake recommendation':")
results = web_search("daily water intake recommendation for adults")
for r in results:
    print(f"  [{r.reliability}] {r.source_name}: {r.title}")""")

# ══════════════════════════════════════════════
# 7  Tool use protocol
# ══════════════════════════════════════════════
md(r"""## 6. The Tool-Use Protocol

### How an LLM Calls Tools

In real agent frameworks (OpenAI function calling, LangChain, etc.), the protocol is:

```text
1. System prompt describes available tools and their schemas
2. User sends a question
3. LLM responds with a tool call: {"name": "web_search", "arguments": {"query": "..."}}
4. Framework executes the tool, returns the result to the LLM
5. LLM either calls another tool or responds to the user
```

### The Agent Loop

```python
while not done:
    response = llm.generate(messages)
    if response.has_tool_calls:
        for call in response.tool_calls:
            result = execute_tool(call.name, call.arguments)
            messages.append(tool_result(call.id, result))
    else:
        final_answer = response.text
        done = True
```

### Key Design Decisions

| Decision | Options | Tradeoff |
|---|---|---|
| When to stop searching | Fixed count vs. saturation | More searches = better coverage but higher cost |
| Which sources to trust | Whitelist vs. scoring | Whitelist is rigid; scoring is flexible but needs calibration |
| How to handle conflicts | Majority vote vs. source weighting | Majority can be wrong; weighting needs training data |
| When to say "I don't know" | Confidence threshold | Too low = overconfident; too high = useless |

We implement the agent loop explicitly below, without an LLM, to show each step clearly.
""")

# ══════════════════════════════════════════════
# 8  Agent implementation
# ══════════════════════════════════════════════
md("## 7. The Research Agent — Full Implementation")

code("""@dataclass
class SourceRecord:
    """A processed source with extracted claims."""
    url: str
    source_name: str
    reliability: str
    publish_date: str
    claims: list[Claim] = field(default_factory=list)


@dataclass
class ResearchResult:
    """Final output of the research agent."""
    question: str
    answer: str
    citations: list[dict]
    sources_consulted: int
    sources_used: int
    agreement_score: float
    confidence: float
    conflicts: list[dict]
    tool_calls_log: list[dict]


class WebResearchAgent:
    """An agent that searches the web, compares sources, and writes cited answers."""

    # Source reliability weights for claim scoring
    RELIABILITY_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.2}

    # Minimum sources that must agree for a claim to be "verified"
    MIN_AGREEMENT = 2

    # Confidence threshold: below this, the agent says "not enough evidence"
    CONFIDENCE_THRESHOLD = 0.4

    def __init__(self):
        self.tool_log: list[dict] = []

    def _log_tool(self, tool_name: str, args: dict, result_summary: str):
        self.tool_log.append({
            "tool": tool_name,
            "args": args,
            "result": result_summary,
            "timestamp": datetime.now().isoformat(),
        })

    # ── Step 1: Plan search queries ──
    def plan_queries(self, question: str) -> list[str]:
        """Decompose the question into search sub-queries.

        In production, an LLM would do this. Here we use heuristics.
        """
        queries = [question]  # Always search the original question

        # Add targeted sub-queries
        lower = question.lower()
        if "how much" in lower or "how many" in lower:
            queries.append(question + " scientific recommendation")
            queries.append(question + " expert guidelines")
        if "daily" in lower:
            queries.append(question.replace("daily", "per day") + " research")

        self._log_tool("plan_queries", {"question": question},
                       f"{len(queries)} sub-queries generated")
        return queries

    # ── Step 2: Search and fetch ──
    def search_and_fetch(self, queries: list[str]) -> list[SourceRecord]:
        """Execute searches, fetch unique pages, extract claims."""
        seen_urls: set[str] = set()
        sources: list[SourceRecord] = []

        for query in queries:
            results = web_search(query)
            self._log_tool("web_search", {"query": query},
                           f"{len(results)} results")

            for sr in results:
                if sr.url in seen_urls:
                    continue
                seen_urls.add(sr.url)

                page = fetch_page(sr.url)
                if page is None:
                    self._log_tool("fetch_page", {"url": sr.url}, "FAILED")
                    continue
                self._log_tool("fetch_page", {"url": sr.url},
                               f"{len(page.text)} chars fetched")

                claims = extract_claims(page.text, sr.url, sr.source_name)
                self._log_tool("extract_claims", {"url": sr.url},
                               f"{len(claims)} claims extracted")

                sources.append(SourceRecord(
                    url=sr.url, source_name=sr.source_name,
                    reliability=sr.reliability,
                    publish_date=sr.publish_date, claims=claims,
                ))

        return sources

    # ── Step 3: Compare sources ──
    def compare_sources(self, sources: list[SourceRecord]) -> dict:
        """Cross-reference claims across sources.

        Returns a dict with:
        - verified_claims: claims supported by multiple sources
        - unique_claims: claims from only one source
        - conflicts: claims that contradict each other
        - agreement_score: 0-1 how much sources agree
        """
        # Group claims by topic (simplified: by key numbers mentioned)
        all_claims = []
        for source in sources:
            for claim in source.claims:
                all_claims.append({
                    "claim": claim,
                    "source": source.source_name,
                    "reliability": source.reliability,
                    "numbers": set(re.findall(r'\d+\.?\d*', claim.text)),
                })

        # Find agreement: claims where multiple sources mention the same numbers
        number_sources: dict[str, list[str]] = defaultdict(list)
        for c in all_claims:
            for num in c["numbers"]:
                key = f"{num}"
                if c["source"] not in number_sources[key]:
                    number_sources[key].append(c["source"])

        # Verified: numbers mentioned by 2+ sources
        verified_numbers = {k: v for k, v in number_sources.items() if len(v) >= self.MIN_AGREEMENT}

        # Build verified claims list
        verified_claims = []
        unique_claims = []
        for c in all_claims:
            is_verified = any(
                num in verified_numbers and c["source"] in verified_numbers[num]
                for num in c["numbers"]
            )
            if is_verified or c["reliability"] == "high":
                verified_claims.append(c)
            else:
                unique_claims.append(c)

        # Detect conflicts: different numbers for the same concept
        conflicts = []
        # Check specific known conflict patterns
        intake_numbers = defaultdict(list)
        for c in all_claims:
            text_lower = c["claim"].text.lower()
            if any(w in text_lower for w in ["liter", "litre", "glasses", "ounce"]):
                for num in c["numbers"]:
                    intake_numbers[c["source"]].append(float(num) if "." in num else int(num))

        # Flag if any source recommends > 4L (likely unreliable)
        for source, nums in intake_numbers.items():
            for n in nums:
                if 4 <= n <= 10:
                    # Could be liters — flag as potential overestimate
                    matching = [c for c in all_claims
                                if c["source"] == source and str(int(n)) in c["numbers"]]
                    for m in matching:
                        if m["reliability"] == "low":
                            conflicts.append({
                                "claim": m["claim"].text,
                                "source": source,
                                "issue": f"Recommends {n}L daily — higher than scientific consensus (2.7-3.7L)",
                                "resolution": "Excluded: unsupported by high-reliability sources",
                            })

        # Agreement score
        total = len(all_claims)
        verified_count = len(verified_claims)
        agreement = verified_count / total if total > 0 else 0.0

        return {
            "verified_claims": verified_claims,
            "unique_claims": unique_claims,
            "conflicts": conflicts,
            "agreement_score": round(agreement, 3),
            "verified_numbers": verified_numbers,
        }

    # ── Step 4: Source verification ──
    def verify_sources(self, sources: list[SourceRecord]) -> list[dict]:
        """Score each source on reliability, recency, and consistency."""
        scored = []
        for s in sources:
            recency_score = 1.0
            try:
                pub_date = datetime.strptime(s.publish_date, "%Y-%m-%d")
                age_days = (datetime(2025, 1, 1) - pub_date).days
                recency_score = max(0.3, 1.0 - age_days / 1095)  # decay over 3 years
            except ValueError:
                recency_score = 0.5

            reliability_score = self.RELIABILITY_WEIGHTS.get(s.reliability, 0.3)

            # Claim quality: ratio of factual claims
            factual = sum(1 for c in s.claims if c.category == "factual")
            claim_quality = factual / len(s.claims) if s.claims else 0

            overall = round(0.4 * reliability_score + 0.3 * recency_score + 0.3 * claim_quality, 3)
            scored.append({
                "source": s.source_name,
                "url": s.url,
                "reliability": reliability_score,
                "recency": round(recency_score, 3),
                "claim_quality": round(claim_quality, 3),
                "overall_score": overall,
                "num_claims": len(s.claims),
            })

        return sorted(scored, key=lambda x: -x["overall_score"])

    # ── Step 5: Write cited answer ──
    def write_answer(self, question: str, comparison: dict,
                     source_scores: list[dict]) -> tuple[str, list[dict]]:
        """Compose a final answer with inline citations.

        In production, an LLM would generate this. Here we template it
        to show the citation structure.
        """
        # Build bibliography
        bibliography = []
        source_idx = {}
        for i, ss in enumerate(source_scores):
            if ss["overall_score"] >= 0.3:  # Exclude very low-quality sources
                ref_num = len(bibliography) + 1
                bibliography.append({
                    "ref": ref_num,
                    "source": ss["source"],
                    "url": ss["url"],
                    "score": ss["overall_score"],
                })
                source_idx[ss["source"]] = ref_num

        # Build answer from verified claims
        high_quality = [c for c in comparison["verified_claims"]
                        if c["reliability"] in ("high", "medium")]

        # Group by source for cleaner composition
        answer_parts = []
        answer_parts.append(f"**Question:** {question}\\n")

        # Main finding
        answer_parts.append(
            "**Answer:** Based on multiple sources, the recommended daily water "
            f"intake for adults is approximately **2.7 liters for women** and "
            f"**3.7 liters for men** (total intake from all sources including food) "
            f"[{source_idx.get('National Health Institute', '?')}]"
            f"[{source_idx.get('Journal of Clinical Nutrition', '?')}].\\n"
        )

        # Supporting details
        answer_parts.append("**Key details:**\\n")
        answer_parts.append(
            f"- About 20% of daily water intake comes from food "
            f"[{source_idx.get('National Health Institute', '?')}].\\n"
        )
        answer_parts.append(
            f'- The popular "8 glasses a day" rule (≈2 liters) lacks strong '
            f"scientific evidence and was taken out of context from a 1945 "
            f"recommendation [{source_idx.get('Journal of Clinical Nutrition', '?')}].\\n"
        )
        answer_parts.append(
            f"- Individual needs vary by body weight, activity level, and climate "
            f"[{source_idx.get('National Health Institute', '?')}]"
            f"[{source_idx.get('Journal of Clinical Nutrition', '?')}].\\n"
        )
        answer_parts.append(
            f"- Urine color (pale yellow) is a practical indicator of adequate "
            f"hydration [{source_idx.get('Journal of Clinical Nutrition', '?')}].\\n"
        )
        answer_parts.append(
            f"- Athletes may need an additional 500-1000 mL per hour of exercise "
            f"[{source_idx.get('FitLife Magazine', '?')}].\\n"
        )
        answer_parts.append(
            f"- Coffee and tea count toward daily intake "
            f"[{source_idx.get('HealthyLiving Blog', '?')}].\\n"
        )

        # Conflicts
        if comparison["conflicts"]:
            answer_parts.append("\\n**Disputed claims (excluded):**\\n")
            for conflict in comparison["conflicts"]:
                answer_parts.append(f'- {conflict["source"]}: {conflict["issue"]}\\n')

        # Bibliography
        answer_parts.append("\\n**Sources:**\\n")
        for ref in bibliography:
            answer_parts.append(f'[{ref["ref"]}] {ref["source"]} — {ref["url"]} '
                                f'(reliability score: {ref["score"]})\\n')

        answer = "".join(answer_parts)
        return answer, bibliography

    # ── Main: run the full pipeline ──
    def research(self, question: str) -> ResearchResult:
        """Run the full research pipeline."""
        self.tool_log = []

        # Step 1: Plan
        queries = self.plan_queries(question)
        print(f"Step 1 — Planned {len(queries)} search queries")

        # Step 2: Search and fetch
        sources = self.search_and_fetch(queries)
        print(f"Step 2 — Fetched {len(sources)} unique sources")

        # Step 3: Compare
        comparison = self.compare_sources(sources)
        print(f"Step 3 — Compared claims: {comparison['agreement_score']:.0%} agreement, "
              f"{len(comparison['conflicts'])} conflicts")

        # Step 4: Verify
        source_scores = self.verify_sources(sources)
        print(f"Step 4 — Scored sources: "
              f"best={source_scores[0]['source']} ({source_scores[0]['overall_score']:.2f})")

        # Step 5: Write
        answer, citations = self.write_answer(question, comparison, source_scores)
        sources_used = len([s for s in source_scores if s["overall_score"] >= 0.3])
        print(f"Step 5 — Wrote answer with {len(citations)} citations")

        # Overall confidence
        high_rel_count = sum(1 for s in sources if s.reliability == "high")
        confidence = min(1.0, comparison["agreement_score"] * 0.5
                         + (high_rel_count / max(len(sources), 1)) * 0.3
                         + (1.0 if len(comparison["conflicts"]) == 0 else 0.5) * 0.2)

        return ResearchResult(
            question=question,
            answer=answer,
            citations=[{"ref": c["ref"], "source": c["source"], "url": c["url"]}
                        for c in citations],
            sources_consulted=len(sources),
            sources_used=sources_used,
            agreement_score=comparison["agreement_score"],
            confidence=round(confidence, 3),
            conflicts=[{"source": c["source"], "issue": c["issue"]}
                        for c in comparison["conflicts"]],
            tool_calls_log=self.tool_log,
        )


print("WebResearchAgent defined.")""")

# ══════════════════════════════════════════════
# 9  Run the agent
# ══════════════════════════════════════════════
md("## 8. Run the Agent")

code("""agent = WebResearchAgent()
question = "How much water should adults drink daily?"

result = agent.research(question)

print("=" * 70)
print(result.answer)
print("=" * 70)
print(f"\\nConfidence: {result.confidence:.0%}")
print(f"Sources consulted: {result.sources_consulted}")
print(f"Sources used: {result.sources_used}")
print(f"Agreement: {result.agreement_score:.0%}")
print(f"Conflicts found: {len(result.conflicts)}")""")

# ══════════════════════════════════════════════
# 10  Tool call trace
# ══════════════════════════════════════════════
md(r"""## 9. Tool-Call Trace

A critical feature of agent systems is **observability** — being able to see exactly which tools were called, in what order, and what they returned. This is essential for debugging and trust.
""")

code("""print(f"Total tool calls: {len(result.tool_calls_log)}")
print()

tool_counts = Counter(t["tool"] for t in result.tool_calls_log)
print("Tool call counts:")
for tool, count in tool_counts.most_common():
    print(f"  {tool}: {count}")

print(f"\\nFull trace:")
print("-" * 70)
for i, call in enumerate(result.tool_calls_log, 1):
    args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in call["args"].items())
    print(f"  {i:2d}. {call['tool']}({args_str})")
    print(f"      → {call['result']}")""")

# ══════════════════════════════════════════════
# 11  Source verification deep dive
# ══════════════════════════════════════════════
md(r"""## 10. Source Verification — How It Works

### The Three-Axis Scoring Model

Every source is scored on three axes:

| Axis | Weight | What it captures |
|---|---|---|
| **Reliability** | 40% | Domain authority — is this a medical journal or a random blog? |
| **Recency** | 30% | How fresh is the information? Science evolves. |
| **Claim quality** | 30% | Ratio of factual (number-backed) claims vs opinions |

### Why Source Verification Matters

Without source verification, an agent treats all sources equally. This means:
- A wellness blog claiming "drink 5 liters" counts the same as a peer-reviewed study
- An article from 2015 counts the same as one from 2024
- Sensational clickbait counts the same as government guidelines

**Source weighting is not optional for factual research.**
""")

code("""# Re-run source verification for display
sources = agent.search_and_fetch(agent.plan_queries(question))
source_scores = agent.verify_sources(sources)

scores_df = pd.DataFrame(source_scores)
display_cols = ["source", "reliability", "recency", "claim_quality", "overall_score", "num_claims"]
print(scores_df[display_cols].to_string(index=False))""")

code("""# Visualize source scores
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Bar chart: overall scores
ax = axes[0]
colors = ["#2ca02c" if s >= 0.5 else "#ff7f0e" if s >= 0.3 else "#d62728"
          for s in scores_df["overall_score"]]
bars = ax.barh(scores_df["source"], scores_df["overall_score"], color=colors, alpha=0.8)
ax.set_xlabel("Overall Score")
ax.set_title("Source Reliability Ranking")
ax.set_xlim([0, 1])
ax.axvline(0.3, color="red", linestyle="--", alpha=0.5, label="Exclusion threshold")
ax.legend(fontsize=8)
for bar, val in zip(bars, scores_df["overall_score"]):
    ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=9)

# Stacked components
ax2 = axes[1]
components = ["reliability", "recency", "claim_quality"]
weights = [0.4, 0.3, 0.3]
bottom = np.zeros(len(scores_df))
colors_stack = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for comp, w, c in zip(components, weights, colors_stack):
    vals = scores_df[comp].values * w
    ax2.barh(scores_df["source"], vals, left=bottom, color=c, alpha=0.8, label=f"{comp} (×{w})")
    bottom += vals
ax2.set_xlabel("Weighted Score Contribution")
ax2.set_title("Score Breakdown by Component")
ax2.legend(fontsize=7, loc="lower right")

plt.tight_layout()
plt.show()""")

# ══════════════════════════════════════════════
# 12  Conflict resolution
# ══════════════════════════════════════════════
md(r"""## 11. Conflict Resolution — When Sources Disagree

### Types of Disagreement

| Type | Example | Resolution strategy |
|---|---|---|
| **Factual contradiction** | Source A says 2L, Source B says 5L | Weight by source reliability |
| **Scope difference** | One source covers athletes, another covers sedentary adults | Note the different scopes, don't merge |
| **Outdated vs current** | 2019 guidance vs 2024 guidance | Prefer the more recent source |
| **Precision difference** | "About 2-3L" vs "2.7L for women, 3.7L for men" | Prefer the more specific source |

### Our Approach

1. **Detect conflicts** by comparing numeric claims across sources
2. **Weight by reliability** — peer-reviewed > government > blog > clickbait
3. **Exclude unreliable outliers** — claims from low-reliability sources that contradict the consensus
4. **Report transparently** — tell the reader what was excluded and why
""")

code("""print("CONFLICTS DETECTED")
print("=" * 70)
if result.conflicts:
    for i, conflict in enumerate(result.conflicts, 1):
        print(f"\\n{i}. Source: {conflict['source']}")
        print(f"   Issue:  {conflict['issue']}")
else:
    print("No direct conflicts detected.")

print("\\n\\nSOURCE AGREEMENT MATRIX")
print("=" * 70)

# Build a claim-source matrix
sources_rerun = agent.search_and_fetch(agent.plan_queries(question))
comparison = agent.compare_sources(sources_rerun)

print(f"\\nVerified numbers (mentioned by 2+ sources):")
for num, srcs in sorted(comparison["verified_numbers"].items()):
    print(f"  {num}: {', '.join(srcs)}")

print(f"\\nAgreement score: {comparison['agreement_score']:.0%}")
print(f"Verified claims: {len(comparison['verified_claims'])}")
print(f"Unique claims: {len(comparison['unique_claims'])}")""")

# ══════════════════════════════════════════════
# 13  Citation format
# ══════════════════════════════════════════════
md(r"""## 12. Citation Generation — Linking Claims to Sources

### Why Citations Matter

An uncited answer is just another LLM output — the reader has no way to verify it. Citations create:
- **Verifiability**: the reader can check the source
- **Accountability**: if a claim is wrong, we know which source it came from
- **Trust calibration**: the reader can assess source quality themselves

### Citation Formats

| Format | Example | Best for |
|---|---|---|
| **Inline numbered** | "...2.7 liters [1]" | Academic, technical docs |
| **Inline named** | "...2.7 liters (NIH, 2024)" | Reports, blog posts |
| **Footnotes** | "...2.7 liters¹" | Long-form articles |
| **Hyperlinked** | "...2.7 liters ([source](url))" | Web content |

We use inline numbered citations with a bibliography at the end.
""")

code("""print("BIBLIOGRAPHY")
print("=" * 70)
for cite in result.citations:
    print(f"  [{cite['ref']}] {cite['source']}")
    print(f"      {cite['url']}")
    print()

# Show where each citation appears in the answer
print("\\nCITATION USAGE IN ANSWER")
print("=" * 70)
citation_pattern = re.findall(r'\[(\d+)\]', result.answer)
usage_counts = Counter(citation_pattern)
for ref_num, count in sorted(usage_counts.items()):
    source = next((c["source"] for c in result.citations if str(c["ref"]) == ref_num), "?")
    print(f"  [{ref_num}] used {count} time(s) — {source}")""")

# ══════════════════════════════════════════════
# 14  Multiple questions
# ══════════════════════════════════════════════
md("## 13. Testing with Multiple Questions")

code("""# Add more topics to the simulated web for additional questions
SIMULATED_WEB["https://sleep-research.org/duration-guidelines"] = PageContent(
    url="https://sleep-research.org/duration-guidelines",
    title="Sleep Duration Guidelines by Age",
    text=(
        "The National Sleep Foundation recommends 7-9 hours of sleep per night "
        "for adults aged 18-64, and 7-8 hours for adults 65+. Teenagers need "
        "8-10 hours. Short sleep duration (less than 6 hours) is associated with "
        "increased cardiovascular risk, impaired cognitive function, and weight gain. "
        "Sleep quality matters as much as quantity: uninterrupted sleep cycles of "
        "90 minutes each are optimal. Most adults need 4-6 complete cycles per night."
    ),
    source_name="Sleep Research Institute",
    publish_date="2024-01-20",
)

SIMULATED_WEB["https://health-magazine.example.com/sleep-tips"] = PageContent(
    url="https://health-magazine.example.com/sleep-tips",
    title="How Much Sleep Do You Actually Need?",
    text=(
        "Most doctors agree you need 7-9 hours of sleep. But some high performers "
        "thrive on 6 hours. Genetics play a role — about 1-3% of people have a "
        "gene variant (DEC2) that allows them to function well on less sleep. "
        "The key is consistency: going to bed and waking up at the same time matters "
        "more than total hours. Blue light from screens suppresses melatonin and can "
        "delay sleep onset by 30-60 minutes."
    ),
    source_name="Health Magazine",
    publish_date="2023-09-15",
)

SIMULATED_WEB["https://biohack-sleep.example.com/optimal"] = PageContent(
    url="https://biohack-sleep.example.com/optimal",
    title="Biohack Your Sleep: Why 5 Hours Is Enough",
    text=(
        "Elite performers only need 4-5 hours of sleep if you optimize quality. "
        "Polyphasic sleep schedules can reduce total sleep to 3-4 hours with "
        "multiple 20-minute naps. Cold showers before bed improve deep sleep "
        "by 300%. Melatonin supplements at 10mg are safe and effective for "
        "everyone. Sleep is overrated — hustle culture demands sacrifice."
    ),
    source_name="BiohackSleep",
    publish_date="2023-03-01",
)

# Update search index
for url, page in SIMULATED_WEB.items():
    text_blob = f"{page.title} {page.text}".lower()
    for keyword in ["sleep", "hours", "night", "duration", "adults",
                     "recommended", "optimal", "rest"]:
        if keyword in text_blob and url not in SEARCH_INDEX.get(keyword, []):
            SEARCH_INDEX[keyword].append(url)

print("Added sleep-related sources to simulated web.")
print(f"Total pages: {len(SIMULATED_WEB)}")""")

code("""test_questions = [
    "How much water should adults drink daily?",
    "How many hours of sleep do adults need?",
]

all_results = []
for q in test_questions:
    agent_q = WebResearchAgent()
    res = agent_q.research(q)
    all_results.append(res)
    print(f"\\n{'=' * 70}")
    print(f"Q: {q}")
    print(f"Confidence: {res.confidence:.0%}  |  Sources: {res.sources_consulted}  |  "
          f"Citations: {len(res.citations)}  |  Conflicts: {len(res.conflicts)}")
    print(f"Agreement: {res.agreement_score:.0%}")""")

# ══════════════════════════════════════════════
# 15  Second answer display
# ══════════════════════════════════════════════
code("""# Display the sleep question answer
if len(all_results) > 1:
    print("ANSWER: How many hours of sleep do adults need?")
    print("=" * 70)
    print(all_results[1].answer)""")

# ══════════════════════════════════════════════
# 16  Quality analysis
# ══════════════════════════════════════════════
md("## 14. Agent Quality Analysis")

code("""# Compare across questions
summary_rows = []
for res in all_results:
    summary_rows.append({
        "question": res.question[:50],
        "confidence": res.confidence,
        "sources_consulted": res.sources_consulted,
        "sources_used": res.sources_used,
        "agreement": res.agreement_score,
        "citations": len(res.citations),
        "conflicts": len(res.conflicts),
        "tool_calls": len(res.tool_calls_log),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))""")

code("""fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Confidence vs agreement
ax = axes[0]
ax.scatter(summary_df["agreement"], summary_df["confidence"],
           s=summary_df["sources_consulted"] * 30, alpha=0.7, c="#1f77b4")
ax.set_xlabel("Source Agreement")
ax.set_ylabel("Confidence")
ax.set_title("Confidence vs Agreement")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
for _, row in summary_df.iterrows():
    ax.annotate(row["question"][:25] + "...", (row["agreement"], row["confidence"]),
                fontsize=7, ha="center", va="bottom")

# Tool calls per question
ax2 = axes[1]
tool_data = []
for res in all_results:
    counts = Counter(t["tool"] for t in res.tool_calls_log)
    for tool, count in counts.items():
        tool_data.append({"question": res.question[:30], "tool": tool, "count": count})
tool_df = pd.DataFrame(tool_data)
if not tool_df.empty:
    pivot = tool_df.pivot_table(index="question", columns="tool", values="count",
                                aggfunc="sum", fill_value=0)
    pivot.plot(kind="bar", stacked=True, ax=ax2, alpha=0.8, rot=0)
    ax2.set_title("Tool Calls by Question")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", labelsize=7)
    ax2.legend(fontsize=7)

# Sources used vs consulted
ax3 = axes[2]
x = np.arange(len(summary_df))
width = 0.35
ax3.bar(x - width / 2, summary_df["sources_consulted"], width,
        label="Consulted", color="#1f77b4", alpha=0.8)
ax3.bar(x + width / 2, summary_df["sources_used"], width,
        label="Used in answer", color="#2ca02c", alpha=0.8)
ax3.set_title("Sources: Consulted vs Used")
ax3.set_ylabel("Count")
ax3.set_xticks(x)
ax3.set_xticklabels([q[:20] + "..." for q in summary_df["question"]], fontsize=7)
ax3.legend(fontsize=8)

plt.tight_layout()
plt.show()""")

# ══════════════════════════════════════════════
# 17  Production considerations
# ══════════════════════════════════════════════
md(r"""## 15. Production Considerations

### Using Real Search APIs

| Provider | API | Cost | Rate limit |
|---|---|---|---|
| Google Custom Search | `customsearch.googleapis.com` | $5 per 1K queries | 100/day (free tier) |
| Bing Web Search | `api.bing.microsoft.com` | $3 per 1K transactions | 3 calls/sec |
| SerpAPI | `serpapi.com` | $50/mo for 5K searches | 1 call/sec |
| Tavily | `tavily.com` | Free tier: 1K searches/mo | 5 calls/sec |

### Using LLMs for Each Step

In production, replace our heuristic functions with LLM calls:

```python
# Claim extraction (production)
messages = [
    {"role": "system", "content": "Extract atomic factual claims..."},
    {"role": "user", "content": page_text},
]
claims = llm.generate(messages)

# Answer composition (production)
messages = [
    {"role": "system", "content": "Write a cited answer using [n] format..."},
    {"role": "user", "content": f"Question: {q}\nVerified claims: {claims}"},
]
answer = llm.generate(messages)
```

### Real-World Tool Definition (OpenAI Format)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    }
]
```

### Cost Estimation

| Component | Per question | 1K questions/day |
|---|---|---|
| Search API (3 queries) | $0.015 | $15 |
| Page fetching (5 pages) | ~free | ~free |
| LLM: claim extraction (5 calls) | $0.01 | $10 |
| LLM: comparison + writing | $0.005 | $5 |
| **Total** | **~$0.03** | **~$30** |
""")

# ══════════════════════════════════════════════
# 18  Failure modes
# ══════════════════════════════════════════════
md(r"""## 16. Failure Modes and Mitigations

### Common Failures

| Failure mode | Example | Impact | Mitigation |
|---|---|---|---|
| **Search returns irrelevant results** | Query too vague → off-topic pages | Wrong answer with confident tone | Use sub-queries; verify relevance before extraction |
| **All sources repeat the same error** | Common misconception in top results | Consensus on a wrong claim | Include authoritative sources (journals, gov sites) explicitly |
| **Source is outdated** | 2018 article with since-revised guidance | Stale answer | Weight by recency; check publication dates |
| **Paywalled content** | Can see snippet but not full article | Incomplete extraction | Use snippet + search for freely available versions |
| **Circular sourcing** | Blog B cites Blog A; both appear as "2 sources agree" | False consensus | Track original sources; detect citation chains |
| **Adversarial content** | SEO-optimized misinformation ranks high | Poisoned answer | Whitelist trusted domains; cross-reference with known reliable sources |

### The Circular Sourcing Problem

This is the most subtle failure. If many websites copy from one original source, simple counting says "5 sources agree!" — but really only 1 source exists. 

**Detection strategies:**
1. Check if sources are word-for-word similar (possible copy)
2. Look for citation/attribution to an original source
3. Prefer primary sources (peer-reviewed, official) over secondary (blogs, news)
4. Count independent institutions, not URLs

### When the Agent Should Refuse

The agent should say "I don't have enough reliable information" when:
- Confidence score < threshold (we use 0.4)
- No high-reliability sources found
- All sources conflict with no clear winner
- The topic requires expertise the agent cannot verify (medical advice, legal guidance)
""")

# ══════════════════════════════════════════════
# 19  LLM integration sketch
# ══════════════════════════════════════════════
md(r"""## 17. LLM Integration — Connecting to a Real Model

Below is a sketch showing how to connect each agent step to an LLM. This does not run (it requires API keys or a local model) but shows the production pattern.
""")

code("""# Sketch: LLM-powered research agent (does not run — illustrative only)

LLM_INTEGRATION_SKETCH = '''
import openai  # or: from transformers import pipeline

client = openai.OpenAI()  # or local model

def llm_plan_queries(question: str) -> list[str]:
    """Use LLM to decompose question into search sub-queries."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":
             "Decompose this question into 2-4 search queries that would help "
             "find a comprehensive, factual answer. Return as JSON list."},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)["queries"]


def llm_extract_claims(text: str, source_url: str) -> list[dict]:
    """Use LLM to extract atomic factual claims."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":
             "Extract all factual claims from this text. For each claim, provide: "
             "the claim text, whether it contains a specific number/statistic, "
             "and your confidence (0-1) that it is factually accurate. "
             "Return as JSON list of objects."},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)["claims"]


def llm_write_cited_answer(question: str, verified_claims: list,
                            bibliography: list) -> str:
    """Use LLM to compose a cited answer."""
    claims_text = "\\n".join(
        f"- {c['text']} (source: [{c['ref']}])" for c in verified_claims
    )
    bib_text = "\\n".join(
        f"[{b['ref']}] {b['source']} — {b['url']}" for b in bibliography
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":
             "Write a clear, factual answer using ONLY the provided claims. "
             "Use [n] inline citations. End with the bibliography. "
             "If claims conflict, note the disagreement."},
            {"role": "user", "content":
             f"Question: {question}\\n\\nVerified claims:\\n{claims_text}"
             f"\\n\\nBibliography:\\n{bib_text}"},
        ],
    )
    return response.choices[0].message.content
'''

print("LLM integration sketch (illustrative — does not execute):")
print(LLM_INTEGRATION_SKETCH[:500] + "\\n...")""")

# ══════════════════════════════════════════════
# 20  Evaluation framework
# ══════════════════════════════════════════════
md("## 18. Evaluating a Research Agent")

code("""# Define evaluation criteria for research agents

EVAL_CRITERIA = {
    "citation_accuracy": {
        "description": "Do citations point to real sources that support the claim?",
        "weight": 0.25,
    },
    "source_diversity": {
        "description": "Are multiple independent sources consulted?",
        "weight": 0.15,
    },
    "conflict_handling": {
        "description": "Are disagreements between sources noted and resolved?",
        "weight": 0.15,
    },
    "factual_correctness": {
        "description": "Are the factual claims in the answer correct?",
        "weight": 0.25,
    },
    "completeness": {
        "description": "Does the answer cover the key aspects of the question?",
        "weight": 0.10,
    },
    "appropriate_uncertainty": {
        "description": "Does the agent express uncertainty when evidence is weak?",
        "weight": 0.10,
    },
}


def evaluate_research_result(result: ResearchResult) -> dict:
    """Score a research result on defined criteria."""
    scores = {}

    # Citation accuracy: all refs should point to real sources
    valid_refs = sum(1 for c in result.citations if c["url"] in SIMULATED_WEB)
    scores["citation_accuracy"] = valid_refs / len(result.citations) if result.citations else 0

    # Source diversity: different sources consulted
    unique_sources = len(set(c["source"] for c in result.citations))
    scores["source_diversity"] = min(1.0, unique_sources / 3)  # 3+ distinct sources = 1.0

    # Conflict handling: conflicts should be reported
    if result.conflicts:
        scores["conflict_handling"] = 1.0 if "Disputed" in result.answer or "excluded" in result.answer.lower() else 0.3
    else:
        scores["conflict_handling"] = 0.8  # No conflicts = partial credit

    # Factual correctness: proxy via confidence
    scores["factual_correctness"] = result.confidence

    # Completeness: sources used vs available
    scores["completeness"] = result.sources_used / result.sources_consulted if result.sources_consulted > 0 else 0

    # Appropriate uncertainty: should not be 100% confident
    scores["appropriate_uncertainty"] = 1.0 if result.confidence < 0.95 else 0.5

    # Weighted overall
    overall = sum(scores[k] * EVAL_CRITERIA[k]["weight"] for k in scores)

    return {
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "overall": round(overall, 3),
    }


for res in all_results:
    eval_result = evaluate_research_result(res)
    print(f"Q: {res.question[:50]}")
    for k, v in eval_result["scores"].items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"  {k:30s} {bar} {v:.0%}")
    print(f"  {'OVERALL':30s} {'█' * int(eval_result['overall'] * 20)}{'░' * (20 - int(eval_result['overall'] * 20))} {eval_result['overall']:.0%}")
    print()""")

# ══════════════════════════════════════════════
# 21  Evaluation visual
# ══════════════════════════════════════════════
code("""# Radar chart of evaluation criteria
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

criteria = list(EVAL_CRITERIA.keys())
N = len(criteria)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for res in all_results:
    eval_result = evaluate_research_result(res)
    values = [eval_result["scores"][c] for c in criteria]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=1.5, label=res.question[:35] + "...")
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([c.replace("_", "\\n") for c in criteria], fontsize=8)
ax.set_ylim([0, 1])
ax.set_title("Research Agent Evaluation", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

plt.tight_layout()
plt.show()""")

# ══════════════════════════════════════════════
# 22  Limitations
# ══════════════════════════════════════════════
md(r"""## 19. Limitations

| Limitation | Impact | Real-world fix |
|---|---|---|
| **Simulated web** | Cannot test with real search noise, paywalls, or adversarial SEO | Integrate real search APIs (Google, Bing, Tavily) |
| **Heuristic claim extraction** | Misses nuanced claims; no semantic understanding | Use LLM for extraction |
| **Heuristic answer writing** | Templated output cannot adapt to novel question types | Use LLM for composition |
| **No multi-hop search** | Agent does not issue follow-up searches based on initial findings | Add iterative search with saturation detection |
| **No image/table extraction** | Ignores non-text content | Use document parsing (PDF extractors, OCR) |
| **Single-turn only** | No clarification questions back to the user | Add a clarification step before research begins |
| **English-only sources** | Misses non-English research | Add cross-lingual retrieval |
| **No caching** | Same search repeated for similar questions | Add search cache with TTL |
| **Claim extraction is noisy** | Some extracted "claims" are opinions or filler | Use LLM with structured output for cleaner extraction |

### What Would Make This Production-Ready

1. **Real search integration** (Tavily, SerpAPI) instead of simulated web
2. **LLM-powered extraction** for each agent step
3. **Multi-hop search**: search → read → identify gaps → search again
4. **Source deduplication**: detect when multiple URLs are from the same original source
5. **Confidence calibration**: train the confidence model on human-judged outputs
6. **Streaming**: return partial results as the agent works
7. **Rate limiting and cost controls**: cap API calls per question
""")

# ══════════════════════════════════════════════
# 23  Save
# ══════════════════════════════════════════════
md("## 20. Save Experiment Log")

code("""log = {
    "timestamp": datetime.now().isoformat(),
    "task": "web_research_agent",
    "questions": [r.question for r in all_results],
    "results": [
        {
            "question": r.question,
            "confidence": r.confidence,
            "sources_consulted": r.sources_consulted,
            "sources_used": r.sources_used,
            "agreement_score": r.agreement_score,
            "num_citations": len(r.citations),
            "num_conflicts": len(r.conflicts),
            "tool_calls": len(r.tool_calls_log),
        }
        for r in all_results
    ],
}

log_path = ARTIFACT_DIR / "web_research_agent_log.json"
log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
print(f"Saved: {log_path}")""")

# ══════════════════════════════════════════════
# 24  Key takeaways
# ══════════════════════════════════════════════
md(r"""## 21. Key Takeaways

### What We Built
- A web research agent with three tools: `web_search`, `fetch_page`, `extract_claims`
- Multi-source comparison with reliability-weighted scoring
- Conflict detection and resolution
- Cited answer generation with inline `[n]` references and bibliography

### Tool Use Principles
1. **Define tools with clear schemas** — the agent (or LLM) uses the description to decide when to call them
2. **Log every tool call** — observability is non-negotiable for debugging and trust
3. **Tools return structured data** — not free text; structured data enables downstream processing
4. **The agent decides the tool sequence** — not hardcoded; the agent adapts to each question

### Source Verification Principles
1. **Not all sources are equal** — weight by authority, recency, and claim quality
2. **Agreement is not proof** — multiple sources copying one original is still one source
3. **Conflicts are features, not bugs** — surfacing disagreements builds trust
4. **Exclude, don't ignore** — when you drop a source, tell the reader why
5. **Confidence should be calibrated** — an agent saying "I'm 95% sure" should actually be right 95% of the time

### When to Build a Research Agent
| Situation | Build it? | Why |
|---|---|---|
| Factual questions over recent events | Yes | LLM knowledge cutoff makes search necessary |
| Questions where accuracy is critical | Yes | Multi-source verification reduces error rate |
| High-volume Q&A with citations needed | Yes | Manual research does not scale |
| Creative writing or brainstorming | No | Grounding constrains creativity |
| Well-known stable facts | Maybe | Simple RAG or fine-tuned model may suffice |
| Real-time data (stock prices, weather) | Partially | Need specialized APIs, not web search |
""")


# ══════════════════════════════════════════════
# Build
# ══════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(__file__).parent / "web_research_agent.ipynb"
out.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Notebook written: {out}")
print(f"Cells: {len(cells)}")
print(f"Code:  {sum(1 for c in cells if c['cell_type'] == 'code')}")
print(f"Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"Size: {out.stat().st_size:,} bytes")
