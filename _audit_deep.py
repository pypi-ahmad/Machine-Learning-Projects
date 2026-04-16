"""
Deep audit: check code syntax, content quality, and local-first compliance
for all 100 Local AI Projects notebooks.
"""
import json, os, re, ast, sys

base = r'E:\Github\Machine-Learning-Projects\100_Local_AI_Projects'

categories = [
    ('Beginner_Local_LLM_Apps', range(1, 11)),
    ('Local_RAG', range(11, 21)),
    ('Advanced_RAG_and_Retrieval_Engineering', range(21, 31)),
    ('LangGraph_Workflows', range(31, 41)),
    ('CrewAI_Multi-Agent_Systems', range(41, 51)),
    ('Local_Tool-Using_Agents', range(51, 61)),
    ('Local_Eval_and_Observability_Projects', range(61, 71)),
    ('Fine-Tuning-Adjacent_Learning_Projects', range(71, 81)),
    ('Multimodal_-_OCR_-_Speech_-_VLM', range(81, 91)),
    ('Coding_and_Developer_Agents', range(91, 101)),
]

# Build list of the 100 main project notebooks only (skip legacy duplicates)
all_projects = []
for cat, proj_range in categories:
    cat_path = os.path.join(base, cat)
    if not os.path.isdir(cat_path):
        continue
    for entry in sorted(os.listdir(cat_path)):
        full = os.path.join(cat_path, entry)
        if not os.path.isdir(full):
            continue
        # Skip legacy duplicates (N__name pattern)
        if re.match(r'^\d+__', entry):
            continue
        match = re.match(r'^(\d+)', entry)
        if match:
            num = int(match.group(1))
            if num in proj_range or num == 100:
                all_projects.append((num, cat, entry, full))

all_projects.sort(key=lambda x: x[0])
print(f'Auditing {len(all_projects)} main project notebooks\n')

issues_by_project = {}
warnings_by_project = {}

for num, cat, folder, full_path in all_projects:
    nb_path = os.path.join(full_path, 'notebook.ipynb')
    issues = []
    warnings = []

    with open(nb_path, 'r', encoding='utf-8-sig') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    md_cells = [c for c in cells if c.get('cell_type') == 'markdown']
    code_cells = [c for c in cells if c.get('cell_type') == 'code']

    # ── 1. Code syntax check ──
    for i, c in enumerate(code_cells):
        src = c.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        # Skip empty cells or cells with only comments/magic
        stripped = '\n'.join(
            line for line in src.split('\n')
            if line.strip() and not line.strip().startswith('#')
            and not line.strip().startswith('!')
            and not line.strip().startswith('%')
        )
        if not stripped.strip():
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            issues.append(f'SyntaxError in code cell {i}: {e.msg} (line {e.lineno})')

    # ── 2. All source combined ──
    all_source = ''
    for c in cells:
        src = c.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        all_source += src + '\n'

    # ── 3. Ollama / local-first check ──
    uses_ollama = 'ollama' in all_source.lower() or 'Ollama' in all_source
    uses_openai = 'OPENAI_API_KEY' in all_source or 'ChatOpenAI' in all_source
    if not uses_ollama:
        warnings.append('Does not reference Ollama (should be local-first)')
    if uses_openai:
        issues.append('References OpenAI API (should be local-first)')

    # ── 4. Check for proper markdown educational content ──
    all_md = ''
    for c in md_cells:
        src = c.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        all_md += src + '\n'

    # Check for "What you'll learn" or "Learning" or "overview"
    has_learning = any(
        kw in all_md.lower()
        for kw in ['learn', 'overview', 'objective', 'goal', 'what we', 'what you']
    )
    if not has_learning:
        warnings.append('No learning objectives or overview found in markdown')

    # Check for recap/summary
    has_recap = any(
        kw in all_md.lower()
        for kw in ['recap', 'summary', 'what you learned', 'key takeaway', 'conclusion', 'next step']
    )
    if not has_recap:
        warnings.append('No recap/summary section found')

    # ── 5. Check for proper imports ──
    has_langchain = 'langchain' in all_source.lower()
    has_llamaindex = 'llama_index' in all_source.lower() or 'llamaindex' in all_source.lower()
    has_langgraph = 'langgraph' in all_source.lower()
    has_crewai = 'crewai' in all_source.lower()
    has_haystack = 'haystack' in all_source.lower()
    has_dspy = 'dspy' in all_source.lower()
    has_pydanticai = 'pydantic_ai' in all_source.lower() or 'pydanticai' in all_source.lower()

    # Verify framework matches category
    if 'LangGraph' in cat and not has_langgraph:
        warnings.append('LangGraph category but no LangGraph imports found')
    if 'CrewAI' in cat and not has_crewai:
        warnings.append('CrewAI category but no CrewAI imports found')

    # ── 6. Check notebook has reasonable size ──
    total_code_chars = sum(
        len(''.join(c['source']) if isinstance(c['source'], list) else c['source'])
        for c in code_cells
    )
    total_md_chars = sum(
        len(''.join(c['source']) if isinstance(c['source'], list) else c['source'])
        for c in md_cells
    )

    if total_code_chars < 300:
        issues.append(f'Very sparse code content: {total_code_chars} chars')
    if total_md_chars < 150:
        warnings.append(f'Sparse markdown: {total_md_chars} chars')

    # ── Print results ──
    status = 'ERROR' if issues else ('WARN' if warnings else 'OK')
    print(f'[{num:3d}] {status:5s} | md={len(md_cells):2d} code={len(code_cells):2d} | code={total_code_chars:5d}ch md={total_md_chars:4d}ch | {folder}')
    if issues:
        for iss in issues:
            print(f'       X {iss}')
        issues_by_project[num] = issues
    if warnings:
        for w in warnings:
            print(f'       W {w}')
        warnings_by_project[num] = warnings

print(f'\n{"="*80}')
print(f'SUMMARY: {len(all_projects)} notebooks audited')
print(f'  Errors: {len(issues_by_project)}')
print(f'  Warnings: {len(warnings_by_project)}')
print(f'  Clean (no issues): {len(all_projects) - len(issues_by_project) - len(set(warnings_by_project.keys()) - set(issues_by_project.keys()))}')

if issues_by_project:
    print(f'\nERROR projects (must fix):')
    for num, iss in sorted(issues_by_project.items()):
        print(f'  [{num:3d}] {len(iss)} error(s)')



