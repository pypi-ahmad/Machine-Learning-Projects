"""Audit script for the 100 Local AI Projects notebooks."""
import json, os, re

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

all_projects = []
for cat, proj_range in categories:
    cat_path = os.path.join(base, cat)
    if not os.path.isdir(cat_path):
        print(f'MISSING CATEGORY: {cat}')
        continue
    for entry in sorted(os.listdir(cat_path)):
        full = os.path.join(cat_path, entry)
        if os.path.isdir(full):
            match = re.match(r'^(\d+)', entry)
            if match:
                num = int(match.group(1))
                if num in proj_range:
                    all_projects.append((num, cat, entry, full))

# project 100
for entry in sorted(os.listdir(os.path.join(base, 'Coding_and_Developer_Agents'))):
    if re.match(r'^100', entry):
        full = os.path.join(base, 'Coding_and_Developer_Agents', entry)
        if os.path.isdir(full) and (100, 'Coding_and_Developer_Agents', entry, full) not in all_projects:
            all_projects.append((100, 'Coding_and_Developer_Agents', entry, full))

all_projects.sort(key=lambda x: x[0])
print(f'Found {len(all_projects)} project folders\n')

issues_summary = []

for num, cat, folder, full_path in all_projects:
    issues = []

    # Check for notebook
    nb_path = os.path.join(full_path, 'notebook.ipynb')
    if not os.path.exists(nb_path):
        ipynbs = [f for f in os.listdir(full_path) if f.endswith('.ipynb')]
        if ipynbs:
            nb_path = os.path.join(full_path, ipynbs[0])
            issues.append(f'Notebook named "{ipynbs[0]}" instead of notebook.ipynb')
        else:
            issues.append('NO NOTEBOOK FOUND')
            issues_summary.append((num, folder, issues))
            continue

    # Check for .py files
    py_files = [f for f in os.listdir(full_path) if f.endswith('.py')]
    if py_files:
        issues.append(f'Has .py files: {py_files}')

    # Parse notebook
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(f'INVALID JSON: {e}')
        issues_summary.append((num, folder, issues))
        continue

    cells = nb.get('cells', [])
    md_cells = [c for c in cells if c.get('cell_type') == 'markdown']
    code_cells = [c for c in cells if c.get('cell_type') == 'code']

    if len(cells) < 5:
        issues.append(f'Too few cells: {len(cells)}')
    if len(md_cells) < 3:
        issues.append(f'Too few markdown cells: {len(md_cells)} (need educational content)')
    if len(code_cells) < 2:
        issues.append(f'Too few code cells: {len(code_cells)}')

    # Get all source
    all_source = ''
    for c in cells:
        src = c.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        all_source += src + '\n'

    # Check for MLOps bootstrap cell
    for i, c in enumerate(cells):
        src = c.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        tags = c.get('metadata', {}).get('tags', [])
        if 'injected-mlops-bootstrap' in tags:
            issues.append(f'Has injected MLOps bootstrap cell (cell {i})')
        if 'mlflow.db' in src and 'Path(__file__)' in src:
            issues.append(f'Cell {i} has Path(__file__) in notebook - will error')

    # Cloud API checks
    if 'OPENAI_API_KEY' in all_source or 'openai.api_key' in all_source:
        issues.append('Uses OpenAI API key (should be local-first)')
    if 'ChatOpenAI' in all_source and 'Ollama' not in all_source:
        issues.append('Uses ChatOpenAI without Ollama')

    # Streamlit check
    if 'import streamlit' in all_source.lower():
        issues.append('Imports Streamlit (not allowed in notebooks)')

    # Title check
    has_title = False
    first_cell = cells[0] if cells else None
    first_is_bootstrap = False
    if first_cell and first_cell.get('cell_type') == 'code':
        src = first_cell.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        if 'mlops' in src.lower() or 'bootstrap' in src.lower() or 'injected' in src.lower():
            first_is_bootstrap = True
            issues.append('First cell is injected bootstrap, not educational title')

    for c in cells:
        if c.get('cell_type') == 'markdown':
            src = c.get('source', '')
            if isinstance(src, list):
                src = ''.join(src)
            if src.strip().startswith('#'):
                has_title = True
            break

    if not has_title:
        issues.append('No markdown title (# heading) found')

    # Check content depth - all source code length
    total_code_len = sum(
        len(''.join(c['source']) if isinstance(c['source'], list) else c['source'])
        for c in code_cells
    )
    total_md_len = sum(
        len(''.join(c['source']) if isinstance(c['source'], list) else c['source'])
        for c in md_cells
    )
    if total_code_len < 200:
        issues.append(f'Very little code content: {total_code_len} chars')
    if total_md_len < 100:
        issues.append(f'Very little markdown content: {total_md_len} chars')

    # Check for common broken patterns
    if "from pathlib import Path" not in all_source and "Path(__file__)" in all_source:
        issues.append('Uses Path(__file__) without importing Path')

    if issues:
        issues_summary.append((num, folder, issues))

    status = 'ISSUES' if issues else 'OK'
    print(f'[{num:3d}] {status:6s} | md={len(md_cells):2d} code={len(code_cells):2d} | {folder}')
    if issues:
        for iss in issues:
            print(f'         > {iss}')

print(f'\n{"="*80}')
print(f'SUMMARY: {len(all_projects)} projects checked, {len(issues_summary)} with issues')
print(f'Projects with no issues: {len(all_projects) - len(issues_summary)}')
if issues_summary:
    print(f'\nProjects with issues:')
    for num, folder, iss_list in issues_summary:
        print(f'  [{num:3d}] {folder}: {len(iss_list)} issue(s)')

