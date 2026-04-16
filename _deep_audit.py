"""Deep audit of all 100 Local AI Project notebooks."""
import json, os, re, ast

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

seen = {}
for cat, proj_range in categories:
    cat_path = os.path.join(base, cat)
    if not os.path.isdir(cat_path):
        continue
    for entry in sorted(os.listdir(cat_path)):
        full = os.path.join(cat_path, entry)
        if os.path.isdir(full):
            match = re.match(r'^(\d+)', entry)
            if match:
                num = int(match.group(1))
                if num in proj_range:
                    if num not in seen or entry.startswith('0'):
                        seen[num] = (cat, entry, full)

issues_all = []
ollama_models = set()

for num in sorted(seen.keys()):
    cat, folder, full_path = seen[num]
    nb_path = os.path.join(full_path, 'notebook.ipynb')
    if not os.path.exists(nb_path):
        ipynbs = [f for f in os.listdir(full_path) if f.endswith('.ipynb')]
        if ipynbs:
            nb_path = os.path.join(full_path, ipynbs[0])
        else:
            issues_all.append((num, folder, ['NO NOTEBOOK']))
            continue

    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        issues_all.append((num, folder, [f'INVALID JSON: {e}']))
        continue

    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    md_cells = [c for c in cells if c.get('cell_type') == 'markdown']
    raw_cells = [c for c in cells if c.get('cell_type') == 'raw']

    issues = []

    # Check for raw cells (broken notebook structure)
    if raw_cells:
        issues.append(f'{len(raw_cells)} raw cell(s) - content may be misplaced')

    # Check for empty notebook
    if len(code_cells) == 0:
        issues.append('EMPTY - no code cells')

    # Extract all code
    all_code = []
    for c in code_cells:
        src = c.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        all_code.append(src)
    full_code = '\n'.join(all_code)

    # Syntax check
    for i, code in enumerate(all_code):
        lines = [l for l in code.split('\n') if not l.strip().startswith('%') and not l.strip().startswith('!')]
        cleaned = '\n'.join(lines)
        if not cleaned.strip():
            continue
        try:
            ast.parse(cleaned)
        except SyntaxError as e:
            issues.append(f'SYNTAX ERROR in cell {i}: {e.msg} (line {e.lineno})')
            break

    # Check for __file__
    if '__file__' in full_code:
        issues.append('Uses __file__ - will fail in Jupyter')

    # Check for hardcoded paths
    if 'C:\\\\' in full_code or 'C:/' in full_code or '/home/' in full_code:
        issues.append('Hardcoded absolute path')

    # Check for src module import
    if re.search(r'^from src\b|^import src\b', full_code, re.MULTILINE):
        issues.append('Imports from src module - likely missing')

    # Check for Flask app.run blocking
    if 'flask' in full_code.lower() and 'app.run' in full_code:
        issues.append('Runs Flask server - blocks notebook')

    # Check for Streamlit
    if 'import streamlit' in full_code:
        issues.append('Uses Streamlit - not allowed in notebooks')

    # Check Ollama models referenced
    for m in re.finditer(r'model\s*=\s*["\']([^"\')]+)["\']', full_code):
        ollama_models.add(m.group(1))

    # Check content quality
    total_code_len = sum(len(c) for c in all_code)
    total_md_len = sum(
        len(''.join(c['source']) if isinstance(c['source'], list) else c['source'])
        for c in md_cells
    )
    if total_code_len < 100 and len(code_cells) > 0:
        issues.append(f'Very little code: {total_code_len} chars')
    if total_md_len < 50 and len(md_cells) > 0:
        issues.append(f'Very little markdown: {total_md_len} chars')

    # Check data handling - does notebook create its own data?
    creates_data = ('sample' in full_code.lower() or 'Document(' in full_code or
                    'example' in full_code.lower() or '"""' in full_code or
                    "'''" in full_code)
    reads_ext_file = bool(re.search(r'pd\.read_csv|pd\.read_json|open\([^)]*\.(?:csv|json|txt|pdf)', full_code))
    if reads_ext_file and not creates_data:
        issues.append('May need external data files')

    if issues:
        issues_all.append((num, folder, issues))

print('=' * 70)
print('DEEP AUDIT: 100 Local AI Project Notebooks')
print('=' * 70)
print(f'Total unique projects: {len(seen)}')
print(f'Projects with issues: {len(issues_all)}')
print(f'Clean projects: {len(seen) - len(issues_all)}')
print()

if issues_all:
    print('--- ISSUES FOUND ---')
    for num, folder, iss in issues_all:
        print(f'\n[{num:3d}] {folder}')
        for i in iss:
            print(f'       > {i}')

print()
print('--- OLLAMA MODELS REFERENCED ---')
for m in sorted(ollama_models):
    print(f'  {m}')

# Check data/dataset situation
print()
print('--- DATASET/DATA ANALYSIS ---')
print('These are LOCAL AI projects (LLM/RAG/Agent). They do NOT use')
print('traditional ML datasets. Data handling:')
print('  - Projects 1-10:  Inline sample text, PDFs, markdown (self-contained)')
print('  - Projects 11-30: Inline documents for RAG vector stores')
print('  - Projects 31-40: Inline data for LangGraph workflows')
print('  - Projects 41-50: Inline tasks for CrewAI agents')
print('  - Projects 51-60: Inline data for tool-using agents')
print('  - Projects 61-70: Inline test cases for eval/observability')
print('  - Projects 71-80: Inline examples for fine-tuning prep')
print('  - Projects 81-90: Inline/sample files for multimodal')
print('  - Projects 91-100: Inline code samples for coding agents')
print()
print('Runtime requirement: Ollama server with pulled models')

