import json, ast, re
from pathlib import Path

EXCLUDE = {'venv', '.venv', 'core', 'data', '__pycache__', '.git', '.github'}

root = Path('.')

# Count pipeline.py files  
pipelines = []
for p in sorted(root.rglob('pipeline.py')):
    parts_lower = {x.lower() for x in p.parts}
    if parts_lower & EXCLUDE:
        continue
    if any('data analysis' in x.lower() for x in p.parts):
        continue
    pipelines.append(p)

# Count generated notebooks
notebooks = []
for p in sorted(root.rglob('*.ipynb')):
    parts_lower = {x.lower() for x in p.parts}
    if parts_lower & EXCLUDE:
        continue
    if any('data analysis' in x.lower() for x in p.parts):
        continue
    if (p.parent / 'pipeline.py').exists():
        notebooks.append(p)

# Find pipelines without a matching notebook
missing = []
for p in pipelines:
    safe = re.sub(r'[<>:"|?*]', '_', p.parent.name)
    nb_path = p.parent / f"{safe}.ipynb"
    if not nb_path.exists():
        missing.append(p.as_posix())

print(f"PIPELINE_COUNT={len(pipelines)}")
print(f"NOTEBOOK_COUNT={len(notebooks)}")
print(f"MISSING_COUNT={len(missing)}")
if missing:
    print("MISSING:")
    for m in missing:
        print(f"  {m}")

# Validate a sample of notebooks
errors = []
sample_details = []
for p in pipelines[:5]:
    safe = re.sub(r'[<>:"|?*]', '_', p.parent.name)
    nb_path = p.parent / f"{safe}.ipynb"
    if not nb_path.exists():
        continue
    nb = json.loads(nb_path.read_text('utf-8'))
    
    n_cells = len(nb['cells'])
    n_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    n_md = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
    
    last_code = [c for c in nb['cells'] if c['cell_type'] == 'code'][-1]
    last_code_src = ''.join(last_code['source']).strip()
    
    has_file_setup = any('__file__' in ''.join(c['source']) for c in nb['cells'])
    has_matplotlib = any('%matplotlib inline' in ''.join(c['source']) for c in nb['cells'])
    
    cell_errors = []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        code = ''.join(c['source'])
        clean = '\n'.join(l for l in code.splitlines() if not l.strip().startswith('%') and not l.strip().startswith('!'))
        if not clean.strip():
            continue
        try:
            compile(clean, f'<{nb_path}:cell{i}>', 'exec')
        except SyntaxError as e:
            cell_errors.append(f"cell {i}: {e}")
    
    sample_details.append({
        'path': nb_path.as_posix(),
        'total_cells': n_cells,
        'code_cells': n_code,
        'markdown_cells': n_md,
        'last_code': last_code_src[:60],
        'has_file_setup': has_file_setup,
        'has_matplotlib': has_matplotlib,
        'compile_errors': cell_errors
    })
    if cell_errors:
        errors.extend(cell_errors)

print(f"\nSAMPLE_DETAILS:")
for d in sample_details:
    print(f"  {d['path']}:")
    print(f"    cells={d['total_cells']} (code={d['code_cells']}, md={d['markdown_cells']})")
    print(f"    last_code='{d['last_code']}'")
    print(f"    __file__={d['has_file_setup']}, matplotlib={d['has_matplotlib']}")
    if d['compile_errors']:
        print(f"    ERRORS: {d['compile_errors']}")

# Full compile validation of ALL notebooks
print("\nFull compile validation of all notebooks...")
full_errors = []
for p in pipelines:
    safe = re.sub(r'[<>:"|?*]', '_', p.parent.name)
    nb_path = p.parent / f"{safe}.ipynb"
    if not nb_path.exists():
        continue
    try:
        nb = json.loads(nb_path.read_text('utf-8'))
    except json.JSONDecodeError as e:
        full_errors.append(f"{nb_path}: JSON error: {e}")
        continue
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        code = ''.join(c['source'])
        clean = '\n'.join(l for l in code.splitlines() if not l.strip().startswith('%') and not l.strip().startswith('!'))
        if not clean.strip():
            continue
        try:
            compile(clean, f'<{nb_path}:cell{i}>', 'exec')
        except SyntaxError as e:
            full_errors.append(f"{nb_path} cell {i}: {e}")

print(f"FULL_COMPILE_ERRORS={len(full_errors)}")
if full_errors:
    for e in full_errors[:20]:
        print(f"  {e}")
