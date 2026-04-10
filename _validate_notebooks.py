import json, re
from pathlib import Path

EXCLUDE = {'venv', '.venv', 'core', 'data', '__pycache__', '.git', '.github'}

root = Path('.')
pipelines = []
for p in sorted(root.rglob('pipeline.py')):
    parts_lower = {x.lower() for x in p.parts}
    if parts_lower & EXCLUDE:
        continue
    if any('data analysis' in x.lower() for x in p.parts):
        continue
    pipelines.append(p)

issues = []
stats = {'total': 0, 'ok': 0, 'missing': 0, 'json_err': 0, 'struct_err': 0, 'compile_err': 0}

for p in pipelines:
    safe = re.sub(r'[<>:"|?*]', '_', p.parent.name)
    nb_path = p.parent / f"{safe}.ipynb"
    stats['total'] += 1

    if not nb_path.exists():
        issues.append(f"MISSING: {nb_path}")
        stats['missing'] += 1
        continue

    try:
        nb = json.loads(nb_path.read_text('utf-8'))
    except Exception as e:
        issues.append(f"JSON_ERR: {nb_path}: {e}")
        stats['json_err'] += 1
        continue

    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c['cell_type'] == 'code']
    md_cells = [c for c in cells if c['cell_type'] == 'markdown']

    errs = []

    if len(cells) < 5:
        errs.append(f"too few cells: {len(cells)}")

    if cells and cells[0]['cell_type'] != 'markdown':
        errs.append("first cell not markdown")
    elif cells:
        first_src = ''.join(cells[0]['source'])
        if not first_src.startswith('#'):
            errs.append("first cell missing title")

    all_code = ' '.join(''.join(c['source']) for c in code_cells)
    if '__file__' not in all_code:
        errs.append("missing __file__ setup")

    if '%matplotlib inline' not in all_code:
        errs.append("missing %matplotlib inline")

    if code_cells:
        last_code_src = ''.join(code_cells[-1]['source']).strip()
        if 'main()' not in last_code_src:
            errs.append(f"last code cell not main(): '{last_code_src[:50]}'")

    if 'matplotlib.use(' in all_code and 'Agg' in all_code:
        errs.append("still has matplotlib.use(Agg)")

    for i, c in enumerate(code_cells):
        code = ''.join(c['source'])
        clean = '\n'.join(l for l in code.splitlines()
                         if not l.strip().startswith('%') and not l.strip().startswith('!'))
        if not clean.strip():
            continue
        try:
            compile(clean, f'<cell{i}>', 'exec')
        except SyntaxError as e:
            errs.append(f"compile error cell {i}: {e}")

    if 'def ' not in all_code:
        errs.append("no function definitions found")

    if 'def main(' not in all_code:
        errs.append("no main() function defined")

    if errs:
        issues.append("STRUCT: " + str(nb_path) + ": " + "; ".join(errs))
        stats['struct_err'] += 1
    else:
        stats['ok'] += 1

print("=" * 60)
print("NOTEBOOK VALIDATION REPORT")
print("=" * 60)
print(f"Total pipelines:     {stats['total']}")
print(f"OK notebooks:        {stats['ok']}")
print(f"Missing notebooks:   {stats['missing']}")
print(f"JSON errors:         {stats['json_err']}")
print(f"Structure errors:    {stats['struct_err']}")
print(f"Compile errors:      {stats['compile_err']}")
print()

if issues:
    print("ISSUES:")
    for iss in issues:
        print(f"  {iss}")
else:
    print("ALL NOTEBOOKS VALID!")
