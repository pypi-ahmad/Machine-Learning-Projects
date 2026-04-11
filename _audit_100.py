"""Audit all 100 notebooks for end-to-end completeness."""
import json, os, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

base = '100_Local_AI_Projects'
results = []
for cat in sorted(os.listdir(base)):
    cat_path = os.path.join(base, cat)
    if not os.path.isdir(cat_path):
        continue
    for proj in sorted(os.listdir(cat_path)):
        if proj[0].isdigit() and '__' in proj[:4]:
            parts = proj.split('__', 1)
            if parts[0].isdigit() and int(parts[0]) <= 10:
                continue
        nb = os.path.join(cat_path, proj, 'notebook.ipynb')
        if not os.path.exists(nb):
            continue
        with open(nb, encoding='utf-8') as f:
            data = json.load(f)
        cells = data['cells']
        code_cells = [c for c in cells if c['cell_type'] == 'code']
        md_cells = [c for c in cells if c['cell_type'] == 'markdown']

        all_code = ' '.join(''.join(c['source']) for c in code_cells)
        has_import = 'import ' in all_code
        has_ollama = 'ollama' in all_code.lower() or 'ChatOllama' in all_code
        has_print = 'print(' in all_code
        has_langchain = 'langchain' in all_code
        first_md = ''.join(md_cells[0]['source']) if md_cells else ''
        has_title = first_md.startswith('#')
        last_md = ''.join(md_cells[-1]['source']) if md_cells else ''
        has_wrapup = any(w in last_md.lower() for w in ['learned', 'summary', 'conclusion', 'takeaway'])
        empty_code = sum(1 for c in code_cells if len(''.join(c['source']).strip()) == 0)

        issues = []
        if not has_import:
            issues.append('no-imports')
        if not has_ollama and not has_langchain:
            issues.append('no-llm-setup')
        if not has_print:
            issues.append('no-output')
        if not has_title:
            issues.append('no-title')
        if not has_wrapup:
            issues.append('no-wrapup')
        if empty_code > 0:
            issues.append(f'{empty_code}-empty-cells')
        if len(code_cells) < 3:
            issues.append('too-few-code-cells')

        num = proj.split('_')[0]
        results.append({
            'num': num, 'name': proj[:45], 'cells': len(cells),
            'code': len(code_cells), 'md': len(md_cells),
            'issues': issues,
        })

total = len(results)
clean = sum(1 for r in results if not r['issues'])
print(f'AUDIT: {total} notebooks')
print(f'Clean (no issues): {clean}/{total}')
print(f'With issues: {total - clean}/{total}')
print()
if total - clean > 0:
    print('ISSUES:')
    for r in results:
        if r['issues']:
            name = r['name'][:40].ljust(40)
            print(f"  #{r['num']} {name} cells={r['cells']:>2} issues={r['issues']}")
print()
print('CELL COUNT DISTRIBUTION:')
counts = Counter(r['cells'] for r in results)
for c in sorted(counts):
    bar = '#' * counts[c]
    print(f'  {c:>2} cells: {bar} ({counts[c]})')
