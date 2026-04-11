"""Check the output of the main() cell in _out_test.ipynb."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

nb_path = Path('Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS/_out_test.ipynb')
nb = json.loads(nb_path.read_text('utf-8', errors='replace'))
cells = nb['cells']

# Find the main() cell (last code cell)
code_cells = [c for c in cells if c.get('cell_type') == 'code']
print(f"Total code cells: {len(code_cells)}")

for i, cell in enumerate(code_cells):
    src = ''.join(cell.get('source', []))
    outputs = cell.get('outputs', [])
    errors = [o for o in outputs if o.get('output_type') == 'error']
    has_output = len(outputs) > 0
    print(f"\n--- Code cell {i+1} (ec={cell.get('execution_count')}) ---")
    print(f"  src: {repr(src[:60])}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e.get('ename')}: {e.get('evalue')[:200]}")
    elif has_output:
        text_outputs = [o for o in outputs if o.get('output_type') in ('stream', 'execute_result')]
        for o in text_outputs[:2]:
            text = ''.join(o.get('text', []))
            print(f"  output: {repr(text[:200])}")
    else:
        print("  (no output)")
