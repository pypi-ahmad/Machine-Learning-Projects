"""Check cells[15] (papermill 1-based Cell 16) for IndentationError."""
import json
import ast
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

nb_path = Path('Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS/Fraud Detection - IEEE-CIS.ipynb')
nb = json.loads(nb_path.read_text('utf-8', errors='replace'))
cells = nb['cells']

# papermill Cell 16 = cells[15] (1-based) OR cells[16] (they may use execution_count)
# Let's check both cells[15] and cells[16]
for idx in [14, 15, 16, 17]:
    cell = cells[idx]
    src = ''.join(cell.get('source', []))
    lines = src.split('\n')
    print(f"\n=== cells[{idx}] type={cell.get('cell_type')}, lines={len(lines)} ===")
    if len(lines) >= 83:
        try:
            ast.parse(src)
            print("  AST: OK")
        except SyntaxError as e:
            print(f"  AST ERROR: {e}")
        # Check lines 78-88
        for i in range(77, min(len(lines), 90)):
            prefix = ">>>" if i == 82 else "   "
            indent = len(lines[i]) - len(lines[i].lstrip()) if lines[i].strip() else 0
            print(f"  {prefix} L{i+1:3d} (indent={indent:2d}): {repr(lines[i][:100])}")
    else:
        print(f"  Only {len(lines)} lines, first 3: {[repr(l[:60]) for l in lines[:3]]}")

# Also, show papermill execution order (execution_count)
print("\n=== Cell execution counts ===")
for i, c in enumerate(cells):
    ec = c.get('execution_count', 'N/A')
    ctype = c.get('cell_type', '?')
    src = ''.join(c.get('source', []))
    print(f"  cells[{i:2d}] type={ctype}, exec_count={ec}, lines={len(src.split(chr(10)))}, preview={repr(src[:50])}")
