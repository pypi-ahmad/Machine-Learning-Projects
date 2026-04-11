"""Check the full context around line 83 in the fraud detection notebooks."""
import json
import ast
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

nb_path = Path('Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS/Fraud Detection - IEEE-CIS.ipynb')
nb = json.loads(nb_path.read_text('utf-8', errors='replace'))
cells = nb['cells']

cell = cells[16]
src = ''.join(cell.get('source', []))
lines = src.split('\n')
print(f"Cell[16] total lines: {len(lines)}")
print("\n--- Lines 60-95 ---")
for i in range(59, min(len(lines), 95)):
    prefix = ">>>" if i == 82 else "   "
    indent = len(lines[i]) - len(lines[i].lstrip()) if lines[i].strip() else 0
    print(f"{prefix} L{i+1:3d} (indent={indent:2d}): {repr(lines[i][:100])}")

print("\n--- AST parse check ---")
try:
    ast.parse(src)
    print("AST parse: OK")
except SyntaxError as e:
    print(f"AST parse ERROR: {e}")

# Also check for mixed tabs/spaces
print("\n--- Tab check ---")
for i, line in enumerate(lines):
    if '\t' in line:
        print(f"  Tab found at L{i+1}: {repr(line[:80])}")
