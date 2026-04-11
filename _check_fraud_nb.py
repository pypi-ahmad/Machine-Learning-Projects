"""Check the fraud detection notebook for the IndentationError at line 83."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

nb_dir = Path('Anomaly detection and fraud detection')
fraud_nbs = [
    'Fraud Detection - IEEE-CIS/Fraud Detection - IEEE-CIS.ipynb',
    'Fraud Detection in Financial Transactions/Fraud Detection in Financial Transactions.ipynb',
    'Fraudulent Credit Card Transaction Detection/Fraudulent Credit Card Transaction Detection.ipynb',
    'Insurance Fraud Detection/Insurance Fraud Detection.ipynb',
]

for nb_rel in fraud_nbs:
    nb_path = nb_dir / nb_rel
    if not nb_path.exists():
        print(f"NOT FOUND: {nb_path}")
        continue
    nb = json.loads(nb_path.read_text('utf-8', errors='replace'))
    cells = nb['cells']
    print(f"\n=== {nb_path.parent.name} ===")
    print(f"Total cells: {len(cells)}")
    
    # papermill uses notebook cell index 16 (0-based)
    if len(cells) > 16:
        cell = cells[16]
        src = ''.join(cell.get('source', []))
        lines = src.split('\n')
        print(f"Cell[16] type: {cell.get('cell_type')}, total lines: {len(lines)}")
        if len(lines) > 82:
            for i in range(max(0, 80), min(len(lines), 90)):
                indent = len(lines[i]) - len(lines[i].lstrip()) if lines[i].strip() else 0
                print(f"  L{i+1}: indent={indent} {repr(lines[i][:100])}")
        else:
            print(f"  Cell[16] has only {len(lines)} lines")
    
    # Also check which cells have 83+ lines to see what's at line 83 for each
    print("--- Cells with 83+ lines ---")
    for cidx, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))
        lines = src.split('\n')
        if len(lines) >= 83:
            line83 = lines[82]
            indent = len(line83) - len(line83.lstrip()) if line83.strip() else 0
            print(f"  cells[{cidx}]: L83 indent={indent} {repr(line83[:80])}")
