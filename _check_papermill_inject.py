"""Check if papermill parameter cell was injected, and what the cell_index offset is."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

nb_path = Path('Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS/Fraud Detection - IEEE-CIS.ipynb')
nb = json.loads(nb_path.read_text('utf-8', errors='replace'))
cells = nb['cells']

print(f"Total cells: {len(cells)}")
print()

for i, cell in enumerate(cells[:5]):
    src = ''.join(cell.get('source', []))
    tags = cell.get('metadata', {}).get('tags', [])
    print(f"cells[{i}]: type={cell.get('cell_type')}, tags={tags}")
    print(f"  src preview: {repr(src[:100])}")
    print()

# Look for any injected parameter cell
print("--- Looking for injected parameters tag ---")
for i, cell in enumerate(cells):
    tags = cell.get('metadata', {}).get('tags', [])
    if 'injected-parameters' in tags or 'parameters' in tags:
        print(f"  cells[{i}] has tags: {tags}")

print()
print("--- Total cells by type ---")
from collections import Counter
ctype_counts = Counter(c.get('cell_type') for c in cells)
print(dict(ctype_counts))
