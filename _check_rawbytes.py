"""Check the raw bytes of cell[16] lines 77-85 to look for hidden characters."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

nb_path = Path('Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS/Fraud Detection - IEEE-CIS.ipynb')
nb = json.loads(nb_path.read_text('utf-8', errors='replace'))
cells = nb['cells']
cell = cells[16]
src_lines = cell.get('source', [])

print(f"Cell[16] source has {len(src_lines)} elements in JSON array")
print("\n--- Raw JSON source lines 76-86 (0-indexed) ---")
for i, line in enumerate(src_lines):
    if 75 <= i <= 87:
        # Show raw bytes
        print(f"src_lines[{i}]: {repr(line)}")

# Also concatenate and show lines 76-86 
src = ''.join(src_lines)
lines = src.split('\n')
print(f"\n--- Lines 76-86 after join and split ---")
for i in range(75, min(len(lines), 88)):
    raw = lines[i]
    indent = len(raw) - len(raw.lstrip()) if raw.strip() else 0
    # Check for non-space whitespace
    leading = raw[:indent] if indent > 0 else ''
    has_tab = '\t' in leading
    has_crlf = '\r' in raw
    print(f"  L{i+1}: indent={indent}, tab={has_tab}, cr={has_crlf}, text={repr(raw[:90])}")

# Also verify by compiling
print("\n--- compile() check for cell[16] ---")
try:
    code = compile(src, '<cell16>', 'exec')
    print("  compile: OK")
except SyntaxError as e:
    print(f"  compile ERROR: {e}")
    print(f"  lineno: {e.lineno}, text: {repr(e.text)}")
