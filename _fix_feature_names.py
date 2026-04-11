"""
Fix LightGBM 'special JSON characters in feature name' errors.

The fix: add column name sanitization after X = df.drop(columns=[TARGET]) 
in preprocess functions across all pipeline.py and notebook cells.

Also handles: brackets, colons, commas from feature names like 'concave points_mean'.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent

# Pattern to find and fix column sanitization
# We look for lines that set X from df and add column sanitization after
SANITIZE_LINE = "# Sanitize column names for LightGBM/XGBoost compatibility"
SANITIZE_CODE = """    # Sanitize column names for LightGBM/XGBoost compatibility
    X.columns = [str(c).replace(' ', '_').replace('[', '_').replace(']', '_')
                 .replace(',', '_').replace(':', '_').replace('{', '_')
                 .replace('}', '_').replace('<', '_').replace('>', '_')
                 .replace('\"', '_') for c in X.columns]"""

# Pattern: X = df.drop(columns=[TARGET]) or X = df.drop(columns=[target])
DROP_TARGET_PATTERN = re.compile(
    r'([ \t]*X\s*=\s*df\.drop\(columns=\[(?:TARGET|target|"[^"]+"|\'[^\']+\')\]\))',
    re.MULTILINE
)

ALREADY_FIXED = SANITIZE_LINE


def fix_source(src: str) -> tuple[str, int]:
    if ALREADY_FIXED in src:
        return src, 0
    
    count = 0
    
    def replacer(m):
        nonlocal count
        line = m.group(1)
        count += 1
        return f"{line}\n{SANITIZE_CODE}"
    
    new_src = DROP_TARGET_PATTERN.sub(replacer, src)
    return new_src, count


def process_notebook(path: Path) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except Exception:
            return False

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        new_src, n = fix_source(src)
        if n > 0:
            lines = new_src.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return True
    return False


def process_py(path: Path) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    new_src, n = fix_source(src)
    if n > 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_src)
        return True
    return False


fixed = 0
skip_dirs = {"venv", ".venv", "__pycache__", ".git"}

for nb_path in sorted(ROOT.rglob("*.ipynb")):
    if any(s in nb_path.parts for s in skip_dirs):
        continue
    if process_notebook(nb_path):
        print(f"fixed nb: {nb_path.relative_to(ROOT)}")
        fixed += 1

for py_path in sorted(ROOT.rglob("pipeline.py")):
    if any(s in py_path.parts for s in skip_dirs):
        continue
    if process_py(py_path):
        print(f"fixed py: {py_path.relative_to(ROOT)}")
        fixed += 1

print(f"\nTotal files fixed: {fixed}")
