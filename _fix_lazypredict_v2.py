"""
Fix LazyPredict MLflow autologging across ALL notebooks.
Uses regex to find and wrap any LazyClassifier/LazyRegressor call 
with mlflow autolog disable + small sample cap.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent

# Pattern to find LazyPredict fit call in notebook cells or pipeline.py
PATTERNS = [
    # Pattern: lazy_models, _ = lazy.fit(X_anything, X_anything, y_anything, y_anything)
    (
        re.compile(
            r'(\s+)(lazy(?:_models)?,\s*_\s*=\s*lazy\.fit\('
            r'(?P<xt>\w+),\s*(?P<xe>\w+),\s*(?P<yt>\w+),\s*(?P<ye>\w+)\))',
            re.MULTILINE
        ),
        True  # needs fixing
    ),
]

WRAPPER_TEMPLATE = """\
{indent}# Disable MLflow sklearn autologging to avoid logging 25+ LazyPredict models
{indent}try:
{indent}    import mlflow as _mlflow; _mlflow.sklearn.autolog(disable=True)
{indent}except Exception:
{indent}    pass
{indent}_lp_max = 5000
{indent}_{xt}_lp = {xt}.iloc[:_lp_max] if hasattr({xt}, 'iloc') and len({xt}) > _lp_max else {xt}
{indent}_{xe}_lp = {xe}.iloc[:_lp_max] if hasattr({xe}, 'iloc') and len({xe}) > _lp_max else {xe}
{indent}_{yt}_lp = {yt}.iloc[:_lp_max] if hasattr({yt}, 'iloc') and len({yt}) > _lp_max else {yt}
{indent}_{ye}_lp = {ye}.iloc[:_lp_max] if hasattr({ye}, 'iloc') and len({ye}) > _lp_max else {ye}
{indent}lazy_models, _ = lazy.fit(_{xt}_lp, _{xe}_lp, _{yt}_lp, _{ye}_lp)
{indent}try:
{indent}    import mlflow as _mlflow; _mlflow.sklearn.autolog(disable=False)
{indent}except Exception:
{indent}    pass"""

ALREADY_FIXED = "Disable MLflow sklearn autologging"

fixed_files = 0
fixed_cells = 0


def fix_source(src: str) -> tuple[str, int]:
    """Fix all LazyPredict fit calls in a source string. Returns (new_src, count)."""
    if ALREADY_FIXED in src:
        return src, 0

    pattern = re.compile(
        r'(?P<indent>[ \t]*)lazy(?:_models)?\s*,\s*_\s*=\s*lazy\.fit\('
        r'(?P<xt>\w+),\s*(?P<xe>\w+),\s*(?P<yt>\w+),\s*(?P<ye>\w+)\)',
        re.MULTILINE
    )

    count = 0
    def replacer(m):
        nonlocal count
        indent = m.group("indent")
        xt = m.group("xt")
        xe = m.group("xe")
        yt = m.group("yt")
        ye = m.group("ye")
        count += 1
        return WRAPPER_TEMPLATE.format(indent=indent, xt=xt, xe=xe, yt=yt, ye=ye)

    new_src = pattern.sub(replacer, src)
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
        if "LazyClassifier" not in src and "LazyRegressor" not in src:
            continue
        new_src, n = fix_source(src)
        if n > 0:
            lines = new_src.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
            changed = True
            global fixed_cells
            fixed_cells += n

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return True
    return False


def process_py(path: Path) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if "LazyClassifier" not in src and "LazyRegressor" not in src:
        return False
    new_src, n = fix_source(src)
    if n > 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_src)
        fixed_cells + n
        return True
    return False


for nb_path in sorted(ROOT.rglob("*.ipynb")):
    skip = {"venv", ".venv", "__pycache__", ".git", "node_modules"}
    if any(s in nb_path.parts for s in skip):
        continue
    if process_notebook(nb_path):
        print(f"fixed nb: {nb_path.relative_to(ROOT)}")
        fixed_files += 1

for py_path in sorted(ROOT.rglob("pipeline.py")):
    skip = {"venv", ".venv", "__pycache__", ".git"}
    if any(s in py_path.parts for s in skip):
        continue
    if process_py(py_path):
        print(f"fixed py: {py_path.relative_to(ROOT)}")
        fixed_files += 1

print(f"\nTotal files fixed: {fixed_files}, LazyPredict call sites fixed: {fixed_cells}")
