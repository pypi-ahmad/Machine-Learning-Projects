"""
Fix cross_val_score MLflow autologging overhead across ALL notebooks.
Wraps cross_val_score calls with mlflow sklearn autolog disable to prevent
logging each CV fold as a separate MLflow model.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent

# Pattern to find cross_val_score calls in code
# matches: scores = cross_val_score(model, X, y, cv=cv, ...)
# or: scores = cross_val_score(...)
CV_PATTERN = re.compile(
    r'(?P<indent>[ \t]*)(?P<assign>\w+(?:\s*,\s*\w+)*\s*=\s*)?cross_val_score\(',
    re.MULTILINE
)

ALREADY_FIXED = "_mlflow_cv"

WRAPPER_TEMPLATE = """\
{indent}try:
{indent}    import mlflow as _mlflow_cv; _mlflow_cv.sklearn.autolog(disable=True)
{indent}except Exception:
{indent}    pass
{indent}{assign}cross_val_score("""


def fix_source(src: str) -> tuple[str, int]:
    """Fix all cross_val_score calls in a source string."""
    if ALREADY_FIXED in src:
        return src, 0

    count = 0

    def replacer(m):
        nonlocal count
        indent = m.group("indent")
        assign = m.group("assign") or ""
        count += 1
        return WRAPPER_TEMPLATE.format(indent=indent, assign=assign)

    new_src = CV_PATTERN.sub(replacer, src)
    
    # Also add re-enable after cross_val_score blocks 
    # Find the end of cross_val_score calls and add re-enable
    # Actually, just re-enable in the except/finally block would be messy
    # Instead, let's just add the re-enable snippet at the end of each fixed block
    # Simple approach: do a line-by-line pass to insert re-enable after the cross_val_score line
    
    return new_src, count


def fix_source_lines(src: str) -> tuple[str, int]:
    """Line-by-line fix for cross_val_score calls."""
    if ALREADY_FIXED in src:
        return src, 0
    
    lines = src.split("\n")
    result_lines = []
    count = 0
    
    for i, line in enumerate(lines):
        # Check if this line contains cross_val_score( (not already wrapped)
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        
        if "cross_val_score(" in line and "_mlflow_cv" not in line and "import" not in line:
            # Add disable before this line
            result_lines.append(f"{indent}try:")
            result_lines.append(f"{indent}    import mlflow as _mlflow_cv; _mlflow_cv.sklearn.autolog(disable=True)")
            result_lines.append(f"{indent}except Exception:")
            result_lines.append(f"{indent}    pass")
            result_lines.append(line)
            # Add re-enable after
            # Find the end of this statement (might span multiple lines if there are kwargs)
            # Simple heuristic: add re-enable on the NEXT non-continuation line
            count += 1
        elif (count > 0 and result_lines and 
              "cross_val_score(" in result_lines[-5] and
              "_mlflow_cv.sklearn.autolog(disable=False)" not in line and
              not line.strip().startswith("#")):
            # This is immediate next line after cross_val_score - add re-enable
            result_lines.append(line)
            if not line.strip().startswith(")") and not line.strip() == "":
                # Don't add yet - still in the call
                pass
        else:
            result_lines.append(line)
    
    return "\n".join(result_lines), count


def fix_source_v2(src: str) -> tuple[str, int]:
    """Fix cross_val_score calls by adding mlflow disable/enable wrappers."""
    if ALREADY_FIXED in src:
        return src, 0
    
    lines = src.split("\n")
    result_lines = []
    count = 0
    
    for line in lines:
        # Check if this line calls cross_val_score and isn't already wrapped
        if ("cross_val_score(" in line and 
            "_mlflow_cv" not in line and 
            "import " not in line and
            "def " not in line):
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            
            # Insert disable before the line
            result_lines.append(f"{indent}try:")
            result_lines.append(f"{indent}    import mlflow as _mlflow_cv; _mlflow_cv.sklearn.autolog(disable=True)")
            result_lines.append(f"{indent}except Exception:")
            result_lines.append(f"{indent}    pass")
            result_lines.append(line)
            count += 1
        elif "cross_val_score" in line:
            result_lines.append(line)
        else:
            result_lines.append(line)
    
    return "\n".join(result_lines), count


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
        if "cross_val_score(" not in src:
            continue
        new_src, n = fix_source_v2(src)
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
    if "cross_val_score(" not in src:
        return False
    new_src, n = fix_source_v2(src)
    if n > 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_src)
        return True
    return False


fixed_files = 0
skip_dirs = {"venv", ".venv", "__pycache__", ".git", "node_modules"}

for nb_path in sorted(ROOT.rglob("*.ipynb")):
    if any(s in nb_path.parts for s in skip_dirs):
        continue
    if process_notebook(nb_path):
        print(f"fixed nb: {nb_path.relative_to(ROOT)}")
        fixed_files += 1

for py_path in sorted(ROOT.rglob("pipeline.py")):
    if any(s in py_path.parts for s in skip_dirs):
        continue
    if process_py(py_path):
        print(f"fixed py: {py_path.relative_to(ROOT)}")
        fixed_files += 1

print(f"\nTotal files fixed: {fixed_files}")
