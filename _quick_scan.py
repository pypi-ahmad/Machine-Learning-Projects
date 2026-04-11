"""Quick scan of notebooks in a given category for common issues."""
import json, re
from pathlib import Path
import sys

ROOT = Path(".")
category = sys.argv[1] if len(sys.argv) > 1 else "Regression"

issues_log = []
total = 0
for f in sorted((ROOT / category).rglob("*.ipynb")):
    if any(p in str(f) for p in ("venv", ".venv", ".git", "_checkpoints")):
        continue
    total += 1
    try:
        nb = json.loads(f.read_text("utf-8"))
        code = "\n".join("".join(c.get("source", "")) for c in nb["cells"] if c["cell_type"] == "code")
        issues = []
        if "../input/" in code:
            issues.append("kaggle_path")
        if "sns.factorplot" in code or "seaborn.factorplot" in code:
            issues.append("factorplot")
        if re.search(r"[A-Z]:\\\\", code):
            issues.append("abs_path")
        if "plt." in code and "import matplotlib" not in code:
            issues.append("no_plt")
        if "from __future__" in code:
            issues.append("py2_future")
        if re.search(r"pd.read_excel|pd.read_csv", code) and not re.search(r"os.path|Path\(", code):
            # Check if the data file exists near the notebook
            for m in re.finditer(r'(?:read_csv|read_excel)\(["\']([^"\']+)["\']', code):
                fname = m.group(1)
                nb_dir = f.parent
                if not (nb_dir / fname).exists() and not fname.startswith(("http", "..", "data")):
                    issues.append(f"missing_data:{fname[:30]}")
        if issues:
            issues_log.append((str(f.relative_to(ROOT)), issues))
    except Exception as e:
        issues_log.append((str(f.relative_to(ROOT)), [f"parse_error:{e}"]))

print(f"Category: {category} | Total notebooks: {total} | With issues: {len(issues_log)}")
for name, iss in issues_log[:25]:
    print(f"  {name}")
    print(f"    issues: {iss}")
