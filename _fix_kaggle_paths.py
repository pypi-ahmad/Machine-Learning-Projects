"""
Fix Kaggle-style ../input/ paths in notebooks to use local data/ paths.
Also fix notebooks with direct HTTP data loading where local copies exist.

Usage: venv\Scripts\python _fix_kaggle_paths.py
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
DATA_DIR  = REPO_ROOT / "data"

# Build a lookup: filename_lower -> actual path
data_files = {}
if DATA_DIR.exists():
    for f in DATA_DIR.rglob("*"):
        if f.is_file():
            data_files[f.name.lower()] = f

def find_local_data(filename: str, nb_path: Path) -> str | None:
    """Try to find a local equivalent for a data file."""
    # Check notebook's own directory first
    local = nb_path.parent / filename
    if local.exists():
        return filename  # relative to notebook dir, just use filename

    # Check sibling data/ folders
    for parent in [nb_path.parent.parent, nb_path.parent.parent.parent]:
        candidate = parent / "data" / filename
        if candidate.exists():
            rel = candidate.relative_to(nb_path.parent)
            return str(rel).replace("\\", "/")

    # Check global data/ directory
    lower = filename.lower()
    if lower in data_files:
        import os
        rel = os.path.relpath(data_files[lower], nb_path.parent)
        return rel.replace("\\", "/")

    return None


def fix_notebook(nb_path: Path) -> int:
    """Replace ../input/dataset/file.csv with local equivalent. Returns number of replacements."""
    try:
        text = nb_path.read_text("utf-8")
    except Exception:
        return 0

    nb = json.loads(text)
    changed = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src_lines = cell.get("source", [])
        if isinstance(src_lines, list):
            src = "".join(src_lines)
        else:
            src = src_lines

        # Find all ../input/dataset/filename references
        new_src = src
        for m in re.finditer(r'\.\./input/[^/"\']+/([^"\')\s]+)', src):
            filename = m.group(1).split("/")[-1]  # take last component
            local = find_local_data(filename, nb_path)
            if local:
                new_src = new_src.replace(m.group(0), local)
                changed += 1
            else:
                # Replace with data/ subdirectory search
                new_src = new_src.replace(
                    m.group(0),
                    f"../../data/{filename}",    # standard project data layout
                )
                changed += 1

        if new_src != src:
            if isinstance(src_lines, list):
                cell["source"] = [new_src]
            else:
                cell["source"] = new_src

    if changed:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), "utf-8")

    return changed


def main() -> None:
    total_nb = 0
    total_fixes = 0
    skip = {"venv", ".venv", ".git", "__pycache__", ".ipynb_checkpoints"}

    for nb_path in sorted(REPO_ROOT.rglob("*.ipynb")):
        if any(p in nb_path.parts for p in skip):
            continue
        try:
            nb_text = nb_path.read_text("utf-8")
        except Exception:
            continue
        if "../input/" not in nb_text:
            continue

        n = fix_notebook(nb_path)
        total_nb += 1
        total_fixes += n
        rel = nb_path.relative_to(REPO_ROOT)
        print(f"Fixed {rel}: {n} replacements")

    print(f"\nTotal notebooks patched: {total_nb}, total replacements: {total_fixes}")


if __name__ == "__main__":
    main()
