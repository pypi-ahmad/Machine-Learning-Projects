"""Fix kaggle download cells that were broken by the multi-line subprocess patch.

Replaces the broken download blocks with a proper guarded version.
"""

import json, glob, os, re, sys


def fix_notebook(nb_path):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))

        if "subprocess" not in src:
            continue
        if "kaggle" not in src.lower():
            continue

        # Extract key from DATA_DIR line
        m = re.search(r"'data',\s*'([^']+)'", src)
        if not m:
            continue
        key = m.group(1)

        # Extract kaggle dataset ID from -d argument
        m2 = re.search(r"-d['\"],\s*['\"]([^'\"]+)['\"]", src)
        if not m2:
            continue
        dataset_id = m2.group(1)

        # Check for --unzip flag
        has_unzip = "--unzip" in src

        # Determine the raw/sub folder
        m3 = re.search(r"DATA_DIR\s*=\s*os\.path\.join\(REPO_DIR,\s*(.+?)\)", src)
        data_dir_expr = m3.group(0) if m3 else f"os.path.join(REPO_DIR, 'data', '{key}', 'raw')"

        # Build clean replacement
        unzip_flag = ", '--unzip'" if has_unzip else ""
        new_src = f"""import os, subprocess, sys, glob

{data_dir_expr}
os.makedirs(DATA_DIR, exist_ok=True)

try:
    import kaggle
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'])

# Skip download if data already present (avoids OOM on large zips)
_existing = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True)
_existing += glob.glob(os.path.join(DATA_DIR, '**', '*.jpg'), recursive=True)
_existing += glob.glob(os.path.join(DATA_DIR, '**', '*.jpeg'), recursive=True)
if len(_existing) < 10:
    subprocess.run(['kaggle', 'datasets', 'download', '-d', '{dataset_id}',
                    '-p', DATA_DIR{unzip_flag}], check=False)
else:
    print(f'Dataset already present ({{len(_existing)}} images) — skipping download')
print(f'Dataset at {{DATA_DIR}}')"""

        src_lines = new_src.split("\n")
        cell["source"] = [l + "\n" for l in src_lines[:-1]] + [src_lines[-1]]
        changed = True

    if changed:
        with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
    return changed


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    nbs = sorted(glob.glob(os.path.join(root, "**", "*_pipeline.ipynb"), recursive=True))
    fixed = 0
    for nb_path in nbs:
        rel = os.path.relpath(nb_path, root)
        try:
            if fix_notebook(nb_path):
                print(f"  ✓ {rel}")
                fixed += 1
        except Exception as e:
            print(f"  ✗ {rel}: {e}")
    print(f"\nFixed {fixed} notebooks")


if __name__ == "__main__":
    main()
