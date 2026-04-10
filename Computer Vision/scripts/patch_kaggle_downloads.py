"""Add skip-if-data-exists guard to kaggle download cells in all notebooks.

This prevents re-downloading large datasets that are already cached,
which can cause OOM kernel deaths (especially with --unzip on multi-GB files).

Idempotent — only modifies cells that contain raw subprocess.run kaggle calls
without the guard.
"""

import json, glob, os, re, sys

GUARD = "already present"  # marker text

def patch_notebook(nb_path):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))

        # Only target cells with raw kaggle subprocess download
        if "subprocess.run" not in src:
            continue
        if "kaggle" not in src.lower():
            continue
        if "download" not in src.lower():
            continue
        if GUARD in src:
            continue  # already patched

        # Extract the dataset key from the path
        m = re.search(r"'data',\s*'([^']+)'", src)
        if not m:
            continue
        key = m.group(1)

        # Build patched source
        lines = src.split("\n")
        new_lines = []
        imports_done = False
        for line in lines:
            if line.startswith("import ") and not imports_done:
                # Ensure glob is imported
                if "glob" not in line:
                    line = line.rstrip() + ", glob"
                imports_done = True
                new_lines.append(line)
            elif "subprocess.run" in line:
                # Indent the subprocess call and wrap it
                indent = len(line) - len(line.lstrip())
                pad = " " * indent
                # Insert guard before subprocess.run
                new_lines.append(f"{pad}# Skip download if data already present (avoids OOM on large zips)")
                new_lines.append(f"{pad}_existing = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True)")
                new_lines.append(f"{pad}_existing += glob.glob(os.path.join(DATA_DIR, '**', '*.jpg'), recursive=True)")
                new_lines.append(f"{pad}_existing += glob.glob(os.path.join(DATA_DIR, '**', '*.jpeg'), recursive=True)")
                new_lines.append(f"{pad}if len(_existing) < 10:")
                new_lines.append(f"{pad}    " + line.lstrip())
                new_lines.append(f"{pad}else:")
                new_lines.append(f"{pad}    print(f'Dataset already present ({{len(_existing)}} images) — skipping download')")
            else:
                new_lines.append(line)

        new_src = "\n".join(new_lines)
        # Rebuild source array
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
    patched = 0
    for nb_path in nbs:
        rel = os.path.relpath(nb_path, root)
        try:
            if patch_notebook(nb_path):
                print(f"  ✓ {rel}")
                patched += 1
        except Exception as e:
            print(f"  ✗ {rel}: {e}")
    print(f"\nPatched {patched} notebooks")


if __name__ == "__main__":
    main()
