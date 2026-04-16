"""
Remove injected MLOps bootstrap cells from 100_Local_AI_Projects notebooks.
These cells contain Path(__file__) which crashes in notebook context.
This is a one-time repair script - delete after use.
"""
import json
import os

base = r'E:\Github\Machine-Learning-Projects\100_Local_AI_Projects'

fixed = []
skipped = []

for root, dirs, files in os.walk(base):
    for fname in files:
        if fname.endswith('.ipynb'):
            fpath = os.path.join(root, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            cells = nb.get('cells', [])
            if not cells:
                skipped.append(fpath)
                continue

            # Check if first cell is the injected bootstrap
            first = cells[0]
            tags = first.get('metadata', {}).get('tags', [])
            if 'injected-mlops-bootstrap' not in tags:
                skipped.append(fpath)
                continue

            # Remove the bootstrap cell
            nb['cells'] = cells[1:]

            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
                f.write('\n')

            rel = os.path.relpath(fpath, base)
            fixed.append(rel)
            print(f'FIXED: {rel}')

print(f'\nDone. Fixed {len(fixed)} notebooks, skipped {len(skipped)}.')

