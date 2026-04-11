import re, pathlib
ROOT = pathlib.Path('E:/Github/Machine-Learning-Projects')
EXCLUDE_DIRS = {'venv', '__pycache__', '.git', 'node_modules', 'outputs', '_templates', 'artifacts'}
nbs = []
for p in sorted(ROOT.rglob('pipeline.py')):
    parts_lower = {x.lower() for x in p.parts}
    if parts_lower & {d.lower() for d in EXCLUDE_DIRS}:
        continue
    if any('data analysis' in x.lower() for x in p.parts):
        continue
    safe_name = re.sub(r'[<>:"|?*]', '_', p.parent.name)
    nb_path = p.parent / f'{safe_name}.ipynb'
    if nb_path.exists() and 'classification' in str(nb_path).lower():
        nbs.append(nb_path)
print(f'Total: {len(nbs)}')
for i, nb in enumerate(nbs, 1):
    print(f'[{i:2d}] {nb.relative_to(ROOT)}')
