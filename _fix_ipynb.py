"""Convert percent-format .ipynb to proper JSON .ipynb — delete after use."""
import json, re

path = r'E:\Github\Machine-Learning-Projects\Classification\Medical Appointment No-Show Prediction\Medical Appointment No-Show Prediction.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    raw = f.read()

# Parse percent-format into cells
blocks = re.split(r'^#%%\s*', raw, flags=re.MULTILINE)
cells = []
for block in blocks:
    block = block.strip()
    if not block:
        continue
    if block.startswith('md\n') or block.startswith('md\r\n'):
        # Markdown cell - remove 'md\n' prefix
        content = re.sub(r'^md\s*\n', '', block, count=1)
        source_lines = content.split('\n')
        source = [line + '\n' for line in source_lines[:-1]] + [source_lines[-1]] if source_lines else []
        cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': source
        })
    else:
        # Code cell
        source_lines = block.split('\n')
        source = [line + '\n' for line in source_lines[:-1]] + [source_lines[-1]] if source_lines else []
        cells.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': source
        })

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
with open(path, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
ct = [c['cell_type'] for c in nb2['cells']]
print(f'Converted to valid .ipynb JSON')
print(f'Total cells: {len(nb2["cells"])}')
print(f'Markdown: {ct.count("markdown")}')
print(f'Code: {ct.count("code")}')

# Print all headings
for c in nb2['cells']:
    if c['cell_type'] == 'markdown':
        src = ''.join(c['source'])
        for line in src.split('\n'):
            if line.startswith('#'):
                print(f'  {line.strip()}')

