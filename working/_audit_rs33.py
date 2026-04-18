#!/usr/bin/env python3
import json

nb_path = r'e:\Github\Machine-Learning-Projects\Computer Vision\Road Segmentation for Autonomous Vehicles\Souce Code\road_segmentation_pipeline.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 70)
print("ROAD SEGMENTATION NOTEBOOK (#33) - CONTENT AUDIT")
print("=" * 70)

# Check first 5 cells
for i, cell in enumerate(nb['cells'][:5]):
    cell_type = cell.get('cell_type', '?')
    if cell_type == 'markdown':
        source = ''.join(cell.get('source', []))[:60]
        print(f"\n[Cell {i}] MARKDOWN: {source}...")
    elif cell_type == 'code':
        source = ''.join(cell.get('source', []))[:60]
        print(f"\n[Cell {i}] CODE: {source}...")

# Check for key phrases in entire notebook
all_text = ''
for cell in nb['cells']:
    source = ''.join(cell.get('source', []))
    all_text += source + '\n'

print("\n" + "=" * 70)
print("CONTENT VERIFICATION")
print("=" * 70)

checks = [
    ('CamVid', 'Dataset source'),
    ('YOLO', 'Model framework'),
    ('mask', 'Segmentation masks'),
    ('kagglehub', 'Kaggle download'),
    ('train', 'Training step'),
    ('mAP', 'Metrics'),
]

for phrase, desc in checks:
    if phrase.lower() in all_text.lower():
        print(f"✓ {desc}: found '{phrase}'")
    else:
        print(f"✗ {desc}: NOT FOUND '{phrase}'")

print("\n" + "=" * 70)
print("✓✓ NOTEBOOK AUDIT COMPLETE ✓✓")
print("=" * 70)
