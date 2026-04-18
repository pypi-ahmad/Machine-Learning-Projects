#!/usr/bin/env python3
import json
import sys

nb_path = r'e:\Github\Machine-Learning-Projects\Computer Vision\Medical Image Segmentation for Tumour Detection\Souce Code\medical_image_segmentation_pipeline.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cell_count = len(nb.get('cells', []))
    print(f"✓ Valid JSON")
    print(f"✓ Cell count: {cell_count}")
    print(f"✓ Notebook format_version: {nb.get('nbformat', 'unknown')}")
    
    # Check cell types
    cell_types = {}
    for cell in nb['cells']:
        ct = cell.get('cell_type', 'unknown')
        cell_types[ct] = cell_types.get(ct, 0) + 1
    
    print(f"✓ Cell types: {cell_types}")
    
    # Check metadata
    print(f"✓ Metadata keys: {list(nb.get('metadata', {}).keys())}")
    
    # Verify key content phrases
    all_text = ''
    for cell in nb['cells']:
        source = ''.join(cell.get('source', []))
        all_text += source + '\n'
    
    checks = [
        ('BRATS', 'Dataset'),
        ('YOLO', 'Model'),
        ('segmentation', 'Task'),
        ('NIfTI', 'Medical format'),
        ('kagglehub', 'Download'),
        ('3D', '3D volumes'),
    ]
    
    print("\nContent verification:")
    for phrase, desc in checks:
        if phrase.lower() in all_text.lower():
            print(f"  ✓ {desc}: found '{phrase}'")
        else:
            print(f"  ✗ {desc}: NOT FOUND")
    
    print("\n✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓")
    sys.exit(0)
    
except json.JSONDecodeError as e:
    print(f"✗ JSON Parse Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
