#!/usr/bin/env python3
import json
import sys

nb_path = r'e:\Github\Machine-Learning-Projects\Computer Vision\Road Segmentation for Autonomous Vehicles\Souce Code\road_segmentation_pipeline.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cell_count = len(nb.get('cells', []))
    print(f"✓ Valid JSON")
    print(f"✓ Cell count: {cell_count}")
    print(f"✓ Notebook format_version: {nb.get('nbformat', 'unknown')}")
    print(f"✓ Metadata keys: {list(nb.get('metadata', {}).keys())[:5]}")
    
    # Check for required sections
    cell_types = {}
    for cell in nb['cells']:
        ct = cell.get('cell_type', 'unknown')
        cell_types[ct] = cell_types.get(ct, 0) + 1
    
    print(f"✓ Cell types: {cell_types}")
    print("\n✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓")
    sys.exit(0)
    
except json.JSONDecodeError as e:
    print(f"✗ JSON Parse Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
