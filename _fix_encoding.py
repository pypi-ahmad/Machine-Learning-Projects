"""Check and fix encoding for notebooks 48 and 100."""
import json
import os

files = [
    r'E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\CrewAI_Multi-Agent_Systems\48_CrewAI_Customer_Success_Crew\notebook.ipynb',
    r'E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\Coding_and_Developer_Agents\100_Local_AI_Ops_Mini_Platform\notebook.ipynb',
]

for fpath in files:
    # Read raw bytes to detect encoding
    with open(fpath, 'rb') as f:
        raw = f.read(10)

    bom_info = 'none'
    if raw[:2] == b'\xff\xfe':
        bom_info = 'UTF-16LE BOM'
    elif raw[:3] == b'\xef\xbb\xbf':
        bom_info = 'UTF-8 BOM'
    elif raw[:2] == b'\xfe\xff':
        bom_info = 'UTF-16BE BOM'

    print(f'{os.path.basename(os.path.dirname(fpath))}: BOM={bom_info}, first bytes={raw[:10].hex()}')

    # If it has a non-UTF-8 encoding, re-encode to UTF-8
    if bom_info != 'none':
        encoding = 'utf-16-le' if 'UTF-16LE' in bom_info else ('utf-16-be' if 'UTF-16BE' in bom_info else 'utf-8-sig')
        with open(fpath, 'r', encoding=encoding) as f:
            content = f.read()

        # Verify it's valid JSON
        nb = json.loads(content)
        print(f'  Parsed OK: {len(nb.get("cells", []))} cells')

        # Re-save as UTF-8 without BOM
        with open(fpath, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')
        print(f'  Re-saved as UTF-8')
    else:
        # Try loading as UTF-8
        with open(fpath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        print(f'  Already UTF-8, {len(nb.get("cells", []))} cells')

print('\nDone.')

