"""
Fix CatBoost 'lr=' → 'learning_rate=' in ALL pipeline.py files.
CatBoost uses 'learning_rate' not 'lr'.
Also fix XGBoost early stopping issue, and notebooks.
"""
import re
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

ROOT = Path('.')
EXCLUDE_DIRS = {'venv', '.venv', 'core', 'data', '__pycache__', '.git', '.github'}

fixed_py = 0
fixed_nb = 0
skipped = 0

# Fix pipeline.py files
for pyfile in sorted(ROOT.rglob('pipeline.py')):
    parts_lower = {x.lower() for x in pyfile.parts}
    if parts_lower & EXCLUDE_DIRS:
        continue
    
    text = pyfile.read_text('utf-8', errors='replace')
    new_text = text
    
    # Fix CatBoost lr= → learning_rate=
    new_text = re.sub(
        r'(CatBoost(?:Classifier|Regressor)\([^)]*)\blr=(\d)',
        r'\1learning_rate=\2',
        new_text
    )
    
    if new_text != text:
        pyfile.write_text(new_text, 'utf-8')
        fixed_py += 1
        print(f"Fixed pipeline.py: {pyfile.relative_to(ROOT)}")
    else:
        skipped += 1

print(f"\nFixed {fixed_py} pipeline.py files ({skipped} unchanged)")

# Now fix notebooks too
fixed_nb = 0
for nb_path in sorted(ROOT.rglob('*.ipynb')):
    parts_lower = {x.lower() for x in nb_path.parts}
    if parts_lower & EXCLUDE_DIRS:
        continue
    if '_out_test' in nb_path.name or 'checkpoint' in str(nb_path).lower():
        continue
    
    try:
        text = nb_path.read_text('utf-8', errors='replace')
        if 'CatBoost' not in text or 'lr=' not in text:
            continue
        
        nb = json.loads(text)
        changed = False
        
        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            src = cell.get('source', [])
            src_text = ''.join(src) if isinstance(src, list) else src
            
            # Fix CatBoost lr= → learning_rate=
            new_src = re.sub(
                r'(CatBoost(?:Classifier|Regressor)\([^)]*)\blr=(\d)',
                r'\1learning_rate=\2',
                src_text
            )
            
            if new_src != src_text:
                if isinstance(src, list):
                    # Rebuild as list - preserve line endings
                    cell['source'] = new_src.splitlines(keepends=True)
                    # Last line shouldn't have \n if original didn't
                    if src and not src[-1].endswith('\n') and cell['source']:
                        cell['source'][-1] = cell['source'][-1].rstrip('\n')
                else:
                    cell['source'] = new_src
                changed = True
        
        if changed:
            nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), 'utf-8')
            fixed_nb += 1
            print(f"Fixed notebook: {nb_path.relative_to(ROOT)}")
    
    except Exception as e:
        print(f"ERROR in {nb_path.name}: {e}")

print(f"\nFixed {fixed_nb} notebook files")
print(f"\nTotal: {fixed_py} pipeline.py + {fixed_nb} notebooks fixed")
