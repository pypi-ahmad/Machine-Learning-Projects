"""Quick fix for remaining hardcoded paths in specific notebooks."""
import json, re

def fix_notebook(path, replacements):
    """Fix source code in a notebook at the JSON level.
    replacements: list of (old_pattern, new_string) tuples (regex).
    """
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changed = False
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell['source'])
        new_src = src
        for pattern, replacement in replacements:
            new_src = re.sub(pattern, replacement, new_src)
        if new_src != src:
            # Split back into lines preserving newlines
            lines = new_src.split('\n')
            cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
            changed = True
    
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
    return changed

ROOT = r"d:\Workspace\Github\Machine-Learning-Projects"

# P24 - Dog vs Cat: glob path
print("Fixing P24...")
ok = fix_notebook(
    f"{ROOT}/Machine Learning Project 24 - Dog Vs Cat Classification/main.ipynb",
    [
        (r"glob\(r'C:\\\\Users\\\\[^']+'\)", "glob(str(DATA_DIR / 'PetImages' / '*'))"),
    ]
)
print(f"  {'Fixed' if ok else 'No changes'}")

# P30 - Pneumonia: check for hardcoded paths
print("Fixing P30...")
ok = fix_notebook(
    f"{ROOT}/Machine Learning Project 30 - Pneumonia Classification/pneumonia.ipynb",
    [
        (r"r?['\"]C:\\\\?Users\\\\?[^'\"]+['\"]", "str(DATA_DIR / 'chest_xray')"),
    ]
)
print(f"  {'Fixed' if ok else 'No changes'}")

# P4 - Indian Dance: already fixed via edit_notebook_file but check
print("Checking P4...")
ok = fix_notebook(
    f"{ROOT}/Machine Learning Project 4 - Indian-classical-dance problem using Machine Learning/main.ipynb",
    [
        # Any remaining C:\Users paths
        (r'r"C:\\\\Users\\\\[^"]+train\\\\kathak\\\\[^"]+\.jpg[^"]*"',
         'str(DATA_DIR / "train" / os.listdir(str(DATA_DIR / "train"))[0] / os.listdir(str(DATA_DIR / "train" / os.listdir(str(DATA_DIR / "train"))[0]))[0])'),
    ]
)
print(f"  {'Fixed' if ok else 'No changes'}")

# P92 - Movie Recommendation: check
print("Checking P92...")
ok = fix_notebook(
    f"{ROOT}/Machine Learning Projects 92 - Movie Recommendation Engine/Movie_Recommendation_Engine.ipynb",
    [
        (r"r?['\"]C:\\\\?Users\\\\?[^'\"]+['\"]", "'should_not_match'"),
    ]
)
print(f"  {'Fixed' if ok else 'No changes'}")

# P6 - IMDB: check
print("Checking P6...")
ok = fix_notebook(
    f"{ROOT}/Machine Learning Project 6 - Imdb sentiment review Analysis using ML/Untitled.ipynb",
    []
)
print(f"  {'Fixed' if ok else 'No changes'}")

# P12 - Hand digit: check
print("Checking P12...")
ok = fix_notebook(
    f"{ROOT}/Machine Learning Project 12 - Hand Digit Recognition Using ML/Untitled.ipynb",
    []
)
print(f"  {'Fixed' if ok else 'No changes'}")

print("\nDone!")
