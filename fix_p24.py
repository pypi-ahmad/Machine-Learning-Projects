"""Fix P24 glob cell at JSON level."""
import json

path = r"d:\Workspace\Github\Machine-Learning-Projects\Machine Learning Project 24 - Dog Vs Cat Classification\main.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 5 has the glob path
cell = nb['cells'][5]
old_src = ''.join(cell['source'])
print("BEFORE:", repr(old_src[:200]))

new_lines = []
for line in cell['source']:
    if "glob(r'C:\\" in line or "glob(r\"C:\\\\" in line:
        line = "folders = glob(str(DATA_DIR / 'PetImages' / '*'))\n"
    new_lines.append(line)
cell['source'] = new_lines

new_src = ''.join(cell['source'])
print("AFTER:", repr(new_src[:200]))

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved!")
