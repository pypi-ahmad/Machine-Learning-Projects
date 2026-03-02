"""Fix P4 cell 6 directly."""
import json

path = r"d:\Workspace\Github\Machine-Learning-Projects\Machine Learning Project 4 - Indian-classical-dance problem using Machine Learning\main.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell6 = nb['cells'][6]
src_lines = cell6['source']
print("BEFORE:")
for l in src_lines[:5]:
    print(f"  {repr(l)}")

# Find and replace the Image.open line
new_lines = []
for line in src_lines:
    if 'PIL.Image.open(r"C:\\Users' in line or "PIL.Image.open(r'C:\\Users" in line:
        # Replace with dynamic path
        new_lines.append("test_class = os.listdir(str(DATA_DIR / 'test'))[0]\n")
        new_lines.append("test_img = os.listdir(str(DATA_DIR / 'test' / test_class))[0]\n")
        new_lines.append("image = PIL.Image.open(DATA_DIR / 'test' / test_class / test_img)\n")
    else:
        new_lines.append(line)

cell6['source'] = new_lines

print("\nAFTER:")
for l in new_lines[:6]:
    print(f"  {repr(l)}")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\nSaved!")
