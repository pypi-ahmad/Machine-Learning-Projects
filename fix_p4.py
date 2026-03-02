"""Fix P4 Indian Dance remaining hardcoded paths at JSON level."""
import json, re

path = r"d:\Workspace\Github\Machine-Learning-Projects\Machine Learning Project 4 - Indian-classical-dance problem using Machine Learning\main.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 4: PIL.Image.open(r"C:\Users\...\train\kathak\kathak_original_1.jpg_...")
cell4 = nb['cells'][4]
cell4['source'] = [
    "# Display a sample training image\n",
    "sample_class = os.listdir(str(DATA_DIR / 'train'))[0]\n",
    "sample_img = os.listdir(str(DATA_DIR / 'train' / sample_class))[0]\n",
    "PIL.Image.open(DATA_DIR / 'train' / sample_class / sample_img)"
]
print("Fixed cell 4")

# Cell 5: PIL.Image.open(r"C:\Users\...\train\kathak\kathak_original_58.jpg...")
cell5 = nb['cells'][5]
cell5['source'] = [
    "# Display another sample training image\n",
    "sample_img2 = os.listdir(str(DATA_DIR / 'train' / sample_class))[1]\n",
    "PIL.Image.open(DATA_DIR / 'train' / sample_class / sample_img2)"
]
print("Fixed cell 5")

# Cell 6: image = PIL.Image.open(r"C:\Users\...\test\80.jpg") + datagens + flow_from_directory
cell6 = nb['cells'][6]
src = ''.join(cell6['source'])
# Replace only the Image.open line
src = re.sub(
    r'image = PIL\.Image\.open\(r"C:\\\\Users\\\\[^"]+"\)',
    'test_class = os.listdir(str(DATA_DIR / \'test\'))[0]\n'
    'test_img = os.listdir(str(DATA_DIR / \'test\' / test_class))[0]\n'
    'image = PIL.Image.open(DATA_DIR / \'test\' / test_class / test_img)',
    src
)
# Split back
lines = src.split('\n')
cell6['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
print("Fixed cell 6")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved!")
