"""Fix P7 and P92 at JSON level."""
import json

ROOT = r"d:\Workspace\Github\Machine-Learning-Projects"

# ── P7: Hyperparameter Tunning ──
path7 = f"{ROOT}/Machine Learning Project 7 - Advandced Hyperparameter Tunning/Hyper_Parameter_Tunning.ipynb"
with open(path7, 'r', encoding='utf-8') as f:
    nb7 = json.load(f)

# Cell 3: read_csv('diabetes.csv') -> needs DATA_DIR + proper path
cell3 = nb7['cells'][3]
src = ''.join(cell3['source'])
print(f"P7 cell 3 BEFORE: {repr(src[:200])}")

cell3['source'] = [
    "from pathlib import Path\n",
    "DATA_DIR = Path.cwd().parent / 'data' / 'advandced_hyperparameter_tunning'\n",
    "\n",
    "import pandas as pd\n",
    "\n",
    "## Read dataset\n",
    "df = pd.read_csv(DATA_DIR / 'diabetes.csv')"
]
print("P7: Fixed cell 3")

with open(path7, 'w', encoding='utf-8') as f:
    json.dump(nb7, f, indent=1, ensure_ascii=False)
print("P7: Saved!")

# ── P92: Movie Recommendation Engine ──
path92 = f"{ROOT}/Machine Learning Projects 92 - Movie Recommendation Engine/Movie_Recommendation_Engine.ipynb"
with open(path92, 'r', encoding='utf-8') as f:
    nb92 = json.load(f)

# Find cells that load movies.csv and ratings.csv
for i, cell in enumerate(nb92['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'movies.csv' in src or 'ratings.csv' in src or 'read_csv' in src:
        print(f"P92 cell {i}: {repr(src[:200])}")

# Cell 0 should have imports; find the cell that loads movies and ratings
for i, cell in enumerate(nb92['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    
    # Add DATA_DIR to the imports cell (first code cell)
    if i == 0 and 'DATA_DIR' not in src:
        cell['source'] = [
            "from pathlib import Path\n",
            "DATA_DIR = Path.cwd().parent / 'data' / 'movie_recommendation_engine'\n",
            "\n",
        ] + cell['source']
        print(f"P92: Added DATA_DIR to cell {i}")
    
    # Fix movies.csv loading
    if "movies.csv" in src and "DATA_DIR" not in src:
        new_lines = []
        for line in cell['source']:
            if "movies.csv" in line and "read_csv" in line:
                line = line.replace("'movies.csv'", "DATA_DIR / 'movies.csv'")
                line = line.replace('"movies.csv"', 'DATA_DIR / "movies.csv"')
            new_lines.append(line)
        cell['source'] = new_lines
        print(f"P92: Fixed movies.csv in cell {i}")
    
    # Fix ratings.csv loading
    if "ratings.csv" in src and "DATA_DIR" not in src:
        new_lines = []
        for line in cell['source']:
            if "ratings.csv" in line and "read_csv" in line:
                line = line.replace("'ratings.csv'", "DATA_DIR / 'ratings.csv'")
                line = line.replace('"ratings.csv"', 'DATA_DIR / "ratings.csv"')
            new_lines.append(line)
        cell['source'] = new_lines
        print(f"P92: Fixed ratings.csv in cell {i}")

with open(path92, 'w', encoding='utf-8') as f:
    json.dump(nb92, f, indent=1, ensure_ascii=False)
print("P92: Saved!")
