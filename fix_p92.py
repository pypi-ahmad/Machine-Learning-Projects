"""Fix P92 Movie Recommendation properly."""
import json

path = r"d:\Workspace\Github\Machine-Learning-Projects\Machine Learning Projects 92 - Movie Recommendation Engine\Movie_Recommendation_Engine.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0 is markdown. Find first code cell to add DATA_DIR
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell['source'])
        if 'DATA_DIR' not in src:
            cell['source'] = [
                "from pathlib import Path\n",
                "DATA_DIR = Path.cwd().parent / 'data' / 'movie_recommendation_engine'\n",
                "\n",
            ] + cell['source']
            print(f"Added DATA_DIR to code cell {i}")
        break

# Cell 3: movie_file = "data\movie_dataset\movies.csv"
cell3 = nb['cells'][3]
cell3['source'] = [
    "movie_file = DATA_DIR / 'movies.csv'\n",
    "movie_data = pd.read_csv(movie_file, usecols = [0, 1])\n",
    "movie_data.head()"
]
print("Fixed cell 3 (movies.csv)")

# Cell 4: ratings_file = "data\\movie_dataset\\ratings.csv"
cell4 = nb['cells'][4]
cell4['source'] = [
    "ratings_file = DATA_DIR / 'ratings.csv'\n",
    "ratings_info = pd.read_csv(ratings_file, usecols = [0, 1, 2])\n",
    "ratings_info.head()"
]
print("Fixed cell 4 (ratings.csv)")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved!")
