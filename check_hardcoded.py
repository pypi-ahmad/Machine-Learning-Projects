"""Check for remaining C:\\Users paths in all notebooks that were flagged."""
import json, os, re

ROOT = r"d:\Workspace\Github\Machine-Learning-Projects"

notebooks = [
    ("P24", "Machine Learning Project 24 - Dog Vs Cat Classification", "main.ipynb"),
    ("P30", "Machine Learning Project 30 - Pneumonia Classification", "pneumonia.ipynb"),
    ("P4", "Machine Learning Project 4 - Indian-classical-dance problem using Machine Learning", "main.ipynb"),
    ("P92", "Machine Learning Projects 92 - Movie Recommendation Engine", "Movie_Recommendation_Engine.ipynb"),
    ("P6", "Machine Learning Project 6 - Imdb sentiment review Analysis using ML", "Untitled.ipynb"),
    ("P12", "Machine Learning Project 12 - Hand Digit Recognition Using ML", "Untitled.ipynb"),
]

for tag, proj, nb_name in notebooks:
    path = os.path.join(ROOT, proj, nb_name)
    if not os.path.exists(path):
        print(f"{tag}: FILE NOT FOUND")
        continue
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    found = []
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if re.search(r'C:\\\\?Users\\\\?', src):
            found.append((i, src[:150]))
    
    if found:
        print(f"{tag}: {len(found)} cells with C:\\Users in source:")
        for idx, snippet in found:
            print(f"  Cell {idx}: {snippet}...")
    else:
        print(f"{tag}: CLEAN ✓")
