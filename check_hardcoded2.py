"""Check remaining WARNed notebooks for C:\\Users in source cells."""
import json, re, os

ROOT = r"d:\Workspace\Github\Machine-Learning-Projects"

notebooks = [
    ("P11", "Machine Learning Project 11 - Creadi Card Fraud Problem Handling Imbalanced Dataset", "Hanling_Imbalanced_Data.ipynb"),
    ("P117", "Machine Learning Projects 117 - Recommender Systems - The Fundamentals", "Recommender Systems - The Fundamentals.ipynb"),
    ("P7", "Machine Learning Project 7 - Advandced Hyperparameter Tunning", "Hyper_Parameter_Tunning.ipynb"),
    ("P92", "Machine Learning Projects 92 - Movie Recommendation Engine", "Movie_Recommendation_Engine.ipynb"),
    ("P49", "Machine Learning Project 49 - Cifar 10", "04_image_classification_with_CNN(Colab).ipynb"),
    ("P19", "Machine Learning Project 19 - Fashion Mnist Data Analysis Using ML", "F_Mnist_model.ipynb"),
    ("P4", "Machine Learning Project 4 - Indian-classical-dance problem using Machine Learning", "main.ipynb"),
]

for tag, proj, nb_name in notebooks:
    path = os.path.join(ROOT, proj, nb_name)
    if not os.path.exists(path):
        print(f"{tag}: FILE NOT FOUND")
        continue
    
    # Check both .ipynb JSON and .py files
    if nb_name.endswith('.ipynb'):
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
            # Check in outputs too
            out_count = 0
            for cell in nb['cells']:
                for out in cell.get('outputs', []):
                    text = ''.join(out.get('text', []))
                    if not text:
                        # Check data in output
                        data = out.get('data', {})
                        for mime, content in data.items():
                            if isinstance(content, list):
                                text += ''.join(content)
                    if re.search(r'C:\\\\?Users\\\\?', text):
                        out_count += 1
            if out_count:
                print(f"{tag}: CLEAN source ✓ (but {out_count} outputs have old C:\\Users paths - cosmetic only)")
            else:
                print(f"{tag}: CLEAN ✓")
    else:
        text = open(path, 'r', encoding='utf-8').read()
        if re.search(r'C:\\\\?Users', text):
            print(f"{tag}: Has C:\\Users in .py file")
        else:
            print(f"{tag}: CLEAN ✓")
