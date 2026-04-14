import json, os

base = r'E:\Github\Machine-Learning-Projects\NLP'

files_to_fix = {
    os.path.join(base, 'Stopword Impact Study', 'stopword_impact_study.ipynb'): None,
    os.path.join(base, 'Stemming vs Lemmatization Project', 'stemming_vs_lemmatization.ipynb'): None,
    os.path.join(base, 'N-gram Explorer', 'ngram_explorer.ipynb'): None,
}

for fpath in files_to_fix:
    with open(fpath, 'r', encoding='utf-8') as fh:
        content = fh.read()

    first_50 = repr(content[:80])
    print(f"File: {os.path.basename(fpath)}")
    print(f"  Length: {len(content)}")
    print(f"  First 80: {first_50}")

    try:
        nb = json.loads(content)
        if len(nb.get('cells', [])) == 1 and nb['cells'][0].get('cell_type') == 'raw':
            print("  STATUS: Still has old raw-cell structure")
        else:
            print(f"  STATUS: Has {len(nb.get('cells', []))} cells, first type: {nb['cells'][0].get('cell_type','?')}")
    except json.JSONDecodeError as e:
        print(f"  STATUS: Invalid JSON at pos {e.pos}")
    print()

