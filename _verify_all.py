import json, os

base = r'E:\Github\Machine-Learning-Projects\NLP'
files = [
    ('Custom Tokenizer Comparison Lab', 'custom_tokenizer_comparison_lab.ipynb'),
    ('Stopword Impact Study', 'stopword_impact_study.ipynb'),
    ('Stemming vs Lemmatization Project', 'stemming_vs_lemmatization.ipynb'),
    ('N-gram Explorer', 'ngram_explorer.ipynb'),
    ('TF-IDF vs Embeddings Benchmark', 'tfidf_vs_embeddings.ipynb'),
]

for folder, fname in files:
    fp = os.path.join(base, folder, fname)
    try:
        with open(fp, 'r', encoding='utf-8') as fh:
            nb = json.load(fh)
        n_md = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
        n_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
        first = nb['cells'][0]['source'][0][:50]
        print(f"OK: {fname}: {len(nb['cells'])} cells ({n_md} md, {n_code} code) - {first}")
    except Exception as e:
        print(f"ERROR: {fname}: {e}")

