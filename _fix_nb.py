import json

files = [
    r'E:\Github\Machine-Learning-Projects\NLP\Stopword Impact Study\stopword_impact_study.ipynb',
    r'E:\Github\Machine-Learning-Projects\NLP\Stemming vs Lemmatization Project\stemming_vs_lemmatization.ipynb',
    r'E:\Github\Machine-Learning-Projects\NLP\N-gram Explorer\ngram_explorer.ipynb',
]
for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        content = fh.read()
    idx = content.find('"nbformat_minor":5}')
    if idx > 0:
        clean = content[:idx + len('"nbformat_minor":5}')]
        try:
            json.loads(clean)
            with open(f, 'w', encoding='utf-8') as fh:
                fh.write(clean)
            print(f'Fixed: {f}')
        except json.JSONDecodeError as e:
            print(f'JSON error in {f}: {e}')
    else:
        print(f'Pattern not found in {f}')

