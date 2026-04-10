"""Test loading various replacement data sources."""
import sys

tests = []

# Test canonical HF names for legacy-script datasets
def test_hf(name, split="train", config=None):
    from datasets import load_dataset
    kwargs = {"split": split}
    if config:
        kwargs["name"] = config
    ds = load_dataset(name, **kwargs)
    print(f"  HF OK  {name}{f' ({config})' if config else ''}: {len(ds)} rows, cols={ds.column_names[:5]}")

def test_url(url, desc=""):
    import pandas as pd
    df = pd.read_csv(url, nrows=5)
    print(f"  URL OK  {desc}: cols={list(df.columns)[:8]}")

def test_openml(data_id, desc=""):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    cols = list(d.frame.columns)
    target_name = d.target.name if hasattr(d.target, 'name') else 'N/A'
    print(f"  OML OK  {data_id} ({desc}): target='{target_name}' shape={d.frame.shape} cols={cols[:5]}...{cols[-2:]}")

# === Test HF canonical names ===
print("=== HF canonical names ===")
for name, config in [
    ("eriktks/conll2003", None),
    ("takala/financial_phrasebank", "sentences_50agree"),
    ("odegiber/hate_speech18", None),
    ("daily_dialog", None),
    ("cornell-movie-review-data/rotten_tomatoes", None),
    ("CSTR-Edinburgh/vctk", None),
]:
    try:
        test_hf(name, config=config)
    except Exception as e:
        print(f"  HF BAD {name}: {str(e)[:100]}")
    sys.stdout.flush()

# === Test URLs for tabular datasets ===
print("\n=== URLs for missing tabular datasets ===")
url_tests = [
    ("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv", "insurance"),
    ("https://raw.githubusercontent.com/dsrscientist/dataset1/master/advertising.csv", "advertising"),
    ("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", "titanic"),
]
for url, desc in url_tests:
    try:
        test_url(url, desc)
    except Exception as e:
        print(f"  URL BAD {desc}: {str(e)[:100]}")
    sys.stdout.flush()

# === Test OpenML replacements ===
print("\n=== OpenML replacements ===")
openml_tests = [
    (42178, "telecom-churn"),  # for aai510-group1/telecom-churn-dataset
    (1480, "wilt"),  # possible alternative
    (44, "spambase"),
]
for oid, desc in openml_tests:
    try:
        test_openml(oid, desc)
    except Exception as e:
        print(f"  OML BAD {oid} ({desc}): {str(e)[:100]}")
    sys.stdout.flush()

print("\nDone.")
