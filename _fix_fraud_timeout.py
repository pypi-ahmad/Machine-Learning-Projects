"""
Fix fraud detection pipelines and notebooks to add PyOD subsampling for large datasets.
Prevents timeout on 284k-row creditcard.csv datasets.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OLD_PYOD_FIT = '            pm.fit(X_train.values if hasattr(X_train, "values") else X_train)'

NEW_PYOD_FIT = '''            # Subsample large datasets for PyOD (CPU-bound, becomes slow > 50k rows)
            MAX_PYOD = 50_000
            if len(X_train) > MAX_PYOD:
                _idx = np.random.RandomState(42).choice(len(X_train), MAX_PYOD, replace=False)
                _Xp = X_train.iloc[_idx] if hasattr(X_train, "iloc") else X_train[_idx]
            else:
                _Xp = X_train
            pm.fit(_Xp.values if hasattr(_Xp, "values") else _Xp)'''

OLD_FLAML_BUDGET = "automl.fit(X_train, y_train, task=\"classification\", time_budget=120,"
NEW_FLAML_BUDGET = """# Scale FLAML time budget with dataset size
            _flaml_budget = 60 if len(X_train) > 100_000 else 120
            automl.fit(X_train, y_train, task="classification", time_budget=_flaml_budget,"""

root = Path(__file__).parent

# --- Fix pipeline.py files ---
fraud_pipelines = [
    "Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS/pipeline.py",
    "Anomaly detection and fraud detection/Fraud Detection in Financial Transactions/pipeline.py",
    "Anomaly detection and fraud detection/Fraudulent Credit Card Transaction Detection/pipeline.py",
    "Anomaly detection and fraud detection/Insurance Fraud Detection/pipeline.py",
    "Classification/Advanced Credit Card Fraud Detection/pipeline.py",
    "Classification/Credit Card Fraud - Imbalanced Dataset/pipeline.py",
    "Classification/Fraud Detection/pipeline.py",
]

pipeline_fixed = 0
for rel in fraud_pipelines:
    f = root / rel
    if not f.exists():
        print(f"NOT FOUND: {rel}")
        continue
    content = f.read_text("utf-8", errors="replace")
    changed = False
    if OLD_PYOD_FIT in content:
        content = content.replace(OLD_PYOD_FIT, NEW_PYOD_FIT)
        changed = True
    if OLD_FLAML_BUDGET in content:
        content = content.replace(OLD_FLAML_BUDGET, NEW_FLAML_BUDGET)
        changed = True
    if changed:
        f.write_text(content, "utf-8")
        print(f"Fixed pipeline: {rel}")
        pipeline_fixed += 1
    else:
        print(f"No changes needed: {rel}")

print(f"\nPipelines fixed: {pipeline_fixed}")

# --- Fix notebook files ---
notebook_paths = [
    "Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS/Fraud Detection - IEEE-CIS.ipynb",
    "Anomaly detection and fraud detection/Fraud Detection in Financial Transactions/Fraud Detection in Financial Transactions.ipynb",
    "Anomaly detection and fraud detection/Fraudulent Credit Card Transaction Detection/Fraudulent Credit Card Transaction Detection.ipynb",
    "Anomaly detection and fraud detection/Insurance Fraud Detection/Insurance Fraud Detection.ipynb",
    "Classification/Advanced Credit Card Fraud Detection/Advanced Credit Card Fraud Detection.ipynb",
    "Classification/Credit Card Fraud - Imbalanced Dataset/Credit Card Fraud - Imbalanced Dataset.ipynb",
    "Classification/Fraud Detection/Fraud Detection.ipynb",
]

nb_fixed = 0
for rel in notebook_paths:
    nb_path = root / rel
    if not nb_path.exists():
        print(f"NOT FOUND: {rel}")
        continue
    try:
        content = nb_path.read_text("utf-8", errors="replace")
        changed = False
        if OLD_PYOD_FIT in content:
            content = content.replace(OLD_PYOD_FIT, NEW_PYOD_FIT)
            changed = True
        if OLD_FLAML_BUDGET in content:
            content = content.replace(OLD_FLAML_BUDGET, NEW_FLAML_BUDGET)
            changed = True
        if changed:
            # Re-parse and re-serialize the notebook properly
            nb = json.loads(content)
            nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), "utf-8")
            print(f"Fixed notebook: {rel}")
            nb_fixed += 1
        else:
            print(f"No changes needed in notebook: {rel}")
    except Exception as e:
        print(f"ERROR {rel}: {e}")

print(f"\nNotebooks fixed: {nb_fixed}")
print(f"Total fixed: {pipeline_fixed + nb_fixed}")
