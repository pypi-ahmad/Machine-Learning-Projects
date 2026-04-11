"""
Fix fraud detection notebooks at the cell level (not raw JSON text).
Adds PyOD subsampling and reduces FLAML budget for large datasets.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OLD_PYOD_FIT = '            pm.fit(X_train.values if hasattr(X_train, "values") else X_train)'

NEW_PYOD_FIT = '''            # Subsample large datasets for PyOD (CPU-bound, slow > 50k rows)
            MAX_PYOD = 50_000
            if len(X_train) > MAX_PYOD:
                _idx = np.random.RandomState(42).choice(len(X_train), MAX_PYOD, replace=False)
                _Xp = X_train.iloc[_idx] if hasattr(X_train, "iloc") else X_train[_idx]
            else:
                _Xp = X_train
            pm.fit(_Xp.values if hasattr(_Xp, "values") else _Xp)'''

OLD_FLAML = 'automl.fit(X_train, y_train, task="classification", time_budget=120,'
NEW_FLAML = '''# Scale FLAML budget with dataset size
            _budget = 60 if len(X_train) > 100_000 else 120
            automl.fit(X_train, y_train, task="classification", time_budget=_budget,'''

root = Path(__file__).parent

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
        nb = json.loads(nb_path.read_text("utf-8", errors="replace"))
        changed = False
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            new_src = src
            if OLD_PYOD_FIT in new_src:
                new_src = new_src.replace(OLD_PYOD_FIT, NEW_PYOD_FIT)
                changed = True
            if OLD_FLAML in new_src:
                new_src = new_src.replace(OLD_FLAML, NEW_FLAML)
                changed = True
            if new_src != src:
                cell["source"] = [new_src]
        
        if changed:
            nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), "utf-8")
            print(f"Fixed: {nb_path.name}")
            nb_fixed += 1
        else:
            print(f"No changes: {nb_path.name}")
    except Exception as e:
        print(f"ERROR {nb_path.name}: {e}")

print(f"\nTotal notebooks fixed: {nb_fixed}")
