"""
Fix fraud detection notebooks - correct version with proper indentation.
Adds PyOD subsampling and updates FLAML budget.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# PyOD fix: replace the pm.fit line (12-space indented) with subsampled version
OLD_PYOD = '            pm.fit(X_train.values if hasattr(X_train, "values") else X_train)'
NEW_PYOD = '''            # Subsample large datasets for PyOD (CPU-bound, slow > 50k rows)
            _pyod_max = 50_000
            if len(X_train) > _pyod_max:
                _pidx = np.random.RandomState(42).choice(len(X_train), _pyod_max, replace=False)
                _Xp = X_train.iloc[_pidx] if hasattr(X_train, "iloc") else X_train[_pidx]
            else:
                _Xp = X_train
            pm.fit(_Xp.values if hasattr(_Xp, "values") else _Xp)'''

# FLAML fix v2: replace only time_budget=120 with adaptive budget
# This is surgical and preserves indentation automatically
OLD_FLAML_120 = "time_budget=120,"
NEW_FLAML_ADAPTIVE = "time_budget=(60 if len(X_train) > 100_000 else 120),"

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
            
            # Fix PyOD subsampling (was broken in previous fix script)
            if OLD_PYOD in new_src:
                new_src = new_src.replace(OLD_PYOD, NEW_PYOD)
                changed = True
            
            # Fix the broken FLAML indentation from previous script
            # First, undo the bad fix (12-space `_budget` lines)
            bad_flaml = (
                '        # Scale FLAML budget with dataset size\n'
                '            _budget = 60 if len(X_train) > 100_000 else 120\n'
                '            automl.fit(X_train, y_train, task="classification", time_budget=_budget,'
            )
            if bad_flaml in new_src:
                # Revert to clean FLAML call first
                new_src = new_src.replace(
                    bad_flaml,
                    'automl.fit(X_train, y_train, task="classification", time_budget=120,'
                )
                changed = True
            
            # Also fix if it has the "pipeline.py" style bad fix
            bad_flaml2 = (
                '# Scale FLAML time budget with dataset size\n'
                '            _flaml_budget = 60 if len(X_train) > 100_000 else 120\n'
                '            automl.fit(X_train, y_train, task="classification", time_budget=_flaml_budget,'
            )
            if bad_flaml2 in new_src:
                new_src = new_src.replace(
                    bad_flaml2,
                    'automl.fit(X_train, y_train, task="classification", time_budget=120,'
                )
                changed = True
            
            # Now apply the clean FLAML fix (surgical replacement of just the time_budget value)
            if OLD_FLAML_120 in new_src and 'automl.fit' in new_src:
                new_src = new_src.replace(OLD_FLAML_120, NEW_FLAML_ADAPTIVE, 1)  # Only first occurrence
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
