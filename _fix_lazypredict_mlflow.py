"""Fix LazyPredict MLflow autologging issue in Classification notebooks."""
import json
from pathlib import Path

OLD_BLOCK = """    # ── LazyPredict (quick sweep benchmark) ──
    try:
        from lazypredict.Supervised import LazyClassifier
        t0 = time.perf_counter()
        lazy = LazyClassifier(verbose=0, ignore_warnings=True)
        lazy_models, _ = lazy.fit(X_train, X_test, y_train, y_test)
        timings["LazyPredict"] = time.perf_counter() - t0
        print(f"\\nLazyPredict — Top 5 classifiers:  ({timings['LazyPredict']:.1f}s)")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"LazyPredict: {e}")"""

NEW_BLOCK = """    # ── LazyPredict (quick sweep benchmark) ──
    try:
        from lazypredict.Supervised import LazyClassifier
        import mlflow
        mlflow.sklearn.autolog(disable=True)
        t0 = time.perf_counter()
        lazy = LazyClassifier(verbose=0, ignore_warnings=True)
        _lp_max = 5000
        _lp_xt = X_train.iloc[:_lp_max] if len(X_train) > _lp_max else X_train
        _lp_xe = X_test.iloc[:_lp_max] if len(X_test) > _lp_max else X_test
        _lp_yt = y_train.iloc[:_lp_max] if len(y_train) > _lp_max else y_train
        _lp_ye = y_test.iloc[:_lp_max] if len(y_test) > _lp_max else y_test
        lazy_models, _ = lazy.fit(_lp_xt, _lp_xe, _lp_yt, _lp_ye)
        mlflow.sklearn.autolog(disable=False)
        timings["LazyPredict"] = time.perf_counter() - t0
        print(f"\\nLazyPredict — Top 5 classifiers:  ({timings['LazyPredict']:.1f}s)")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"LazyPredict: {e}")"""

fixed = 0

for nb_path in Path(".").rglob("*.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except Exception:
            continue

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if OLD_BLOCK in src:
            new_src = src.replace(OLD_BLOCK, NEW_BLOCK)
            lines = new_src.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
            changed = True

    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"Fixed: {nb_path}")
        fixed += 1

# Also fix pipeline.py files
for py_path in Path(".").rglob("pipeline.py"):
    with open(py_path, "r", encoding="utf-8") as f:
        src = f.read()
    if OLD_BLOCK in src:
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(src.replace(OLD_BLOCK, NEW_BLOCK))
        print(f"Fixed pipeline.py: {py_path}")
        fixed += 1

print(f"\nTotal fixed: {fixed}")
