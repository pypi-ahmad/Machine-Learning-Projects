#!/usr/bin/env python
"""Task 0 audit: full project inventory."""
import importlib.util, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.registry import PROJECT_REGISTRY
for p in sorted(ROOT.glob("CV */modern.py")):
    spec = importlib.util.spec_from_file_location(f"_a_{p.parent.name}", str(p))
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        try: spec.loader.exec_module(mod)
        except: pass

rows = []
for reg_name, cls in sorted(PROJECT_REGISTRY.items()):
    pk = reg_name.removesuffix("_v2")
    cat = getattr(cls, "category", "opencv_utility")
    task_map = {"detection":"detect","pose":"pose","segmentation":"seg","classification":"cls","opencv_utility":"utility"}
    task = task_map.get(cat, cat)
    has_cfg = (ROOT / "configs" / "datasets" / f"{pk}.yaml").exists()
    folder = None
    for f in ROOT.glob("CV */modern.py"):
        txt = f.read_text(encoding="utf-8")
        if f'@register("{reg_name}")' in txt:
            folder = f.parent.name
            break
    has_train = (ROOT / folder / "train.py").exists() if folder else False
    fw = "ultralytics" if task in ("detect","pose","seg","cls","obb") else "opencv"
    rows.append({"reg": reg_name, "pk": pk, "task": task, "fw": fw, "cfg": has_cfg, "train": has_train, "folder": folder or "?"})

hdr = f"{'reg':<35s} {'pk':<30s} {'task':<8s} {'fw':<12s} {'cfg':>3s} {'trn':>3s} folder"
print(hdr)
print("-" * len(hdr) + "-" * 50)
for r in rows:
    c = "Y" if r["cfg"] else "-"
    t = "Y" if r["train"] else "-"
    print(f"{r['reg']:<35s} {r['pk']:<30s} {r['task']:<8s} {r['fw']:<12s} {c:>3s} {t:>3s} {r['folder']}")
print(f"\nTotal: {len(rows)} projects")
ul_count = sum(1 for r in rows if r["fw"] == "ultralytics")
cv_count = sum(1 for r in rows if r["fw"] == "opencv")
cfg_count = sum(1 for r in rows if r["cfg"])
trn_count = sum(1 for r in rows if r["train"])
print(f"Ultralytics: {ul_count}  OpenCV-only: {cv_count}  Has config: {cfg_count}  Has train: {trn_count}")
