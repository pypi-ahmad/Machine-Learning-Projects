#!/usr/bin/env python
"""Phase 3B-3 Smoke Test — Model Registry + YOLO26 Migration."""
import sys, pathlib, ast, importlib.util, glob, inspect

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results = []

def check(name, fn):
    try:
        fn()
        results.append(("PASS", name))
    except Exception as e:
        results.append(("FAIL", f"{name}: {e}"))

# 1
def t1():
    from models.registry import ModelRegistry, get_registry, get_active, resolve, YOLO26_DEFAULTS, register_model
    from models import ModelRegistry as M2
check("registry_imports", t1)

# 2
def t2():
    from models.registry import resolve, YOLO26_DEFAULTS
    for task, expected in YOLO26_DEFAULTS.items():
        w, v, d = resolve("_test", task)
        assert w == expected and v is None and d is True, f"{task}: {w}"
check("resolve_defaults", t2)

# 3
def t3():
    from utils.yolo import load_yolo, load_yolo_pose, load_yolo_seg, load_yolo_cls
    assert "yolo26n.pt" in str(inspect.signature(load_yolo))
    assert "yolo26n-pose.pt" in str(inspect.signature(load_yolo_pose))
    assert "yolo26n-seg.pt" in str(inspect.signature(load_yolo_seg))
    assert "yolo26n-cls.pt" in str(inspect.signature(load_yolo_cls))
check("yolo_defaults", t3)

# 4
def t4():
    from benchmarks.run_all import benchmark_project
    src = inspect.getsource(benchmark_project)
    for f in ["model_path", "used_pretrained_default", "model_version"]:
        assert f in src, f"missing {f}"
check("benchmark_fields", t4)

# 5
def t5():
    from core.base import CVProject
    from core.registry import PROJECT_REGISTRY, register, list_registered
check("core_imports", t5)

# 6
def t6():
    files = sorted(set(
        list(ROOT.glob("models/*.py"))
        + [ROOT / "utils/yolo.py", ROOT / "utils/datasets.py"]
        + list(ROOT.glob("benchmarks/*.py"))
        + list(ROOT.glob("core/*.py"))
        + list(ROOT.glob("**/modern.py"))
        + list(ROOT.glob("train/*.py"))
        + list(ROOT.glob("CV Project*/train.py"))
        + list(ROOT.glob("CV Projects*/train.py"))
    ))
    for f in files:
        ast.parse(f.read_text(encoding="utf-8"))
    assert len(files) >= 70, f"only {len(files)} files"
check("ast_parse_all", t6)

# 7
def t7():
    files = sorted(glob.glob(str(ROOT / "**/modern.py"), recursive=True))
    for f in files:
        name = pathlib.Path(f).parent.name.replace(" ", "_").replace("-", "_")
        spec = importlib.util.spec_from_file_location(f"mod_{name}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    assert len(files) == 50, f"got {len(files)}"
check("all_50_modern_import", t7)

# 8
def t8():
    needle = "yolo" + "11"  # avoid self-match
    for f in ROOT.rglob("*.py"):
        if f.name == "smoke_3b3.py":
            continue
        try:
            if needle in f.read_text(encoding="utf-8").lower():
                raise AssertionError(f"{needle} in {f.relative_to(ROOT)}")
        except UnicodeDecodeError:
            pass
check("no_yolo11_refs", t8)

# 9
def t9():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!models/registry.py" in gi
    assert "!models/metadata.json" in gi
    assert "!models/__init__.py" in gi
check("gitignore_models", t9)

# Summary
passed = sum(1 for s, _ in results if s == "PASS")
total = len(results)
for status, name in results:
    print(f"  [{status}] {name}")
print(f"\nSMOKE TEST SUMMARY: {passed}/{total} passed")
sys.exit(0 if passed == total else 1)
