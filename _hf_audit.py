"""Extract all HuggingFace dataset names used in _hf_load() calls."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent
exclude = {"venv", ".venv", "core", "data", "__pycache__", ".git", ".github"}

hf_datasets = {}
for p in sorted(root.rglob("pipeline.py")):
    parts_lower = {x.lower() for x in p.parts}
    if parts_lower & exclude:
        continue
    text = p.read_text("utf-8", errors="ignore")
    for m in re.finditer(r"""_hf_load\(["']([^"']+)["']""", text):
        ds = m.group(1)
        family = p.parent.parent.name
        project = p.parent.name
        hf_datasets.setdefault(ds, []).append(f"{family}/{project}")

print("=== HuggingFace dataset names used across pipelines ===")
for ds, projects in sorted(hf_datasets.items()):
    print(f"  {ds}  ({len(projects)} projects)")
print(f"\nTotal unique HF datasets: {len(hf_datasets)}")
print(f"Total pipelines using HF: {sum(len(v) for v in hf_datasets.values())}")
