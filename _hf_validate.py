"""Check which HuggingFace dataset names actually exist on the Hub."""
from huggingface_hub import dataset_info
import re
from pathlib import Path

root = Path(__file__).resolve().parent
exclude = {"venv", ".venv", "core", "data", "__pycache__", ".git", ".github"}

hf_datasets = set()
for p in sorted(root.rglob("pipeline.py")):
    parts_lower = {x.lower() for x in p.parts}
    if parts_lower & exclude:
        continue
    text = p.read_text("utf-8", errors="ignore")
    for m in re.finditer(r'_hf_load\(["\']([^"\']+)["\']\s*[\),]', text):
        hf_datasets.add(m.group(1))

print(f"Found {len(hf_datasets)} unique HF dataset references")
ok = []
fail = []
for i, ds in enumerate(sorted(hf_datasets), 1):
    try:
        dataset_info(ds)
        ok.append(ds)
        print(f"  [{i}/{len(hf_datasets)}] OK: {ds}")
    except Exception:
        fail.append(ds)
        print(f"  [{i}/{len(hf_datasets)}] FAIL: {ds}")

print(f"\nVALID: {len(ok)} / {len(hf_datasets)}")
print(f"INVALID: {len(fail)} / {len(hf_datasets)}")
print("\n=== INVALID DATASETS ===")
for d in fail:
    print(f"  {d}")
print("\n=== VALID DATASETS ===")
for d in ok:
    print(f"  {d}")
