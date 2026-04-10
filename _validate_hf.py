"""Validate all HF dataset names from the generator against HuggingFace Hub."""
import re
import requests

src = open('_overhaul_v2.py', 'r', encoding='utf-8').read()
hf_names = sorted(set(re.findall(r'_hf\("([^"]+)"', src)))

valid = []
invalid = []

for name in hf_names:
    try:
        resp = requests.head(f"https://huggingface.co/api/datasets/{name}", timeout=10)
        if resp.status_code == 200:
            valid.append(name)
            print(f"  OK  {name}")
        else:
            invalid.append(name)
            print(f"  BAD {name} (HTTP {resp.status_code})")
    except Exception as e:
        invalid.append(name)
        print(f"  ERR {name} ({e})")

print(f"\nValid: {len(valid)}, Invalid: {len(invalid)}")
print("\nInvalid datasets:")
for n in invalid:
    print(f"  {n}")
