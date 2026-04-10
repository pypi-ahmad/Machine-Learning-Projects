"""Re-validate HF datasets - HEAD with follow_redirects, then try load_dataset_builder for 307s."""
import re, requests

src = open('_overhaul_v2.py', 'r', encoding='utf-8').read()
hf_names = sorted(set(re.findall(r'_hf\("([^"]+)"', src)))

for name in hf_names:
    try:
        resp = requests.get(f"https://huggingface.co/api/datasets/{name}", timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  OK  {name} (id={data.get('id','')})")
        else:
            print(f"  BAD {name} (HTTP {resp.status_code})")
    except Exception as e:
        print(f"  ERR {name} ({e})")
