import re
src = open('_overhaul_v2.py', 'r', encoding='utf-8').read()
hf = sorted(set(re.findall(r'_hf\("([^"]+)"', src)))
print(f'Total: {len(hf)}')
for d in hf:
    print(d)
