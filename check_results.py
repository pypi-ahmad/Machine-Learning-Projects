import json
from collections import Counter

results = json.load(open("execution_results.json"))
print(f"Cached results: {len(results)}")
sc = Counter(r["status"] for r in results)
for s, c in sc.most_common():
    print(f"  {s}: {c}")
print()
for r in results:
    tag = r.get("error_type", "")
    print(f"  {r['status']:8s} {r['time_s']:6.1f}s  {r['family']}/{r['project']}  {tag}")
