import json

r = json.load(open('execution_results.json'))
s = [x for x in r if x['status'] == 'success']
f = [x for x in r if x['status'] != 'success']
print(f"Total: {len(r)}, Success: {len(s)}, Fail: {len(f)}")
for x in f:
    nb = x['notebook'].replace('\\', '/').split('/')[-1]
    err = x.get('error', '')[:150]
    print(f"  FAIL: {nb} -> {err}")
print()
print("Last 10 results:")
for x in r[-10:]:
    nb = x['notebook'].replace('\\', '/').split('/')[-1]
    print(f"  {x['status']:10s} {x['time_s']:7.0f}s {nb}")
