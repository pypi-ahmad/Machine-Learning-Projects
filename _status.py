import json, os
r = json.load(open('execution_results.json'))
ok = [x for x in r if x['status']=='success']
fail = [x for x in r if x['status']!='success']
print(f"Total: {len(r)}, OK: {len(ok)}, FAIL: {len(fail)}")
print("\n=== FAILURES ===")
for x in fail:
    nb = os.path.basename(x['notebook'])
    t = x['time_s']
    e = x.get('error','')[:200]
    print(f"  [{t:.0f}s] {nb}")
    print(f"    Error: {e}")
print("\n=== LAST 10 OK ===")
for x in ok[-10:]:
    nb = os.path.basename(x['notebook'])
    print(f"  [{x['time_s']:.0f}s] {nb}")
