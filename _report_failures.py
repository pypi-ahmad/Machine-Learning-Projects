import json

r = json.load(open("execution_results.json"))
fail = [x for x in r if x["status"] != "success"]
ok = [x for x in r if x["status"] == "success"]
print(f"Total: {len(r)}, OK: {len(ok)}, FAIL: {len(fail)}")

crash = [x for x in fail if "322122" in x.get("error", "")]
other = [x for x in fail if "322122" not in x.get("error", "")]

print(f"\n=== Clustering crashes ({len(crash)}) ===")
for x in crash:
    print(f"  {x['notebook'].split(chr(92))[-1]}")

print(f"\n=== Other failures ({len(other)}) ===")
for x in other:
    nb = x["notebook"].split(chr(92))[-1]
    err = x.get("error", "")[:120]
    t = x["time_s"]
    print(f"  [{t:.0f}s] {nb}: {err}")
