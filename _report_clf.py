import json
r = json.load(open("execution_results.json"))
fail = [x for x in r if x["status"]!="success" and "322122" not in x.get("error","")]
for x in fail:
    nb = x["notebook"].split(chr(92))[-1]
    err = x.get("error","")[:300]
    t = x["time_s"]
    print(f"[{t:.0f}s] {nb}")
    print(f"  ERROR: {err}")
    print()
