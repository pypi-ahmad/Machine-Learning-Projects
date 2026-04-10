import json, os
d = json.load(open("execution_results.json"))
for e in d:
    if e["status"] != "success":
        print("=" * 60)
        print(f"NB: {os.path.basename(e['notebook'])}")
        print(f"Status: {e['status']}")
        print(f"Elapsed: {e.get('elapsed_seconds', 0)}s")
        err = e.get("error", "")
        print(f"Error:\n{err[:3000]}")
        print()
