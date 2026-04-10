"""Wait for new notebook results and report them."""
import json, time, sys, os

last_count = 17
target = int(sys.argv[1]) if len(sys.argv) > 1 else 25

while True:
    d = json.load(open("execution_results.json"))
    count = len(d)
    if count > last_count:
        for entry in d[last_count:]:
            nb = os.path.basename(entry["notebook"])
            elapsed = entry.get("elapsed_seconds", 0)
            status = entry["status"]
            err = ""
            if status == "fail":
                err = f" | {entry.get('error','')[:120]}"
            print(f"  {status:>7s} {elapsed:>6.0f}s {nb}{err}")
        last_count = count
        print(f"  --- Total: {count} (success={sum(1 for x in d if x['status']=='success')}, fail={sum(1 for x in d if x['status']=='fail')})")
        print(f"  --- Time: {time.strftime('%H:%M:%S')}")
        if count >= target:
            break
    time.sleep(10)
