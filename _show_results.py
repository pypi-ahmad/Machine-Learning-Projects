import json
r = json.loads(open('_notebook_results.json').read())
for k, v in r.items():
    name = k.split('/')[-1]
    status = v.get('status', '?')
    error = v.get('error', '')[:80] if v.get('error') else ''
    print(f"{status}: {name} | {error}")
