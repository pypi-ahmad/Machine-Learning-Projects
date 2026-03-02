import json
r = json.load(open('audit_phase5/standardization_report.json'))
ok = [d for d in r['details'] if d['status'] == 'OK']
print(f"{len(ok)} projects to stress-test\n")
for d in ok:
    print(f"  P{d['project']:03d}  {d['task']:16s}  target={str(d.get('target_col','?')):30s}  nb={d['notebook']}")
