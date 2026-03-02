"""Check data availability for each of the 43 standardized projects."""
import json, csv, os, glob

base = r'D:\Workspace\Github\Machine-Learning-Projects'
report = json.load(open(os.path.join(base, 'audit_phase5/standardization_report.json')))
ok_projects = [d for d in report['details'] if d['status'] == 'OK']
ok_pnums = {d['project'] for d in ok_projects}

# Load phase3 dataset status
import re as _re
p3_status = {}
with open(os.path.join(base, 'audit_phase3/phase3_dataset_status.csv')) as f:
    for row in csv.DictReader(f):
        m = _re.search(r'Project[s]?\s+(\d+)', row['project'])
        if m:
            pnum = int(m.group(1))
            p3_status[pnum] = row.get('status', '?')

# Check each
ready = []
blocked = []
for d in ok_projects:
    pnum = d['project']
    ds = p3_status.get(pnum, 'UNKNOWN')
    # Also check if notebook file exists
    nb = d['notebook']
    # Find project dir
    pattern = os.path.join(base, f'Machine Learning Project*{pnum}*')
    dirs = glob.glob(pattern)
    # Filter to exact match
    exact_dirs = []
    for dd in dirs:
        dirname = os.path.basename(dd)
        # Extract number from dirname
        import re
        m = re.search(r'Project[s]?\s+(\d+)', dirname)
        if m and int(m.group(1)) == pnum:
            exact_dirs.append(dd)
    pdir = exact_dirs[0] if exact_dirs else (dirs[0] if dirs else None)
    nb_exists = os.path.exists(os.path.join(pdir, nb)) if pdir else False
    
    status_ok = ds in ('OK_LOCAL', 'OK_BUILTIN', 'DOWNLOADED')
    entry = f"P{pnum:03d}  data={ds:20s}  nb={nb_exists}  task={d['task']}"
    if status_ok and nb_exists:
        ready.append((pnum, entry))
    else:
        blocked.append((pnum, entry, ds, nb_exists))

print(f"READY to execute: {len(ready)}")
for _, e in ready:
    print(f"  {e}")
print(f"\nBLOCKED: {len(blocked)}")
for _, e, ds, nb in blocked:
    reasons = []
    if not nb: reasons.append("missing_notebook")
    if ds not in ('OK_LOCAL', 'OK_BUILTIN', 'DOWNLOADED'): reasons.append(f"data_{ds}")
    print(f"  {e}  reason={','.join(reasons)}")
