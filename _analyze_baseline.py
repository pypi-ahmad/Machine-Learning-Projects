"""Analyze Phase 6 baseline results in detail."""
import json
import re
from collections import Counter

r = json.load(open('audit_phase6/phase6_stress_report.json'))

# Separate PyCaret failures from real failures
pycaret_errors = 0
real_errors = []
error_patterns = Counter()

for proj in r['projects']:
    pnum = proj['project']
    for run in proj['runs']:
        for f in run.get('failures', []):
            msg = f.get('error_message', '') or ''
            etype = f.get('error_type', '') or ''
            
            if f.get('is_pycaret'):
                pycaret_errors += 1
                continue
            
            # Classify the actual error pattern
            if 'FileNotFoundError' in etype or 'FileNotFoundError' in msg:
                pattern = 'MISSING_FILE'
            elif 'ModuleNotFoundError' in etype:
                # Extract module name
                m = re.search(r"No module named '([^']+)'", msg)
                mod = m.group(1) if m else '?'
                pattern = f'MISSING_MODULE:{mod}'
            elif 'NameError' in etype:
                m = re.search(r"name '([^']+)' is not defined", msg)
                var = m.group(1) if m else '?'
                pattern = f'NAME_ERROR:{var}'
            elif 'ValueError' in etype:
                pattern = f'VALUE_ERROR:{msg[:80]}'
            elif 'KeyError' in etype:
                pattern = f'KEY_ERROR:{msg[:80]}'
            elif 'TypeError' in etype:
                pattern = f'TYPE_ERROR:{msg[:80]}'
            elif 'AttributeError' in etype:
                pattern = f'ATTR_ERROR:{msg[:80]}'
            elif 'IndexError' in etype:
                pattern = f'INDEX_ERROR:{msg[:80]}'
            else:
                pattern = f'{etype}:{msg[:60]}'
            
            error_patterns[pattern] += 1
            real_errors.append({
                'project': pnum,
                'cell': f['index'],
                'pattern': pattern,
                'is_std': f.get('is_standardized'),
                'is_lazy': f.get('is_lazypredict'),
            })

print(f"Total PyCaret errors (expected): {pycaret_errors}")
print(f"Total real errors: {len(real_errors)}")
print(f"\n{'='*80}")
print(f"Error Pattern Distribution:")
print(f"{'='*80}")
for pattern, count in error_patterns.most_common(30):
    print(f"  {count:4d}  {pattern}")

# Separate standardized cell errors from original cell errors
std_errors = [e for e in real_errors if e['is_std'] or e['is_lazy']]
orig_errors = [e for e in real_errors if not e['is_std'] and not e['is_lazy']]

print(f"\n{'='*80}")
print(f"Standardized cell errors: {len(std_errors)}")
for pattern, count in Counter(e['pattern'] for e in std_errors).most_common():
    print(f"  {count:4d}  {pattern}")

print(f"\nOriginal notebook cell errors: {len(orig_errors)}")
for pattern, count in Counter(e['pattern'] for e in orig_errors).most_common(20):
    print(f"  {count:4d}  {pattern}")

# Per-project summary excluding PyCaret
print(f"\n{'='*80}")
print(f"Per-project real errors (excluding PyCaret):")
print(f"{'='*80}")
proj_errors = Counter()
for e in real_errors:
    proj_errors[e['project']] += 1

for pnum, count in sorted(proj_errors.items()):
    patterns = set(e['pattern'].split(':')[0] for e in real_errors if e['project'] == pnum)
    print(f"  P{pnum:03d}: {count:3d} errors  [{', '.join(sorted(patterns))}]")
