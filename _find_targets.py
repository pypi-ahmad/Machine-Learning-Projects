"""Find suspicious config entries in _overhaul_v2.py"""
with open('_overhaul_v2.py', 'r', encoding='utf-8') as f:
    text = f.read()
    lines = text.split('\n')

# Find lines with target definitions
import re
for i, line in enumerate(lines):
    if '"target"' in line or "'target'" in line:
        # Extract the target value
        m = re.search(r'["\']target["\']\s*:\s*["\']([^"\']+)["\']', line)
        if m:
            target = m.group(1)
            # Print context
            start = max(0, i-2)
            ctx = lines[start:i+1]
            print(f"Line {i+1}, target='{target}':")
            for c in ctx:
                print(f"  {c.rstrip()}")
            print()
