"""Remove the orphaned old gen_ner code that was incorrectly merged."""
with open('_overhaul_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the boundaries:
# Line after gen_ner's closing ''')\n\n  (search for the pattern)
# The junk starts with "def gen_nlp_similarity(path, cfg):\n    from gliner" around line 2326
# The real gen_nlp_similarity starts around line 2641 with the regular pattern

# Find first gen_nlp_similarity (the fake one)
first_sim = None
second_sim = None
for i, line in enumerate(lines):
    if 'def gen_nlp_similarity(path, cfg):' in line:
        if first_sim is None:
            first_sim = i
        else:
            second_sim = i
            break

if first_sim is None or second_sim is None:
    print("Could not find boundaries")
    exit(1)

print(f"Removing lines {first_sim+1} to {second_sim} (0-indexed: {first_sim} to {second_sim-1})")
print(f"  First fake gen_nlp_similarity at line {first_sim+1}")
print(f"  Real gen_nlp_similarity at line {second_sim+1}")

# Remove the junk lines
new_lines = lines[:first_sim] + lines[second_sim:]

with open('_overhaul_v2.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Removed {second_sim - first_sim} lines")
print(f"New file has {len(new_lines)} lines")
