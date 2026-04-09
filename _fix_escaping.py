"""Fix \n escaping issues in gen_ner template."""
import re

with open('_overhaul_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find gen_ner scope
start = content.find('def gen_ner(')
end = content.find('\ndef gen_nlp_similarity(')
if start == -1 or end == -1:
    print("Could not find gen_ner boundaries")
    exit(1)

portion = content[start:end]

# Replace print("\n...") -> print() \n    print("...")
# and    print(f"\n...") -> print() \n    print(f"...")
# Handle both \n (actual backslash-n in file) and \\n variants

# First handle the \\n cases (actual two chars in file: backslash backslash n)
portion = portion.replace('print("\\\\n', 'print()\n    print("')
portion = portion.replace("print('\\\\n", "print()\n    print('")
portion = portion.replace('print(f"\\\\n', 'print()\n    print(f"')

# Then handle the \n cases (actual backslash-n in file = two chars)
portion = portion.replace('print("\\n', 'print()\n    print("')
portion = portion.replace("print('\\n", "print()\n    print('")
portion = portion.replace('print(f"\\n', 'print()\n    print(f"')

content = content[:start] + portion + content[end:]

with open('_overhaul_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Fixed escaping in gen_ner")
