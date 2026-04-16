"""Rebuild notebook 100 from scratch."""
import json

# Read the broken file, strip the #%% raw prefix
path100 = r'E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\Coding_and_Developer_Agents\100_Local_AI_Ops_Mini_Platform\notebook.ipynb'
with open(path100, 'r', encoding='utf-8') as f:
    content = f.read()

# Strip the #%% raw prefix if present
if content.startswith('#%% raw\n'):
    content = content[len('#%% raw\n'):]
elif content.startswith('#%% raw\r\n'):
    content = content[len('#%% raw\r\n'):]

nb = json.loads(content)
print(f'Parsed: {len(nb.get("cells", []))} cells')

# Re-save properly
with open(path100, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')
print('Re-saved notebook 100')

# Verify
with open(path100, 'r', encoding='utf-8') as f:
    check = json.load(f)
print(f'Verified: {len(check["cells"])} cells')

