"""Fix syntax errors in notebooks 41 and 48."""
import json
import sys

log = open(r'E:\Github\Machine-Learning-Projects\_fix_log.txt', 'w', encoding='utf-8')

def log_print(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

# ── Fix Project 41 ──
path41 = r'E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\CrewAI_Multi-Agent_Systems\41_CrewAI_Startup_Validation_Crew\notebook.ipynb'
try:
    with open(path41, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = cell['source']
        if isinstance(src, list):
            new_src = []
            for line in src:
                if "role='Devil" in line and "Advocate'" in line:
                    line = '    role="Devil\'s Advocate",\n'
                new_src.append(line)
            cell['source'] = new_src

    with open(path41, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    log_print('Fixed project 41')
except Exception as e:
    log_print(f'Error fixing 41: {e}')

# ── Inspect Project 48 ──
path48 = r'E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\CrewAI_Multi-Agent_Systems\48_CrewAI_Customer_Success_Crew\notebook.ipynb'
try:
    with open(path48, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    for ci, (i, c) in enumerate(code_cells):
        src = c['source']
        if isinstance(src, list):
            full = ''.join(src)
        else:
            full = src
        try:
            compile(full, f'<cell_{ci}>', 'exec')
        except SyntaxError as e:
            log_print(f'Project 48 - code cell {ci} (nb cell {i}): {e}')
            if isinstance(src, list):
                for li, line in enumerate(src):
                    log_print(f'  {li:3d}: {repr(line)}')
    log_print('Inspected project 48')
except Exception as e:
    log_print(f'Error inspecting 48: {e}')

log.close()
