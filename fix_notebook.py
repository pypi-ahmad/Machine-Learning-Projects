import nbformat

nb_path = r'Computer Vision/Oxford Flowers 102 Classification/Oxford Flowers 102 Classification.ipynb'
nb = nbformat.read(nb_path, as_version=4)

# Find and fix the configuration cell
for i, cell in enumerate(nb.cells):
    if 'CONFIGURATION & CONSTANTS' in cell.source and cell.cell_type == 'code':
        print(f'Found config cell at index {i}')
        
        # Replace the problematic line
        old_line = 'SAVE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))'
        new_line = 'SAVE_DIR = Path.cwd() / "Computer Vision" / "Oxford Flowers 102 Classification"'
        
        cell.source = cell.source.replace(old_line, new_line)
        print('Fixed SAVE_DIR')
        break

# Save
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Notebook updated successfully')
