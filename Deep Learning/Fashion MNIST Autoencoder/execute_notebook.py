#!/usr/bin/env python
"""Execute the Fashion MNIST Autoencoder notebook end-to-end."""
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import sys

notebook_path = "Fashion MNIST Autoencoder.ipynb"

print(f"Loading notebook: {notebook_path}")
with open(notebook_path, encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print(f"Executing notebook (timeout: 600s)...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

try:
    out, resources = ep.preprocess(nb, {'metadata': {'path': '.'}})
    print("\n✅ Notebook executed successfully")
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(out, f)
    print("✅ Notebook saved with execution results")
    
    for cell in out.cells:
        if hasattr(cell, 'outputs'):
            for output in cell.outputs:
                if hasattr(output, 'text') and 'EXECUTION_COMPLETE' in output.text:
                    print("✅ EXECUTION_COMPLETE marker found")
                    sys.exit(0)
    
    print("⚠️ EXECUTION_COMPLETE marker not found, but notebook executed without errors")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Execution error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
