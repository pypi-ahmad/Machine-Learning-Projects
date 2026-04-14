"""Validate notebooks 14 and 15."""
import json, sys, os

for nb_num, nb_dir in [(14, "14_Local_Financial_Report_Analyst"), (15, "15_Local_Contract_Clause_Finder")]:
    path = os.path.join("E:/Github/Machine-Learning-Projects/100_Local_AI_Projects/Local_RAG", nb_dir, "notebook.ipynb")
    nb = json.load(open(path, encoding="utf-8"))
    cc = [c for c in nb["cells"] if c["cell_type"] == "code"]
    mc = [c for c in nb["cells"] if c["cell_type"] == "markdown"]

    ns = {}
    ok = True
    # Run first 7 non-pip cells (everything up to LLM calls)
    count = 0
    for i, cell in enumerate(cc):
        if count >= 7:
            break
        src = "".join(cell["source"])
        if "!pip" in src:
            print(f"  NB{nb_num} Cell {i}: SKIP (pip)")
            continue
        try:
            exec(compile(src, f"cell_{i}", "exec"), ns)
            print(f"  NB{nb_num} Cell {i}: OK")
        except Exception as e:
            print(f"  NB{nb_num} Cell {i}: ERROR - {type(e).__name__}: {str(e)[:100]}")
            ok = False
            break
        count += 1

    status = "PASS" if ok else "FAIL"
    print(f"NB{nb_num}: {len(cc)} code, {len(mc)} md — {status}\n")

