import json, os, shutil

BASE = r"E:\Github\Machine-Learning-Projects\LangGraph"

old_folders = ["36_Job_Application_Assistant","37_Procurement_Approval_Copilot","38_Medical_Triage_Note_Workflow","39_Research_Workflow_Memory","40_Travel_Planner_Checkpoints"]
for folder in old_folders:
    p = os.path.join(BASE, folder)
    if os.path.exists(p):
        shutil.rmtree(p)
        print("Deleted: " + folder)

nbs = {"36_Procurement_Review_Flow":"procurement_review_flow.ipynb","37_Travel_Planner_Flow":"travel_planner_flow.ipynb","38_Research_Workflow_Memory":"research_workflow_memory.ipynb","39_Ticket_Escalation_Router":"ticket_escalation_router.ipynb","40_Compliance_Checklist_Flow":"compliance_checklist_flow.ipynb"}
for folder, fn in nbs.items():
    path = os.path.join(BASE, folder, fn)
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    if len(cells) == 1 and cells[0].get("cell_type") == "raw":
        src = "".join(cells[0].get("source", []))
        real = json.loads(src)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(real, f, indent=1, ensure_ascii=False)
        rc = real.get("cells", [])
        md = sum(1 for c in rc if c["cell_type"] == "markdown")
        co = sum(1 for c in rc if c["cell_type"] == "code")
        print("FIXED " + fn + ": " + str(len(rc)) + " cells (" + str(md) + " md, " + str(co) + " code)")
    else:
        md = sum(1 for c in cells if c["cell_type"] == "markdown")
        co = sum(1 for c in cells if c["cell_type"] == "code")
        print("OK " + fn + ": " + str(len(cells)) + " cells (" + str(md) + " md, " + str(co) + " code)")
print("Done!")

