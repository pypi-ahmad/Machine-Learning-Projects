import json, sys

files = {
    "36": r"E:\Github\Machine-Learning-Projects\LangGraph\36_Procurement_Review_Flow\procurement_review_flow.ipynb",
    "37": r"E:\Github\Machine-Learning-Projects\LangGraph\37_Travel_Planner_Flow\travel_planner_flow.ipynb",
    "38": r"E:\Github\Machine-Learning-Projects\LangGraph\38_Research_Workflow_Memory\research_workflow_memory.ipynb",
    "39": r"E:\Github\Machine-Learning-Projects\LangGraph\39_Ticket_Escalation_Router\ticket_escalation_router.ipynb",
    "40": r"E:\Github\Machine-Learning-Projects\LangGraph\40_Compliance_Checklist_Flow\compliance_checklist_flow.ipynb",
}

for num, path in files.items():
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        nb = json.loads(raw)
        cells = nb["cells"]
        md = 0
        code = 0
        for c in cells:
            if c["cell_type"] == "markdown":
                md += 1
            elif c["cell_type"] == "code":
                code += 1
        print(f"{num}: {len(cells)} cells ({md} md, {code} code) - VALID")
    except Exception as e:
        print(f"{num}: ERROR - {e}")

