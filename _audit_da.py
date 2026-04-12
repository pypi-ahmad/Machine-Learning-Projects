import json, pathlib

base = pathlib.Path("Data Analysis")
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    nbs = list(d.glob("*.ipynb"))
    for p in nbs:
        try:
            nb = json.loads(p.read_text("utf-8"))
            cells = nb.get("cells", [])
            md = " ".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown").lower()
            code = " ".join("".join(c["source"]) for c in cells if c["cell_type"] == "code").lower()
            checks = {
                "learning_obj": "learning objectives" in md,
                "common_mistakes": "common mistakes" in md,
                "mini_challenge": "mini challenge" in md,
                "kaggle_dl": "kaggle" in code,
                "key_findings": "key findings" in md or "key takeaways" in md,
                "limitations": "limitations" in md,
            }
            ok = all(checks.values())
            status = "OK" if ok else "NEEDS"
            missing = [k for k, v in checks.items() if not v]
            extra = f" missing={missing}" if not ok else ""
            total = len(cells)
            md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
            print(f"{status}: {d.name}/{p.name} ({total} cells, {md_count} md){extra}")
        except Exception as e:
            print(f"ERROR: {d.name}/{p.name}: {e}")
