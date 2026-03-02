#!/usr/bin/env python3
"""
Phase 3b — Step 7: Final Dataset Fix Report Generator
Generates a comprehensive Markdown report summarizing all dataset resolution work.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "audit_phase3"
REPORT_DIR.mkdir(exist_ok=True)

# Load validation results
val_path = REPORT_DIR / "dataset_validation_report.json"
results = json.load(open(val_path, encoding="utf-8"))

ok = [r for r in results if r["status"] == "OK"]
warn = [r for r in results if r["status"] == "WARN"]
fail = [r for r in results if r["status"] == "FAIL"]

report = []
report.append("# Phase 3b — Dataset Resolution Report")
report.append("")
report.append("## Executive Summary")
report.append("")
report.append(f"- **Total blocked projects**: 31")
report.append(f"- **Fully resolved**: {len(ok)}")
report.append(f"- **Warnings (known limitations)**: {len(warn)}")
report.append(f"- **Failed**: {len(fail)}")
report.append("")

report.append("## Resolution Methods Applied")
report.append("")
report.append("| Method | Count | Description |")
report.append("|--------|-------|-------------|")
report.append("| Kaggle Download | 12 | Downloaded via `kaggle datasets download` |")
report.append("| Direct URL | 6 | Downloaded from UCI, GitHub, or other direct links |")
report.append("| Archive Extract | 3 | Downloaded + extracted (ZIP, RAR, TGZ) |")
report.append("| Built-in Loader | 3 | Already use sklearn/tfds built-in datasets (no fix needed) |")
report.append("| Self-Download | 2 | Framework auto-downloads (torchvision FashionMNIST, CIFAR10) |")
report.append("| Sibling Copy | 2 | Shared data with another project (creditcard.csv, MovieLens) |")
report.append("| Path Fix Only | 3 | Data existed, only notebook paths needed fixing |")
report.append("")

report.append("## Data Directory Structure")
report.append("")
report.append("All datasets centralized under `data/<project_slug>/`:")
report.append("")
report.append("```")
# List actual data dirs
data_dir = ROOT / "data"
if data_dir.exists():
    slugs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    for slug in slugs[:40]:
        report.append(f"  data/{slug}/")
    if len(slugs) > 40:
        report.append(f"  ... and {len(slugs) - 40} more")
report.append("```")
report.append("")

report.append("## Notebook Modifications")
report.append("")
report.append("Every patched notebook follows this pattern:")
report.append("")
report.append("```python")
report.append("from pathlib import Path")
report.append("DATA_DIR = Path.cwd().parent / 'data' / '<project_slug>'")
report.append("")
report.append("# Then all file loads use DATA_DIR:")
report.append("df = pd.read_csv(DATA_DIR / 'data.csv')")
report.append("```")
report.append("")

report.append("## Validation Results")
report.append("")
report.append("| # | Project | Status | Notes |")
report.append("|---|---------|--------|-------|")
for i, r in enumerate(results, 1):
    name = r["project"].replace("|", "\\|")
    status = {"OK": "✅ OK", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL"}[r["status"]]
    notes = r.get("notes", "").replace("|", "\\|")
    if not notes and r["status"] == "OK":
        notes = f"loader={r['loader']}"
    report.append(f"| {i} | {name} | {status} | {notes} |")
report.append("")

report.append("## Known Limitations")
report.append("")
report.append("1. **P129 Weather Data** (`minute_weather.csv`): Source URL is bot-protected (Google Drive). ")
report.append("   Must be manually downloaded from: https://drive.google.com/...")
report.append("   Place file at: `data/weather_data_clustering_using_k-means/minute_weather.csv`")
report.append("")
report.append("2. **P125 GloVe Embeddings** (~862 MB): Skipped due to size. Download from:")
report.append("   https://nlp.stanford.edu/data/glove.6B.zip")
report.append("   Place `glove.6B.100d.txt` at: `data/text_classification_keras_consumer_complaints/`")
report.append("")
report.append("3. **Cell outputs**: Some notebooks still contain old `C:\\Users\\` paths in ")
report.append("   cached output cells (not source). These are cosmetic and do not affect execution.")
report.append("")

report.append("## Files Generated This Phase")
report.append("")
report.append("| File | Purpose |")
report.append("|------|---------|")
report.append("| `dataset_downloader.py` | Master download/extract script |")
report.append("| `notebook_patcher.py` | Bulk notebook path patcher (22 projects) |")
report.append("| `validate_datasets.py` | Validation script (31 projects) |")
report.append("| `audit_phase3/dataset_validation_report.json` | Machine-readable validation results |")
report.append("| `audit_phase3/phase3b_report.md` | This report |")
report.append("")

report.append("## Directory Fixes Applied")
report.append("")
report.append("- **P21 Face Expression**: Flattened `train/train/<class>/` → `train/<class>/`")
report.append("- **P4 Indian Dance**: Added idempotent cell to reorganize flat images into class subfolders")
report.append("- **P117 Book Crossing**: Copied `BX-Book-Ratings.csv` from nested Kaggle download to data root")
report.append("")

out_path = REPORT_DIR / "phase3b_report.md"
out_path.write_text("\n".join(report), encoding="utf-8")
print(f"Report written to: {out_path.relative_to(ROOT)}")
print(f"  {len(ok)} OK / {len(warn)} WARN / {len(fail)} FAIL")
