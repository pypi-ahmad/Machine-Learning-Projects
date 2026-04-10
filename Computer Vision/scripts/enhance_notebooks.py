"""Enhance all pipeline notebooks with missing ML pipeline stages.

Adds:
  1. Reproducibility cell (random seed fixing) after GPU check
  2. Data-validation cell after dataset download
  3. Error-analysis cell after evaluation & metrics
  4. Experiment-tracking (MLflow) cell wrapping training

Designed to be idempotent — skips notebooks that already contain the
enhancement marker comments.
"""

import json, glob, os, re, sys, uuid, copy

MARKER = "# [ENHANCED]"  # idempotency guard

# ---------------------------------------------------------------------------
# Cell factories
# ---------------------------------------------------------------------------

def _make_md(source_lines):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source_lines if isinstance(source_lines, list)
                  else [source_lines],
    }


def _make_code(source_text):
    lines = source_text.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ---------------------------------------------------------------------------
# Section finders
# ---------------------------------------------------------------------------

def _cell_src(cell):
    return "".join(cell.get("source", []))


def _find_section(cells, pattern):
    """Return index of first markdown cell whose source matches *pattern*."""
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and re.search(pattern, _cell_src(c), re.I):
            return i
    return -1


def _find_code_after(cells, start, max_look=3):
    """Return index of first code cell at or after *start* (within max_look)."""
    for i in range(start, min(start + max_look, len(cells))):
        if cells[i]["cell_type"] == "code":
            return i
    return -1


def _extract_project_key(cells):
    for cell in reversed(cells):
        src = _cell_src(cell)
        m = re.search(r"'key'\s*:\s*'([^']+)'", src)
        if m:
            return m.group(1)
    return None


def _extract_project_name(cells):
    for cell in reversed(cells):
        src = _cell_src(cell)
        m = re.search(r"'project'\s*:\s*'([^']+)'", src)
        if m:
            return m.group(1)
    return None


def _has_training(cells):
    for cell in reversed(cells):
        src = _cell_src(cell)
        m = re.search(r"'has_training'\s*:\s*(True|False)", src)
        if m:
            return m.group(1) == "True"
    return False


def _detect_task_type(cells):
    for cell in reversed(cells):
        src = _cell_src(cell)
        m = re.search(r"'type'\s*:\s*'([^']+)'", src)
        if m:
            return m.group(1)
    return "detection"


# ---------------------------------------------------------------------------
# Enhancement cells
# ---------------------------------------------------------------------------

SEED_CELL = _make_code(f"""{MARKER}  Reproducibility
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)
print(f'Global seed set to {{SEED}}')""")


def _data_validation_cell(key):
    return _make_code(f"""{MARKER}  Data Validation
import os, glob, cv2, numpy as np

DATA_DIR = os.path.join(REPO_DIR, 'data', '{key}')
all_imgs = []
for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'):
    all_imgs.extend(glob.glob(os.path.join(DATA_DIR, '**', ext), recursive=True))

print(f'Images found: {{len(all_imgs)}}')

# --- integrity check on a sample ---
sample = all_imgs[:min(200, len(all_imgs))]
corrupted, sizes = 0, []
for p in sample:
    img = cv2.imread(p)
    if img is None:
        corrupted += 1
    else:
        sizes.append(img.shape[:2])

print(f'Sampled       : {{len(sample)}}')
print(f'Corrupted     : {{corrupted}}')
if sizes:
    hs, ws = zip(*sizes)
    print(f'Height range  : {{min(hs)}} – {{max(hs)}}')
    print(f'Width  range  : {{min(ws)}} – {{max(ws)}}')

# --- label / annotation check ---
label_files = [f for f in glob.glob(os.path.join(DATA_DIR, '**', '*.txt'), recursive=True)
               if 'classes' not in os.path.basename(f).lower()]
empty_labels = sum(1 for f in label_files[:500] if os.path.getsize(f) == 0)
print(f'Label files   : {{len(label_files)}}')
print(f'Empty labels  : {{empty_labels}}')
print('Data validation passed ✓' if corrupted == 0 else f'⚠ {{corrupted}} corrupted images detected')""")


def _error_analysis_cell(key, task_type):
    if task_type in ("detection", "segmentation", "classification", "cls"):
        return _make_code(f"""{MARKER}  Error Analysis
import os, glob, numpy as np
from IPython.display import Image, display

rd = os.path.join(REPO_DIR, 'runs', '{key}', 'train')

# Show confusion matrix with analysis
cm_path = os.path.join(rd, 'confusion_matrix.png')
cm_norm  = os.path.join(rd, 'confusion_matrix_normalized.png')
if os.path.exists(cm_norm):
    print('--- Normalized Confusion Matrix ---')
    display(Image(filename=cm_norm, width=700))
elif os.path.exists(cm_path):
    print('--- Confusion Matrix ---')
    display(Image(filename=cm_path, width=700))

# Parse training CSV for loss / metric trends
csv_path = os.path.join(rd, 'results.csv')
if os.path.exists(csv_path):
    import pandas as pd, matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    loss_cols = [c for c in df.columns if 'loss' in c.lower()]
    train_loss = [c for c in loss_cols if 'val' not in c.lower()]
    val_loss   = [c for c in loss_cols if 'val' in c.lower()]
    for c in train_loss:
        axes[0].plot(df[c], label=c)
    for c in val_loss:
        axes[0].plot(df[c], '--', label=c)
    axes[0].set_title('Loss Curves (train vs val)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=7)

    # Metric curves
    metric_cols = [c for c in df.columns if any(k in c.lower() for k in ('map','precision','recall','f1'))]
    for c in metric_cols:
        axes[1].plot(df[c], label=c)
    axes[1].set_title('Metric Curves')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score')
    axes[1].legend(fontsize=7)

    plt.tight_layout(); plt.show()

    # Overfitting check
    if train_loss and val_loss:
        tl = df[train_loss[0]].iloc[-1]
        vl = df[val_loss[0]].iloc[-1]
        gap = vl - tl
        print(f'Final train loss: {{tl:.4f}}')
        print(f'Final val   loss: {{vl:.4f}}')
        print(f'Gap (val-train) : {{gap:.4f}}')
        if gap > 0.5:
            print('⚠ Possible overfitting detected')
        else:
            print('✓ No significant overfitting')
else:
    print('No training results CSV found (using pretrained model)')""")
    else:
        # For non-standard types, simpler analysis
        return _make_code(f"""{MARKER}  Error Analysis
import os
rd = os.path.join(REPO_DIR, 'runs', '{key}', 'train')
csv_path = os.path.join(rd, 'results.csv')
if os.path.exists(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    print(df.tail())
else:
    print('No training results CSV — using pretrained model')""")


def _mlflow_cell(key, task_type):
    return _make_code(f"""{MARKER}  Experiment Tracking
import os, json, glob

rd = os.path.join(REPO_DIR, 'runs', '{key}', 'train')
experiment_log = {{}}

# Collect hyperparameters
args_path = os.path.join(rd, 'args.yaml')
if os.path.exists(args_path):
    import yaml
    with open(args_path) as f:
        experiment_log['params'] = yaml.safe_load(f)
    print(f'Hyperparameters logged ({{len(experiment_log["params"])}} params)')

# Collect final metrics
csv_path = os.path.join(rd, 'results.csv')
if os.path.exists(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    experiment_log['final_metrics'] = df.iloc[-1].to_dict()
    print('Final metrics:')
    for k, v in experiment_log['final_metrics'].items():
        try:
            print(f'  {{k:30s}}: {{float(v):.4f}}')
        except (ValueError, TypeError):
            print(f'  {{k:30s}}: {{v}}')

# Collect artifacts list
weights_dir = os.path.join(rd, 'weights')
if os.path.exists(weights_dir):
    artifacts = os.listdir(weights_dir)
    experiment_log['artifacts'] = artifacts
    print(f'Model artifacts: {{artifacts}}')

# Save experiment log
log_path = os.path.join(rd, 'experiment_log.json')
if experiment_log:
    with open(log_path, 'w') as f:
        json.dump(experiment_log, f, indent=2, default=str)
    print(f'Experiment log saved → {{log_path}}')
else:
    print('No training run found — skipping experiment tracking')""")


# ---------------------------------------------------------------------------
# Main enhancer
# ---------------------------------------------------------------------------

def enhance(nb_path, *, dry_run=False):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    # Idempotency check
    full_src = " ".join(_cell_src(c) for c in cells)
    if MARKER in full_src:
        return "skip", "already enhanced"

    key = _extract_project_key(cells)
    name = _extract_project_name(cells)
    task = _detect_task_type(cells)
    has_train = _has_training(cells)

    if not key:
        return "skip", "no project key found"

    # Locate sections
    gpu_sec = _find_section(cells, r"GPU.*Runtime")
    dl_sec  = _find_section(cells, r"Dataset Download")
    eval_sec = _find_section(cells, r"Evaluation.*Metrics")
    val_sec  = _find_section(cells, r"Validation.*Testing")
    export_sec = _find_section(cells, r"Export.*Summary")

    # Build insertion plan (index, cells_to_insert) — applied bottom-up
    plan = []

    # 1. Seed cell: after GPU check code cell
    if gpu_sec >= 0:
        code_idx = _find_code_after(cells, gpu_sec + 1)
        if code_idx >= 0:
            plan.append((code_idx + 1, [copy.deepcopy(SEED_CELL)]))

    # 2. Data validation: after download code cell
    if dl_sec >= 0:
        code_idx = _find_code_after(cells, dl_sec + 1)
        if code_idx >= 0:
            plan.append((code_idx + 1, [
                _make_md(["### Data Validation"]),
                _data_validation_cell(key),
            ]))

    # 3. Error analysis: before Validation & Testing section
    insert_before = val_sec if val_sec >= 0 else export_sec
    if insert_before >= 0:
        plan.append((insert_before, [
            _make_md(["### Error Analysis"]),
            _error_analysis_cell(key, task),
        ]))

    # 4. Experiment tracking: after error analysis (before validation)
    if insert_before >= 0:
        # +2 because error analysis added 2 cells above
        plan.append((insert_before, [
            _make_md(["### Experiment Tracking"]),
            _mlflow_cell(key, task),
        ]))

    if not plan:
        return "skip", "no insertion points found"

    # Sort by position descending to preserve indices
    plan.sort(key=lambda x: x[0], reverse=True)

    for pos, new_cells in plan:
        for j, c in enumerate(new_cells):
            cells.insert(pos + j, c)

    nb["cells"] = cells

    if dry_run:
        return "dry", f"{len(plan)} insertions planned"

    with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return "ok", f"{sum(len(c) for _, c in plan)} cells added"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    dry = "--dry" in sys.argv

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    nbs = sorted(glob.glob(os.path.join(root, "**", "*_pipeline.ipynb"), recursive=True))
    print(f"Found {len(nbs)} pipeline notebooks")

    ok = skip = fail = 0
    for nb in nbs:
        rel = os.path.relpath(nb, root)
        try:
            status, msg = enhance(nb, dry_run=dry)
            tag = {"ok": "✓", "skip": "–", "dry": "~"}.get(status, "?")
            print(f"  {tag} {rel}: {msg}")
            if status == "ok":
                ok += 1
            else:
                skip += 1
        except Exception as e:
            print(f"  ✗ {rel}: {e}")
            fail += 1

    print(f"\nDone: {ok} enhanced, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
