# Phase 3B-4 Report — Dataset Registry + Reproducible Accuracy Evaluation

**Date:** 2026-03-03  
**Scope:** Dataset config layer, accuracy evaluation runner, registry eval writeback, provenance metadata

---

## 1. What Was Added

### 1.1 Dataset Config Layer (`configs/datasets/`)

Small tracked YAML files that describe **where** each project's dataset lives
and **how** it should be evaluated.  Actual data files live under `data/`
(git-ignored).

| File | Purpose |
|---|---|
| `configs/datasets/README.md` | Schema docs and usage instructions |
| `configs/datasets/<project_key>.yaml` | Per-project dataset config (17 files) |
| `scripts/bootstrap_dataset_configs.py` | Auto-generates template configs |
| `scripts/validate_datasets.py` | Checks which datasets actually exist on disk |

**Config schema:**

```yaml
project_key: "face_detection"
task: "detect"
framework: "ultralytics"
dataset:
  name: "Face Detection (YOLO)"
  kind: "ultralytics_yaml"
  root: "data/face_detection"
  data_yaml: "data/face_detection/data.yaml"
  train: null
  val: null
  test: null
  notes: "Place Ultralytics-format dataset under data/face_detection/"
metrics:
  primary: "map50-95"
  promote_on_eval: false
```

### 1.2 Accuracy Evaluation Runner (`benchmarks/evaluate_accuracy.py`)

A CLI tool that evaluates model accuracy for every trainable project:

- Reads dataset configs from `configs/datasets/`
- Resolves model weights via `models.registry.resolve()`
- Runs `model.val()` for Ultralytics tasks (detect/seg/pose/obb/cls)
- Supports torchvision ImageFolder classification as fallback
- **Skips cleanly** when datasets or weights are missing
- Outputs structured CSV + JSON + provenance metadata

### 1.3 Registry Eval Writeback (`models/registry.py`)

New `record_eval()` method on `ModelRegistry`:

```python
reg.record_eval(
    project="face_detection",
    version="v1",
    dataset={"name": "WIDER Face", "kind": "ultralytics_yaml"},
    metrics={"map50": 0.82, "map50-95": 0.65},
    primary_name="map50-95",
    primary_value=0.65,
)
```

Stores under the version entry as:
```json
{
  "eval": {
    "evaluated_at": "2026-03-03T12:00:00+00:00",
    "dataset": {"name": "WIDER Face", "kind": "ultralytics_yaml"},
    "metrics": {"map50": 0.82, "map50-95": 0.65},
    "primary": {"name": "map50-95", "value": 0.65}
  }
}
```

Only triggers when `--write-registry` flag is provided AND a non-null version exists.

---

## 2. How to Bootstrap Configs

```bash
# Generate template configs for all 17 trainable projects
python scripts/bootstrap_dataset_configs.py

# Preview without writing
python scripts/bootstrap_dataset_configs.py --dry-run

# Overwrite existing configs
python scripts/bootstrap_dataset_configs.py --force
```

Output: one YAML per trainable project in `configs/datasets/`.

---

## 3. How to Run Accuracy Evaluation

```bash
# Evaluate all projects (skips when datasets missing)
python -m benchmarks.evaluate_accuracy

# Evaluate a single project
python -m benchmarks.evaluate_accuracy --project face_detection

# Limit to first N projects
python -m benchmarks.evaluate_accuracy --limit 3

# Force CPU
python -m benchmarks.evaluate_accuracy --device cpu

# Write eval metrics back into models/metadata.json
python -m benchmarks.evaluate_accuracy --write-registry

# Custom output directory
python -m benchmarks.evaluate_accuracy --outdir ./my_results
```

---

## 4. Output Schema

### CSV Columns (`benchmarks/results/accuracy_results.csv`)

| Column | Type | Description |
|---|---|---|
| `timestamp_iso` | str | ISO 8601 timestamp |
| `repo_name` | str | Repository folder name |
| `project_key` | str | Registry project key |
| `task` | str | detect / seg / pose / cls / obb |
| `dataset_name` | str | Human-readable dataset name |
| `dataset_kind` | str | ultralytics_yaml / imagefolder / custom |
| `dataset_ref` | str | data_yaml path or val path |
| `model_version` | str? | Active version tag or null |
| `model_path` | str | Weights file (trained or pretrained name) |
| `used_pretrained_default` | bool | True if YOLO26 fallback |
| `metric_primary_name` | str | e.g. "map50-95", "acc_top1" |
| `metric_primary_value` | float? | Numeric value or null |
| `metrics_json` | str | Compact JSON of all metrics |
| `status` | str | ok / missing_dataset_config / missing_dataset_files / ... |
| `error` | str | Short error message (if any) |

### Provenance Metadata (`benchmarks/results/accuracy_run_meta.json`)

```json
{
  "timestamp_iso": "2026-03-02T20:37:06.828204+00:00",
  "python_version": "3.13.12",
  "torch_version": "2.10.0+cu130",
  "ultralytics_version": "8.4.19",
  "cuda_available": true,
  "cuda_device_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "git_commit": "d9e4875",
  "command_args": { "project": null, "limit": null, "device": "auto", ... },
  "platform": "Windows-11-10.0.26200-SP0"
}
```

### JSON Results (`benchmarks/results/accuracy_results.json`)

Array of result dicts matching CSV columns.

---

## 5. Status Values

| Status | Meaning |
|---|---|
| `ok` | Evaluation completed successfully |
| `missing_dataset_config` | No YAML in `configs/datasets/` |
| `missing_dataset_files` | Config exists but data files not found on disk |
| `weights_unavailable` | Weights file not found / download failed |
| `unsupported_cls_backend` | Cannot determine classification eval backend |
| `unsupported_task` | Unknown task type |
| `eval_error` | Runtime error during evaluation |

---

## 6. Registry Writeback Behavior

- Only active when `--write-registry` CLI flag is passed
- Only writes for projects with `status == "ok"` AND `model_version is not None`
- Stores eval data under `versions.<version>.eval` in `metadata.json`
- Preserves all existing keys (non-breaking extension)
- Does **not** auto-promote active version (by design)

---

## 7. Files Created / Modified

### New Files (8)

| File | Purpose |
|---|---|
| `configs/datasets/README.md` | Documentation for dataset configs |
| `configs/datasets/*.yaml` (×17) | Per-project dataset configs |
| `scripts/bootstrap_dataset_configs.py` | Config generator |
| `scripts/validate_datasets.py` | Config + data path validator |
| `benchmarks/evaluate_accuracy.py` | Accuracy evaluation runner |
| `PHASE3B_4_REPORT.md` | This report |

### Modified Files (2)

| File | Change |
|---|---|
| `models/registry.py` | Added `record_eval()` method (~45 lines) |
| `.gitignore` | Added `benchmarks/results/` exclusion + configs note |

---

## 8. Smoke Test Summary

### Phase 3B-3 Smoke Tests (existing)
```
  [PASS] registry_imports
  [PASS] resolve_defaults
  [PASS] yolo_defaults
  [PASS] benchmark_fields
  [PASS] core_imports
  [PASS] ast_parse_all
  [PASS] all_50_modern_import
  [PASS] no_yolo11_refs
  [PASS] gitignore_models

SMOKE TEST SUMMARY: 9/9 passed
```

### Phase 3B-4 Validation
```
Bootstrap:  17 created, 0 skipped, 17 total trainable projects
Validator:  17 configs, 17 configured, 0 with data present
Evaluator:  17 projects → 0 evaluated, 17 skipped (missing_dataset_files)
            No crashes, proper CSV/JSON/META output generated
record_eval: Tested with temp registry — stores eval data correctly
```

---

## 9. Architecture

```
configs/
└── datasets/
    ├── README.md
    ├── face_detection.yaml      ─┐
    ├── object_detection.yaml     │  17 dataset configs
    ├── image_segmentation.yaml   │  (tracked in git)
    └── ...                      ─┘

benchmarks/
├── evaluate_accuracy.py     ← NEW: accuracy evaluation CLI
├── run_all.py               ← existing: performance benchmark
└── results/                 ← git-ignored
    ├── accuracy_results.csv
    ├── accuracy_results.json
    └── accuracy_run_meta.json

models/
├── registry.py              ← UPDATED: +record_eval()
├── __init__.py
└── metadata.json

scripts/
├── bootstrap_dataset_configs.py  ← NEW
├── validate_datasets.py          ← NEW
└── smoke_3b3.py                  ← existing
```

---

*Phase 3B-4 complete.  Dataset configs bootstrapped, accuracy evaluator runs cleanly (skips when data missing), registry writeback tested.*
