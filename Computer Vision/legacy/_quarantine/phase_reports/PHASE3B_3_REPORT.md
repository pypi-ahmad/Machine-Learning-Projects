# Phase 3B-3 Report — Model Registry, Versioning & YOLO26 Migration

**Date:** 2025-07-13  
**Scope:** Model Registry API, YOLO26 migration (all 50 projects), benchmark integration, training defaults

---

## 1. What Was Implemented

### 1.1 Model Registry (`models/registry.py`)

A centralized **ModelRegistry** class that manages trained model lifecycle:

| Feature | Description |
|---|---|
| **Deterministic paths** | `models/<project>/<version>/best.pt` |
| **Version tracking** | Per-version metrics stored in `models/metadata.json` |
| **Auto-promotion** | Highest primary metric (mAP, accuracy, etc.) becomes active |
| **`resolve()` API** | Single function returns `(weights, version, used_pretrained_default)` — always succeeds |
| **Thread-safe persistence** | Atomic JSON writes with tmp-file swap |
| **CLI** | `python -m models.registry list|info|active|set-active` |

### 1.2 `resolve()` — The Core API

```python
from models.registry import resolve

weights, version, is_default = resolve("face_detection", "detect")
# Trained model registered  → ("models/face_detection/v2/best.pt", "v2", False)
# No trained model exists   → ("yolo26n.pt", None, True)
```

Every `modern.py` `load()` method now uses this pattern:

```python
def load(self):
    _key, _task = "object_detection", "detect"
    try:
        from models.registry import resolve
        weights, version, is_default = resolve(_key, _task)
    except Exception:
        weights, version, is_default = "yolo26n.pt", None, True
    print(f"  [{_key}] version={version}  weights={weights}  pretrained_fallback={is_default}")
    self.model = load_yolo(weights)
```

The `except Exception` fallback ensures inference **never crashes** even if the registry module is missing or corrupt.

### 1.3 YOLO26 Pretrained Defaults

All references to `yolo11` have been migrated to **YOLO26** across the entire codebase.

```python
YOLO26_DEFAULTS = {
    "detect": "yolo26n.pt",
    "seg":    "yolo26n-seg.pt",
    "pose":   "yolo26n-pose.pt",
    "cls":    "yolo26n-cls.pt",
    "obb":    "yolo26n-obb.pt",
}
```

---

## 2. Files Created / Modified

### New Files
| File | Purpose |
|---|---|
| `models/registry.py` | ModelRegistry class + resolve() + CLI (~340 lines) |
| `models/__init__.py` | Package exports: `ModelRegistry, YOLO26_DEFAULTS, resolve, get_registry, get_active, register_model` |
| `models/metadata.json` | Persistent version store (starts empty `{}`, populated by training) |
| `scripts/smoke_3b3.py` | 9-check smoke test suite |

### Modified Files (summary)

| Scope | Count | Change |
|---|---|---|
| `modern.py` (DL projects) | 18 | Added `resolve()` pattern with YOLO26 fallback |
| `modern.py` (utility projects) | 32 | Docstring yolo11→yolo26 where applicable |
| `train/*.py` (core pipelines) | 2 | Default weights yolo11→yolo26 |
| `train.py` (per-project) | 15 | Default weights + docstrings yolo11→yolo26 |
| `utils/yolo.py` | 1 | All 4 loader defaults → yolo26 |
| `utils/datasets.py` | 1 | Dataset registry entries yolo11→yolo26 |
| `benchmarks/run_all.py` | 1 | Added `model_version`, `model_path`, `used_pretrained_default` fields |
| `.gitignore` | 1 | Added `!models/registry.py`, `!models/__init__.py`, `!models/metadata.json` exceptions |
| `PHASE3_MODERNIZATION.md` | 1 | All yolo11 references → yolo26 |

**Total: 72 files touched**

---

## 3. Per-Project Registry Integration

### Detection Projects (11)
| Project | Registry Key | Fallback |
|---|---|---|
| P3 — Face Detection | `face_detection` | `yolo26n.pt` |
| P12 — Object Detection | `object_detection` | `yolo26n.pt` |
| P13 — Sudoku Solver | `sudoku_solver` | `yolo26n.pt` |
| P16 — Car Detection | `car_detection` | `yolo26n.pt` |
| P18 — Ball Tracking | `ball_tracking` | `yolo26n.pt` |
| P24 — Custom Object Detection | `custom_object_detection` | `yolo26n.pt` |
| P37 — Number Plate Detection | `number_plate_detection` | `yolo26n.pt` |
| P46 — Face Detection (Haar) | `face_detection_haar` | `yolo26n.pt` |
| P47 — Face Mask Detection | `face_mask_detection` | `yolo26n.pt` |
| P48 — Face Attributes | `face_attributes` | `yolo26n.pt` |
| P49 — Text Detection | `text_detection` | `yolo26n.pt` |

### Pose Projects (6)
| Project | Registry Key | Fallback |
|---|---|---|
| P4 — Facial Landmarks | `facial_landmarks` | `yolo26n-pose.pt` |
| P5 — Finger Counter | `finger_counter` | `yolo26n-pose.pt` |
| P6 — Hand Tracking | `hand_tracking` | `yolo26n-pose.pt` |
| P10 — Pose Detection | `pose_detection` | `yolo26n-pose.pt` |
| P17 — Blink Detection | `blink_detection` | `yolo26n-pose.pt` |
| P21 — Volume Controller | `volume_controller` | `yolo26n-pose.pt` |

### Segmentation Projects (1)
| Project | Registry Key | Fallback |
|---|---|---|
| P41 — Image Segmentation | `image_segmentation` | `yolo26n-seg.pt` |

### No-YOLO Projects (1 + 31 utility)
| Project | Notes |
|---|---|
| P9 — Virtual Painter | Pure OpenCV (no YOLO model) |
| P1, P2, P7, P8, P11, P14, P15, P19–P20, P22–P36, P38–P40, P42–P45, P50 | Utility/OpenCV-only — no model loading |

---

## 4. How to Train with Versioning

```bash
# Example: train a face mask detection model
python train/train_detection.py \
    --data datasets/face_mask/data.yaml \
    --model yolo26n.pt \
    --epochs 100 \
    --registry-project face_mask_detection

# What happens:
# 1. Trains YOLO26n on your dataset
# 2. Automatically registers best.pt in models/metadata.json
# 3. Sets as active if mAP > previous active version
# 4. Next time modern.py runs, resolve() returns trained weights
```

After training, `models/metadata.json` will contain:

```json
{
  "face_mask_detection": {
    "active": "v1",
    "versions": {
      "v1": {
        "path": "models/face_mask_detection/v1/best.pt",
        "metrics": {
          "date": "2025-07-13",
          "mAP": 0.81,
          "mAP50": 0.92
        }
      }
    }
  }
}
```

Manual version promotion:
```bash
python -m models.registry set-active face_mask_detection v2
python -m models.registry list
python -m models.registry info face_mask_detection
```

---

## 5. How Inference Selects the Best Model

```
modern.py load()
    │
    ├─ resolve(project_key, task)
    │     │
    │     ├─ Registry has active version with existing weights?
    │     │     └─ YES → return (trained_path, version, False)
    │     │
    │     └─ NO → return (YOLO26_DEFAULTS[task], None, True)
    │
    └─ Fallback (if registry import fails)
          └─ return ("yolo26n.pt" / task variant, None, True)
```

**Key guarantees:**
- Inference **never crashes** due to missing models
- Trained models are preferred over pretrained when available
- Pretrained YOLO26 fallback is always correct for the task type
- Diagnostic print shows exactly which weights were loaded

---

## 6. Benchmark Fields Added

`benchmarks/run_all.py` now resolves registry metadata for each project:

```
CSV columns (12 total):
  project, model_version, model_path, used_pretrained_default,
  load_time_s, avg_latency_ms, min_latency_ms, max_latency_ms,
  fps, mem_before_mb, mem_after_mb, status
```

New fields:
| Field | Type | Description |
|---|---|---|
| `model_version` | str | Active version tag (e.g. `"v2"`) or `""` if pretrained |
| `model_path` | str | Weights file path (trained or fallback name) |
| `used_pretrained_default` | bool | `True` when using YOLO26 pretrained fallback |

The benchmark runner automatically determines the correct task type (`detect`/`pose`/`seg`/`cls`) from the project's `category` attribute.

---

## 7. Smoke Test Results

**Script:** `scripts/smoke_3b3.py`

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

| Check | What it verifies |
|---|---|
| `registry_imports` | `ModelRegistry`, `resolve`, `YOLO26_DEFAULTS`, etc. importable |
| `resolve_defaults` | `resolve()` returns correct YOLO26 default for all 5 task types |
| `yolo_defaults` | `utils/yolo.py` function signatures contain yolo26 defaults |
| `benchmark_fields` | `benchmark_project()` source contains all 3 new fields |
| `core_imports` | `CVProject`, `PROJECT_REGISTRY`, `register`, `list_registered` importable |
| `ast_parse_all` | 77+ Python files parse without syntax errors |
| `all_50_modern_import` | All 50 `modern.py` files import successfully |
| `no_yolo11_refs` | Zero `yolo11` references in any `.py` file |
| `gitignore_models` | `.gitignore` has exceptions for registry source files |

---

## 8. .gitignore Safety

```gitignore
# Models directory — ignore weights, keep registry source
models/
!models/registry.py
!models/__init__.py
!models/metadata.json
models/**/best.*
```

This ensures:
- Downloaded/trained `.pt` weight files are not committed
- Registry source code (`registry.py`, `__init__.py`) IS committed
- Metadata (`metadata.json`) IS committed so version history is shared
- Per-project `best.*` weights are explicitly excluded

---

## 9. Architecture Summary

```
models/
├── __init__.py          # Package exports (resolve, YOLO26_DEFAULTS, etc.)
├── registry.py          # ModelRegistry class + resolve() + CLI
├── metadata.json        # Persistent JSON store (auto-populated by training)
└── <project>/           # Created by training pipeline
    └── <version>/
        └── best.pt      # Trained weights

utils/yolo.py            # load_yolo(), load_yolo_pose(), load_yolo_seg(), load_yolo_cls()
                         #   All defaults: yolo26n*.pt

benchmarks/run_all.py    # CSV includes model_version, model_path, used_pretrained_default

train/                   # All defaults: yolo26n*.pt
├── train_detection.py   #   --registry-project flag for auto-registration
└── train_segmentation.py

CV Project **/modern.py  # 50 files, 18 with resolve() pattern
```

---

*Phase 3B-3 complete. All 50 projects migrated to YOLO26 with registry-aware model loading.*
