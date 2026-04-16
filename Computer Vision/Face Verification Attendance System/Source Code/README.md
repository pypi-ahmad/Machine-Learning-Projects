# Face Verification Attendance System

> **Task:** Face Verification &nbsp;|&nbsp; **Key:** `face_verification_attendance` &nbsp;|&nbsp; **Framework:** InsightFace + YOLO

---

## Overview

Enrollment-based face verification system for automated attendance tracking. Uses InsightFace ArcFace embeddings for identity matching with cosine similarity, a configurable similarity threshold, unknown-face handling, session-level dedup, and full CSV/JSON export of attendance logs.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Face Verification / Attendance |
| **Detection** | YOLO face detector (primary) / InsightFace RetinaFace (fallback) |
| **Embeddings** | InsightFace ArcFace (buffalo_l, 512-d) |
| **Matching** | Cosine similarity with configurable threshold (default: 0.45) |
| **Dataset** | LFW multi-identity face images (automatic local bootstrap) |
| **Key Metrics** | Verification accuracy, FAR, FRR |

## Dataset

- **Source:** `sklearn.datasets.fetch_lfw_people` (LFW). If that path is unavailable, the bootstrap falls back to the repo's shared dataset helper.
- **Layout:** ImageFolder — `identity_name/img1.jpg, img2.jpg, ...`
- **Prepared Path:** `data/face_verification_attendance/processed/identities/`
- **Download:** Automatic on first `python train.py` or `python data_bootstrap.py` run
- **Force re-download:** `python train.py --force-download` or `python data_bootstrap.py --force`

## Project Structure

```
Face Verification Attendance System/
└── Source Code/
    ├── config.py          # FaceAttendanceConfig dataclass
    ├── face_detector.py   # YOLO / InsightFace face detection
    ├── embedder.py        # InsightFace ArcFace embedding extraction
    ├── enrollment.py      # Gallery management — enroll / save / load
    ├── matcher.py         # Cosine similarity matching against gallery
    ├── attendance_log.py  # Timestamped attendance with session dedup
    ├── parser.py          # Full pipeline: detect → embed → match → log
    ├── validator.py       # Quality checks and validation warnings
    ├── visualize.py       # Overlay renderer with attendance panel
    ├── export.py          # JSON / CSV export of results
    ├── infer.py           # CLI — enrollment and verification modes
    ├── modern.py          # CVProject subclass — @register("face_verification_attendance")
    ├── train.py           # Dataset download + verification evaluation
    ├── data_bootstrap.py  # Idempotent dataset bootstrap
    ├── requirements.txt   # Project dependencies
    └── README.md          # This file
```

## Quick Start

### 1. Enroll Identities

```bash
# Single image enrollment
python infer.py --mode enroll --identity "Alice" --source alice.jpg

# Directory of images for one identity
python infer.py --mode enroll --identity "Alice" --source alice_images/

# Directory of identities (ImageFolder layout)
python infer.py --mode enroll --source faces/
#   faces/Alice/img1.jpg, img2.jpg
#   faces/Bob/img1.jpg, img2.jpg
```

### 2. Verify / Take Attendance

```bash
# Single image
python infer.py --mode verify --source test.jpg

# Webcam (live)
python infer.py --mode verify --source 0

# Video file
python infer.py --mode verify --source classroom.mp4

# With exports
python infer.py --mode verify --source 0 --export-csv attendance.csv --export-json results.json
```

### 3. Programmatic API

```python
from core import discover_projects
from core.registry import PROJECT_REGISTRY

discover_projects()
proj = PROJECT_REGISTRY["face_verification_attendance"]()
proj.load()

# Enroll
proj.enroll("Alice", "path/to/alice.jpg")
proj.enroll("Bob", "path/to/bob.jpg")
proj.save_gallery()

# Verify
proj.load_gallery()
result = proj.predict("path/to/test.jpg")
for face in result["faces"]:
    print(f"{face['identity']}: {face['similarity']:.3f} (matched={face['matched']})")

# Export
proj.export_attendance_csv("attendance.csv")
```

### 4. Evaluation

```bash
cd "Face Verification Attendance System/Source Code"
python data_bootstrap.py                     # prepare LFW locally
python train.py                              # download + evaluate
python train.py --force-download             # re-download dataset
python train.py --max-identities 100         # more identities
python train.py --threshold 0.40             # lower threshold
```

## CLI Reference

```
python infer.py [OPTIONS]

Required:
  --source PATH         Image, directory, video, or '0' for webcam

Mode:
  --mode {enroll,verify} Pipeline mode (default: verify)

Enrollment:
  --identity NAME       Identity name (single-image enrollment)
  --gallery-dir DIR     Gallery directory (default: gallery/)

Verification:
  --threshold FLOAT     Cosine similarity threshold (default: 0.45)

Export:
  --export-json PATH    JSON export path
  --export-csv PATH     CSV export path
  --save-annotated      Save annotated images
  --output-dir DIR      Output directory (default: output/)

Other:
  --config PATH         YAML/JSON config file
  --no-display          Disable GUI windows
  --force-download      Force dataset re-download
```

## Features

- **InsightFace ArcFace embeddings** — 512-d normalized vectors (buffalo_l)
- **Dual detection backend** — YOLO face detector (fast) with InsightFace fallback
- **Multi-image enrollment** — Average embeddings for robust identity representation
- **Gallery persistence** — Save/load enrolled identities as JSON
- **Session dedup** — Configurable cooldown prevents duplicate attendance entries
- **Unknown face handling** — Faces below threshold labelled as "Unknown"
- **Live webcam mode** — Real-time attendance from camera feed
- **Attendance panel** — Visual overlay showing recently logged identities
- **CSV/JSON export** — Full attendance records export

## Configuration

Create a YAML config file to override defaults:

```yaml
# face_attendance_config.yaml
similarity_threshold: 0.45
dedup_cooldown_sec: 300
gallery_dir: gallery
det_confidence: 0.4
embedding_model: buffalo_l
show_attendance_panel: true
```

```bash
python infer.py --mode verify --source 0 --config face_attendance_config.yaml
```

## Dependencies

```bash
pip install -r requirements.txt
```

Optional GPU runtime:

```bash
pip install onnxruntime-gpu
```
