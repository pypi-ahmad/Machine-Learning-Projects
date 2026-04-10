# Interactive Video Object Cutout Studio

Promptable object segmentation for images and videos using **SAM 2**
(Segment Anything Model 2).  Click a point or draw a bounding box on any
object, and the model produces a precise mask that can be propagated
across every frame of a video.

| Feature | Detail |
|---|---|
| **Model** | SAM 2.1 (Segment Anything Model 2) via HuggingFace |
| **Prompts** | Point clicks (foreground / background) and bounding boxes |
| **Video** | Mask propagation across all frames |
| **Exports** | Alpha masks, transparent cutouts (RGBA), overlays |
| **Modes** | Interactive demo, batch CLI, benchmark on DAVIS 2017 |
| **Works on** | Any local image or video — no dataset required |

---

## Architecture

Prompt UI and segmentation engine are **fully separated**:

```
┌──────────────┐        ┌──────────────┐
│  prompt_ui   │        │   engine     │
│ (OpenCV GUI) │  ───►  │ (SAM 2 Img)  │
│  points/box  │        │  set_image   │
│  collection  │        │  predict     │
└──────────────┘        └──────┬───────┘
                               │
                     ┌─────────┴─────────┐
                     │    controller      │
                     │  orchestrates      │
                     │  image + video     │
                     └─────────┬─────────┘
                               │
                     ┌─────────┴─────────┐
                     │   propagator      │
                     │  (SAM 2 Video)    │
                     │  extract frames   │
                     │  propagate masks  │
                     └─────────┬─────────┘
                               │
                     ┌─────────┴─────────┐
                     │     export        │
                     │  masks / cutouts  │
                     │  overlays / video │
                     └───────────────────┘
```

---

## Quick Start

### Installation

```bash
# 1. Install SAM 2 (requires Python ≥ 3.10, PyTorch ≥ 2.5.1)
pip install git+https://github.com/facebookresearch/sam2.git

# 2. Install remaining dependencies
pip install -r "Interactive Video Object Cutout Studio/requirements.txt"
```

SAM 2 model weights are downloaded automatically from HuggingFace on
first run.

### Interactive Demo

```bash
# Segment an object in an image
python "Interactive Video Object Cutout Studio/Source Code/demo.py" \
    --source photo.jpg --save-cutout

# Segment & propagate in a video
python "Interactive Video Object Cutout Studio/Source Code/demo.py" \
    --source clip.mp4 --save-cutout --save-mask
```

**Controls in the prompt window:**

| Action | Effect |
|---|---|
| Left-click | Add foreground point (green) |
| Right-click | Add background point (red) |
| Shift + drag | Draw bounding box (cyan) |
| `u` | Undo last action |
| `c` | Clear all prompts |
| Enter | Confirm & segment |
| Esc | Cancel |

### Batch CLI

```bash
# Image with point prompt
python "Interactive Video Object Cutout Studio/Source Code/infer.py" \
    --source image.jpg --point 320,240 --save-cutout --save-mask

# Image with box prompt
python "Interactive Video Object Cutout Studio/Source Code/infer.py" \
    --source image.jpg --box 50,100,400,350 --save-overlay

# Video with point prompt (propagated from frame 0)
python "Interactive Video Object Cutout Studio/Source Code/infer.py" \
    --source video.mp4 --point 400,300 --save-cutout --save-mask

# Directory of images with box prompt
python "Interactive Video Object Cutout Studio/Source Code/infer.py" \
    --source images/ --box 100,100,500,400 --save-cutout
```

### Benchmark on DAVIS 2017

```bash
# Download DAVIS & evaluate (first-frame GT box → per-frame IoU)
python "Interactive Video Object Cutout Studio/Source Code/benchmark.py" \
    --eval --max-sequences 10
```

---

## CLI Reference

### `demo.py` — Interactive

| Argument | Default | Description |
|---|---|---|
| `--source` | *(required)* | Image or video path |
| `--config` | — | JSON/YAML config file |
| `--frame-idx` | `0` | Video frame to prompt on |
| `--save-mask` | off | Export binary alpha masks |
| `--save-cutout` | off | Export RGBA cutouts (transparent bg) |

### `infer.py` — Batch

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Image, directory, video, or webcam index |
| `--point` | — | Point prompt: `x,y` or `x,y;x2,y2` or `x,y,label` |
| `--box` | — | Box prompt: `x1,y1,x2,y2` |
| `--frame-idx` | `0` | Video frame for prompt |
| `--config` | — | JSON/YAML config file |
| `--no-display` | off | Suppress GUI window |
| `--save-mask` | off | Save alpha masks |
| `--save-cutout` | off | Save transparent cutouts |
| `--save-overlay` | off | Save overlay visualisations |

### `benchmark.py` — Evaluation

| Argument | Default | Description |
|---|---|---|
| `--eval` | off | Run DAVIS benchmark |
| `--max-sequences` | `10` | Max sequences to evaluate |
| `--force-download` | off | Force re-download DAVIS |

---

## Configuration

All parameters tuneable via `CutoutConfig` or a JSON/YAML file:

```json
{
    "model_id": "facebook/sam2.1-hiera-small",
    "device": null,
    "multimask_output": true,
    "mask_threshold": 0.0,
    "max_frames": 0,
    "frame_stride": 1,
    "overlay_alpha": 0.45,
    "overlay_color": [255, 144, 30]
}
```

### Model Variants

| Model ID | Size | Speed |
|---|---|---|
| `facebook/sam2.1-hiera-tiny` | 39 MB | Fastest |
| `facebook/sam2.1-hiera-small` | 46 MB | **Default** — good balance |
| `facebook/sam2.1-hiera-base-plus` | 81 MB | Higher quality |
| `facebook/sam2.1-hiera-large` | 224 MB | Best quality |

---

## Project Structure

```
Interactive Video Object Cutout Studio/
├── Source Code/
│   ├── config.py           # CutoutConfig dataclass
│   ├── engine.py           # SAM2Engine — image segmentation
│   ├── propagator.py       # VideoPropagator — video mask propagation
│   ├── prompt_ui.py        # PromptCollector — interactive OpenCV UI
│   ├── export.py           # Alpha masks, cutouts, overlays
│   ├── controller.py       # CutoutController orchestrator
│   ├── validator.py        # Input validation
│   ├── infer.py            # Batch CLI entry point
│   ├── demo.py             # Interactive demo script
│   ├── benchmark.py        # DAVIS 2017 evaluation
│   ├── modern.py           # CVProject registry entry
│   └── data_bootstrap.py   # DAVIS dataset download
├── requirements.txt
└── README.md
```

---

## Dataset

**DAVIS 2017 trainval 480p** — standard video object segmentation
benchmark with dense per-frame annotations.

- **Source:** ETH Zurich (CC BY-NC)
- **Size:** ~600 MB
- **Sequences:** 60+ video clips with pixel-accurate masks
- **Purpose:** Benchmarking only — the tool works on any local media

Auto-downloaded on first benchmark run via `data_bootstrap.py`.

---

## How It Works

### Image Segmentation

1. SAM 2 encodes the image into an embedding via the Hiera vision
   transformer.
2. User prompts (points / boxes) are passed to the mask decoder.
3. The model returns up to 3 mask candidates with confidence scores.
4. The best-scoring mask is selected for export.

### Video Propagation

1. Video frames are extracted to a temporary JPEG directory.
2. SAM 2's memory-augmented video predictor encodes all frames.
3. User prompts on a single frame seed the object's mask.
4. The predictor propagates the mask forward through all remaining
   frames using streaming memory attention.
5. Per-frame masks are exported as alpha masks, cutouts, or overlays.

---

## Requirements

```
torch>=2.5.1
torchvision>=0.20.1
opencv-python>=4.10.0,<5
numpy>=1.26.0
huggingface-hub>=0.20.0
```

SAM 2 must be installed separately:

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```
