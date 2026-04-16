# Sports Ball Possession Tracker

YOLO26m detection + ByteTrack tracking for player/ball detection with nearest-player possession estimation.

## Classes

- `player` -- detected persons on the field
- `ball` -- the sports ball

## How Possession Works

1. Detect all players and the ball each frame via YOLO + tracking.
2. Find the nearest player within `possession_radius_px` of the ball centre.
3. That player gains possession.
4. Possession is "sticky" -- held for `possession_hold_frames` after the ball leaves proximity.
5. Cumulative possession time is tracked per player ID.

## Quick Start

```bash
# Train on sports dataset (synthetic fallback if real data unavailable)
python train.py --epochs 50 --batch 8 --imgsz 1280

# Evaluate detection quality
python evaluate.py --model runs/train/weights/best.pt

# Run possession tracking on a video
python infer.py --source match.mp4 --export-json results.json --export-csv timeline.csv
python infer.py --source match.mp4 --save-video output/annotated.mp4 --no-display
```

## Files

| File | Purpose |
|------|---------|
| config.py | PossessionConfig dataclass and YAML/JSON loader |
| data_bootstrap.py | Dataset download with synthetic fallback |
| tracker.py | YOLO detect + ByteTrack multi-object tracking |
| possession.py | Nearest-player possession estimator |
| train.py | Training via shared train_detection |
| evaluate.py | Validation metrics (mAP, per-class AP) |
| infer.py | CLI video inference pipeline |
| export.py | JSON + CSV possession timeline export |
| visualize.py | Overlay renderer (boxes, trails, possession bar) |
| modern.py | CVProject adapter for registry |
| possession_config.yaml | Sample YAML configuration |

## Configuration

Edit `possession_config.yaml` or pass CLI flags:

- `--model` -- YOLO weights (default: yolo26m.pt)
- `--imgsz` -- input resolution (default: 1280)
- `--conf` -- confidence threshold
- `--export-json` / `--export-csv` -- export paths
- `--save-video` -- save annotated video
- `--no-display` -- headless mode

## Possession tunables

- `possession_radius_px` -- max ball-to-player distance (default: 120px)
- `possession_hold_frames` -- sticky hold duration (default: 5 frames)
- `min_ball_conf` -- minimum ball detection confidence (default: 0.20)
