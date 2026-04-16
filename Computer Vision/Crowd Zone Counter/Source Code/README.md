# Crowd Zone Counter

YOLO26m person detection with configurable polygon zones, per-zone counting, and overcrowding alerts.

## Quick Start

```bash
# Train person detector (synthetic data auto-generated if real data unavailable)
python train.py --epochs 50 --batch 16 --imgsz 640

# Evaluate
python evaluate.py --model runs/train/weights/best.pt

# Run on video with zone counting
python infer.py --source crowd.mp4 --config crowd_config.yaml --no-display
python infer.py --source stadium.jpg --export-json output/results.json

# Webcam
python infer.py --source 0 --config crowd_config.yaml
```

## How It Works

1. YOLO detects all persons in each frame.
2. Each person's foot-point (bottom-centre of bbox) is tested against configured polygon zones.
3. Per-zone counts are computed; overcrowding alerts fire when count exceeds `max_capacity`.
4. Alert cooldown prevents repeated warnings.

## Files

| File | Purpose |
|------|---------|
| config.py | CrowdConfig / ZoneConfig dataclasses, YAML loader |
| data_bootstrap.py | Dataset download with synthetic fallback |
| detector.py | PersonDetector (YOLO, person-class filter) |
| zone_counter.py | Zone assignment, counting, overcrowding alerts |
| train.py | Training via shared train_detection |
| evaluate.py | Validation metrics |
| infer.py | CLI inference (image, video, webcam) |
| export.py | JSON + CSV export with alert log |
| visualize.py | Zone polygons, count labels, alert banners, dashboard |
| modern.py | CVProject adapter for registry |
| crowd_config.yaml | Sample zone configuration |

## Zone Configuration

Define zones in `crowd_config.yaml`:

```yaml
zones:
  - name: "Entrance"
    polygon: [[50,50], [400,50], [400,400], [50,400]]
    max_capacity: 15
  - name: "Main-Area"
    polygon: [[450,50], [900,50], [900,650], [450,650]]
    max_capacity: 40
```

Key tunables: `conf_threshold`, `alert_cooldown_frames`, `zone_alpha`, `show_alerts`.
