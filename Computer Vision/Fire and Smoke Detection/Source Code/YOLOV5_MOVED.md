# YOLOv5 Clone Relocated

The embedded YOLOv5 directories and files that were previously here have been moved to:

```
/legacy/yolov5_clones/fire_and_smoke_detection/
```

Moved items: `.github/`, `classify/`, `data/`, `models/`, `runs/`, `segment/`, `utils/`,
`benchmarks.py`, `detect.py`, `export.py`, `hubconf.py`, `setup.cfg`, `train.py`, `val.py`

Project-specific files remain: `main.py`, `best.pt`, `*.jpg`, `requirements.txt`, `How to run.txt`

**Note:** `main.py` uses `torch.hub.load('.', 'custom', 'best.pt', source='local')` which
requires the YOLOv5 repo structure in the working directory. This project needs migration
to `ultralytics` before it will run again.
