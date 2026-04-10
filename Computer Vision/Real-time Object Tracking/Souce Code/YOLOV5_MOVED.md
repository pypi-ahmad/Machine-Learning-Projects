# YOLOv5 Clone Relocated

The embedded YOLOv5 directories and files that were previously here have been moved to:

```
/legacy/yolov5_clones/realtime_object_tracking/
```

Moved items: `classify/`, `data/`, `models/`, `runs/`, `segment/`, `utils/`,
`benchmarks.py`, `detect.py`, `export.py`, `hubconf.py`, `setup.cfg`, `train.py`, `val.py`

Project-specific files remain: `webapp.py`, `best.pt`, `static/`, `templates/`,
`test images and videos/`, `uploads/`, `requirements.txt`, `How to run.txt`

**Note:** `webapp.py` calls `detect.py` via subprocess, which now lives in the legacy
directory. This project needs migration to `ultralytics` before it will run again.
