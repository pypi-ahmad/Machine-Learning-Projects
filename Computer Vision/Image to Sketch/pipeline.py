"""
Modern CV Object Detection Pipeline (April 2026)

Primary : YOLO26m (Ultralytics) — real-time object detection / tracking.
Data    : Auto-downloads sample images at runtime; also scans local dir.
Timing  : Wall-clock per inference batch.
Export  : metrics.json with detection counts, classes, and timing.
"""
import os, json, time, warnings
from pathlib import Path
import urllib.request

warnings.filterwarnings("ignore")

TASK = "detect"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
]


def download_samples():
    save_dir = Path(SAVE_DIR) / "sample_images"
    save_dir.mkdir(exist_ok=True)
    paths = []
    for url in SAMPLE_URLS:
        fname = save_dir / url.split("/")[-1]
        if not fname.exists():
            urllib.request.urlretrieve(url, str(fname))
        paths.append(fname)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for ext in exts:
        paths.extend([p for p in Path(SAVE_DIR).rglob(f"*{ext}") if p not in paths])
    print(f"{len(paths)} images available")
    return paths


def run_detection(files):
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    out_dir = os.path.join(SAVE_DIR, "detections")
    os.makedirs(out_dir, exist_ok=True)
    metrics = {"model": "yolo26m", "task": "detect", "images": []}
    t0 = time.perf_counter()
    for f in files[:20]:
        results = model(str(f))
        for r in results:
            r.save(filename=os.path.join(out_dir, f.name))
            n_boxes = len(r.boxes) if r.boxes is not None else 0
            classes = []
            if r.boxes is not None:
                classes = [r.names[int(c)] for c in r.boxes.cls.tolist()]
            metrics["images"].append({"file": f.name, "detections": n_boxes,
                                       "classes": dict(sorted({c: classes.count(c) for c in set(classes)}.items()))})
            if n_boxes:
                print(f"  {f.name}: {n_boxes} objects detected")
    elapsed = time.perf_counter() - t0
    metrics["time_s"] = round(elapsed, 1)
    metrics["total_images"] = len(metrics["images"])
    metrics["total_detections"] = sum(i["detections"] for i in metrics["images"])
    print(f"  Detection: {metrics['total_images']} images, {metrics['total_detections']} objects in {elapsed:.1f}s")
    print(f"  Results saved to {out_dir}")
    return metrics


def run_tracking(files):
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    video_files = [f for f in files if f.suffix in (".mp4", ".avi", ".mov")]
    if not video_files:
        print("  No video files found. Running detection on images instead.")
        return run_detection(files)
    metrics = {"model": "yolo26m", "task": "track", "videos": []}
    t0 = time.perf_counter()
    for v in video_files[:3]:
        model.track(str(v), persist=True, save=True, project=SAVE_DIR, name="tracking")
        metrics["videos"].append({"file": v.name})
        print(f"  Tracked: {v.name}")
    elapsed = time.perf_counter() - t0
    metrics["time_s"] = round(elapsed, 1)
    print(f"  Tracking: {len(metrics['videos'])} videos in {elapsed:.1f}s")
    return metrics


def main():
    print("=" * 60)
    print(f"CV DETECTION | Task: {TASK} | Model: YOLO26m")
    print("=" * 60)
    files = download_samples()
    if TASK == "track":
        metrics = run_tracking(files)
    else:
        metrics = run_detection(files)

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
