"""
Modern CV Detection / Tracking Pipeline (April 2026)

Primary : YOLO26m (Ultralytics) for detection and tracking.
Export  : metrics.json with file-level detections + validation.json with output checks.
Data    : Auto-downloads demo files at runtime.
"""
import os, json, time, warnings
from pathlib import Path
import urllib.request

warnings.filterwarnings("ignore")

TASK = "track"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_samples():
    from pathlib import Path
    return [p for p in Path(SAVE_DIR).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mov")]


def run_detection(files):
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    image_files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    out_dir = os.path.join(SAVE_DIR, "detections")
    os.makedirs(out_dir, exist_ok=True)
    metrics = {"model": "yolo26m", "task": "detect", "images": []}
    t0 = time.perf_counter()
    for f in image_files[:20]:
        results = model(str(f))
        for r in results:
            r.save(filename=os.path.join(out_dir, f.name))
            n_boxes = len(r.boxes) if r.boxes is not None else 0
            classes = [int(b.cls) for b in r.boxes] if r.boxes is not None else []
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


def run_eda(files, save_dir):
    """Input file summary for detection."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"  Input files: {len(files)}")
    if files:
        total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        print(f"  Total size: {total_size / 1024:.1f} KB")
    print("EDA complete.")


def validate_results(metrics, files, save_dir):
    """Validate output coverage for detection / tracking demos."""
    validation = {
        "task": metrics.get("task", TASK),
        "input_files": len(files),
        "processed": int(metrics.get("total_images", len(metrics.get("videos", [])))),
        "time_s": round(float(metrics.get("time_s", 0)), 1),
    }
    if metrics.get("task") == "track":
        validation["processed"] = len(metrics.get("videos", []))
        validation["passed"] = validation["processed"] > 0 and validation["time_s"] >= 0
    else:
        validation["total_detections"] = int(metrics.get("total_detections", 0))
        validation["passed"] = validation["processed"] > 0 and validation["time_s"] >= 0
    out_path = os.path.join(save_dir, "validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"Validation saved to {out_path}")
    return validation


def main():
    print("=" * 60)
    print(f"CV DETECTION | Task: {TASK} | Model: YOLO26m")
    print("=" * 60)
    files = download_samples()
    run_eda(files, SAVE_DIR)
    if TASK == "track":
        metrics = run_tracking(files)
    else:
        metrics = run_detection(files)
    metrics["validation"] = validate_results(metrics, files, SAVE_DIR)

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
