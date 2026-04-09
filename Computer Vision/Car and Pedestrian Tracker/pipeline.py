"""
Modern CV Object Detection Pipeline (April 2026)
Model: YOLO26m (Ultralytics) — auto-downloads model + sample images
Data: Auto-downloaded at runtime
"""
import os, warnings
from pathlib import Path
import urllib.request

warnings.filterwarnings("ignore")

TASK = "track"

SAMPLE_URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
]


def download_samples():
    save_dir = Path(os.path.dirname(__file__)) / "sample_images"
    save_dir.mkdir(exist_ok=True)
    paths = []
    for url in SAMPLE_URLS:
        fname = save_dir / url.split("/")[-1]
        if not fname.exists():
            urllib.request.urlretrieve(url, str(fname))
        paths.append(fname)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for ext in exts:
        paths.extend([p for p in Path(os.path.dirname(__file__)).rglob(f"*{ext}") if p not in paths])
    print(f"{len(paths)} images available")
    return paths


def run_detection(files):
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    save_dir = os.path.join(os.path.dirname(__file__), "detections")
    os.makedirs(save_dir, exist_ok=True)
    for f in files[:20]:
        results = model(str(f))
        for r in results:
            r.save(filename=os.path.join(save_dir, f.name))
            if r.boxes is not None:
                print(f"  ✓ {f.name}: {len(r.boxes)} objects detected")
    print(f"Results saved to {save_dir}")


def run_tracking(files):
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    video_files = [f for f in files if f.suffix in (".mp4", ".avi")]
    if not video_files:
        print("No video files found. Running detection on images instead.")
        run_detection(files)
        return
    for v in video_files[:3]:
        model.track(str(v), persist=True, save=True, project=os.path.dirname(__file__), name="tracking")
        print(f"  ✓ Tracked: {v.name}")


def main():
    print("=" * 60)
    print(f"CV DETECTION — YOLO26m | Task: {TASK}")
    print("=" * 60)
    files = download_samples()
    if TASK == "track": run_tracking(files)
    else: run_detection(files)


if __name__ == "__main__":
    main()
