"""Dataset bootstrap for Number Plate Reader Pro.

Downloads and prepares a public license plate detection dataset for
training when available and falls back to a synthetic YOLO-format
dataset with plate text metadata for local validation.

Usage::

    from data_bootstrap import ensure_plate_dataset

    data_root = ensure_plate_dataset()              # idempotent
    data_root = ensure_plate_dataset(force=True)    # force re-download
"""

from __future__ import annotations

import logging
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("plate_reader.data_bootstrap")

PROJECT_KEY = "number_plate_reader_pro"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SPLIT_SIZES = {
    "train": 16,
    "val": 4,
    "test": 4,
}

PLATE_TEXTS = [
    "KA01AB1234",
    "MH12DE1433",
    "DL8CAF5031",
    "TN10Q7788",
    "RJ14TC0911",
    "UP32HK4400",
    "GJ05MN8080",
    "AP09ZX1200",
    "WB20K3321",
    "HR26DL9001",
    "PB10ET7744",
    "TS11AA1900",
    "MP04CJ2201",
    "KL07BH7080",
    "CH01AX5500",
    "OD02PL6607",
    "CG15RX2088",
    "BR01DN3110",
    "GA03NT7722",
    "JK02AR1044",
    "AS06PT5566",
    "PY01CC7007",
    "SK08PL2020",
    "UK07HJ3330",
]


def ensure_plate_dataset(*, force: bool = False) -> Path:
    """Download and prepare the detection dataset (idempotent)."""
    ready_marker = DATA_ROOT / ".ready"
    data_yaml = DATA_ROOT / "data.yaml"

    if ready_marker.exists() and data_yaml.exists() and not force:
        log.info("[%s] Dataset ready at %s -- skipping", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    if force and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    download_error = ""

    try:
        from scripts.download_data import ensure_dataset as _ensure

        data_path = _ensure(PROJECT_KEY, force=force)
        public_yaml = data_path / "data.yaml"
        if public_yaml.exists():
            sample_count = _count_images(data_path)
            _write_dataset_info(
                source="roboflow-license-plate-recognition-rxg4e",
                license_note="See the upstream dataset page for current license terms.",
                sample_count=sample_count,
                fallback_used=False,
                download_error="",
            )
            ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            log.info("[%s] Public dataset ready at %s", PROJECT_KEY, data_path)
            return data_path
    except Exception as exc:
        download_error = str(exc)
        log.warning(
            "[%s] Public dataset bootstrap failed (%s); generating synthetic ALPR data",
            PROJECT_KEY,
            exc,
        )

    sample_count = _generate_synthetic_dataset(DATA_ROOT)
    _write_dataset_info(
        source="synthetic-license-plates-v1",
        license_note="Generated locally for testing and validation.",
        sample_count=sample_count,
        fallback_used=True,
        download_error=download_error,
    )
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


def _count_images(data_path: Path) -> int:
    count = 0
    for img in data_path.rglob("*"):
        if img.suffix.lower() in IMAGE_EXTS:
            count += 1
    return count


def _generate_synthetic_dataset(data_root: Path) -> int:
    rng = random.Random(42)
    annotations: dict[str, dict[str, object]] = {}
    total = 0

    for split_name in ("train", "val", "test"):
        images_dir = data_root / split_name / "images"
        labels_dir = data_root / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        count = SPLIT_SIZES[split_name]
        for index in range(count):
            plate_text = PLATE_TEXTS[total % len(PLATE_TEXTS)]
            image, box = _render_synthetic_scene(plate_text, rng, total)
            image_name = f"{split_name}_{index:04d}.jpg"
            label_name = f"{split_name}_{index:04d}.txt"
            image_path = images_dir / image_name
            label_path = labels_dir / label_name

            cv2.imwrite(str(image_path), image)
            label_path.write_text(_yolo_label(box, image.shape[1], image.shape[0]), encoding="utf-8")
            annotations[image_name] = {
                "split": split_name,
                "plate_text": plate_text,
                "box": list(box),
            }
            total += 1

    (data_root / "ocr_labels.json").write_text(
        __import__("json").dumps(annotations, indent=2),
        encoding="utf-8",
    )
    _write_data_yaml(data_root)
    return total


def _render_synthetic_scene(
    plate_text: str,
    rng: random.Random,
    seed: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    width = 960
    height = 540
    image = np.full((height, width, 3), 210, dtype=np.uint8)

    noise = np.random.RandomState(seed).randint(0, 10, image.shape, dtype=np.uint8)
    image = np.clip(image.astype(np.int16) - noise, 0, 255).astype(np.uint8)

    # Simple rear-view vehicle patch to keep the detector task easy.
    rear_w = rng.randint(520, 620)
    rear_h = rng.randint(240, 290)
    rear_x = width // 2 - rear_w // 2 + rng.randint(-20, 20)
    rear_y = height // 2 - rear_h // 2 + rng.randint(-15, 15)
    rear_color = (
        rng.randint(45, 95),
        rng.randint(55, 125),
        rng.randint(85, 165),
    )
    cv2.rectangle(image, (rear_x, rear_y), (rear_x + rear_w, rear_y + rear_h), rear_color, -1)
    cv2.rectangle(image, (rear_x + 35, rear_y + 25), (rear_x + rear_w - 35, rear_y + 95), (rear_color[0] + 20, rear_color[1] + 20, rear_color[2] + 20), -1)
    cv2.circle(image, (rear_x + 70, rear_y + rear_h - 45), 28, (40, 40, 200), -1)
    cv2.circle(image, (rear_x + rear_w - 70, rear_y + rear_h - 45), 28, (40, 40, 200), -1)

    plate_w = rng.randint(280, 330)
    plate_h = rng.randint(88, 110)
    plate_x = width // 2 - plate_w // 2 + rng.randint(-12, 12)
    plate_y = rear_y + rear_h // 2 - plate_h // 2 + rng.randint(-10, 10)
    box = (plate_x, plate_y, plate_x + plate_w, plate_y + plate_h)

    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (248, 248, 248), -1)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (25, 25, 25), 3)
    cv2.putText(
        image,
        plate_text,
        (box[0] + 16, box[1] + plate_h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.28,
        (5, 5, 5),
        3,
        cv2.LINE_AA,
    )

    return image, box


def _yolo_label(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x_c = ((x1 + x2) / 2) / width
    y_c = ((y1 + y2) / 2) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"0 {x_c:.6f} {y_c:.6f} {box_w:.6f} {box_h:.6f}\n"


def _write_data_yaml(data_root: Path) -> None:
    yaml_text = (
        f"path: {str(data_root).replace('\\', '/')}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names:\n"
        "  0: license_plate\n"
    )
    (data_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def _write_dataset_info(
    *,
    source: str,
    license_note: str,
    sample_count: int,
    fallback_used: bool,
    download_error: str,
) -> None:
    payload = {
        "dataset_key": PROJECT_KEY,
        "source": source,
        "license_note": license_note,
        "fallback_used": fallback_used,
        "sample_count": sample_count,
        "data_yaml": str((DATA_ROOT / 'data.yaml').resolve()),
        "ocr_labels_json": str((DATA_ROOT / 'ocr_labels.json').resolve()),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if download_error:
        payload["download_error"] = download_error
    (DATA_ROOT / "dataset_info.json").write_text(
        __import__("json").dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    path = ensure_plate_dataset()
    print(path)


if __name__ == "__main__":
    main()
