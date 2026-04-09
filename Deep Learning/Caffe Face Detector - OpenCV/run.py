#!/usr/bin/env python3
"""Project 26 -- Face Detection (OpenCV DNN)

Model  : OpenCV SSD face detector (pre-trained)
Task   : Inference-only face detection

Usage:
    python run.py --image path/to/image.jpg
    python run.py --smoke-test
"""
import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from shared.utils import ensure_dir, url_download, save_metrics

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"



TASK_TYPE = 'inference_only'

def get_model():
    ensure_dir(DATA_DIR)
    proto = url_download(PROTO, DATA_DIR, "deploy.prototxt")
    model = url_download(MODEL, DATA_DIR, "face_model.caffemodel")
    return cv2.dnn.readNetFromCaffe(str(proto), str(model))


def detect_faces(net, image_path, conf=0.5):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104, 177, 123))
    net.setInput(blob)
    dets = net.forward()
    faces = []
    for i in range(dets.shape[2]):
        c = dets[0, 0, i, 2]
        if c > conf:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            faces.append((x1, y1, x2, y2, float(c)))
    return img, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()

    ensure_dir(OUTPUT_DIR)
    net = get_model()

    if args.download_only:
        print("  Model files ready. Exiting (--download-only).")
        return

    if args.smoke_test:
        # Create a tiny test image and run inference
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(OUTPUT_DIR / "smoke_test_input.jpg"), dummy)
        result, faces = detect_faces(net, OUTPUT_DIR / "smoke_test_input.jpg", conf=0.3)
        cv2.imwrite(str(OUTPUT_DIR / "smoke_test_output.jpg"), result)
        save_metrics({"faces_detected": len(faces), "smoke_test": True}, OUTPUT_DIR)
        print(f"  [SMOKE] Inference OK — {len(faces)} face(s) on dummy image.")
        return

    if args.image:
        result, faces = detect_faces(net, args.image)
        out_path = OUTPUT_DIR / "detected.jpg"
        cv2.imwrite(str(out_path), result)
        save_metrics({"faces_detected": len(faces)}, OUTPUT_DIR)
        print(f"  Found {len(faces)} face(s) -> {out_path}")
    else:
        print("  Pass --image <path> to run. Example:")
        print("    python run.py --image data/sample.jpg")


if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
