"""Sign Language Alphabet Recognizer — training pipeline.
"""Sign Language Alphabet Recognizer — training pipeline.

Workflow:
1. Load dataset images (auto-downloaded via bootstrap).
2. Run MediaPipe Hands on each image → extract landmarks.
3. Convert landmarks to normalised feature vectors.
4. Train an MLP classifier via sklearn.
5. Evaluate on a held-out test split.
6. Save model + evaluation report.
"""
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(
    force_download: bool = False,
    model_out: str = "model/sign_lang_clf.pkl",
    test_size: float = 0.2,
    max_iter: int = 500,
    max_images_per_class: int = 0,
) -> None:
    """End-to-end training entry point."""
    import cv2
    from sklearn.model_selection import train_test_split

    from classifier import SignClassifier
    from config import ASL_STATIC_LABELS
    from data_bootstrap import ensure_sign_lang_dataset
    from evaluator import evaluate_model
    from feature_extractor import extract_features
    from hand_detector import HandDetector

    # 1. Ensure dataset is available
    data_dir = ensure_sign_lang_dataset(force=force_download)
    print(f"Dataset ready: {data_dir}")

    # 2. Discover class directories
    # Expect structure: <data_dir>/processed/by_letter/<A|B|C|...>/*.png
    by_letter = data_dir / "processed" / "by_letter"
    if not by_letter.exists():
        print(f"No processed/by_letter directory found at {by_letter}")
        sys.exit(1)

    labels = sorted(
        d.name
        for d in by_letter.iterdir()
        if d.is_dir() and d.name in ASL_STATIC_LABELS
    )
    if not labels:
        print("No valid letter directories found.")
        sys.exit(1)
    print(f"Found {len(labels)} classes: {', '.join(labels)}")

    label_to_idx = {l: i for i, l in enumerate(labels)}

    # 3. Extract features
    detector = HandDetector(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        static_image_mode=True,
    )
    detector.load()  # explicit load for static-mode usage

    features_list: list[np.ndarray] = []
    targets_list: list[int] = []
    skipped = 0

    for label in labels:
        class_dir = by_letter / label
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in exts
        )
        if max_images_per_class > 0:
            images = images[:max_images_per_class]
        print(f"  {label}: {len(images)} images ...", end=" ", flush=True)
        cls_ok = 0
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                skipped += 1
                continue
            hand = detector.detect_for_image(frame)
            if hand is None:
                skipped += 1
                continue
            feat = extract_features(hand)
            features_list.append(feat)
            targets_list.append(label_to_idx[label])
            cls_ok += 1
        print(f"{cls_ok} OK")

    detector.close()

    if not features_list:
        print("No features extracted -- aborting.")
        sys.exit(1)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(targets_list, dtype=np.int64)
    print(f"\nTotal samples: {len(X)} ({skipped} images skipped)")

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )
    print(f"Train: {len(X_train)},  Test: {len(X_test)}")

    # 5. Train
    clf = SignClassifier()
    metrics = clf.train(X_train, y_train, labels, max_iter=max_iter)
    print(f"Train accuracy: {metrics['train_accuracy']:.4f}")

    # 6. Save model
    clf.save(model_out)
    print(f"Model saved to {model_out}")

    # 7. Evaluate on test set
    report = evaluate_model(clf, X_test, y_test, labels)
    report_path = Path("eval_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nTest accuracy: {report['accuracy']:.4f}")
    print(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    main(force_download=force)
