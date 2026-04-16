"""Sign Language Alphabet Recognizer -- training pipeline.

Workflow:
1. Download and prepare a public ASL alphabet subset.
2. Run hand-landmark detection on each image.
3. Convert landmarks to normalised feature vectors.
4. Train an MLP classifier via sklearn.
5. Evaluate on the prepared test split.
6. Save model and evaluation artifacts.
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
    from classifier import SignClassifier
    from config import ASL_STATIC_LABELS
    from data_bootstrap import ensure_sign_lang_dataset
    from evaluator import (
        evaluate_model,
        print_confusion_matrix,
        save_confusion_matrix_image,
    )
    from hand_detector import HandDetector

    # 1. Ensure dataset is available
    data_dir = ensure_sign_lang_dataset(force=force_download)
    print(f"Dataset ready: {data_dir}")

    processed_root = data_dir / "processed"
    train_root = processed_root / "train"
    test_root = processed_root / "test"
    if not train_root.exists():
        print(f"No processed/train directory found at {train_root}")
        sys.exit(1)

    # 2. Discover class directories
    labels = [
        label
        for label in ASL_STATIC_LABELS
        if (train_root / label).exists() or (test_root / label).exists()
    ]
    if not labels:
        print("No valid letter directories found.")
        sys.exit(1)
    print(f"Found {len(labels)} classes: {', '.join(labels)}")

    label_to_idx = {l: i for i, l in enumerate(labels)}

    # 3. Extract features
    detector = HandDetector(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        static_image_mode=True,
    )
    detector.load()
    try:
        X_train, y_train, train_stats = _extract_split_features(
            train_root,
            labels,
            label_to_idx,
            detector,
            max_images_per_class,
        )
        X_test, y_test, test_stats = _extract_split_features(
            test_root,
            labels,
            label_to_idx,
            detector,
            max_images_per_class,
        )
    finally:
        detector.close()

    if len(X_train) == 0:
        print("No training features extracted -- aborting.")
        sys.exit(1)

    if len(X_test) == 0:
        from sklearn.model_selection import train_test_split

        print("No prepared test split detected; falling back to train_test_split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train,
            y_train,
            test_size=test_size,
            random_state=42,
            stratify=y_train,
        )
        test_stats["split_strategy"] = "fallback_train_test_split"

    print(
        f"\nTrain samples: {len(X_train)} ({train_stats['skipped']} skipped), "
        f"Test samples: {len(X_test)} ({test_stats['skipped']} skipped)"
    )

    # 4. Train
    clf = SignClassifier()
    metrics = clf.train(X_train, y_train, labels, max_iter=max_iter)
    print(f"Train accuracy: {metrics['train_accuracy']:.4f}")

    # 5. Save model
    clf.save(model_out)
    print(f"Model saved to {model_out}")

    # 6. Evaluate on test set
    report = evaluate_model(clf, X_test, y_test, labels)
    report["dataset_dir"] = str(data_dir)
    report["model_path"] = str(Path(model_out).resolve())
    report["train_samples"] = int(len(X_train))
    report["test_samples"] = int(len(X_test))
    report["kept_per_class"] = {
        "train": train_stats["kept_per_class"],
        "test": test_stats["kept_per_class"],
    }
    report["skipped_images"] = {
        "train": train_stats["skipped"],
        "test": test_stats["skipped"],
    }

    report_path = Path("eval_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cm_path = Path("confusion_matrix.png")
    save_confusion_matrix_image(report["confusion_matrix"], labels, str(cm_path))

    print(f"\nTest accuracy: {report['accuracy']:.4f}")
    print_confusion_matrix(report["confusion_matrix"], labels)
    print(f"Evaluation report saved to {report_path}")
    print(f"Confusion matrix image saved to {cm_path}")


def _extract_split_features(
    split_root: Path,
    labels: list[str],
    label_to_idx: dict[str, int],
    detector: HandDetector,
    max_images_per_class: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    import cv2

    from feature_extractor import extract_features

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    features_list: list[np.ndarray] = []
    targets_list: list[int] = []
    kept_per_class: dict[str, int] = {}
    skipped = 0

    if not split_root.exists():
        return (
            np.empty((0, 42), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {"kept_per_class": kept_per_class, "skipped": skipped},
        )

    for label in labels:
        class_dir = split_root / label
        if not class_dir.exists():
            kept_per_class[label] = 0
            continue

        images = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in exts
        )
        if max_images_per_class > 0:
            images = images[:max_images_per_class]

        print(f"  {split_root.name}/{label}: {len(images)} images ...", end=" ", flush=True)
        kept = 0
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                skipped += 1
                continue
            hand = detector.detect_for_image(frame)
            if hand is None:
                skipped += 1
                continue
            features_list.append(extract_features(hand))
            targets_list.append(label_to_idx[label])
            kept += 1
        kept_per_class[label] = kept
        print(f"{kept} OK")

    if not features_list:
        return (
            np.empty((0, 42), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {"kept_per_class": kept_per_class, "skipped": skipped},
        )

    return (
        np.array(features_list, dtype=np.float32),
        np.array(targets_list, dtype=np.int64),
        {"kept_per_class": kept_per_class, "skipped": skipped},
    )


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    main(force_download=force)
