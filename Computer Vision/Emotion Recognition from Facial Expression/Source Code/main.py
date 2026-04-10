"""Entry point for Emotion Recognition from facial expression.

Usage:
    python main.py            # run inference on test video
    python main.py --webcam   # use webcam instead
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse


def main():
    parser = argparse.ArgumentParser(description="Emotion Recognition from facial expression")
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of video file")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    args = parser.parse_args()

    if args.train:
        print("[INFO] Starting training...")
        import TrainEmotionDetector  # noqa: F401
    elif args.evaluate:
        print("[INFO] Starting evaluation...")
        import EvaluateEmotionDetector  # noqa: F401
    else:
        print("[INFO] Starting inference...")
        import TestEmotionDetector  # noqa: F401


if __name__ == "__main__":
    main()
