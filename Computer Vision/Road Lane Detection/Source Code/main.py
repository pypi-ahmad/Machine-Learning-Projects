"""Entry point for Road Lane Detection.

Usage:
    python main.py              # run video detection (default)
    python main.py --image      # run image detection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse


def main():
    parser = argparse.ArgumentParser(description="Road Lane Detection")
    parser.add_argument("--image", action="store_true", help="Run detection on image instead of video")
    args = parser.parse_args()

    if args.image:
        print("[INFO] Running lane detection on image...")
        import detection_on_image
        detection_on_image.main()
    else:
        print("[INFO] Running lane detection on video...")
        import detection_on_vid
        detection_on_vid.main()


if __name__ == "__main__":
    main()
