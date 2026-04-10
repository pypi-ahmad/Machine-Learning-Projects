"""Entry point for Licence Plate Detector.

Usage:
    python main.py --image path/to/image.jpg
    python main.py --video path/to/video.mp4
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from deploy import main as deploy_main
    import argparse
    parser = argparse.ArgumentParser(description="Licence Plate Detector (YOLOv5 + EasyOCR)")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--video", type=str, default=None, help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    args = parser.parse_args()

    deploy_main(img_path=args.image, vid_path=args.video, vid_out=args.output)


if __name__ == "__main__":
    main()
